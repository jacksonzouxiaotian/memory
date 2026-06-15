#!/usr/bin/env python3

import random

import numpy as np


ACTIONS = ("Commit", "Explore", "Recover", "Reject")
COMMIT, EXPLORE, RECOVER, REJECT = range(len(ACTIONS))


class NarrowPassageMemoryEnv:
    """Small Gym-style environment for high-level narrow-passage decisions.

    The environment is intentionally lightweight and dependency-free. It models
    passage feasibility, memory hits, dynamic clutter, rough terrain, and stuck
    recovery at the mode-decision level rather than simulating continuous robot
    dynamics.
    """

    observation_keys = (
        "width",
        "clearance",
        "body_margin",
        "risk",
        "passability",
        "memory_hit",
        "repeat_count",
        "stuck",
        "dynamic_density",
        "terrain_roughness",
    )

    def __init__(self, seed=0, max_steps=8, scenario_mix=None):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.scenario_mix = scenario_mix or {
            "dynamic_obstacle": 1.0,
            "pedestrian": 1.0,
            "rough_terrain": 1.0,
            "post_collision": 1.0,
            "entrance_misclassification": 1.0,
            "repeat_failure": 1.0,
        }
        self.state = None
        self.steps = 0
        self.failed_once = False
        self.recovered = False

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.steps = 0
        self.failed_once = False
        self.recovered = False
        scenario = self._sample_scenario()
        feasible = self.rng.random() > 0.42
        width = self.rng.uniform(0.35, 1.0 if feasible else 0.72)
        body_margin = self.rng.uniform(0.42, 0.62)
        clearance = max(0.0, width - body_margin + self.rng.uniform(-0.08, 0.12))
        dynamic_density = self.rng.uniform(0.0, 0.25)
        terrain_roughness = self.rng.uniform(0.0, 0.25)
        stuck = 0.0
        memory_hit = self.rng.uniform(0.0, 0.25)

        if scenario == "dynamic_obstacle":
            dynamic_density = self.rng.uniform(0.45, 0.95)
        elif scenario == "pedestrian":
            dynamic_density = self.rng.uniform(0.55, 1.0)
            feasible = feasible and self.rng.random() > 0.2
        elif scenario == "rough_terrain":
            terrain_roughness = self.rng.uniform(0.5, 1.0)
        elif scenario == "post_collision":
            stuck = self.rng.uniform(0.45, 0.9)
            self.failed_once = True
        elif scenario == "entrance_misclassification":
            feasible = False
            width = self.rng.uniform(0.7, 1.0)
            clearance = self.rng.uniform(0.0, 0.14)
        elif scenario == "repeat_failure":
            feasible = False
            width = self.rng.uniform(0.82, 1.0)
            clearance = self.rng.uniform(0.28, 0.42)
            dynamic_density = self.rng.uniform(0.0, 0.18)
            terrain_roughness = self.rng.uniform(0.0, 0.18)
            memory_hit = self.rng.uniform(0.65, 1.0)

        risk = self._risk(width, clearance, dynamic_density, terrain_roughness)
        passability = max(0.0, min(1.0, 1.0 - risk + self.rng.uniform(-0.08, 0.08)))
        self.state = {
            "scenario": scenario,
            "feasible": feasible,
            "width": width,
            "clearance": clearance,
            "body_margin": body_margin,
            "risk": risk,
            "passability": passability,
            "memory_hit": memory_hit,
            "repeat_count": 1.0 if self.failed_once else 0.0,
            "stuck": stuck,
            "dynamic_density": dynamic_density,
            "terrain_roughness": terrain_roughness,
        }
        return self.observation(), {"scenario": scenario}

    def step(self, action):
        self.steps += 1
        reward = -0.02
        terminated = False
        info = {
            "success": False,
            "collision": False,
            "repeat_failure": False,
            "correct_reject": False,
            "false_reject": False,
            "recovery_success": False,
        }

        feasible = self.state["feasible"]
        risk = self.state["risk"]
        memory_hit = self.state["memory_hit"]
        stuck = self.state["stuck"]

        if action == COMMIT:
            fail_probability = min(
                0.95,
                risk * 0.65
                + self.state["dynamic_density"] * 0.15
                + self.state["terrain_roughness"] * 0.1
                + (0.35 if not feasible else 0.0),
            )
            fails = self.rng.random() < fail_probability
            if not fails and feasible:
                reward += 10.0
                info["success"] = True
                terminated = True
            else:
                reward -= 10.0
                info["collision"] = True
                if memory_hit > 0.5 or self.failed_once:
                    reward -= 6.0 * max(memory_hit, 0.5)
                    info["repeat_failure"] = True
                self.failed_once = True
                self.state["repeat_count"] += 1.0
                self.state["memory_hit"] = max(self.state["memory_hit"], 0.85)
                self.state["risk"] = max(self.state["risk"], 0.9)
        elif action == EXPLORE:
            reward += 0.4
            self.state["risk"] = max(0.0, self.state["risk"] - 0.18)
            self.state["passability"] = min(1.0, self.state["passability"] + 0.12)
            if not feasible:
                self.state["memory_hit"] = max(self.state["memory_hit"], 0.75)
        elif action == RECOVER:
            if self.failed_once or stuck > 0.35:
                reward += 3.0
                info["recovery_success"] = True
                self.recovered = True
                self.state["stuck"] = 0.0
                self.state["repeat_count"] = 0.0
                self.state["risk"] = max(0.0, self.state["risk"] - 0.2)
            else:
                reward -= 0.8
        elif action == REJECT:
            terminated = True
            if not feasible or memory_hit > 0.6 or risk > 0.75:
                reward += 2.0
                info["correct_reject"] = True
                info["success"] = True
            else:
                reward -= 1.0
                info["false_reject"] = True
        else:
            raise ValueError(f"unknown action: {action}")

        if action == COMMIT:
            reward -= 3.0 * risk
        reward -= 0.01 * self.steps

        truncated = self.steps >= self.max_steps and not terminated
        if truncated:
            reward -= 2.0

        return self.observation(), reward, terminated, truncated, info

    def observation(self):
        return np.array([self.state[key] for key in self.observation_keys], dtype=float)

    def _sample_scenario(self):
        names = list(self.scenario_mix)
        weights = [self.scenario_mix[name] for name in names]
        total = sum(weights)
        pick = self.rng.random() * total
        running = 0.0
        for name, weight in zip(names, weights):
            running += weight
            if pick <= running:
                return name
        return names[-1]

    @staticmethod
    def _risk(width, clearance, dynamic_density, terrain_roughness):
        geometry_risk = max(0.0, 0.65 - width) + max(0.0, 0.2 - clearance) * 1.5
        risk = geometry_risk + 0.35 * dynamic_density + 0.25 * terrain_roughness
        return max(0.0, min(1.0, risk))
