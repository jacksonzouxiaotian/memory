#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np

try:
    from .narrow_passage_env import ACTIONS, COMMIT
except ImportError:
    from narrow_passage_env import ACTIONS, COMMIT


class MemoryRegularizedSoftmaxPolicy:
    """Linear softmax policy with risk/memory commit regularization."""

    def __init__(
        self,
        obs_dim,
        action_dim=len(ACTIONS),
        learning_rate=0.03,
        memory_beta=0.8,
        risk_alpha=0.4,
        seed=0,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.memory_beta = memory_beta
        self.risk_alpha = risk_alpha
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0.0, 0.02, size=(obs_dim, action_dim))
        self.bias = np.zeros(action_dim, dtype=float)

    def action_probs(self, obs):
        logits = obs @ self.weights + self.bias
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def act(self, obs, rng):
        probs = self.action_probs(obs)
        return int(rng.choice(self.action_dim, p=probs)), probs

    def update(self, trajectory):
        returns = self._discounted_returns([step["reward"] for step in trajectory])
        if returns.std() > 1e-8:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        for step, advantage in zip(trajectory, returns):
            obs = step["obs"]
            action = step["action"]
            probs = step["probs"]
            grad_logits = -probs
            grad_logits[action] += 1.0

            memory_hit = obs[5]
            risk = obs[3]
            commit_penalty = self.memory_beta * memory_hit + self.risk_alpha * risk
            grad_logits[COMMIT] -= commit_penalty * probs[COMMIT]

            self.weights += self.learning_rate * advantage * np.outer(obs, grad_logits)
            self.bias += self.learning_rate * advantage * grad_logits

    @staticmethod
    def _discounted_returns(rewards, gamma=0.96):
        returns = np.zeros(len(rewards), dtype=float)
        running = 0.0
        for index in reversed(range(len(rewards))):
            running = rewards[index] + gamma * running
            returns[index] = running
        return returns

    def save(self, path):
        payload = {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "learning_rate": self.learning_rate,
            "memory_beta": self.memory_beta,
            "risk_alpha": self.risk_alpha,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        policy = cls(
            obs_dim=len(payload["weights"]),
            action_dim=len(payload["bias"]),
            learning_rate=payload.get("learning_rate", 0.03),
            memory_beta=payload.get("memory_beta", 0.8),
            risk_alpha=payload.get("risk_alpha", 0.4),
        )
        policy.weights = np.array(payload["weights"], dtype=float)
        policy.bias = np.array(payload["bias"], dtype=float)
        return policy


class RulePolicy:
    def act(self, obs, rng=None):
        risk = obs[3]
        memory_hit = obs[5]
        repeat_count = obs[6]
        stuck = obs[7]
        if stuck > 0.45 or repeat_count > 0.5:
            return 2, None
        if memory_hit > 0.7 or risk > 0.82:
            return 3, None
        if risk > 0.55:
            return 1, None
        return 0, None


class RiskOnlyPolicy:
    def act(self, obs, rng=None):
        risk = obs[3]
        stuck = obs[7]
        if stuck > 0.45:
            return 2, None
        if risk > 0.82:
            return 3, None
        if risk > 0.55:
            return 1, None
        return 0, None


class FMRSRulePolicy:
    """Deterministic failure-memory guided risk-sensitive mode policy."""

    def act(self, obs, rng=None):
        risk = obs[3]
        passability = obs[4]
        memory_hit = obs[5]
        repeat_count = obs[6]
        stuck = obs[7]
        dynamic_density = obs[8]
        terrain_roughness = obs[9]
        if stuck > 0.35:
            return 2, None
        if repeat_count > 0.5 and memory_hit > 0.55:
            return 3, None
        if memory_hit > 0.62:
            return 3, None
        if risk > 0.82 and passability < 0.45:
            return 3, None
        if dynamic_density > 0.55 or terrain_roughness > 0.6 or risk > 0.5:
            return 1, None
        return 0, None
