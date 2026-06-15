#!/usr/bin/env python3

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from memory_regularized_policy import MemoryRegularizedSoftmaxPolicy
from narrow_passage_env import NarrowPassageMemoryEnv


def train(args):
    env = NarrowPassageMemoryEnv(seed=args.seed, max_steps=args.max_steps)
    obs, _ = env.reset()
    policy = MemoryRegularizedSoftmaxPolicy(
        obs_dim=len(obs),
        learning_rate=args.learning_rate,
        memory_beta=args.memory_beta,
        risk_alpha=args.risk_alpha,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    rows = []
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        trajectory = []
        total_reward = 0.0
        terminal_info = {}
        for _ in range(args.max_steps):
            action, probs = policy.act(obs, rng)
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            trajectory.append({
                "obs": obs,
                "action": action,
                "reward": reward,
                "probs": probs,
            })
            total_reward += reward
            obs = next_obs
            terminal_info = step_info
            if terminated or truncated:
                break
        policy.update(trajectory)
        rows.append({
            "Episode": episode,
            "Scenario": info["scenario"],
            "Reward": round(total_reward, 4),
            "Steps": len(trajectory),
            "Success": terminal_info.get("success", False),
            "Collision": terminal_info.get("collision", False),
            "RepeatFailure": terminal_info.get("repeat_failure", False),
            "CorrectReject": terminal_info.get("correct_reject", False),
            "FalseReject": terminal_info.get("false_reject", False),
            "RecoverySuccess": terminal_info.get("recovery_success", False),
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.save(output_dir / "fm_rs_rl_policy.json")
    with (output_dir / "fm_rs_rl_training.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--memory-beta", type=float, default=0.8)
    parser.add_argument("--risk-alpha", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results_rl")
    args = parser.parse_args()
    rows = train(args)
    print(rows[-1])


if __name__ == "__main__":
    main()

