#!/usr/bin/env python3

import argparse
import csv
import random
from pathlib import Path

import numpy as np

from memory_regularized_policy import (
    FMRSRulePolicy,
    MemoryRegularizedSoftmaxPolicy,
    RiskOnlyPolicy,
    RulePolicy,
)
from narrow_passage_env import NarrowPassageMemoryEnv


def evaluate_policy(policy, episodes, seed, max_steps):
    env = NarrowPassageMemoryEnv(seed=seed, max_steps=max_steps)
    rng = np.random.default_rng(seed)
    random.seed(seed)
    rows = []
    for episode in range(episodes):
        obs, info = env.reset(seed=seed + episode)
        total_reward = 0.0
        episode_info = {
            "success": False,
            "collision": False,
            "repeat_failure": False,
            "correct_reject": False,
            "false_reject": False,
            "recovery_success": False,
        }
        actions = []
        for _ in range(max_steps):
            action, _ = policy.act(obs, rng)
            actions.append(action)
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += reward
            for key in episode_info:
                episode_info[key] = episode_info[key] or step_info.get(key, False)
            if terminated or truncated:
                break
        rows.append({
            "Scenario": info["scenario"],
            "Reward": total_reward,
            "Steps": len(actions),
            "Success": episode_info["success"],
            "Collision": episode_info["collision"],
            "RepeatFailure": episode_info["repeat_failure"],
            "CorrectReject": episode_info["correct_reject"],
            "FalseReject": episode_info["false_reject"],
            "RecoverySuccess": episode_info["recovery_success"],
        })
    return rows


def summarize(name, rows):
    count = len(rows)
    return {
        "Policy": name,
        "Episodes": count,
        "SuccessRate": round(sum(row["Success"] for row in rows) / count, 4),
        "CollisionRate": round(sum(row["Collision"] for row in rows) / count, 4),
        "RepeatFailureRate": round(sum(row["RepeatFailure"] for row in rows) / count, 4),
        "RejectAccuracy": round(sum(row["CorrectReject"] for row in rows) / count, 4),
        "FalseRejectRate": round(sum(row["FalseReject"] for row in rows) / count, 4),
        "RecoverySuccessRate": round(sum(row["RecoverySuccess"] for row in rows) / count, 4),
        "MeanAttemptsPerTask": round(sum(row["Steps"] for row in rows) / count, 4),
        "MeanReward": round(sum(row["Reward"] for row in rows) / count, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-path", default="results_rl/fm_rs_rl_policy.json")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", default="results_rl/fm_rs_rl_eval_summary.csv")
    args = parser.parse_args()

    policies = {
        "RulePolicy": RulePolicy(),
        "RiskOnlyPolicy": RiskOnlyPolicy(),
        "FM-RS-RL rule": FMRSRulePolicy(),
    }
    if Path(args.policy_path).exists():
        policies["FM-RS-RL"] = MemoryRegularizedSoftmaxPolicy.load(args.policy_path)

    summary_rows = [
        summarize(
            name,
            evaluate_policy(policy, args.episodes, args.seed, args.max_steps),
        )
        for name, policy in policies.items()
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
