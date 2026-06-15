# Supplementary RL Experiment Design

## Positioning

The reinforcement-learning component is a supplementary experiment, not a
replacement for the planner, passage memory, or hardware-facing navigation
stack. Its role is to test whether the proposed failure-aware decision logic
generalizes under disturbances that are difficult to exhaustively cover in
deterministic benchmark scenes.

We use the following framing:

```text
FM-RS-RL:
Failure-Memory Guided Risk-Sensitive Reinforcement Learning
for Narrow-Passage Quadruped Navigation
```

The policy is a high-level mode selector:

```text
action in {Commit, Explore, Recover, Reject}
```

It does not output continuous velocity commands. A body-aware local planner or
controller still executes the selected mode.

## System Role

```text
Passage Geometry Encoder
        ↓
Risk / Feasibility Estimator
        ↓
Passage-Centric Failure Memory
        ↓
FM-RS-RL Mode Policy
        ↓
Commit / Explore / Recover / Reject
        ↓
Body-Aware Local Planner or Controller
        ↓
Execution + Failure Update
```

This isolates the RL contribution: learning when to commit, probe, recover, or
reject by using geometry, estimated risk, and passage-level failure memory.

## MDP

### State

The policy receives structured navigation features rather than raw images:

| Symbol | Feature | Description |
|---|---|---|
| `width` | passage width | Estimated local opening width |
| `clearance` | clearance | Minimum lateral clearance |
| `body_margin` | body margin | Robot safety envelope |
| `risk` | risk score | Estimated probability of unsafe traversal |
| `passability` | passability score | Estimated feasibility |
| `memory_hit` | memory match score | Similarity to previously failed passages |
| `repeat_count` | repeat failures | Number of recent failed attempts |
| `stuck` | stuck score | Local oscillation or blocked-motion evidence |
| `dynamic_density` | dynamic clutter | Moving-obstacle interference |
| `terrain_roughness` | terrain roughness | Locomotion disturbance proxy |

The memory feature is the key algorithmic input:

```text
memory_hit = max_i sim(z_current, z_failed_i)
```

where `z_current` is the current passage embedding and `z_failed_i` are stored
failure-memory embeddings.

### Actions

| Action | Meaning |
|---|---|
| `Commit` | Traverse the passage with the local planner |
| `Explore` | Probe slowly to improve confidence |
| `Recover` | Back out or replan after blocked motion |
| `Reject` | Avoid this passage and select an alternative |

### Reward

The reward emphasizes task success and penalizes repeated failed commitment:

```text
R = R_goal + R_progress
    - lambda_collision R_collision
    - lambda_stuck R_stuck
    - lambda_repeat R_repeat
    - lambda_risk R_risk
    - lambda_reject R_false_reject
    - lambda_time R_time
```

The most important term is:

```text
R_repeat = 1[action = Commit] * memory_hit * 1[failure]
```

This discourages the policy from committing to passages that look similar to
past failures.

## Training Scenarios

The supplementary RL simulation samples disturbances from the following groups:

| Scenario | What it validates |
|---|---|
| Dynamic obstacles | Robustness to moving-obstacle timing changes |
| Pedestrian interference | Recovery under social/dynamic disturbance |
| Uneven terrain | Sensitivity to locomotion and perception noise |
| Post-collision recovery | Ability to recover after blocked motion |
| Entrance misclassification | Avoiding falsely promising passages |
| Multiple failed attempts | Strategy change after repeated failure |

## Baselines

| Method | Purpose |
|---|---|
| Rule policy | Hand-coded threshold mode decision |
| Risk-only policy | Uses risk/passability but not memory |
| Memory-only policy | Uses failure memory but not risk |
| FM-RS-RL | Uses risk, memory, and recovery state |

The strongest evidence comes from comparing risk-only and memory-aware policies
under the same simulated disturbances.

## Metrics

| Metric | Interpretation |
|---|---|
| `SuccessRate` | Task-level completion rate |
| `CollisionRate` | Unsafe traversal frequency |
| `RepeatFailureRate` | Re-entering a failed passage |
| `RejectAccuracy` | Correctly rejecting infeasible passages |
| `FalseRejectRate` | Rejecting feasible passages unnecessarily |
| `RecoverySuccessRate` | Successful recovery after stuck/collision |
| `MeanAttemptsPerTask` | Efficiency of mode decisions |
| `MeanReward` | RL objective value |

## Paper Claim

The RL claim should remain conservative:

> FM-RS-RL provides supplementary evidence that passage-level failure memory can
> improve high-level mode decisions under complex disturbances. It complements
> the deterministic planner and memory benchmark rather than replacing real
> robot or high-fidelity simulation experiments.

