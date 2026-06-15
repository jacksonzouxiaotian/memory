# Failure-Aware Narrow-Passage Navigation

This repository contains the paper-facing benchmark and supplementary
experiments for **failure-aware narrow-passage navigation**. The central question
is:

> Can a robot use passage-level geometry and prior traversal failures to avoid
> repeatedly entering geometrically infeasible narrow passages?

The main contribution is not a new low-level controller. It is a
failure-aware decision layer that can sit above classical, optimization-based,
or learning-based local planners.

## Method Overview

The paper method should be read as a modular stack:

```text
Passage Geometry Encoder
        ↓
Risk / Feasibility Estimator
        ↓
Passage-Centric Failure Memory
        ↓
Mode Decision: Commit / Explore / Recover / Reject
        ↓
Body-Aware Local Planner or Controller
        ↓
Execution + Failure Update
```

`OursPlanner` in `benchmark.py` is a lightweight memory-biased A* proxy used for
controlled experiments. The full paper method is broader: it combines passage
geometry, risk estimation, failure memory, mode decisions, and body-aware local
execution.

## What Is Implemented

| Component | Purpose | Main files |
|---|---|---|
| Controlled narrow-passage benchmark | Mechanism validation for failure memory and passage transfer | `paper_experiments.py`, `benchmark.py` |
| Recent local/trajectory planner proxies | Paper-facing baselines for RPP, ATR, DDP, RTEB, MPC uncertainty | `benchmark.py`, `paper_experiments.py` |
| Public dataset adapter | Run baselines on `map.npy` / `map.pgm` plus task CSV files | `dataset_adapter.py` |
| MovingAI public-map validation | External map validation plus hidden-blockage stress tests | `results_movingai_room_baseline_*.csv` |
| Supplementary RL | FM-RS-RL high-level mode-policy experiment | `rl/`, `RL_EXPERIMENT_DESIGN.md` |

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run all paper-facing deterministic experiments:

```bash
python paper_experiments.py --experiment all
```

Run individual experiment groups:

```bash
python paper_experiments.py --experiment trigger
python paper_experiments.py --experiment precision
python paper_experiments.py --experiment transfer
python paper_experiments.py --experiment baselines
```

Fast smoke test:

```bash
python paper_experiments.py \
  --experiment all \
  --batches 2 \
  --tasks 2 \
  --seeds 5 \
  --positive-cases 5 \
  --negative-cases 8
```

## Paper Experiment Map

| Paper item | Command | Outputs |
|---|---|---|
| Failure-memory trigger | `python paper_experiments.py --experiment trigger` | `results_failure_memory_trigger_summary.csv`, `results_failure_memory_trigger_trials.csv` |
| Memory precision/recall | `python paper_experiments.py --experiment precision` | `results_memory_precision_recall_summary.csv`, `results_memory_precision_recall_trials.csv` |
| Passage-memory transfer | `python paper_experiments.py --experiment transfer` | `results_passage_memory_transfer_summary.csv`, `results_passage_memory_transfer_trials.csv` |
| Latest baseline proxies | `python paper_experiments.py --experiment baselines` | `results_latest_baseline_summary.csv`, `results_latest_baseline_trials.csv` |
| Public dataset run | `python paper_experiments.py --experiment baselines --dataset path/to/dataset` | `results_latest_baseline_summary.csv`, `results_latest_baseline_trials.csv` |
| Supplementary RL | `python rl/train_fm_rs_rl.py`, `python rl/eval_fm_rs_rl.py` | `results_rl*/fm_rs_rl_*.csv` |

## Public Dataset Format

The public dataset adapter expects:

```text
path/to/dataset/
  map.npy                # 2D occupancy grid: 0=free, nonzero=occupied
  # or map.pgm           # PGM map: bright pixels free, dark/unknown occupied
  tasks.csv              # start_x,start_y,goal_x,goal_y
  dynamic_obstacles.csv  # optional: id,t,x,y,radius
```

Run:

```bash
python paper_experiments.py \
  --experiment baselines \
  --dataset path/to/dataset
```

For public grid maps, the adapter also creates a `dataset-hidden-blockage`
stress suite when possible. It keeps the public map and task endpoints, then
injects a small hidden truth-only blockage on the initially preferred route.
This simulates entrance misclassification or a locally infeasible passage.

The stress suite includes mechanism controls such as:

```text
RPP proxy
RPP proxy + random retry
RPP proxy + failure memory
```

These compare the same underlying planner with the same attempt budget,
separating the algorithmic effect of failure memory from generic replanning or
random perturbation.

## Current Public-Map Result

The repository includes a MovingAI Room benchmark validation run using
`32room_000.map` and medium-length public scenario tasks. Results are saved in:

```text
results_movingai_room_baseline_summary.csv
results_movingai_room_baseline_trials.csv
```

The key hidden-blockage stress result is:

| Method | ExecutableRate | CollisionRate | FailedPassageSelections |
|---|---:|---:|---:|
| RPP proxy | 0.0 | 1.0 | 10 |
| RPP proxy + random retry | 0.0 | 1.0 | 10 |
| RPP proxy + failure memory | 1.0 | 0.5 | 5 |
| DDP proxy | 0.0 | 1.0 | 10 |
| DDP proxy + failure memory | 1.0 | 0.5 | 5 |
| ATR proxy | 0.2 | 0.8889 | 8 |
| ATR proxy + failure memory | 1.0 | 0.4444 | 4 |
| RTEB proxy | 0.2 | 0.8889 | 8 |
| RTEB proxy + failure memory | 1.0 | 0.4444 | 4 |
| Ours full | 1.0 | 0.5 | 5 |

This result supports the mechanism claim: the gain comes from using failure
memory, not merely from retrying or random perturbation.

## Baseline Scope

The executable benchmark includes lightweight paper proxies for:

- RPP: path-tracking behavior.
- ATR: narrow-passage segment refinement.
- DDP: gradual dynamics-constraint relaxation.
- RTEB: recovery-oriented trajectory smoothing.
- MPC uncertainty: uncertainty-aware dynamic obstacle prediction.

These are not official ROS/Nav2 plugin implementations. They are controlled
benchmark proxies designed to compare planner behaviors inside the same
lightweight Python testbed.

For baseline rationale and paper table design, see:

```text
BASELINE_EXPERIMENT_PLAN.md
```

## Supplementary RL

The RL component is framed as:

```text
FM-RS-RL:
Failure-Memory Guided Risk-Sensitive Reinforcement Learning
```

It is a high-level policy over:

```text
Commit / Explore / Recover / Reject
```

It does not replace the local planner. It tests whether failure-memory features
improve high-level decisions under dynamic obstacles, pedestrian interference,
rough terrain, post-collision recovery, entrance misclassification, and repeated
failed attempts.

Run a lightweight training and evaluation smoke test:

```bash
python rl/train_fm_rs_rl.py --episodes 1000 --output-dir results_rl_smoke
python rl/eval_fm_rs_rl.py \
  --policy-path results_rl_smoke/fm_rs_rl_policy.json \
  --episodes 200 \
  --output results_rl_smoke/fm_rs_rl_eval_summary.csv
```

Current smoke result:

| Policy | SuccessRate | CollisionRate | RepeatFailureRate | MeanAttemptsPerTask |
|---|---:|---:|---:|---:|
| RulePolicy | 1.0 | 0.485 | 0.100 | 2.335 |
| RiskOnlyPolicy | 1.0 | 0.725 | 0.340 | 2.150 |
| FM-RS-RL rule | 0.810 | 0.260 | 0.085 | 3.045 |
| FM-RS-RL | 0.925 | 0.355 | 0.095 | 1.450 |

The intended paper claim is conservative: FM-RS-RL is supplementary evidence
that failure-memory features reduce repeated failed commitments under complex
disturbances.

For the full RL design, see:

```text
RL_EXPERIMENT_DESIGN.md
```

## Metrics

Important paper-facing metrics include:

- `ExecutableRate`
- `FailedShortPassageAttempts`
- `RepeatFailureRate`
- `FailedPassageSelections`
- `RejectRate`
- `MeanAttemptsPerTask`
- `MeanExecutableLength`
- `Precision`, `Recall`, `FalsePositiveRate`
- `CollisionRate`
- `RecoverySuccessRate`

These are more relevant to failure-aware quadruped passage navigation than
generic path length alone.

## Repository Layout

```text
benchmark.py
  Historical planners, map utilities, trajectory proxy baselines, dynamic tools.

paper_experiments.py
  Main deterministic paper-facing experiment entry point.

dataset_adapter.py
  Minimal public dataset loader for map.npy / map.pgm and task CSV files.

rl/
  Lightweight FM-RS-RL supplementary experiment code.

BASELINE_EXPERIMENT_PLAN.md
  Baseline rationale and experiment-table planning.

RL_EXPERIMENT_DESIGN.md
  Supplementary RL experiment design for the paper.

results_*.csv
  Generated paper-facing summaries and trial-level outputs.
```

## Verification

Useful checks:

```bash
python -m py_compile \
  benchmark.py paper_experiments.py dataset_adapter.py \
  rl/narrow_passage_env.py rl/memory_regularized_policy.py \
  rl/train_fm_rs_rl.py rl/eval_fm_rs_rl.py
```

`pytest` is listed in `requirements.txt`, but it may need to be installed in the
active environment before running the test suite.
