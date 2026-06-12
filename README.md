# Failure-Aware Narrow-Passage Navigation Benchmark

This repository contains the paper-facing benchmark for **failure-aware narrow-passage navigation**. The central question is:

> Can a quadruped robot use passage-level geometry and prior traversal failures to avoid repeatedly entering geometrically infeasible narrow passages?

The repository is intentionally scoped around narrow passages, body-aware planning, and failure memory. Older dynamic-obstacle, Kalman, semantic-migration, and Space-Time A* experiments are kept as optional extensions, but they are not the main paper story.

## Main Method Story

The paper method should be read as a modular navigation stack:

```text
Passage Geometry Encoder
        ↓
Risk / Feasibility Estimator
        ↓
Passage-Centric Failure Memory
        ↓
Mode Decision: Commit / Recover / Reject
        ↓
Body-Aware Local Planning or Control
```

`benchmark.py` still contains the historical planners and utilities. The reproducible paper-facing entry point is now `paper_experiments.py`.

## What This Repository Supports

Main experiments:

1. **Body-aware narrow-passage planning**
   Checks whether a robot footprint can actually traverse a narrow passage, rather than treating the robot as a point.

2. **Failure-memory ablation**
   Tests whether remembering failed passage regions reduces repeated infeasible attempts.

3. **Passage-memory transfer**
   Tests whether passage-anchored memory transfers better than absolute-cell memory under map shifts, noise, and hard negatives.

4. **Optional dynamic-memory extension**
   Dynamic obstacles, Kalman prediction, uncertainty radius calibration, and Space-Time A* are retained as appendix material.

## Repository Files

```text
benchmark.py
  Historical planner, map, memory, dynamic-obstacle, and visualization code.

paper_experiments.py
  Reproducible paper experiments with argparse.

results_*paper*.csv / results_failure_* / results_memory_* / results_passage_*
  Current paper experiment outputs.

figures/
  Existing visualizations from earlier benchmark runs.

BENCHMARK_REPORT.md
  Older benchmark report, useful as background but not the current paper narrative.
```

## Quick Start

Create the environment:

```bash
pip install -r requirements.txt
```

Run all paper experiments:

```bash
python paper_experiments.py --experiment all
```

Run each paper experiment separately:

```bash
python paper_experiments.py --experiment trigger
python paper_experiments.py --experiment precision
python paper_experiments.py --experiment transfer
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

## Paper Experiment Mapping

| Paper item | Script | Output |
|---|---|---|
| Failure-memory trigger | `paper_experiments.py --experiment trigger` | `results_failure_memory_trigger_summary.csv`, `results_failure_memory_trigger_trials.csv` |
| Memory precision / recall | `paper_experiments.py --experiment precision` | `results_memory_precision_recall_summary.csv`, `results_memory_precision_recall_trials.csv` |
| Passage-memory transfer | `paper_experiments.py --experiment transfer` | `results_passage_memory_transfer_summary.csv`, `results_passage_memory_transfer_trials.csv` |
| Optional old benchmark suite | `benchmark.py` | legacy `results_*.csv`, `figures/*.png` |

## Current Paper-Facing Metrics

The important robotics metrics are:

- `ExecutableRate`
- `FailedShortPassageAttempts`
- `RepeatFailureRate`
- `FailedPassageSelections`
- `RejectRate`
- `MeanAttemptsPerTask`
- `RobotInflationRadius`
- `MeanExecutableLength`
- `Precision`, `Recall`, `FalsePositiveRate`

These metrics are more relevant to a quadruped narrow-passage system than generic path length or planning success alone.

## Naming and Scope

`OursPlanner` in `benchmark.py` should be interpreted as a **memory-biased A*** baseline: A* with clearance and memory penalties. The full paper method is broader and should be described as:

```text
Failure-Aware Narrow-Passage Navigator
```

The full method combines passage geometry, risk estimation, failure memory, mode decision, and body-aware local execution. Do not present the memory-penalized A* alone as the entire contribution.

## Notes on Results

The updated `paper_experiments.py` avoids relying on very small or overly perfect sanity checks:

- Precision/recall now supports 100+ positive cases and 100+ hard negatives.
- Passage transfer includes body footprint, map noise, open-passage hard negatives, and configurable memory dropout.
- Transfer results are expected to show a realistic improvement, not a brittle `0% vs 100%` demo.

Example full run:

```bash
python paper_experiments.py \
  --experiment all \
  --batches 30 \
  --tasks 5 \
  --seeds 100 \
  --positive-cases 100 \
  --negative-cases 120 \
  --memory-dropout 0.25
```

## Optional Appendix Material

The following are useful, but should be treated as appendix or future-work material unless the paper explicitly studies dynamic planning:

- `ShortTermMemory`
- `OnlineTrajectoryPredictor`
- `KalmanTrajectoryPredictor`
- `SpaceTimeAStarPlanner`
- dynamic obstacle replanning
- nonlinear trajectory prediction
- uncertainty radius calibration

They are retained because they may support future extensions, but the main paper narrative should remain narrow-passage failure-aware navigation.
