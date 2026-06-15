# Baseline and Supplementary Experiment Plan

This note extends the current paper-facing benchmark with stronger local-planner
and trajectory-planner baselines. The existing repository evaluates the core
failure-aware narrow-passage mechanism in a lightweight 2D grid benchmark. The
next paper version should keep those controlled experiments, then add ROS/Nav2
or simulator-level baselines for local trajectory execution.

## Baseline Groups

| Group | Baseline | Role in the paper | Why include it | Evaluation tier |
|---|---|---|---|---|
| Classical search | A*, Dijkstra, RRT, RRT*, Hybrid-A* | Historical/global-planning sanity checks | Shows that the map and footprint constraints are meaningful before local control is introduced | Current 2D benchmark |
| ROS local planner | DWB | Standard dynamic-window local trajectory rollout baseline | Common Nav2 controller; useful for showing whether short-horizon velocity sampling repeatedly commits to infeasible narrow openings | ROS2/Nav2 sim and, if available, robot |
| ROS local planner | TEB | Optimization-based timed trajectory baseline | Strong classical local planner for kinodynamic and obstacle constraints; good contrast against passage-memory recovery | ROS1/ROS2-compatible sim or reimplementation wrapper |
| ROS local planner | MPPI | Sampling-based model-predictive trajectory optimizer | Strong modern Nav2 controller; handles nonlinear costs and dynamic constraints better than DWB-style rollout | ROS2/Nav2 sim and robot |
| ROS local planner | RPP | Regulated Pure Pursuit tracking baseline | Lightweight Nav2 path tracker with speed regulation in constrained spaces; separates path-following quality from memory/mode decisions | ROS2/Nav2 sim and robot |
| Recent trajectory planner | Adaptive Trajectory Refinement (ATR) | Narrow-passage refinement baseline | Directly targets narrow passages by refining risky segments and correcting unsafe poses | Simulator-level reproduction or paper-reported comparison |
| Recent trajectory planner | Decremental Dynamics Planning (DDP) | Dynamics-aware global-to-local planning baseline | Addresses the gap between globally feasible-looking paths and locally executable robot dynamics | Simulator-level reproduction or paper-reported comparison |
| Recent trajectory planner | MPC with obstacle-prediction uncertainty | Dynamic-obstacle uncertainty baseline | Compares against a recent ROS2/Nav2-style MPC planner that reasons over predicted dynamic obstacle distributions | Gazebo/Nav2 simulation |
| Recent trajectory planner | Resilient TEB (RTEB) | Robust replanning/refinement baseline | Recent TEB extension for planner failure recovery, narrow passages, and replanning consistency | Simulator-level reproduction or paper-reported comparison |
| Proposed method | Failure-Aware Narrow-Passage Navigator | Main method | Uses passage geometry, failure memory, reject/recover decisions, and body-aware local execution | Current benchmark plus ROS/Nav2 sim/robot |

## Recommended Main Comparison Table

| Method | Planner family | Body footprint | Dynamic constraints | Dynamic obstacles | Uncertainty-aware | Failure memory | Reject/recover mode | Expected strength | Expected weakness |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| DWB | Dynamic window rollout | Partial via costmap footprint | Yes, short horizon | Reactive | No | No | No | Fast, standard Nav2 baseline | Can repeatedly choose deceptive narrow openings without memory |
| TEB | Timed elastic-band optimization | Yes | Yes | Limited/reactive depending setup | No | No | Limited | Strong smooth local trajectories | May converge poorly when the selected homotopy is infeasible |
| MPPI | Sampling-based model-predictive control | Yes | Yes | Reactive, cost-based | Limited unless extended | No | No | Strong modern local control under nonlinear costs | Still local and memoryless without task-level failure reasoning |
| RPP | Regulated path tracking | Yes via collision checking/costmap | Speed regulation, not full optimization | Reactive | No | No | No | Simple, robust path tracking in constrained spaces | Depends heavily on global path quality |
| ATR | Optimization/refinement | Yes | Depends on local optimizer | Primarily static/cluttered | No | No | No | Designed for narrow-passage trajectory repair | Does not model repeated failed-passage memory |
| DDP | Dynamics-aware planning paradigm | Yes | Strong | Depends on implementation | No | No | No | Reduces global/local dynamics mismatch | Not focused on semantic passage-level failure recall |
| MPC-uncertainty | MPC | Yes | Strong | Predictive | Yes | No | No | Strong dynamic-obstacle comparison | Focuses on moving-obstacle risk, not static deceptive passages |
| RTEB | Robust TEB variant | Yes | Yes | Replanning-capable | No | No | Recovery-oriented | Good recent recovery/refinement baseline | Recovery is trajectory-level, not passage-memory based |
| Ours | Failure-aware passage navigation | Yes | Via selected local controller | Compatible extension | Compatible extension | Yes | Yes | Avoids repeat infeasible passage attempts and supports recovery | Needs reliable passage anchoring and failure feedback |

## Experiment Matrix

| Experiment | Purpose | Required baselines | Optional strong baselines | Primary metrics |
|---|---|---|---|---|
| Narrow-passage feasibility | Show footprint-aware failure in passages that look traversable to point planners | A*, Hybrid-A*, DWB, TEB, MPPI, RPP, Ours | ATR, RTEB | ExecutableRate, collision rate, min clearance, traversal time |
| Repeated failed passage | Show benefit of remembering failed local regions | DWB, TEB, MPPI, RPP, Ours without memory, Ours full | ATR, RTEB | RepeatFailureRate, FailedShortPassageAttempts, MeanAttemptsPerTask |
| Passage-memory transfer | Show that passage-anchored memory transfers across map shifts/noise | Absolute-cell memory, passage-anchor memory, full method | DDP | ExecutableRate, FalsePositiveRate, FailedPassageSelections |
| Dynamic-obstacle stress | Show method remains useful when local execution is perturbed by pedestrians or moving obstacles | DWB, TEB, MPPI, RPP, Ours+same controller | MPC-uncertainty | collision rate, near-miss distance, recovery success, task time |
| Uncertain/deceptive obstacle | Test obstacle prediction uncertainty and entrance misclassification | MPPI, MPC-uncertainty, Ours+MPPI | ATR | unsafe-entry rate, reject accuracy, recovery latency |
| Real or high-fidelity robot run | Demonstrate deployability | DWB, MPPI, RPP, Ours+selected controller | TEB if integration is stable | success rate, interventions, traversal time, repeat failures |

## How to Report the Results

Use two tables in the paper rather than one overloaded table:

1. **Core method table**: current controlled benchmark results for memory trigger,
   memory precision/recall, and passage-memory transfer. This table should stay
   focused on `ExecutableRate`, `RepeatFailureRate`, `FailedPassageSelections`,
   `RejectRate`, and `MeanAttemptsPerTask`.
2. **Trajectory baseline table**: ROS/Nav2 or simulator results comparing DWB,
   TEB, MPPI, RPP, ATR/DDP/MPC-uncertainty/RTEB where available, and the proposed
   method wrapped around the same local controller.

The strongest paper narrative is not that the proposed method replaces MPPI or
TEB. The claim should be that failure-aware passage memory and mode decisions
are complementary to local trajectory optimization: they prevent repeated
commitment to the wrong passage before a local controller is asked to execute it.

## Reinforcement-Learning Supplement

The RL simulation should remain a supplementary experiment, not a replacement
for real or high-fidelity robot experiments. Its purpose is to test whether the
navigation policy generalizes under disturbances that are expensive or unsafe to
create on hardware.

| RL scenario | What it tests | Suggested success metric |
|---|---|---|
| Dynamic obstacles | Generalization under moving obstacle timing changes | collision-free success, near-miss distance |
| Pedestrian interference | Social/dynamic disturbance tolerance | yield/replan success, pedestrian collision rate |
| Uneven terrain | Robustness to locomotion and perception noise | traversal success, slip/stall recovery |
| Post-collision recovery | Whether the policy can recover after contact or blocked motion | recovery success within time budget |
| Entrance misclassification | Whether the policy rejects a falsely promising passage | unsafe-entry rate, correct rejection rate |
| Multiple failed attempts | Whether the policy changes strategy after repeated failure | attempts to success, repeated-failure rate |

Recommended RL protocol:

- Train only in simulation; report it as appendix or supplementary evidence.
- Randomize obstacle trajectories, passage width, terrain roughness, sensor noise,
  actuator delay, and false entrance detections.
- Compare `Ours + RL recovery policy` against `Ours without RL`, `MPPI`, and
  `RPP` in the same disturbed scenes.
- Keep the main claim conservative: RL validates robustness under complex
  disturbances; it does not replace the main physical or deterministic benchmark.

## Source Notes for Baseline Selection

- Nav2 documents DWB, MPPI, and Regulated Pure Pursuit as controller plugins:
  https://docs.nav2.org/configuration/packages/configuring-dwb-controller.html,
  https://docs.nav2.org/configuration/packages/configuring-mppic.html, and
  https://docs.nav2.org/configuration/packages/configuring-regulated-pp.html.
- Regulated Pure Pursuit is also described in Macenski et al., 2023:
  https://arxiv.org/abs/2305.20026.
- Adaptive Trajectory Refinement for narrow passages was released in 2025:
  https://arxiv.org/abs/2510.26142.
- Decremental Dynamics Planning was released in 2025 and targets the
  global-to-local dynamics mismatch: https://arxiv.org/abs/2503.20521.
- MPC with dynamic-obstacle prediction uncertainty in ROS2/Nav2 was released in
  2025: https://arxiv.org/abs/2504.19193.
- Resilient TEB was released in 2024 as a robust replanning/refinement baseline:
  https://arxiv.org/abs/2412.03174.
