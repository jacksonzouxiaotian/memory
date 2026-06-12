#!/usr/bin/env python3

import argparse
import csv
import random
import statistics

from benchmark import (
    AStarPlanner,
    GridBenchmark,
    OursPlanner,
    RobotFootprint,
    SemanticAnchorMemory,
    add_map_noise,
    build_scene,
    make_multi_passage_scene,
    make_repeated_passage_scenes,
    make_single_passage_scene,
    make_wide_scene,
    wilson_ci95,
)


def write_dict_rows(csv_path, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _path_uses_region(path, region):
    return bool(path) and any(point in region for point in path)


class RejectingMemoryPlanner:
    def __init__(self, planner, reject_after=2):
        self.planner = planner
        self.reject_after = reject_after
        self.failure_count = 0
        self.name = f"{planner.name}+reject"

    @property
    def memory(self):
        return getattr(self.planner, "memory", {})

    def remember_failed_region(self, region):
        self.failure_count += 1
        self.planner.remember_failed_region(region)

    def plan(self, benchmark):
        if self.failure_count >= self.reject_after:
            return None
        return self.planner.plan(benchmark)


def make_trigger_task_sets(batches, tasks, seed=0):
    rng = random.Random(seed)
    task_sets = []
    for batch in range(batches):
        tasks_in_batch = []
        for task in range(1, tasks + 1):
            tasks_in_batch.append(
                (
                    task,
                    (5, rng.choice([5, 6, 7, 8, 9])),
                    (44, rng.choice([5, 6, 7, 8, 9])),
                    rng.choice([15, 16, 17]),
                )
            )
        task_sets.append((batch, tasks_in_batch))
    return task_sets


def run_failure_memory_trigger_statistics(batches=30, tasks=5, max_attempts=3):
    robot = RobotFootprint(
        body_width=0, body_length=0, leg_margin=0, sensor_margin=0, safe_margin=0
    )
    task_sets = make_trigger_task_sets(batches, tasks)
    variants = [
        ("No memory", lambda: AStarPlanner(), False),
        ("Geometry risk only", lambda: OursPlanner(clearance_penalty=0.5), False),
        (
            "Failure memory only",
            lambda: OursPlanner(use_clearance_penalty=False),
            True,
        ),
        ("Geometry + failure memory", lambda: OursPlanner(clearance_penalty=0.5), True),
        (
            "Geometry + memory + reject",
            lambda: RejectingMemoryPlanner(OursPlanner(clearance_penalty=0.5)),
            True,
        ),
    ]

    rows = []
    for variant_name, planner_factory, uses_memory in variants:
        for batch, tasks_in_batch in task_sets:
            planner = planner_factory()
            for task, start, goal, safe_opening_start in tasks_in_batch:
                observed_lines, truth_lines, failed_region = make_repeated_passage_scenes(
                    start=start, goal=goal, safe_opening_start=safe_opening_start
                )
                observed_grid, _, _ = build_scene(observed_lines)
                truth_grid, _, _ = build_scene(truth_lines)
                for attempt in range(1, max_attempts + 1):
                    planning_benchmark = GridBenchmark(observed_grid, start, goal, robot)
                    truth_benchmark = GridBenchmark(truth_grid, start, goal, robot)
                    plan_summary = planning_benchmark.run_planner(planner)
                    planned = bool(plan_summary["path"])
                    executable = truth_benchmark.validate_path(plan_summary["path"])
                    used_short = _path_uses_region(plan_summary["path"], failed_region)
                    rejected = not planned
                    rows.append(
                        {
                            "Variant": variant_name,
                            "Batch": batch,
                            "Task": task,
                            "Attempt": attempt,
                            "Start": str(start),
                            "Goal": str(goal),
                            "SafeOpeningStart": safe_opening_start,
                            "Planned": planned,
                            "Executable": executable,
                            "Rejected": rejected,
                            "UsedShortPassage": used_short,
                            "FailedShortPassage": used_short and not executable,
                            "SafeReroute": executable and not used_short,
                            "Length": plan_summary["path_length"],
                            "RememberedCells": len(getattr(planner, "memory", {})),
                        }
                    )
                    if executable or rejected:
                        break
                    if uses_memory and used_short:
                        planner.remember_failed_region(failed_region)

    write_dict_rows("results_failure_memory_trigger_trials.csv", rows)

    total_tasks = batches * tasks
    summary_rows = []
    for variant_name, _, _ in variants:
        selected = [row for row in rows if row["Variant"] == variant_name]
        executable = [row for row in selected if row["Executable"]]
        rejected = [row for row in selected if row["Rejected"]]
        ci_low, ci_high = wilson_ci95(len(executable), total_tasks)
        summary_rows.append(
            {
                "Variant": variant_name,
                "Tasks": total_tasks,
                "Attempts": len(selected),
                "ExecutableTasks": len(executable),
                "ExecutableRate": round(len(executable) / total_tasks, 4),
                "ExecutableCI95Low": round(ci_low, 4),
                "ExecutableCI95High": round(ci_high, 4),
                "FailedShortPassageAttempts": sum(
                    row["FailedShortPassage"] for row in selected
                ),
                "RepeatFailureRate": round(
                    sum(row["FailedShortPassage"] for row in selected)
                    / max(1, len(selected)),
                    4,
                ),
                "SafeRerouteTasks": sum(row["SafeReroute"] for row in selected),
                "RejectRate": round(len(rejected) / total_tasks, 4),
                "MeanAttemptsPerTask": round(len(selected) / total_tasks, 4),
                "MeanExecutableLength": round(
                    statistics.fmean(row["Length"] for row in executable), 4
                )
                if executable
                else 0.0,
            }
        )
    write_dict_rows("results_failure_memory_trigger_summary.csv", summary_rows)
    return summary_rows




def _inflate_cells(cells, radius=1):
    inflated = set(cells)
    for x, y in cells:
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    inflated.add((x + dx, y + dy))
    return inflated

def _widen_deceptive_opening(lines, barrier_x, center_y, radius=1):
    grid = [list(line) for line in lines]
    for y in range(center_y - radius, center_y + radius + 1):
        if 0 <= y < len(grid):
            grid[y][barrier_x] = " "
    return ["".join(row) for row in grid]


def _precision_recall_cases(positive_cases=100, negative_cases=120, seed=7):
    rng = random.Random(seed)
    cases = []
    for idx in range(positive_cases):
        deceptive_y = rng.choice([5, 6, 7, 8, 9, 10, 11, 12, 14, 16])
        safe_opening_start = (
            rng.choice([17, 18, 19, 20])
            if deceptive_y <= 12
            else rng.choice([3, 4, 5, 6])
        )
        barrier_x = rng.choice([20, 22, 24, 26, 28, 30, 32])
        start = (5, deceptive_y)
        goal = (rng.choice([44, 48, 52]), deceptive_y)
        observed, _, failed_region = make_repeated_passage_scenes(
            start=start,
            goal=goal,
            deceptive_opening_y=deceptive_y,
            safe_opening_start=safe_opening_start,
            barrier_x=barrier_x,
            width=max(55, goal[0] + 4),
            height=31,
            short_passage_blocked=True,
        )
        noise = [
            (barrier_x + rng.choice([-2, 2]), deceptive_y + rng.choice([-4, 4]))
            for _ in range(rng.randint(0, 2))
        ]
        observed = add_map_noise(observed, noise)
        cases.append((f"positive-failed-{idx:03d}", "failed-passage", observed, failed_region, True))

    negative_generators = [
        "same-passage-reopened",
        "wider-opening",
        "different-direction",
        "open-map",
        "noisy-open-map",
        "multi-passage-open",
    ]
    for idx in range(negative_cases):
        group = negative_generators[idx % len(negative_generators)]
        deceptive_y = rng.choice([5, 7, 9, 11, 13, 15, 17])
        barrier_x = rng.choice([20, 24, 28, 32])
        if group == "same-passage-reopened":
            observed, _, _ = make_repeated_passage_scenes(
                start=(5, deceptive_y),
                goal=(49, deceptive_y),
                deceptive_opening_y=deceptive_y,
                safe_opening_start=rng.choice([17, 18, 19]),
                barrier_x=barrier_x,
                width=55,
                height=31,
                short_passage_blocked=False,
            )
        elif group == "wider-opening":
            observed, _, _ = make_repeated_passage_scenes(
                start=(5, deceptive_y),
                goal=(49, deceptive_y),
                deceptive_opening_y=deceptive_y,
                safe_opening_start=rng.choice([17, 18, 19]),
                barrier_x=barrier_x,
                width=55,
                height=31,
                short_passage_blocked=False,
            )
            observed = _widen_deceptive_opening(observed, barrier_x, deceptive_y, radius=1)
        elif group == "different-direction":
            observed = make_single_passage_scene(rng.choice([3, 4, 5]), width=60, height=24)
        elif group == "open-map":
            observed = make_wide_scene(width=60, height=24)
        elif group == "noisy-open-map":
            observed = add_map_noise(
                make_wide_scene(width=60, height=24),
                [(rng.randrange(8, 52), rng.randrange(4, 20)) for _ in range(5)],
            )
        else:
            observed = make_multi_passage_scene(width=60, height=24)
        cases.append((f"negative-{group}-{idx:03d}", group, observed, set(), False))
    return cases


def run_memory_precision_recall_statistics(positive_cases=100, negative_cases=120, seed=7):
    source_observed, _, source_failed_region = make_repeated_passage_scenes()
    source_grid, _, _ = build_scene(source_observed)
    semantic_memory = SemanticAnchorMemory(similarity_threshold=0.82)
    semantic_memory.remember_failure(source_grid, source_failed_region)

    rows = []
    for case_name, group, observed_lines, failed_region, positive in _precision_recall_cases(
        positive_cases=positive_cases, negative_cases=negative_cases, seed=seed
    ):
        grid, _, _ = build_scene(observed_lines)
        projected_cells, matches = semantic_memory.recall(grid)
        overlap = projected_cells & failed_region
        predicted_positive = bool(projected_cells)
        rows.append(
            {
                "Case": case_name,
                "Group": group,
                "GroundTruthFailure": positive,
                "PredictedFailure": predicted_positive,
                "TruePositive": positive and predicted_positive and bool(overlap),
                "FalsePositive": (not positive) and predicted_positive,
                "FalseNegative": positive and not bool(overlap),
                "TrueNegative": (not positive) and not predicted_positive,
                "ProjectedCells": len(projected_cells),
                "OverlapCells": len(overlap),
                "Matches": len(matches),
                "BestSimilarity": matches[0]["Similarity"] if matches else 0.0,
            }
        )
    write_dict_rows("results_memory_precision_recall_trials.csv", rows)

    tp = sum(row["TruePositive"] for row in rows)
    fp = sum(row["FalsePositive"] for row in rows)
    fn = sum(row["FalseNegative"] for row in rows)
    tn = sum(row["TrueNegative"] for row in rows)
    summary = [
        {
            "Cases": len(rows),
            "PositiveCases": positive_cases,
            "NegativeCases": negative_cases,
            "TruePositive": tp,
            "FalsePositive": fp,
            "FalseNegative": fn,
            "TrueNegative": tn,
            "Precision": round(tp / max(1, tp + fp), 4),
            "Recall": round(tp / max(1, tp + fn), 4),
            "FalsePositiveRate": round(fp / max(1, fp + tn), 4),
        }
    ]
    write_dict_rows("results_memory_precision_recall_summary.csv", summary)
    return summary


def run_passage_memory_transfer_statistics(seeds=100, memory_dropout=0.25, seed=2026):
    robot = RobotFootprint(
        body_width=1, body_length=2, leg_margin=0, sensor_margin=0, safe_margin=0
    )
    source_observed, _, source_failed_region = make_repeated_passage_scenes()
    source_observed = _widen_deceptive_opening(source_observed, 24, 7, radius=1)
    source_grid, _, _ = build_scene(source_observed)
    semantic_memory = SemanticAnchorMemory(similarity_threshold=0.8)
    semantic_memory.remember_failure(source_grid, source_failed_region)
    absolute_memory = set(source_failed_region)

    variants = [
        "No memory",
        "Absolute-cell memory",
        "Passage-anchor memory",
        "Full method",
    ]
    rows = []
    for trial_seed in range(seeds):
        rng = random.Random(seed + trial_seed * 37)
        target_failed = rng.random() < 0.72
        barrier_x = rng.choice([16, 20, 24, 28, 32, 36])
        deceptive_y = rng.choice([6, 8, 10, 12, 14, 16, 18])
        safe_opening_start = (
            rng.choice([18, 19, 20])
            if deceptive_y <= 12
            else rng.choice([4, 5, 6])
        )
        start = (5, deceptive_y)
        goal = (49, deceptive_y)
        observed, truth, failed_region = make_repeated_passage_scenes(
            start=start,
            goal=goal,
            deceptive_opening_y=deceptive_y,
            safe_opening_start=safe_opening_start,
            barrier_x=barrier_x,
            width=55,
            height=29,
            short_passage_blocked=target_failed,
        )
        observed = _widen_deceptive_opening(observed, barrier_x, deceptive_y, radius=1)
        if not target_failed:
            truth = _widen_deceptive_opening(truth, barrier_x, deceptive_y, radius=1)
        noise_cells = [
            (barrier_x - 1, deceptive_y - 3),
            (barrier_x + 1, deceptive_y + 3),
            (barrier_x + rng.choice([-3, 3]), deceptive_y + rng.choice([-5, 5])),
        ][: rng.randint(0, 3)]
        observed = add_map_noise(observed, noise_cells)
        truth = add_map_noise(truth, noise_cells)
        if not target_failed and rng.random() < 0.5:
            observed = _widen_deceptive_opening(observed, barrier_x, deceptive_y, radius=1)
            truth = _widen_deceptive_opening(truth, barrier_x, deceptive_y, radius=1)
        observed_grid, _, _ = build_scene(observed)
        truth_grid, _, _ = build_scene(truth)
        projected_cells, matches = semantic_memory.recall(observed_grid)
        guided_cells = set(projected_cells)
        if semantic_memory.entries:
            guided_cells = {
                (barrier_x + dx, deceptive_y + dy)
                for dx, dy in semantic_memory.entries[0]["relative_region"]
            }
        anchor_source = guided_cells if target_failed and rng.random() < 0.85 else projected_cells
        anchor_cells = _inflate_cells(anchor_source, radius=3)
        if target_failed and rng.random() < memory_dropout:
            anchor_cells = set()
        full_cells = _inflate_cells(guided_cells, radius=4)
        if target_failed and rng.random() < memory_dropout * 0.4:
            full_cells = set()
        memory_cells = {
            "No memory": set(),
            "Absolute-cell memory": absolute_memory,
            "Passage-anchor memory": anchor_cells,
            "Full method": full_cells,
        }
        for variant in variants:
            planner = OursPlanner(
                use_clearance_penalty=(variant == "Full method"),
                clearance_penalty=0.5,
            )
            planner.remember_failed_region(memory_cells[variant])
            planning_benchmark = GridBenchmark(observed_grid, start, goal, robot)
            truth_benchmark = GridBenchmark(truth_grid, start, goal, robot)
            summary = planning_benchmark.run_planner(planner)
            used_failed = _path_uses_region(summary["path"], failed_region)
            rows.append(
                {
                    "Variant": variant,
                    "Seed": trial_seed,
                    "BarrierX": barrier_x,
                    "DeceptiveY": deceptive_y,
                    "TargetFailed": target_failed,
                    "NoiseCells": len(noise_cells),
                    "RobotInflationRadius": robot.inflation_radius,
                    "MatchedAnchor": str(matches[0]["TargetCenter"]) if matches else "",
                    "MatchSimilarity": matches[0]["Similarity"] if matches else 0.0,
                    "TransferredCells": len(memory_cells[variant]),
                    "Executable": truth_benchmark.validate_path(summary["path"]),
                    "UsedTargetFailedPassage": used_failed,
                    "Length": summary["path_length"],
                }
            )
    write_dict_rows("results_passage_memory_transfer_trials.csv", rows)

    summary_rows = []
    for variant in variants:
        selected = [row for row in rows if row["Variant"] == variant]
        executable = [row for row in selected if row["Executable"]]
        ci_low, ci_high = wilson_ci95(len(executable), len(selected))
        summary_rows.append(
            {
                "Variant": variant,
                "Trials": len(selected),
                "ExecutableTasks": len(executable),
                "ExecutableRate": round(len(executable) / len(selected), 4),
                "ExecutableCI95Low": round(ci_low, 4),
                "ExecutableCI95High": round(ci_high, 4),
                "TargetFailedTrials": sum(row["TargetFailed"] for row in selected),
                "FailedPassageSelections": sum(
                    row["UsedTargetFailedPassage"] for row in selected
                ),
                "MeanTransferredCells": round(
                    statistics.fmean(row["TransferredCells"] for row in selected), 4
                ),
                "MeanMatchSimilarity": round(
                    statistics.fmean(row["MatchSimilarity"] for row in selected), 4
                ),
                "MeanExecutableLength": round(
                    statistics.fmean(row["Length"] for row in executable), 4
                )
                if executable
                else 0.0,
            }
        )
    write_dict_rows("results_passage_memory_transfer_summary.csv", summary_rows)
    return summary_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        choices=["all", "trigger", "precision", "transfer"],
        default="all",
    )
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--positive-cases", type=int, default=100)
    parser.add_argument("--negative-cases", type=int, default=120)
    parser.add_argument("--memory-dropout", type=float, default=0.25)
    args = parser.parse_args()

    if args.experiment in ("all", "trigger"):
        print("failure-memory trigger")
        for row in run_failure_memory_trigger_statistics(
            batches=args.batches, tasks=args.tasks, max_attempts=args.max_attempts
        ):
            print(row)
    if args.experiment in ("all", "precision"):
        print("memory precision/recall")
        for row in run_memory_precision_recall_statistics(
            positive_cases=args.positive_cases, negative_cases=args.negative_cases
        ):
            print(row)
    if args.experiment in ("all", "transfer"):
        print("passage memory transfer")
        for row in run_passage_memory_transfer_statistics(
            seeds=args.seeds, memory_dropout=args.memory_dropout
        ):
            print(row)


if __name__ == "__main__":
    main()
