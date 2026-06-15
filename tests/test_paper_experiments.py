import csv
from pathlib import Path

import numpy as np

from dataset_adapter import load_dataset
import paper_experiments as pe


def test_precision_recall_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary = pe.run_memory_precision_recall_statistics(
        positive_cases=4, negative_cases=6, seed=3
    )
    assert summary[0]["Cases"] == 10
    assert Path("results_memory_precision_recall_summary.csv").exists()
    with Path("results_memory_precision_recall_trials.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10


def test_transfer_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary = pe.run_passage_memory_transfer_statistics(
        seeds=4, memory_dropout=0.25, seed=5
    )
    variants = {row["Variant"] for row in summary}
    assert "Passage-anchor memory" in variants
    assert "Full method" in variants
    assert Path("results_passage_memory_transfer_summary.csv").exists()


def test_latest_baseline_smoke(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary = pe.run_latest_baseline_statistics(
        static_seeds=2, dynamic_seeds=1, missions_per_seed=1
    )
    variants = {row["Variant"] for row in summary}
    assert "RPP proxy" in variants
    assert "ATR proxy" in variants
    assert "DDP proxy" in variants
    assert "RTEB proxy" in variants
    assert "MPC uncertainty proxy" in variants
    assert Path("results_latest_baseline_summary.csv").exists()
    with Path("results_latest_baseline_trials.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert rows


def test_dataset_adapter_and_baseline_smoke(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    grid = np.zeros((12, 18), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    np.save(dataset_dir / "map.npy", grid)
    (dataset_dir / "tasks.csv").write_text(
        "start_x,start_y,goal_x,goal_y\n2,2,15,9\n"
    )
    (dataset_dir / "dynamic_obstacles.csv").write_text(
        "id,t,x,y,radius\nped0,0,8,1,0.8\nped0,8,8,10,0.8\n"
    )

    dataset = load_dataset(dataset_dir)
    assert len(dataset.tasks) == 1
    assert len(dataset.dynamic_obstacles) == 1

    monkeypatch.chdir(tmp_path)
    summary = pe.run_latest_baseline_statistics(dataset_path=dataset_dir)
    suites = {row["Suite"] for row in summary}
    assert "dataset-static" in suites
    assert "dataset-dynamic" in suites
    assert Path("results_latest_baseline_summary.csv").exists()
