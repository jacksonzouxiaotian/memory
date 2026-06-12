import csv
from pathlib import Path

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
