import csv
import json
import shutil
from pathlib import Path

import pytest

import iterative_self_improving_engine as engine


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    source = tmp_path / "sample.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date", "Winning Number 1", "2", "3", "4", "5", "6", "7"])
        for idx in range(25):
            writer.writerow([f"2026-01-{idx + 1:02d}", idx + 1, idx + 2, idx + 3, idx + 4, idx + 5, idx + 6, idx + 7])
    return source


def test_split_holdout_rows_creates_train_and_holdout(sample_csv: Path, tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    holdout_path = tmp_path / "holdout.csv"

    train_rows, holdout_rows = engine.split_holdout_rows(sample_csv, holdout_size=5, train_path=train_path, holdout_path=holdout_path)

    assert len(train_rows) == 20
    assert len(holdout_rows) == 5
    assert train_path.exists()
    assert holdout_path.exists()
    assert holdout_rows[-1][0] == "2026-01-25"


def test_update_adaptive_weights_normalizes_values() -> None:
    weights = engine.update_adaptive_weights(None, {"Step 6": 0.8, "Step 14": 0.2})

    assert set(weights) == {"Step 6", "Step 14"}
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_run_iterative_self_improving_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path
    (workspace / "Emirates_Draw_MEGA7.csv").write_text("Date,Winning Number 1,2,3,4,5,6,7\n2026-01-01,1,2,3,4,5,6,7\n", encoding="utf-8")
    (workspace / "Emirates_Draw_MEGA7_backup.csv").write_text("", encoding="utf-8")

    monkeypatch.chdir(workspace)
    monkeypatch.setattr(engine, "build_prediction_from_current_state", lambda workspace, adaptive_weights=None: {"ticket": [1, 2, 3, 4, 5, 6, 7], "pool": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "prob": [0.1] * 37, "rank_percentile": 0.5})
    monkeypatch.setattr(engine, "run_full_pipeline", lambda workspace, dry_run=False: (True, "ok"))

    result = engine.run_iterative_self_improving(holdout_size=1, max_iterations=1, dry_run=True)

    assert result["iterations"][0]["ticket_hits"] >= 0
    assert result["iterations"][0]["success"] is True


def test_run_iterative_self_improving_reset_state_rebuilds_holdout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path
    backup_content = "Date,Winning Number 1,2,3,4,5,6,7\n2026-01-01,1,2,3,4,5,6,7\n2026-01-02,2,3,4,5,6,7,8\n"
    (workspace / "Emirates_Draw_MEGA7.csv").write_text(backup_content, encoding="utf-8")
    (workspace / "Emirates_Draw_MEGA7_backup.csv").write_text(backup_content, encoding="utf-8")
    (workspace / "Emirates_Draw_MEGA7_training.csv").write_text("stale", encoding="utf-8")
    (workspace / "Emirates_Draw_MEGA7_holdout.csv").write_text("stale", encoding="utf-8")

    monkeypatch.chdir(workspace)
    monkeypatch.setattr(engine, "build_prediction_from_current_state", lambda workspace, adaptive_weights=None: {"ticket": [1, 2, 3, 4, 5, 6, 7], "pool": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], "prob": [0.1] * 37, "rank_percentile": 0.5})
    monkeypatch.setattr(engine, "run_full_pipeline", lambda workspace, dry_run=False: (True, "ok"))

    result = engine.run_iterative_self_improving(holdout_size=1, max_iterations=1, dry_run=True, reset_state=True)

    assert len(result["iterations"]) == 1
    assert (workspace / "Emirates_Draw_MEGA7_holdout.csv").exists()


def test_run_iterative_self_improving_uses_step_scores_for_weight_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path
    (workspace / "Emirates_Draw_MEGA7.csv").write_text("Date,Winning Number 1,2,3,4,5,6,7\n2026-01-01,1,2,3,4,5,6,7\n2026-01-02,2,3,4,5,6,7,8\n", encoding="utf-8")
    (workspace / "Emirates_Draw_MEGA7_backup.csv").write_text("Date,Winning Number 1,2,3,4,5,6,7\n2026-01-01,1,2,3,4,5,6,7\n2026-01-02,2,3,4,5,6,7,8\n", encoding="utf-8")

    monkeypatch.chdir(workspace)
    monkeypatch.setattr(engine, "build_prediction_from_current_state", lambda workspace, adaptive_weights=None: {
        "ticket": [1, 2, 3, 4, 5, 6, 7],
        "pool": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        "prob": [0.1] * 37,
        "rank_percentile": 0.5,
        "step_tickets": {
            "Step 6": [1, 2, 3, 4, 5, 6, 7],
            "Step 7": [1, 2, 3, 4, 5, 6, 8],
            "Step 8": [1, 2, 3, 4, 5, 6, 9],
            "Step 9": [2, 3, 4, 5, 6, 7, 8],
            "Step 10": [1, 2, 3, 4, 5, 6, 10],
            "Step 11": [1, 2, 3, 4, 5, 6, 11],
            "Step 14 Meta": [1, 2, 3, 4, 5, 6, 7],
        },
    })
    monkeypatch.setattr(engine, "run_full_pipeline", lambda workspace, dry_run=False: (True, "ok"))

    result = engine.run_iterative_self_improving(holdout_size=1, max_iterations=1, dry_run=True)

    weights = result["iterations"][0]["adaptive_weights"]
    assert set(weights) == {"Step 6", "Step 7", "Step 8", "Step 9", "Step 10", "Step 11", "Step 14 Meta"}
    assert max(weights.values()) >= min(weights.values())
