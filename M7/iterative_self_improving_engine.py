from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import load_data

DATA_FILE = "Emirates_Draw_MEGA7.csv"
BACKUP_FILE = "Emirates_Draw_MEGA7_backup.csv"
HOLDOUT_FILE = "Emirates_Draw_MEGA7_holdout.csv"
TRAINING_FILE = "Emirates_Draw_MEGA7_training.csv"
WEIGHTS_FILE = "iterative_self_improving_weights.json"
LOG_FILE = "iterative_self_improving_log.json"


def resolve_workspace(workspace: Optional[Path] = None) -> Path:
    """Resolve the active project workspace, preferring the current working directory when it contains the data file."""
    if workspace is not None:
        return workspace
    cwd = Path.cwd()
    if (cwd / DATA_FILE).exists():
        return cwd
    return Path(__file__).resolve().parent


def split_holdout_rows(source_path: str | Path, holdout_size: int = 20,
                       train_path: str | Path | None = None,
                       holdout_path: str | Path | None = None) -> Tuple[List[List[str]], List[List[str]]]:
    """Split the last N rows from the source CSV into a holdout file and keep the rest as training data."""
    source_path = Path(source_path)
    if not source_path.exists() or source_path.stat().st_size == 0:
        raise ValueError(f"Source CSV is missing or empty: {source_path}")

    train_path = Path(train_path) if train_path is not None else source_path
    holdout_path = Path(holdout_path) if holdout_path is not None else Path(HOLDOUT_FILE)

    with source_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    if len(rows) <= 1:
        raise ValueError("Source CSV does not contain enough rows for a holdout split")

    header = rows[0]
    data_rows = rows[1:]
    if len(data_rows) == 1:
        train_rows = []
        holdout_rows = data_rows
    else:
        if holdout_size <= 0 or holdout_size >= len(data_rows):
            raise ValueError("holdout_size must be between 1 and len(data_rows) - 1")
        train_rows = data_rows[:-holdout_size]
        holdout_rows = data_rows[-holdout_size:]

    with train_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(train_rows)

    with holdout_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(holdout_rows)

    return train_rows, holdout_rows


def update_adaptive_weights(previous_weights: Optional[Dict[str, float]],
                            recent_rank_scores: Dict[str, float]) -> Dict[str, float]:
    """Reweight models so that better recent performers receive more influence."""
    if not recent_rank_scores:
        return previous_weights or {"Step 6": 0.2, "Step 7": 0.2, "Step 8": 0.2, "Step 9": 0.2, "Step 10": 0.2, "Step 11": 0.2, "Step 14 Meta": 0.2}

    if previous_weights is None:
        previous_weights = {name: 1.0 / len(recent_rank_scores) for name in recent_rank_scores}

    def normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
        values = list(scores.values())
        max_value = max(values) if values else 1.0
        if max_value <= 0:
            return {name: 1.0 / len(scores) for name in scores}
        return {name: float(value / max_value) for name, value in scores.items()}

    normalized = normalize_scores(recent_rank_scores)
    updated = {}
    for name in recent_rank_scores:
        prior = previous_weights.get(name, 1.0 / len(recent_rank_scores))
        learned = normalized[name]
        updated[name] = (prior * 0.4) + (learned * 0.6)

    total_updated = sum(updated.values())
    return {name: value / total_updated for name, value in updated.items()}


def parse_holdout_draw(row: List[str]) -> List[int]:
    return [int(value) for value in row[2:9]]


def run_full_pipeline(workspace: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """Invoke the existing project pipeline and capture its output."""
    if dry_run:
        return True, "Dry run enabled; pipeline execution skipped"

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    try:
        result = subprocess.run(
            [sys.executable, "run_all.py"],
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=1800,
        )
        output = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, output
    except (subprocess.TimeoutExpired, UnicodeError, OSError, ValueError) as exc:
        return False, f"pipeline_error: {exc}"


def build_prediction_from_current_state(workspace: Path, adaptive_weights: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """Use the existing self-improving test engine to build a structured prediction for the current training CSV."""
    sys.path.insert(0, str(workspace))
    from self_improving_test_engine import predict_for_dataset, get_rank_percentile

    df = load_data(str(workspace / DATA_FILE))
    pred = predict_for_dataset(df, custom_weights=adaptive_weights)
    meta_ticket = pred["meta_top7_ticket"]
    meta_pool = pred["meta_top14_pool"]
    meta_prob = pred["meta_prob"]
    return {
        "ticket": meta_ticket,
        "pool": meta_pool,
        "prob": meta_prob,
        "rank_percentile": get_rank_percentile(meta_prob, meta_ticket),
    }


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_iterative_self_improving(holdout_size: int = 20, max_iterations: int = 20, dry_run: bool = False,
                                 reset_state: bool = False, use_full_pipeline: bool = False) -> Dict[str, object]:
    workspace = resolve_workspace()
    backup_path = workspace / BACKUP_FILE
    data_path = workspace / DATA_FILE
    holdout_path = workspace / HOLDOUT_FILE
    training_path = workspace / TRAINING_FILE
    weights_path = workspace / WEIGHTS_FILE
    log_path = workspace / LOG_FILE

    if reset_state:
        for state_path in (training_path, holdout_path, weights_path, log_path):
            if state_path.exists():
                state_path.unlink()
        if backup_path.exists() and backup_path.stat().st_size > 0:
            shutil.copyfile(backup_path, data_path)
        elif data_path.exists():
            shutil.copyfile(data_path, backup_path)

    if not backup_path.exists():
        shutil.copyfile(data_path, backup_path)

    if reset_state or not training_path.exists() or not holdout_path.exists():
        source_for_split = backup_path if backup_path.exists() and backup_path.stat().st_size > 0 else data_path
        split_holdout_rows(source_for_split,
                           holdout_size=holdout_size,
                           train_path=training_path,
                           holdout_path=holdout_path)
        shutil.copyfile(training_path, data_path)

    if weights_path.exists():
        with weights_path.open("r", encoding="utf-8") as handle:
            adaptive_weights = json.load(handle)
    else:
        adaptive_weights = None

    with holdout_path.open("r", encoding="utf-8", newline="") as handle:
        holdout_rows = list(csv.reader(handle))[1:]

    iterations: List[Dict[str, object]] = []
    recent_rank_scores: Dict[str, float] = {}

    for iteration_idx, row in enumerate(holdout_rows[:max_iterations], start=1):
        target_date = row[1]
        actual_draw = parse_holdout_draw(row)

        if use_full_pipeline:
            run_ok, run_output = run_full_pipeline(workspace, dry_run=dry_run)
        else:
            run_ok, run_output = True, "lightweight_mode: using iterative prediction engine only"
        row_prediction = build_prediction_from_current_state(workspace, adaptive_weights=adaptive_weights)
        prediction_ticket = row_prediction["ticket"]
        prediction_pool = row_prediction["pool"]
        prediction_rank = row_prediction["rank_percentile"]
        ticket_hits = len(set(prediction_ticket) & set(actual_draw))
        pool_hits = len(set(prediction_pool) & set(actual_draw))
        success = ticket_hits >= 1 or pool_hits >= 3

        step_scores = {
            "Step 6": float(max(0.0, 3.0 - abs(ticket_hits - 1))) if ticket_hits >= 0 else 0.0,
            "Step 7": float(max(0.0, 3.0 - abs(pool_hits - 2))) if pool_hits >= 0 else 0.0,
            "Step 8": float(max(0.0, 3.0 - abs(ticket_hits - 2))) if ticket_hits >= 0 else 0.0,
            "Step 9": float(max(0.0, 3.0 - abs(pool_hits - 3))) if pool_hits >= 0 else 0.0,
            "Step 10": float(max(0.0, 3.0 - abs(ticket_hits - 1))) if ticket_hits >= 0 else 0.0,
            "Step 11": float(max(0.0, 3.0 - abs(pool_hits - 2))) if pool_hits >= 0 else 0.0,
            "Step 14 Meta": float(max(0.0, 4.0 - abs(prediction_rank / 10.0))) if prediction_rank is not None else 0.0,
        }
        recent_rank_scores = {**recent_rank_scores, **step_scores}
        adaptive_weights = update_adaptive_weights(adaptive_weights, recent_rank_scores)

        iteration_record = {
            "iteration": iteration_idx,
            "date": target_date,
            "actual_draw": actual_draw,
            "predicted_ticket": prediction_ticket,
            "predicted_pool": prediction_pool,
            "ticket_hits": ticket_hits,
            "pool_hits": pool_hits,
            "rank_percentile": round(prediction_rank, 2),
            "success": success,
            "adaptive_weights": adaptive_weights,
            "run_all_ok": run_ok,
            "run_all_output_excerpt": (run_output[:4000] if run_output else ""),
        }
        iterations.append(iteration_record)
        save_json(weights_path, adaptive_weights)
        save_json(log_path, {"iterations": iterations, "adaptive_weights": adaptive_weights})

        with data_path.open("r", encoding="utf-8", newline="") as handle:
            current_rows = list(csv.reader(handle))

        current_rows.append([target_date, *actual_draw])
        with data_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerows(current_rows)
        shutil.copyfile(data_path, training_path)

    aggregate = {
        "iterations_completed": len(iterations),
        "success_rate": round(sum(1 for item in iterations if item["success"]) / len(iterations), 3) if iterations else 0.0,
        "avg_ticket_hits": round(sum(item["ticket_hits"] for item in iterations) / len(iterations), 3) if iterations else 0.0,
        "avg_pool_hits": round(sum(item["pool_hits"] for item in iterations) / len(iterations), 3) if iterations else 0.0,
        "best_rank_percentile": round(min(item["rank_percentile"] for item in iterations), 3) if iterations else 0.0,
        "worst_rank_percentile": round(max(item["rank_percentile"] for item in iterations), 3) if iterations else 0.0,
    }
    return {"iterations": iterations, "adaptive_weights": adaptive_weights, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the iterative self-improving MEGA7 learning loop")
    parser.add_argument("--holdout-size", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true", help="Skip the expensive pipeline run")
    parser.add_argument("--reset-state", action="store_true", help="Reset holdout/training state before starting")
    parser.add_argument("--full-pipeline", action="store_true", help="Invoke run_all.py on every iteration")
    args = parser.parse_args()

    result = run_iterative_self_improving(holdout_size=args.holdout_size, max_iterations=args.max_iterations,
                                         dry_run=args.dry_run, reset_state=args.reset_state,
                                         use_full_pipeline=args.full_pipeline)
    summary = {
        "aggregate": result.get("aggregate", {}),
        "latest_iteration": result["iterations"][-1] if result["iterations"] else None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
