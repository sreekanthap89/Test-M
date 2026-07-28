"""
utils.py — Shared utilities for the MEGA7 prediction learning project.
"""

import os
import itertools
import numpy as np
import pandas as pd
from datetime import datetime

# ── Constants ─────────────────────────────────────────────────────────────────
CSV_FILE  = "Emirates_Draw_MEGA7.csv"
WIN_COLS  = ["Winning Number 1", "2", "3", "4", "5", "6", "7"]
POOL      = 37
DRAW_SIZE = 7

# ── Session folder (shared across all steps when run via run_all.py) ──────────
_SESSION_FOLDER: str | None = None


def set_session_folder(path: str) -> None:
    """Called once by run_all.py to pin all scripts to a shared output folder."""
    global _SESSION_FOLDER
    _SESSION_FOLDER = path
    os.makedirs(path, exist_ok=True)


def get_run_folder(base: str = "runs") -> str:
    """
    Return the output folder for this run's charts.

    • When called from run_all.py  → returns the shared session folder.
    • When called from a standalone script → creates & returns a new
      timestamped folder:  runs/YYYY-MM-DD_HH-MM-SS/
    """
    if _SESSION_FOLDER is not None:
        return _SESSION_FOLDER
    timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_folder = os.path.join(base, timestamp)
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


# ── Centralized Data Loading ──────────────────────────────────────────────────

def load_data(path: str = CSV_FILE) -> pd.DataFrame:
    """
    Load and clean the MEGA7 CSV file (single source of truth for all steps).

    Returns a DataFrame with columns:
      - Date (datetime)
      - Winning Number 1 .. 7 (numeric)
      - numbers (list of 7 sorted ints)
    """
    df = pd.read_csv(path, skipfooter=1, engine="python")
    df = df[df["Date"].notna() & df["Date"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in WIN_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["numbers"] = df[WIN_COLS].apply(
        lambda row: sorted([int(v) for v in row if pd.notna(v)]), axis=1
    )
    df["sum"]    = df["numbers"].apply(sum)
    df["n_high"] = df["numbers"].apply(lambda nums: sum(1 for n in nums if n > 18))
    df["n_low"]  = df["numbers"].apply(lambda nums: sum(1 for n in nums if n <= 18))
    return df


# ── Centralized Covering Wheel Generator ──────────────────────────────────────

def generate_covering_wheel(candidates: list[int],
                            ticket_size: int = 7,
                            match_guarantee: int = 3) -> list[tuple]:
    """
    Greedy Set Cover algorithm to generate a mathematical wheeling system.
    Returns the minimum set of tickets to guarantee `match_guarantee`
    if `match_guarantee` winning numbers are in the `candidates` pool.
    """
    subsets_to_cover = set(itertools.combinations(candidates, match_guarantee))
    all_possible_tickets = list(itertools.combinations(candidates, ticket_size))

    ticket_coverage = {}
    for t in all_possible_tickets:
        ticket_coverage[t] = set(itertools.combinations(t, match_guarantee))

    chosen_tickets = []
    uncovered = set(subsets_to_cover)

    while uncovered and all_possible_tickets:
        best_ticket = max(all_possible_tickets,
                          key=lambda t: len(ticket_coverage[t] & uncovered))
        newly_covered = ticket_coverage[best_ticket] & uncovered

        if not newly_covered:
            break

        chosen_tickets.append(best_ticket)
        uncovered -= newly_covered
    return chosen_tickets


# ── Structured JSON & Persistence Export ─────────────────────────────────────

import json

def save_run_data(data: dict, filename: str = "predictions_summary.json", folder: str | None = None) -> str:
    """
    Saves predictions, probabilities, candidate pools, and model parameters
    into a structured JSON file inside the timestamped run directory.
    """
    out_dir = folder if folder else get_run_folder()
    out_path = os.path.join(out_dir, filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return out_path


