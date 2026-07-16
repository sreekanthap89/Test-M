"""
run_all.py — Run all 7 EASY6 prediction steps in sequence.

Usage:
    python run_all.py

What it does:
  1. Creates ONE shared timestamped output folder (runs/YYYY-MM-DD_HH-MM-SS/)
  2. Runs steps 01 → 07 in order, all charts saved to that folder
  3. Prints a clean final summary with the predicted ticket
"""

import os
import sys
import time
import importlib.util
import traceback
from datetime import datetime

# ── Force UTF-8 output on Windows ────────────────────────────────────────────
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Non-interactive matplotlib backend (no pop-up windows) ───────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda: None          # silence any plt.show() calls inside steps

# ── Create ONE shared folder for this entire run ─────────────────────────────
import utils
timestamp  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_FOLDER = os.path.join("runs", timestamp)
utils.set_session_folder(RUN_FOLDER)

# ── Step definitions ──────────────────────────────────────────────────────────
STEPS = [
    ("Step 1 — Data Exploration",        "01_data_explorer.py"),
    ("Step 2 — Frequency Analysis",      "02_frequency_analysis.py"),
    ("Step 3 — Probability Distributions","03_probability_distributions.py"),
    ("Step 4 — Monte Carlo Simulation",  "04_monte_carlo_simulation.py"),
    ("Step 5 — Markov Chain",            "05_markov_chain.py"),
    ("Step 6 — Prediction Report",       "06_prediction_report.py"),
    ("Step 7 — Advanced Prediction",     "07_advanced_prediction.py"),
    ("Step 8 — Deep Learning and Wheeling", "08_deep_learning_and_wheeling.py"),
]


def run_step(label: str, filename: str) -> bool:
    """Import and run a single step. Returns True on success."""
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {label}")
    print(f"  File: {filename}")
    print(bar)
    t0 = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location(
            filename.replace(".py", ""), filename
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.main()
        elapsed = time.perf_counter() - t0
        print(f"\n  [OK] Finished in {elapsed:.1f}s")
        return True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"\n  [FAILED] {filename} crashed after {elapsed:.1f}s")
        traceback.print_exc()
        return False


def collect_results() -> dict:
    """
    Re-import step 7 (the advanced prediction) in silent mode and
    extract the final prediction values without re-running everything.
    """
    import pandas as pd
    import numpy as np

    CSV_FILE = "Emirates_Draw_EASY6.csv"
    WIN_COLS = ["Winning Number 1", "2", "3", "4", "5", "6"]
    POOL, DRAW_SIZE, SEED = 40, 6, 42

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_FILE, skipfooter=1, engine="python")
    df = df[df["Date"].notna() & df["Date"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in WIN_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["numbers"] = df[WIN_COLS].apply(
        lambda row: sorted([int(v) for v in row if pd.notna(v)]), axis=1
    )
    df["sum"]    = df["numbers"].apply(sum)
    df["n_high"] = df["numbers"].apply(lambda nums: sum(1 for n in nums if n > 20))
    df["n_low"]  = df["numbers"].apply(lambda nums: sum(1 for n in nums if n <= 20))

    # ── Pull pre-computed results from step 7 module (already ran) ────────────
    spec = importlib.util.spec_from_file_location("step7", "07_advanced_prediction.py")
    mod7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod7)

    p1 = mod7.phase1_weighted_probability(df, recent_n=30)
    p3 = mod7.phase3_number_markov(df)
    p4_tuple = mod7.phase4_feedback_loop(df, p1, lookback=5)
    p4_adj, sum_pat, zone_pat, bal_pat = p4_tuple

    W1, W3, W4 = 0.40, 0.35, 0.25
    ens = W1 * p1 + W3 * p3 + W4 * p4_adj
    ens /= ens.sum()

    rng     = np.random.default_rng(SEED)
    numbers = np.arange(1, POOL + 1)
    mc_freq = np.zeros(POOL)
    for _ in range(100_000):
        draw = rng.choice(numbers, size=DRAW_SIZE, replace=False, p=ens)
        for n in draw:
            mc_freq[n - 1] += 1

    top6   = sorted((np.argsort(mc_freq)[::-1][:6] + 1).tolist())
    top1   = int(np.argmax(mc_freq) + 1)

    # raw freq stats
    raw_freq  = np.zeros(POOL)
    for row in df["numbers"]:
        for n in row:
            raw_freq[n - 1] += 1
    expected   = len(df) * DRAW_SIZE / POOL
    deviations = ((raw_freq - expected) / expected) * 100
    hot_nums   = sorted((numbers[deviations >= 20]).tolist())
    cold_nums  = sorted((numbers[deviations <= -20]).tolist())

    return {
        "last_draw"   : df["numbers"].iloc[-1],
        "last_date"   : df["Date"].iloc[-1].date(),
        "n_draws"     : len(df),
        "hot_numbers" : hot_nums,
        "cold_numbers": cold_nums,
        "phase1_top6" : sorted((np.argsort(p1)[::-1][:6] + 1).tolist()),
        "phase3_top6" : sorted((np.argsort(p3)[::-1][:6] + 1).tolist()),
        "sum_pattern" : sum_pat,
        "zone_pattern": zone_pat,
        "bal_pattern" : bal_pat,
        "top6_ticket" : top6,
        "top1_number" : top1,
    }


def print_final_report(results: dict, run_folder: str, images: list[str]) -> None:
    """Print the clean final summary to the terminal."""
    sep = "=" * 62

    print(f"""
{sep}
  EMIRATES DRAW EASY6 — FULL RUN COMPLETE
{sep}

  Run folder : {run_folder}/
  Data used  : {results['n_draws']} draws
  Last draw  : {results['last_date']}  ->  {results['last_draw']}

{sep}
  STATISTICAL ANALYSIS RESULTS
{sep}

  Hot numbers (>=+20% above expected):
    {results['hot_numbers']}

  Cold numbers (<=−20% below expected):
    {results['cold_numbers']}

{sep}
  PHASE-BY-PHASE PREDICTION
{sep}

  Phase 1  (Frequency-based top 6)  :  {results['phase1_top6']}
  Phase 3  (Markov top 6)           :  {results['phase3_top6']}
  Phase 4  Feedback loop detected:
    Sum    : {results['sum_pattern']}
    Zone   : {results['zone_pattern']}
    Balance: {results['bal_pattern']}

{sep}
  *** FINAL PREDICTED TICKET ***
{sep}

    {results['top6_ticket']}

  Most probable single number  :  #{results['top1_number']}

  (Educational output only — lottery draws are random)
{sep}

  Charts saved ({len(images)} files):""")

    for img in images:
        print(f"    {img}")
    print()


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    wall_start = time.perf_counter()

    print(f"""
{'='*62}
  EMIRATES DRAW EASY6 — FULL PREDICTION RUN
{'='*62}
  Output folder : {RUN_FOLDER}/
  Steps to run  : {len(STEPS)}
  Started at    : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
""")

    # ── Run all steps ─────────────────────────────────────────────────────────
    status = {}
    for label, filename in STEPS:
        ok = run_step(label, filename)
        status[filename] = ok

    # ── Collect images saved to the run folder ────────────────────────────────
    images = sorted([
        os.path.join(RUN_FOLDER, f)
        for f in os.listdir(RUN_FOLDER)
        if f.endswith(".png")
    ])

    # ── Final result ──────────────────────────────────────────────────────────
    results = collect_results()

    total_time = time.perf_counter() - wall_start
    print(f"\n  Total run time: {total_time:.1f}s")

    print_final_report(results, RUN_FOLDER, images)

    # ── Step summary ──────────────────────────────────────────────────────────
    print("  Step status:")
    for label, filename in STEPS:
        icon = "[OK]" if status[filename] else "[FAIL]"
        print(f"    {icon}  {label}")
    print()


if __name__ == "__main__":
    main()
