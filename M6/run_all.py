"""
run_all.py — Run all 13 EASY6 prediction steps in sequence.

Usage:
    python run_all.py

What it does:
  1. Creates ONE shared timestamped output folder (runs/YYYY-MM-DD_HH-MM-SS/)
  2. Runs steps 01 → 13 in order, all charts saved to that folder
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
    ("Step 01 — Data Exploration",                 "01_data_explorer.py"),
    ("Step 02 — Frequency Analysis",               "02_frequency_analysis.py"),
    ("Step 03 — Probability Distributions",        "03_probability_distributions.py"),
    ("Step 04 — Monte Carlo Simulation",           "04_monte_carlo_simulation.py"),
    ("Step 05 — Markov Chain",                     "05_markov_chain.py"),
    ("Step 06 — Prediction Report",                "06_prediction_report.py"),
    ("Step 07 — Advanced Prediction",              "07_advanced_prediction.py"),
    ("Step 08 — Deep Learning and Wheeling",       "08_deep_learning_and_wheeling.py"),
    ("Step 09 — Ultra Stacking Ensemble",          "09_ultra_stacking_ensemble.py"),
    ("Step 10 — Quantum Science Engine",          "10_advanced_quantum_signal_engine.py"),
    ("Step 11 — BlackRock Quant Engine",          "11_blackrock_quant_engine.py"),
    ("Step 12 — Feature & Metric Depth",          "12_enhanced_prediction_features_and_metrics.py"),
    ("Step 13 — GNN & Hawkes Meta Engine",        "13_gnn_hawkes_meta_learning_engine.py"),
    ("Step 14 — Master AI Meta-Ensemble",         "14_master_ai_meta_ensemble.py"),
    ("Step 15 — Randomness Audit & Wheeling Suite", "15_randomness_audit_and_wheeling.py"),
    ("Step 16 — Final Tabular Infographic Report",  "16_final_tabular_report_chart.py"),
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
        if hasattr(mod, "main"):
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
    POOL, DRAW_SIZE, SEED = 39, 6, 42

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
    df["n_high"] = df["numbers"].apply(lambda nums: sum(1 for n in nums if n > 19))
    df["n_low"]  = df["numbers"].apply(lambda nums: sum(1 for n in nums if n <= 19))

    # ── Pull pre-computed results from step 7 module ──────────────────────────
    spec = importlib.util.spec_from_file_location("step7", "07_advanced_prediction.py")
    mod7 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod7)

    p1 = mod7.phase1_weighted_probability(df, recent_n=15)
    p3 = mod7.phase3_number_markov(df)
    p4_tuple = mod7.phase4_feedback_loop(df, p1, lookback=5)
    p4_adj, sum_pat, zone_pat, bal_pat = p4_tuple
    p_cold = mod7.signal_cold_due(df, lookback=8)

    w = mod7.DEFAULT_WEIGHTS
    ens = w["w1"] * p1 + w["w3"] * p3 + w["w4"] * p4_adj + w["wc"] * p_cold
    ens /= ens.sum()

    rng     = np.random.default_rng(SEED)
    numbers = np.arange(1, POOL + 1)
    mc_freq = np.zeros(POOL)
    for _ in range(100_000):
        draw = rng.choice(numbers, size=DRAW_SIZE, replace=False, p=ens)
        for n in draw:
            mc_freq[n - 1] += 1

    top6    = mod7.diversity_select(ens, k=DRAW_SIZE, candidate_pool=mod7.DEFAULT_POOL_SIZE)
    df_sums = df["numbers"].apply(sum)
    top6    = mod7.validate_sum_range(top6, ens, df_sums, candidate_pool=mod7.DEFAULT_POOL_SIZE)

    raw_freq  = np.zeros(POOL)
    for row in df["numbers"]:
        for n in row:
            raw_freq[n - 1] += 1
    expected   = len(df) * DRAW_SIZE / POOL
    deviations = ((raw_freq - expected) / expected) * 100
    hot_nums   = sorted((numbers[deviations >= 20]).tolist())
    cold_nums  = sorted((numbers[deviations <= -20]).tolist())

    # ── Dynamically harvest predictions from Master AI Ensemble (Step 14) ──
    spec14 = importlib.util.spec_from_file_location("step14", "14_master_ai_meta_ensemble.py")
    mod14 = importlib.util.module_from_spec(spec14)
    spec14.loader.exec_module(mod14)

    signals_dict = mod14.harvest_all_signals(df)
    meta_prob, _, _ = mod14.meta_ai_blend(signals_dict, df=df)

    step7_ticket  = sorted((np.argsort(signals_dict["7. Step 7 Ensemble"])[::-1][:DRAW_SIZE] + 1).tolist())
    step8_ticket  = sorted((np.argsort(signals_dict["8. Step 8 MLP NN"])[::-1][:DRAW_SIZE] + 1).tolist())
    step9_ticket  = sorted((np.argsort(signals_dict["9. Step 9 Stacking"])[::-1][:DRAW_SIZE] + 1).tolist())
    step10_ticket = sorted((np.argsort(signals_dict["10. Step 10 Quantum"])[::-1][:DRAW_SIZE] + 1).tolist())
    step11_ticket = sorted((np.argsort(signals_dict["11. BlackRock HRP V2"])[::-1][:DRAW_SIZE] + 1).tolist())
    step12_ticket = sorted((np.argsort(meta_prob)[::-1][:DRAW_SIZE] + 1).tolist())
    step12_pool   = sorted((np.argsort(meta_prob)[::-1][:14] + 1).tolist())
    top1_num      = int(np.argmax(meta_prob) + 1)

    return {
        "last_draw"   : [int(x) for x in df["numbers"].iloc[-1]],
        "last_date"   : df["Date"].iloc[-1].date(),
        "n_draws"     : len(df),
        "hot_numbers" : [int(x) for x in hot_nums],
        "cold_numbers": [int(x) for x in cold_nums],
        "phase1_top6" : sorted([int(x) for x in (np.argsort(p1)[::-1][:DRAW_SIZE] + 1)]),
        "phase2_top6" : [int(x) for x in top6],
        "phase3_top6" : sorted([int(x) for x in (np.argsort(p3)[::-1][:DRAW_SIZE] + 1)]),
        "sum_pattern" : sum_pat,
        "zone_pattern": zone_pat,
        "bal_pattern" : bal_pat,
        "step7_ticket" : [int(x) for x in step7_ticket],
        "step8_ticket" : [int(x) for x in step8_ticket],
        "step9_ticket" : [int(x) for x in step9_ticket],
        "step10_ticket": [int(x) for x in step10_ticket],
        "step11_ticket": [int(x) for x in step11_ticket],
        "step12_ticket": [int(x) for x in step12_ticket],
        "step12_pool"  : [int(x) for x in step12_pool],
        "top1_number"  : int(top1_num),
    }


def print_final_report(results: dict, run_folder: str, images: list[str]) -> None:
    """Print the clean final summary to the terminal."""
    sep = "=" * 62

    print(f"""
{sep}
  EMIRATES DRAW EASY6 — FULL RUN COMPLETE (V2 INSTITUTIONAL)
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
  GRAND UNIFIED PREDICTION REPORT (ALL 12 STEPS HARVESTED)
{sep}

  Step 7 Ensemble Markov Ticket      :  {results['step7_ticket']}
  Step 8 Deep Learning MLP Ticket    :  {results['step8_ticket']}
  Step 9 Ultra Stacking ML Ticket   :  {results['step9_ticket']}
  Step 10 Quantum Science Ticket     :  {results['step10_ticket']}
  Step 11 BlackRock Quant V2 Ticket  :  {results['step11_ticket']}
  Step 12 Master AI Meta V2 Ticket   :  {results['step12_ticket']}

{sep}
  *** ULTIMATE RECOMMENDED GRAND MASTER TICKET (V2) ***
{sep}

    *** {results['step12_ticket']} ***
    Candidate Pool (14 Balls): {results['step12_pool']}

  Most probable single number        :  #{results['top1_number']}
  Highest Win Guarantee Wheeling System:  3-if-3 Covering Wheel

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

    status = {}
    for label, filename in STEPS:
        ok = run_step(label, filename)
        status[filename] = ok

    expected_images = sorted([
        os.path.join(RUN_FOLDER, f)
        for f in os.listdir(RUN_FOLDER)
        if f.endswith(".png")
    ])

    images = []
    for img_path in expected_images:
        if os.path.exists(img_path):
            images.append(img_path)
        else:
            print(f"  ⚠️  Warning: {img_path} not found")

    results = collect_results()

    total_time = time.perf_counter() - wall_start
    print(f"\n  Total run time: {total_time:.1f}s")

    print_final_report(results, RUN_FOLDER, images)

    print("  Step status:")
    for label, filename in STEPS:
        icon = "[OK]" if status[filename] else "[FAIL]"
        print(f"    {icon}  {label}")
    print()


if __name__ == "__main__":
    main()
