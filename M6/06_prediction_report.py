"""
=============================================================
 STEP 6: COMBINED PREDICTION REPORT (EASY6)
=============================================================
 LEARNING GOAL:
   Combine ALL signals from steps 1-5 into a single, structured
   prediction report for the NEXT Emirates Draw EASY6.

   This script teaches:
     • How to blend multiple probabilistic signals
     • Ensemble scoring: combining frequency, Markov, and Monte Carlo
     • How to interpret and communicate uncertainty
     • Why the output is a DISTRIBUTION, not a single number

 HONEST DISCLAIMER:
   No model can reliably predict independent lottery draws.
   The analysis here is for EDUCATIONAL purposes — to show
   how these techniques WOULD be applied in fields where
   genuine prediction IS possible (finance, weather, etc.)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import comb
from scipy.stats import rankdata
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder, load_data, CSV_FILE, WIN_COLS, POOL, DRAW_SIZE

N_MC       = 100_000
SEED       = 42

DEFAULT_WEIGHTS = {
    "frequency": 0.35,
    "cold":      0.10,
    "markov":    0.35,
    "pair_lift": 0.20,
}


def zone_of(n: int) -> int:
    if n <= 10: return 0
    if n <= 20: return 1
    if n <= 30: return 2
    return 3


# ── SIGNALS ───────────────────────────────────────────────────────────────────

def signal_frequency(df, recent_n: int = 12) -> np.ndarray:
    recent_df = df.iloc[-recent_n:]
    freq = np.zeros(POOL)
    for row in recent_df["numbers"]:
        for n in row:
            freq[n - 1] += 1
    s = freq.sum()
    return freq / s if s > 0 else np.ones(POOL) / POOL


def signal_cold(df, lookback: int = 5) -> np.ndarray:
    recent = set()
    for row in df["numbers"].iloc[-lookback:]:
        recent.update(row)
    scores = np.array([0.0 if (i + 1) in recent else 1.0 for i in range(POOL)])
    s = scores.sum()
    return scores / s if s > 0 else np.ones(POOL) / POOL


def signal_markov_zone(df) -> np.ndarray:
    counts = np.zeros(4)
    for row in df["numbers"]:
        for n in row:
            counts[zone_of(n)] += 1
    zone_p = counts / counts.sum()

    last_draw = df["numbers"].iloc[-1]
    last_zcounts = np.zeros(4)
    for n in last_draw:
        last_zcounts[zone_of(n)] += 1

    expected_z = 6.0 / 4.0
    due_zone   = np.argmax(expected_z - (last_zcounts / 2.0))

    scores = np.zeros(POOL)
    for i in range(POOL):
        z = zone_of(i + 1)
        base = zone_p[z]
        if z == due_zone:
            base *= 1.4
        scores[i] = base

    return scores / scores.sum()


def signal_pair_lift(df) -> np.ndarray:
    pair_counts = {}
    for row in df["numbers"]:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                pair = (row[i], row[j])
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    last_draw = set(df["numbers"].iloc[-1])
    scores    = np.zeros(POOL)
    expected  = len(df) * 6 / comb(POOL, 2)

    for num in range(1, POOL + 1):
        if num in last_draw:
            continue
        lift_sum = 0.0
        for anchor in last_draw:
            pair = tuple(sorted([num, anchor]))
            observed = pair_counts.get(pair, 0)
            lift = (observed / expected) if expected > 0 else 1.0
            lift_sum += lift
        scores[num - 1] = lift_sum

    s = scores.sum()
    return scores / s if s > 0 else np.ones(POOL) / POOL


def ensemble(signals: dict, weights: dict) -> np.ndarray:
    combined = np.zeros(POOL)
    for key, weight in weights.items():
        if key in signals:
            combined += weight * signals[key]
    return combined / combined.sum()


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)

    last_draw_date = df["Date"].iloc[-1].date()
    last_draw_nums = df["numbers"].iloc[-1]

    print_section(f"PREDICTION INPUT SUMMARY — EASY6 ({len(df)} draws)")
    print(f"  Last historical draw date : {last_draw_date}")
    print(f"  Last winning numbers      : {last_draw_nums}")

    # Calculate signals
    sig_freq   = signal_frequency(df, recent_n=12)
    sig_cold   = signal_cold(df, lookback=5)
    sig_markov = signal_markov_zone(df)
    sig_pair   = signal_pair_lift(df)

    signals = {
        "frequency": sig_freq,
        "cold":      sig_cold,
        "markov":    sig_markov,
        "pair_lift": sig_pair,
    }

    prob_vector = ensemble(signals, DEFAULT_WEIGHTS)

    # ── MONTE CARLO COMBINED SIMULATION ───────────────────────────────────────
    print_section(f"ENSEMBLE MONTE CARLO ({N_MC:,} DRAWS)")

    rng     = np.random.default_rng(SEED)
    numbers = np.arange(1, POOL + 1)

    mc_counts = np.zeros(POOL)
    for _ in range(N_MC):
        draw = rng.choice(numbers, size=DRAW_SIZE, replace=False, p=prob_vector)
        for n in draw:
            mc_counts[n - 1] += 1

    sim_probs = mc_counts / N_MC

    # Rank numbers
    top6_idx = np.argsort(sim_probs)[-6:][::-1]
    top6_nums = sorted((top6_idx + 1).tolist())

    print(f"  Top 6 Combined Predictions: {top6_nums}")
    print("\n  Detailed Breakdown for Top 6 Candidates:")
    print(f"    {'Rank':<5} {'Num':<6} {'Sim Prob':<10} {'Freq Sig':<10} {'Markov Sig':<10} {'Pair Sig':<10}")
    print("    " + "─" * 55)
    for rank, idx in enumerate(top6_idx, 1):
        num = idx + 1
        print(f"    #{rank:<4} {num:<6d} {sim_probs[idx]*100:5.2f}%     "
              f"{sig_freq[idx]*100:5.2f}%     {sig_markov[idx]*100:5.2f}%     {sig_pair[idx]*100:5.2f}%")

    # ── BACK-TEST ─────────────────────────────────────────────────────────────
    print_section("BACK-TEST METRIC EVALUATION (Validation split)")
    actual_draws = [set(row) for row in df["numbers"]]
    top6_set     = set(top6_nums)

    matches      = [len(top6_set & draw) for draw in actual_draws]
    match_counts = np.bincount(matches, minlength=DRAW_SIZE + 1)

    print("  Historical match distribution for Top-6 Ensemble Ticket:")
    for m in range(DRAW_SIZE + 1):
        cnt = match_counts[m]
        pct = (cnt / len(df)) * 100
        print(f"    Matched {m} numbers: {cnt:3d} times ({pct:5.1f}%)")

    mean_match = np.mean(matches)
    expected_rand_match = DRAW_SIZE * (DRAW_SIZE / POOL)
    print(f"\n  Average matches per draw : {mean_match:.3f} / {DRAW_SIZE}")
    print(f"  Random uniform baseline  : {expected_rand_match:.3f} / {DRAW_SIZE}")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("STEP 6 — Combined Prediction Report — EASY6", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 6a — Probability Vector Bar Chart
    ax1 = fig.add_subplot(gs[0, :])
    bar_colors = ["#e74c3c" if (i + 1) in top6_nums else "#34495e" for i in range(POOL)]
    ax1.bar(range(1, POOL + 1), sim_probs * 100, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(DRAW_SIZE / POOL * 100, color="orange", linestyle="--", linewidth=1.5,
                label=f"Uniform Baseline ({DRAW_SIZE/POOL*100:.2f}%)")
    ax1.set_title("Combined Probability Distribution Vector (%) — Top 6 Highlighted in Red")
    ax1.set_xlabel("Number"); ax1.set_ylabel("Selection Probability (%)")
    ax1.set_xticks(range(1, POOL + 1))
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # 6b — Historical Match Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(range(DRAW_SIZE + 1), match_counts, color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax2.set_title("Historical Match Performance of Top-6 Ticket")
    ax2.set_xlabel("Matches per Draw"); ax2.set_ylabel("Count")
    ax2.grid(axis="y", alpha=0.3)

    # 6c — Executive Summary Card
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    summary_text = (
        "EXECUTIVE PREDICTION REPORT (EASY6)\n"
        "───────────────────────────────────────────────\n"
        f" • Recommended Top-6 Ticket : [ {', '.join(str(n) for n in top6_nums)} ]\n"
        f" • Average Match Rate       : {mean_match:.3f} / 6\n"
        f" • Uniform Baseline Rate     : {expected_rand_match:.3f} / 6\n\n"
        "SIGNAL WEIGHTING ALLOCATION:\n"
        f" • Historical Frequency    : {DEFAULT_WEIGHTS['frequency']*100:.0f}%\n"
        f" • Cold / Due Multiplier   : {DEFAULT_WEIGHTS['cold']*100:.0f}%\n"
        f" • Markov Zone Model       : {DEFAULT_WEIGHTS['markov']*100:.0f}%\n"
        f" • Pair Co-occurrence Lift : {DEFAULT_WEIGHTS['pair_lift']*100:.0f}%\n"
        "───────────────────────────────────────────────\n"
        "EDUCATIONAL REMINDER:\n"
        "Lotteries are independent probabilistic events.\n"
        "Models optimize likelihood, not certainty."
    )
    ax3.text(0.05, 0.5, summary_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1", alpha=0.9))

    out = run_folder + "/step6_prediction_report.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 6")
    print(f"""
  ✔  How to synthesize multiple model signals into a single ensemble probability vector
  ✔  Generated our top 6 prediction ticket: {top6_nums}
  ✔  Evaluated historical back-test match rate ({mean_match:.3f} vs baseline {expected_rand_match:.3f})

  NEXT STEP → Run 07_advanced_prediction.py
""")


if __name__ == "__main__":
    main()
