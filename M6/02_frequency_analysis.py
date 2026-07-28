"""
=============================================================
 STEP 2: FREQUENCY ANALYSIS — HOT & COLD NUMBERS (EASY6)
=============================================================
 LEARNING GOAL:
   Frequency analysis is the first practical tool in prediction.
   We count how often each number has appeared and ask:
     "Are some numbers drawn more than others?"

 KEY CONCEPTS INTRODUCED:
   * Empirical frequency vs theoretical frequency
   * Hot numbers  — drawn more than expected
   * Cold numbers — drawn less than expected
   * Chi-squared goodness-of-fit test
   * Recency weighting (recent draws matter more)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import chisquare
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder, load_data, CSV_FILE, WIN_COLS, POOL, DRAW_SIZE

DRAWS_PER = DRAW_SIZE


def count_frequencies(df, weight_by_recency=False):
    """
    Count how many times each number 1-39 was drawn.
    If weight_by_recency=True, more recent draws get higher weight.
    """
    freq = np.zeros(POOL + 1, dtype=float)   # index 0 unused; 1..39

    n_rows = len(df)
    for i, row in enumerate(df["numbers"]):
        weight = (i + 1) / n_rows if weight_by_recency else 1.0
        for num in row:
            freq[num] += weight
    return freq[1:]   # return array index 0..38 → numbers 1..39


def print_section(title):
    border = "=" * 60
    print(f"\n{border}\n  {title}\n{border}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)

    n_draws              = len(df)
    n_total_balls_drawn  = n_draws * DRAWS_PER
    expected_per_number  = n_total_balls_drawn / POOL

    raw_freq             = count_frequencies(df, weight_by_recency=False)
    rec_freq             = count_frequencies(df, weight_by_recency=True)

    print_section(f"FREQUENCY STATS ({n_draws} draws, {POOL} numbers)")
    print(f"  Total balls drawn    : {n_total_balls_drawn}")
    print(f"  Expected per number  : {expected_per_number:.1f} times")

    # Percent deviation from expected
    deviations = ((raw_freq - expected_per_number) / expected_per_number) * 100

    # Categorise
    hot_idx  = np.where(deviations >= 20)[0] + 1
    cold_idx = np.where(deviations <= -20)[0] + 1

    print(f"\n  HOT  numbers (>=+20% above expected): {sorted(hot_idx.tolist())}")
    print(f"  COLD numbers (<=-20% below expected): {sorted(cold_idx.tolist())}")

    # Top 6 & Bottom 6
    top6_raw = np.argsort(raw_freq)[-6:][::-1] + 1
    bot6_raw = np.argsort(raw_freq)[:6] + 1

    print("\n  Top 6 most frequent overall:")
    for rank, num in enumerate(top6_raw, 1):
        print(f"    #{rank}: Number {num:2d}  drawn {int(raw_freq[num-1]):3d} times  ({deviations[num-1]:+.1f}%)")

    print("\n  Top 6 least frequent overall:")
    for rank, num in enumerate(bot6_raw, 1):
        print(f"    #{rank}: Number {num:2d}  drawn {int(raw_freq[num-1]):3d} times  ({deviations[num-1]:+.1f}%)")

    # ── CHI-SQUARED TEST ──────────────────────────────────────────────────────
    print_section("STATISTICAL TEST — Chi-Squared Goodness of Fit")
    print("""
  Is the variation between numbers just NORMAL RANDOM LUCK,
  or is the lottery wheel systematically non-uniform?

  Chi-Squared hypothesis test:
    H0 (Null Hypothesis) : All numbers have EQUAL probability (uniform wheel).
    H1 (Alt  Hypothesis) : Numbers do NOT have equal probability (wheel bias).

  Rule of thumb:
    p-value > 0.05  →  Data looks uniform. No proof of wheel bias.
    p-value < 0.05  →  Statistically significant difference from uniform.
""")
    chi2_stat, p_value = chisquare(raw_freq)
    print(f"  Chi-Squared statistic : {chi2_stat:.2f}")
    print(f"  p-value               : {p_value:.4f}")
    print(f"  Degrees of freedom    : {POOL - 1}")
    if p_value > 0.05:
        print("  RESULT: p > 0.05 — Cannot reject H0. The distribution appears consistent with uniform randomness.")
    else:
        print("  RESULT: p < 0.05 — Reject H0! Significant deviation from uniform detected.")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    fig.suptitle("STEP 2 — Frequency Analysis — Hot & Cold Numbers (EASY6)", fontsize=16, fontweight="bold")

    numbers_1to39 = np.arange(1, POOL + 1)

    # Top panel: Raw frequencies
    colors = []
    for d in deviations:
        if d >= 20:   colors.append("#e74c3c")  # Hot  - Red
        elif d <= -20: colors.append("#3498db") # Cold - Blue
        else:          colors.append("#95a5a6") # Normal - Grey

    ax1.bar(numbers_1to39, raw_freq, color=colors, edgecolor="black", linewidth=0.5)
    ax1.axhline(expected_per_number, color="black", linestyle="--", linewidth=1.5,
                label=f"Expected ({expected_per_number:.1f})")
    ax1.set_ylabel("Times Drawn", fontsize=11)
    ax1.set_title("All-Time Frequency per Number (Red = Hot >= +20%, Blue = Cold <= -20%)", fontsize=12)

    hot_patch  = mpatches.Patch(color="#e74c3c", label="Hot (>= +20% expected)")
    cold_patch = mpatches.Patch(color="#3498db", label="Cold (<= -20% expected)")
    norm_patch = mpatches.Patch(color="#95a5a6", label="Normal")
    ax1.legend(handles=[hot_patch, cold_patch, norm_patch], fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # Bottom panel: Recency-weighted frequencies
    ax2.bar(numbers_1to39, rec_freq, color="#8e44ad", edgecolor="black", linewidth=0.5, alpha=0.8)
    ax2.set_ylabel("Recency-Weighted Score", fontsize=11)
    ax2.set_xlabel("Number", fontsize=11)
    ax2.set_title("Recency-Weighted Frequency (Recent draws count for more weight)", fontsize=12)
    ax2.set_xticks(numbers_1to39)
    ax2.grid(axis="y", alpha=0.3)

    out = run_folder + "/step2_frequency_analysis.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 2")
    print("""
  ✔  How to calculate empirical frequencies and deviations
  ✔  Hot numbers are NOT guaranteed to keep appearing (gambler's fallacy)
  ✔  Chi-squared test tells you if variation is statistical noise vs bias
  ✔  Recency weighting highlights numbers with recent momentum

  NEXT STEP → Run 03_probability_distributions.py
""")


if __name__ == "__main__":
    main()
