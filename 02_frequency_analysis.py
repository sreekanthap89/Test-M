"""
=============================================================
 STEP 2: FREQUENCY ANALYSIS — HOT & COLD NUMBERS
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
from utils import get_run_folder

CSV_FILE = "Emirates_Draw_EASY6.csv"
WIN_COLS  = ["Winning Number 1", "2", "3", "4", "5", "6"]
POOL      = 40          # numbers 1-40
DRAWS_PER = 6           # 6 numbers per draw


def load_data(path):
    df = pd.read_csv(path, skipfooter=1, engine="python")
    df = df[df["Date"].notna() & df["Date"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    for col in WIN_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["numbers"] = df[WIN_COLS].apply(
        lambda row: sorted([int(v) for v in row if pd.notna(v)]), axis=1
    )
    return df


def count_frequencies(df, weight_by_recency=False):
    """
    Count how many times each number 1-40 was drawn.
    If weight_by_recency=True, more recent draws get higher weight.
    """
    freq = np.zeros(POOL + 1, dtype=float)   # index 0 unused; 1..40

    n_rows = len(df)
    for i, row in enumerate(df["numbers"]):
        # recency weight: older rows get less weight
        weight = (i + 1) / n_rows if weight_by_recency else 1.0
        for num in row:
            freq[num] += weight
    return freq[1:]   # return array index 0..39 → numbers 1..40


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def main():
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)
    n_draws = len(df)

    # ── CONCEPT 1: What is the expected (theoretical) frequency? ─────────────
    print_section("CONCEPT 1 — Expected vs Actual Frequency")
    print(f"""
  In {n_draws} draws of 6 numbers each, the total balls drawn = {n_draws*DRAWS_PER}.
  With {POOL} numbers in the pool, if the draw is perfectly uniform:

      Expected frequency per number = {n_draws*DRAWS_PER} / {POOL} = {n_draws*DRAWS_PER/POOL:.2f}

  Any number that appears significantly more is called 'hot'.
  Any number that appears significantly less is called 'cold'.
  Whether these differences are real or just random noise is
  tested with the Chi-Squared (χ²) test.
""")

    raw_freq   = count_frequencies(df, weight_by_recency=False)
    expected   = n_draws * DRAWS_PER / POOL

    # ── CONCEPT 2: Chi-Squared test ──────────────────────────────────────────
    print_section("CONCEPT 2 — Chi-Squared Goodness-of-Fit Test")
    print("""
  The Chi-Squared test answers: "Is the difference between what
  we observed and what we expected due to chance, or is it real?"

    H₀ (null hypothesis) : All numbers equally likely
    H₁ (alternative)     : Some numbers are more likely than others

  If p-value > 0.05  → we CANNOT reject H₀ (looks random)
  If p-value < 0.05  → we REJECT H₀ (there IS a bias)
""")
    chi2_stat, p_value = chisquare(raw_freq, f_exp=np.full(POOL, expected))
    print(f"  χ² statistic : {chi2_stat:.3f}")
    print(f"  p-value      : {p_value:.4f}")
    if p_value > 0.05:
        print("  Verdict      : ✔ Cannot reject H₀ — draw appears UNIFORM/RANDOM")
    else:
        print("  Verdict      : ✘ Reject H₀ — there IS a measurable frequency BIAS")

    # ── CONCEPT 3: Hot and cold numbers ─────────────────────────────────────
    print_section("CONCEPT 3 — Hot & Cold Numbers")
    threshold_hot  = expected * 1.20    # 20% above expected
    threshold_cold = expected * 0.80    # 20% below expected

    numbers = np.arange(1, POOL + 1)
    hot_nums   = numbers[raw_freq >= threshold_hot]
    cold_nums  = numbers[raw_freq <= threshold_cold]

    print(f"  Expected frequency per number : {expected:.1f}")
    print(f"  Hot  threshold (≥ +20%)       : {threshold_hot:.1f}")
    print(f"  Cold threshold (≤ −20%)       : {threshold_cold:.1f}")
    print(f"\n  🔥 HOT  numbers  : {sorted(hot_nums.tolist())}")
    print(f"  🧊 COLD numbers  : {sorted(cold_nums.tolist())}")
    print(f"\n  Top 10 most frequent:")
    top10_idx = np.argsort(raw_freq)[::-1][:10]
    for rank, idx in enumerate(top10_idx, 1):
        print(f"    {rank:2d}. Number {idx+1:2d}  ->  {raw_freq[idx]:.0f} times"
              f"  (expected {expected:.1f})")

    # ── FRAMEWORK FORMULA: Deviation % per number ────────────────────────────
    print_section("FRAMEWORK FORMULA — Deviation % per Number")
    print("""
  Exact formula from the prediction framework:

      Deviation_i = ( Actual_i - Expected ) / Expected  x 100%

  Positive = drawn MORE than expected  -> HOT
  Negative = drawn LESS than expected  -> COLD
  Near 0   = drawn as expected         -> NEUTRAL

  Weighted P_i = Actual_i / sum(all Actual)  [used in Monte Carlo]
""")
    print(f"  {'Num':>4}  {'Actual':>7}  {'Expected':>9}  {'Deviation':>11}  {'Weighted P':>11}  {'Status':>8}")
    print("  " + "-" * 60)
    deviations = ((raw_freq - expected) / expected) * 100
    weighted_p  = raw_freq / raw_freq.sum()
    sorted_by_dev = np.argsort(deviations)[::-1]
    for idx in sorted_by_dev:
        n      = idx + 1
        dev    = deviations[idx]
        status = "HOT" if dev >= 20 else ("COLD" if dev <= -20 else "normal")
        print(f"  {n:4d}  {raw_freq[idx]:7.0f}  {expected:9.2f}  {dev:+11.2f}%  {weighted_p[idx]:11.5f}  {status}")

    # ── CONCEPT 4: Recency-weighted frequency ────────────────────────────────

    print_section("CONCEPT 4 — Recency Weighting")
    print("""
  The idea: a number drawn last week is 'more relevant' than
  one drawn 2 years ago.  We give recent draws higher weight.
  This is the foundation of exponential smoothing.

  We compare raw frequency with recency-weighted frequency.
  Numbers that climbed in rank are 'warming up'.
  Numbers that fell in rank are 'cooling down'.
""")
    rec_freq = count_frequencies(df, weight_by_recency=True)
    # Normalise to same scale
    rec_freq_norm = rec_freq / rec_freq.sum() * raw_freq.sum()

    trend_up   = numbers[(rec_freq_norm - raw_freq) >  expected * 0.10]
    trend_down = numbers[(rec_freq_norm - raw_freq) < -expected * 0.10]
    print(f"  📈 Warming up  : {sorted(trend_up.tolist())}")
    print(f"  📉 Cooling down: {sorted(trend_down.tolist())}")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    colors = []
    for i, (rf, wf) in enumerate(zip(raw_freq, rec_freq_norm)):
        n = i + 1
        if n in hot_nums:
            colors.append("#e74c3c")   # hot  → red
        elif n in cold_nums:
            colors.append("#3498db")   # cold → blue
        else:
            colors.append("#2ecc71")   # normal → green

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle("STEP 2 — Frequency Analysis: Hot & Cold Numbers", fontsize=15, fontweight="bold")

    # Top plot: raw frequencies
    ax = axes[0]
    bars = ax.bar(numbers, raw_freq, color=colors, edgecolor="white", linewidth=0.5)
    ax.axhline(expected, color="black",  linestyle="--", linewidth=1.5,
               label=f"Expected = {expected:.1f}")
    ax.axhline(threshold_hot,  color="#e74c3c", linestyle=":", linewidth=1.2, label="Hot threshold (+20%)")
    ax.axhline(threshold_cold, color="#3498db", linestyle=":", linewidth=1.2, label="Cold threshold (−20%)")
    ax.set_title("Raw Frequency of Each Number")
    ax.set_xlabel("Number"); ax.set_ylabel("Count")
    ax.set_xticks(numbers); ax.tick_params(axis="x", labelsize=7)
    hot_patch  = mpatches.Patch(color="#e74c3c", label="Hot (≥+20%)")
    cold_patch = mpatches.Patch(color="#3498db", label="Cold (≤−20%)")
    norm_patch = mpatches.Patch(color="#2ecc71", label="Normal")
    ax.legend(handles=[hot_patch, cold_patch, norm_patch], loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Bottom plot: recency-weighted vs raw
    ax2 = axes[1]
    width = 0.4
    ax2.bar(numbers - width/2, raw_freq,       width=width, label="Raw frequency",      color="#4C72B0", alpha=0.8)
    ax2.bar(numbers + width/2, rec_freq_norm,  width=width, label="Recency-weighted",   color="#C44E52", alpha=0.8)
    ax2.axhline(expected, color="black", linestyle="--", linewidth=1.2)
    ax2.set_title("Raw vs Recency-Weighted Frequency  (bars shifted apart = warming/cooling)")
    ax2.set_xlabel("Number"); ax2.set_ylabel("Weighted count")
    ax2.set_xticks(numbers); ax2.tick_params(axis="x", labelsize=7)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = run_folder + "/step2_frequency_analysis.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 2")
    print("""
  ✔  How to compute empirical frequencies from real draw data
  ✔  How to identify 'hot' and 'cold' numbers
  ✔  How the Chi-Squared test validates (or rejects) randomness
  ✔  What recency weighting means and why it matters

  NEXT STEP → Run 03_probability_distributions.py
""")


if __name__ == "__main__":
    main()
