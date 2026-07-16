"""
=============================================================
 STEP 3: PROBABILITY DISTRIBUTIONS
=============================================================
 LEARNING GOAL:
   Understand the 'shape' of randomness. Instead of asking
   "what number will come next?" we ask "what is the PROBABILITY
   that each number appears in the next draw?"

 KEY CONCEPTS INTRODUCED:
   * Empirical probability (from data)
   * The Uniform distribution  (all numbers equally likely)
   * The Normal distribution   (bell curve)
   * Comparing distributions: KL-Divergence
   * Confidence intervals
   * Pair co-occurrence probabilities
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm, uniform, entropy
from itertools import combinations
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder

CSV_FILE = "Emirates_Draw_EASY6.csv"
WIN_COLS  = ["Winning Number 1", "2", "3", "4", "5", "6"]
POOL      = 40


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


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def kl_divergence(p, q):
    """KL divergence D_KL(p || q) — measures how different two distributions are."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    # Add tiny epsilon so we never take log(0)
    p = p / p.sum()
    q = q / q.sum()
    return float(entropy(p, q))


def main():
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)
    n_draws = len(df)

    all_numbers = [n for row in df["numbers"] for n in row]
    freq = np.zeros(POOL)
    for n in all_numbers:
        freq[n - 1] += 1

    # ── CONCEPT 1: Empirical probability ─────────────────────────────────────
    print_section("CONCEPT 1 — Empirical Probability")
    print(f"""
  Empirical probability = (count of event) / (total trials)

  For example:
    If number 7 appeared {int(freq[6])} times in {n_draws * 6} total draws:
      P(7) = {freq[6]}/{n_draws * 6} = {freq[6]/(n_draws*6):.4f}

  Theoretical P(any number) if uniform = 1/{POOL} = {1/POOL:.4f}

  Empirical probabilities fluctuate around the theoretical value.
  The more data you have, the closer they converge (Law of Large Numbers).
""")
    empirical_prob = freq / freq.sum()
    uniform_prob   = np.full(POOL, 1.0 / POOL)

    top3 = np.argsort(empirical_prob)[::-1][:3]
    bot3 = np.argsort(empirical_prob)[:3]
    print("  Top 3 highest empirical probability:")
    for i in top3:
        print(f"    Number {i+1:2d}  P = {empirical_prob[i]:.4f}  (theoretical {1/POOL:.4f})")
    print("  Top 3 lowest empirical probability:")
    for i in bot3:
        print(f"    Number {i+1:2d}  P = {empirical_prob[i]:.4f}  (theoretical {1/POOL:.4f})")

    # ── CONCEPT 2: KL-Divergence ─────────────────────────────────────────────
    print_section("CONCEPT 2 — KL-Divergence (How Far from Uniform?)")
    print("""
  KL-Divergence (Kullback-Leibler) measures how different your
  observed distribution is from a reference distribution (here: Uniform).

    KL = 0.0   → identical to uniform (perfectly random)
    KL > 0.0   → deviation from uniform; higher = more biased

  This is used heavily in machine learning (information theory).
""")
    kl = kl_divergence(empirical_prob, uniform_prob)
    print(f"  KL-Divergence (empirical || uniform) = {kl:.6f}")
    if kl < 0.005:
        print("  Interpretation: Very close to uniform — draw is essentially random.")
    elif kl < 0.02:
        print("  Interpretation: Slight deviation — tiny bias, within normal noise range.")
    else:
        print("  Interpretation: Significant deviation — real bias in the data!")

    # ── CONCEPT 3: Confidence intervals ─────────────────────────────────────
    print_section("CONCEPT 3 — Confidence Intervals")
    print("""
  A confidence interval (CI) gives a RANGE where we expect
  a value to fall with a given probability (e.g. 95%).

  For a proportion p over n trials:
    CI = p ± z * sqrt(p*(1-p)/n)
    where z = 1.96 for 95% confidence

  Applied to our lottery:
    n = total draws = number of times each number COULD appear
    p = empirical probability of each number
""")
    n_total = n_draws * 6
    z95 = 1.96
    ci_lower = empirical_prob - z95 * np.sqrt(empirical_prob * (1 - empirical_prob) / n_total)
    ci_upper = empirical_prob + z95 * np.sqrt(empirical_prob * (1 - empirical_prob) / n_total)
    print(f"  {'Number':>6}  {'P(obs)':>8}  {'CI lower':>9}  {'CI upper':>9}  {'Includes 1/40?':>14}")
    for i in range(POOL):
        includes = "YES" if ci_lower[i] <= 1/POOL <= ci_upper[i] else "NO ←"
        print(f"    {i+1:4d}     {empirical_prob[i]:.4f}   {ci_lower[i]:.4f}    {ci_upper[i]:.4f}    {includes}")

    # ── CONCEPT 4: Pair co-occurrence ────────────────────────────────────────
    print_section("CONCEPT 4 — Pair Co-occurrence")
    print("""
  Some pairs of numbers may appear together more often than chance.
  If numbers A and B are truly independent:
    P(A and B in same draw) ≈ C(38,4)/C(40,6) ≈ 3/65 ≈ 0.046

  We count actual co-occurrences and find the top pairs.
  (This seeds the 'association rules' idea in data mining.)
""")
    co_count = {}
    for row in df["numbers"]:
        for a, b in combinations(row, 2):
            key = (a, b)
            co_count[key] = co_count.get(key, 0) + 1

    # theoretical: choose 2 from 6 drawn = 15 pairs per draw; P(specific pair) = 15 / C(40,2)
    from math import comb
    expected_co = n_draws * 15 / comb(POOL, 2)
    print(f"  Expected co-occurrences per pair : {expected_co:.2f}")

    top_pairs = sorted(co_count.items(), key=lambda x: x[1], reverse=True)[:10]
    print("  Top 10 most frequent pairs:")
    for rank, ((a, b), cnt) in enumerate(top_pairs, 1):
        print(f"    {rank:2d}. ({a:2d}, {b:2d})  →  {cnt} times  "
              f"(expected {expected_co:.1f})")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    numbers = np.arange(1, POOL + 1)
    fig = plt.figure(figsize=(16, 11))
    fig.suptitle("STEP 3 — Probability Distributions", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 3a — Empirical vs uniform probability
    ax1 = fig.add_subplot(gs[0, :])
    ax1.bar(numbers, empirical_prob, color="#4C72B0", alpha=0.75, label="Empirical P")
    ax1.errorbar(numbers, empirical_prob,
                 yerr=[empirical_prob - ci_lower, ci_upper - empirical_prob],
                 fmt="none", color="#C44E52", linewidth=0.8, capsize=3, label="95% CI")
    ax1.axhline(1/POOL, color="red", linestyle="--", linewidth=1.5,
                label=f"Theoretical uniform (1/{POOL})")
    ax1.set_title("Empirical Probability per Number with 95% Confidence Intervals")
    ax1.set_xlabel("Number"); ax1.set_ylabel("Probability")
    ax1.set_xticks(numbers); ax1.tick_params(axis="x", labelsize=7)
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # 3b — Cumulative distribution
    ax2 = fig.add_subplot(gs[1, 0])
    sorted_nums = np.sort(all_numbers)
    cdf = np.arange(1, len(sorted_nums)+1) / len(sorted_nums)
    ax2.step(sorted_nums, cdf, color="#4C72B0", linewidth=1.5, label="Empirical CDF")
    ax2.plot([1, 40], [0, 1], color="red", linestyle="--", linewidth=1.2, label="Uniform CDF")
    ax2.set_title("Cumulative Distribution Function (CDF)")
    ax2.set_xlabel("Number"); ax2.set_ylabel("Cumulative probability")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

    # 3c — Top pair co-occurrence heatmap style bar
    ax3 = fig.add_subplot(gs[1, 1])
    pair_labels = [f"({a},{b})" for (a, b), _ in top_pairs]
    pair_counts = [cnt for _, cnt in top_pairs]
    colors_bar  = ["#e74c3c" if c > expected_co * 1.5 else "#4C72B0" for c in pair_counts]
    ax3.barh(pair_labels[::-1], pair_counts[::-1], color=colors_bar[::-1])
    ax3.axvline(expected_co, color="black", linestyle="--", linewidth=1.2,
                label=f"Expected ({expected_co:.1f})")
    ax3.set_title("Top 10 Pair Co-occurrences")
    ax3.set_xlabel("Count"); ax3.legend(fontsize=8)
    ax3.grid(axis="x", alpha=0.3)

    out = run_folder + "/step3_probability_distributions.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 3")
    print("""
  ✔  Empirical probability from observed data
  ✔  KL-Divergence — how far data is from uniform
  ✔  Confidence intervals — the range of uncertainty
  ✔  Pair co-occurrence — discovering associations

  NEXT STEP → Run 04_monte_carlo_simulation.py
""")


if __name__ == "__main__":
    main()
