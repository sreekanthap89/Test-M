"""
=============================================================
 STEP 3: PROBABILITY DISTRIBUTIONS (EASY6)
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
from math import comb
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder, load_data, CSV_FILE, WIN_COLS, POOL, DRAW_SIZE

DRAWS_PER = DRAW_SIZE


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def kl_divergence(p, q):
    """KL divergence D_KL(p || q) — measures how different two distributions are."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(entropy(p, q))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)
    n_draws = len(df)

    # ── CONCEPT 1: Single-number probability distribution ────────────────────
    print_section("CONCEPT 1 — Single-Number Probability Distribution")
    print(f"""
  In an ideal fair draw of EASY6:
    P(any specific number is drawn in ONE ball pick)  = 1 / {POOL} ≈ {1/POOL:.4f}
    P(any specific number appears in a draw of {DRAWS_PER} balls) = {DRAWS_PER} / {POOL} ≈ {DRAWS_PER/POOL:.4f}
""")

    raw_freq = np.zeros(POOL + 1)
    for row in df["numbers"]:
        for n in row:
            raw_freq[n] += 1
    raw_freq = raw_freq[1:]  # index 0..38 for numbers 1..39

    # Empirical probability vector
    emp_prob  = raw_freq / (n_draws * DRAWS_PER)      # per-ball-pick probability
    per_draw_p = raw_freq / n_draws                   # probability per draw

    # Theoretical uniform vector
    unif_prob = np.full(POOL, 1.0 / POOL)

    kl = kl_divergence(emp_prob, unif_prob)
    print(f"  Empirical per-ball probability mean : {emp_prob.mean():.4f}")
    print(f"  Theoretical per-ball probability    : {1/POOL:.4f}")
    print(f"  KL Divergence D_KL(Empirical || Uniform) : {kl:.6f}")
    print("  (A KL divergence near 0 means empirical matches theoretical uniform very closely)")

    # ── CONCEPT 2: Sum distribution vs Normal fit ───────────────────────────
    print_section("CONCEPT 2 — Draw Sum Distribution & Normal Fit")
    sums = df["numbers"].apply(sum).values
    mu, std = norm.fit(sums)

    print(f"""
  Central Limit Theorem in action:
    The sum of {DRAWS_PER} numbers drawn from 1–{POOL} follows a bell curve.
    Fitted Normal Distribution:  mean = {mu:.1f},  std = {std:.1f}

  95% Confidence Interval for draw sum:
    [{mu - 1.96*std:.1f}  to  {mu + 1.96*std:.1f}]
  Draws inside this range are 'normal'.
  Draws outside this range are 'extreme outliers'.
""")
    # 95% CI count
    ci_low, ci_high = mu - 1.96*std, mu + 1.96*std
    inside = np.sum((sums >= ci_low) & (sums <= ci_high))
    print(f"  Draws inside 95% CI : {inside} / {n_draws} ({inside/n_draws*100:.1f}%)")

    # ── CONCEPT 3: Pair Co-occurrence probabilities ─────────────────────────
    print_section("CONCEPT 3 — Pair Co-Occurrence Probabilities")
    pairs_per_draw = comb(DRAWS_PER, 2)
    total_pairs    = comb(POOL, 2)
    expected_pair_prob = pairs_per_draw / total_pairs
    expected_pair_count = n_draws * expected_pair_prob

    print(f"""
  With {DRAWS_PER} balls drawn, there are C({DRAWS_PER},2) = {pairs_per_draw} pairs per draw.
  Across {POOL} numbers, there are C({POOL},2) = {total_pairs} possible pairs.
  Theoretical probability of any specific pair appearing = {pairs_per_draw} / {total_pairs} ≈ {expected_pair_prob:.4f}
  Expected occurrences per pair across {n_draws} draws     = {expected_pair_count:.1f} times
""")

    pair_counts = {}
    for row in df["numbers"]:
        for pair in combinations(row, 2):
            pair = tuple(sorted(pair))
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    top_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print("  Top 5 most frequent PAIRS:")
    for (n1, n2), count in top_pairs:
        print(f"    Pair ({n1:2d}, {n2:2d}) : appeared {count:2d} times  (expected ≈ {expected_pair_count:.1f})")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("STEP 3 — Probability Distributions — Empirical vs Theoretical (EASY6)",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 3a — Empirical per-draw probability vs uniform line
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(1, POOL + 1), per_draw_p, color="#34495e", edgecolor="white", linewidth=0.5)
    ax1.axhline(DRAWS_PER / POOL, color="#e74c3c", linestyle="--", linewidth=1.5,
                label=f"Theoretical ({DRAWS_PER/POOL:.3f})")
    ax1.set_title("Per-draw Probability per Number")
    ax1.set_xlabel("Number"); ax1.set_ylabel("Probability")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # 3b — Sum distribution + Fitted Normal PDF
    ax2 = fig.add_subplot(gs[0, 1])
    n_counts, bins, _ = ax2.hist(sums, bins=20, density=True, color="#2ecc71", alpha=0.7, edgecolor="white")
    xmin, xmax = ax2.get_xlim()
    x_axis = np.linspace(xmin, xmax, 100)
    ax2.plot(x_axis, norm.pdf(x_axis, mu, std), "r-", linewidth=2,
             label=f"Normal fit (μ={mu:.1f}, σ={std:.1f})")
    ax2.axvline(ci_low,  color="orange", linestyle=":", label="95% CI lower")
    ax2.axvline(ci_high, color="orange", linestyle=":", label="95% CI upper")
    ax2.set_title("Draw Sum Distribution vs Fitted Normal PDF")
    ax2.set_xlabel("Sum"); ax2.set_ylabel("Density")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

    # 3c — Pair co-occurrence histogram
    ax3 = fig.add_subplot(gs[1, 0])
    all_pair_counts = list(pair_counts.values())
    # Add zeros for pairs that never appeared
    n_never = total_pairs - len(all_pair_counts)
    all_counts_padded = all_pair_counts + [0] * n_never
    ax3.hist(all_counts_padded, bins=max(all_counts_padded)+1, color="#9b59b6", edgecolor="white")
    ax3.axvline(expected_pair_count, color="red", linestyle="--", linewidth=1.5,
                label=f"Expected ({expected_pair_count:.1f})")
    ax3.set_title("Pair Co-occurrence Counts Distribution")
    ax3.set_xlabel("Times pair appeared together"); ax3.set_ylabel("Number of pairs")
    ax3.legend(fontsize=9); ax3.grid(axis="y", alpha=0.3)

    # 3d — Summary statistics card
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    stats_text = (
        "SUMMARY OF PROBABILITY METRICS\n"
        "───────────────────────────────────────────────\n"
        f" • KL Divergence (empirical vs unif) : {kl:.6f}\n"
        f" • Fitted Normal Mean (Sum)          : {mu:.2f}\n"
        f" • Fitted Normal Std (Sum)           : {std:.2f}\n"
        f" • 95% Confidence Interval (Sum)     : [{ci_low:.1f}, {ci_high:.1f}]\n"
        f" • Total Unique Pairs Observed       : {len(pair_counts)} / {total_pairs}\n"
        f" • Most Frequent Pair Count          : {top_pairs[0][1]} times\n"
        "───────────────────────────────────────────────\n"
        "KEY TAKEAWAY:\n"
        "The draw sum is highly structured (Normal bell-curve),\n"
        "making sum filtering a powerful rule in predictions!"
    )
    ax4.text(0.05, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1", alpha=0.8))

    out = run_folder + "/step3_probability_distributions.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 3")
    print("""
  ✔  Empirical probability vectors tell us how often numbers show up per pick
  ✔  KL-divergence measures how close data is to pure uniform randomness
  ✔  The sum of 6 numbers forms a bell-curve (Normal distribution)
  ✔  Outlier sums (<95 or >145) are mathematically rare — filter them out!
  ✔  Pair co-occurrence patterns identify numbers that 'like' to appear together

  NEXT STEP → Run 04_monte_carlo_simulation.py
""")


if __name__ == "__main__":
    main()
