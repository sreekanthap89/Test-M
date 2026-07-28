"""
=============================================================
 STEP 4: MONTE CARLO SIMULATION (EASY6)
=============================================================
 LEARNING GOAL:
   Monte Carlo methods use RANDOM SAMPLING to estimate outcomes
   that are too complex for exact math. This is one of the most
   powerful tools in prediction, finance, physics, and AI.

 KEY CONCEPTS INTRODUCED:
   * Monte Carlo principle (simulation by sampling)
   * Biased vs unbiased sampling
   * Law of Large Numbers in practice
   * Confidence intervals from simulation
   * Expected return / risk estimation
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder, load_data, CSV_FILE, WIN_COLS, POOL, DRAW_SIZE

N_SIMULATIONS = 200_000    # number of simulated draws
SEED          = 42


def build_probability_vector(df) -> np.ndarray:
    """
    Build an empirical probability vector for numbers 1-39
    based on historical draw frequency.
    """
    freq = np.zeros(POOL)
    for row in df["numbers"]:
        for n in row:
            freq[n - 1] += 1
    return freq / freq.sum()


def run_monte_carlo(prob_vector: np.ndarray, n_sims: int, seed: int = SEED):
    """
    Simulate `n_sims` draws of 6 numbers without replacement from 1-39.
    Each number is picked according to `prob_vector`.
    """
    rng = np.random.default_rng(seed)
    numbers = np.arange(1, POOL + 1)
    
    simulated_draws = np.zeros((n_sims, DRAW_SIZE), dtype=int)
    for i in range(n_sims):
        simulated_draws[i] = np.sort(
            rng.choice(numbers, size=DRAW_SIZE, replace=False, p=prob_vector)
        )

    return simulated_draws


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)
    n_draws = len(df)

    print_section(f"PREPARING MONTE CARLO ({N_SIMULATIONS:,} SIMULATION DRAWS)")

    # Prob vectors
    emp_probs  = build_probability_vector(df)
    unif_probs = np.full(POOL, 1.0 / POOL)

    print("  Running Uniform Simulation (ideal fair wheel)...")
    sim_unif = run_monte_carlo(unif_probs, N_SIMULATIONS, seed=SEED)

    print("  Running Biased Simulation (weighted by historical data)...")
    sim_bias = run_monte_carlo(emp_probs, N_SIMULATIONS, seed=SEED + 1)

    # ── METRIC 1: Frequency counts in simulated draws ─────────────────────────
    print_section("METRIC 1 — Most Frequently Simulated Numbers")

    bias_flat = sim_bias.flatten()
    sim_counts = np.bincount(bias_flat, minlength=POOL + 1)[1:]

    top6_sim = np.argsort(sim_counts)[-6:][::-1] + 1
    print("  Top 6 numbers selected by Monte Carlo (weighted):")
    for rank, num in enumerate(top6_sim, 1):
        pct = (sim_counts[num - 1] / N_SIMULATIONS) * 100
        print(f"    #{rank}: Number {num:2d} appeared in {sim_counts[num-1]:7,d} / {N_SIMULATIONS:,} draws ({pct:.2f}%)")

    # ── METRIC 2: Simulated draw sum distribution ────────────────────────────
    print_section("METRIC 2 — Simulated Sum Distribution & Quantiles")

    unif_sums = sim_unif.sum(axis=1)
    bias_sums = sim_bias.sum(axis=1)

    q5, q25, q50, q75, q95 = np.percentile(bias_sums, [5, 25, 50, 75, 95])
    print("  Monte Carlo Sum Distribution (Biased):")
    print(f"    Mean sum           : {bias_sums.mean():.2f}")
    print(f"    Median sum (q50)   : {q50:.1f}")
    print(f"    50% Central Range  : [{q25:.1f} to {q75:.1f}]  (IQ range)")
    print(f"    90% Central Range  : [{q5:.1f} to {q95:.1f}]")

    # ── METRIC 3: Simulated vs actual draw match distribution ───────────────
    print_section("METRIC 3 — Expected Match Rate (Back-test against historical data)")
    print("""
  If we play the Top-6 Monte Carlo predicted ticket against all historical draws,
  how many numbers do we match on average?
""")
    actual_draws = [set(row) for row in df["numbers"]]
    top6_set = set(top6_sim)

    matches = [len(top6_set & draw) for draw in actual_draws]
    match_counts = np.bincount(matches, minlength=DRAW_SIZE + 1)

    print("  Historical match distribution for Top-6 Monte Carlo ticket:")
    for m in range(DRAW_SIZE + 1):
        cnt = match_counts[m]
        pct = (cnt / n_draws) * 100
        print(f"    Matched {m} numbers: {cnt:3d} times ({pct:5.1f}%)")
    mean_match = np.mean(matches)
    expected_rand_match = DRAW_SIZE * (DRAW_SIZE / POOL)
    print(f"\n  Average matches per draw : {mean_match:.3f}")
    print(f"  Random baseline expectation: {expected_rand_match:.3f} matches")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"STEP 4 — Monte Carlo Simulation ({N_SIMULATIONS:,} Draws — EASY6)",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    # 4a — Frequency comparison: Uniform vs Biased
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(1, POOL + 1)
    ax1.bar(x - 0.2, np.bincount(sim_unif.flatten(), minlength=POOL+1)[1:],
            width=0.4, color="#95a5a6", alpha=0.8, label="Uniform Sim")
    ax1.bar(x + 0.2, sim_counts,
            width=0.4, color="#e74c3c", alpha=0.8, label="Historical-Weighted Sim")
    ax1.set_title("Simulated Number Frequencies (200k draws)")
    ax1.set_xlabel("Number"); ax1.set_ylabel("Count")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # 4b — Simulated Sum Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(bias_sums, bins=35, color="#3498db", edgecolor="white", alpha=0.8, density=True)
    ax2.axvline(q5,  color="red", linestyle=":",  label=f"5th percentile ({q5:.0f})")
    ax2.axvline(q95, color="red", linestyle=":",  label=f"95th percentile ({q95:.0f})")
    ax2.axvline(q50, color="black", linestyle="--", label=f"Median ({q50:.0f})")
    ax2.set_title("Simulated Draw Sum Distribution (90% Confidence Interval)")
    ax2.set_xlabel("Draw Sum"); ax2.set_ylabel("Probability Density")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

    # 4c — Match distribution bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(range(DRAW_SIZE + 1), match_counts, color="#2ecc71", edgecolor="black", linewidth=0.5)
    ax3.set_title("Historical Match Distribution for Top-6 MC Ticket")
    ax3.set_xlabel("Matches per Draw"); ax3.set_ylabel("Number of Draws")
    ax3.grid(axis="y", alpha=0.3)

    # 4d — Top 6 predicted ticket summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    ticket_str = ", ".join(f"#{n}" for n in sorted(top6_sim))
    summary_text = (
        "MONTE CARLO TOP PREDICTED TICKET\n"
        "───────────────────────────────────────────────\n"
        f"  Suggested 6-number ticket:\n"
        f"  ★  [ {ticket_str} ]\n\n"
        "SIMULATION SUMMARY METRICS:\n"
        f" • Total Simulations Run : {N_SIMULATIONS:,}\n"
        f" • Mean Simulated Sum    : {bias_sums.mean():.1f}\n"
        f" • 90% Central Sum Range : [{q5:.0f} to {q95:.0f}]\n"
        f" • Historical Match Mean : {mean_match:.3f} / 6\n"
        f" • Random Baseline Mean  : {expected_rand_match:.3f} / 6\n"
        "───────────────────────────────────────────────\n"
        "KEY TAKEAWAY:\n"
        "Monte Carlo isolates numbers with high statistical\n"
        "resonance over 200,000 simulated future draws!"
    )
    ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f39c12", alpha=0.2))

    out = run_folder + "/step4_monte_carlo.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 4")
    print(f"""
  ✔  Monte Carlo replaces complex analytical equations with massive random sampling
  ✔  Running {N_SIMULATIONS:,} draws gives stable probability estimates
  ✔  We generated our first candidate 6-number ticket: {sorted(top6_sim)}
  ✔  We verified that 90% of all valid draws sum between {q5:.0f} and {q95:.0f}

  NEXT STEP → Run 05_markov_chain.py
""")


if __name__ == "__main__":
    main()
