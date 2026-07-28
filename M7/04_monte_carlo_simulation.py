"""
=============================================================
 STEP 4: MONTE CARLO SIMULATION (MEGA7)
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
from utils import get_run_folder

CSV_FILE      = "Emirates_Draw_MEGA7.csv"
WIN_COLS      = ["Winning Number 1", "2", "3", "4", "5", "6", "7"]
POOL          = 37
DRAW_SIZE     = 7
N_SIMULATIONS = 200_000    # number of simulated draws
SEED          = 42


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


def build_probability_vector(df) -> np.ndarray:
    """
    Build an empirical probability vector for numbers 1-37
    based on historical draw frequency.
    """
    freq = np.zeros(POOL)
    for row in df["numbers"]:
        for n in row:
            freq[n - 1] += 1
    return freq / freq.sum()


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def simulate_draws(prob_vector: np.ndarray, n_sims: int, strategy: str) -> list:
    """
    Simulate n_sims lottery draws.

    strategy = 'uniform'  → every number equally likely (pure chance)
    strategy = 'biased'   → sampling probabilities come from historical data
    """
    rng = np.random.default_rng(SEED)
    numbers = np.arange(1, POOL + 1)

    if strategy == "uniform":
        weights = None
    else:
        weights = prob_vector

    draws = []
    for _ in range(n_sims):
        draw = sorted(rng.choice(numbers, size=DRAW_SIZE, replace=False, p=weights).tolist())
        draws.append(draw)
    return draws


def score_draw(draw, target):
    """Return how many numbers in 'draw' appear in 'target'."""
    return len(set(draw) & set(target))


def main():
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)
    prob_vector = build_probability_vector(df)

    # ── CONCEPT 1: Monte Carlo principle ─────────────────────────────────────
    print_section("CONCEPT 1 — What is Monte Carlo?")
    print(f"""
  Imagine you cannot solve a problem mathematically.
  Instead you SIMULATE it thousands of times and measure
  what happens on average.

  Example: "If I buy a ticket with 7 numbers,
            what is the probability of matching ≥ 3 out of 7?"

  Monte Carlo answer: run {N_SIMULATIONS:,} random draws, count
  how many match ≥ 3 numbers, divide by {N_SIMULATIONS:,}.

  This is exactly how casinos, banks (VaR), and physicists work.
""")

    # ── Simulate: uniform vs biased ──────────────────────────────────────────
    print(f"  Simulating {N_SIMULATIONS:,} draws (uniform)  …")
    uniform_draws = simulate_draws(prob_vector, N_SIMULATIONS, "uniform")

    print(f"  Simulating {N_SIMULATIONS:,} draws (empirically biased) …")
    biased_draws  = simulate_draws(prob_vector, N_SIMULATIONS, "biased")

    target = df["numbers"].iloc[-1]
    print(f"\n  Target ticket (most recent actual draw): {target}")

    # ── CONCEPT 2: Match distribution ────────────────────────────────────────
    print_section("CONCEPT 2 — Simulated Match Distribution")
    print("""
  For each simulated draw, we count how many numbers match
  our target ticket.  This gives us a probability distribution
  over 'how many matches' we can realistically expect.
""")
    def match_distribution(draws, target):
        scores = [score_draw(d, target) for d in draws]
        dist   = {k: 0 for k in range(DRAW_SIZE + 1)}
        for s in scores:
            dist[s] += 1
        return dist, scores

    unif_dist, unif_scores = match_distribution(uniform_draws, target)
    bias_dist, bias_scores = match_distribution(biased_draws,  target)

    header = f"  {'Matches':>7}  {'Uniform count':>14}  {'Uniform %':>10}  {'Biased %':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for k in range(DRAW_SIZE + 1):
        uc = unif_dist[k]; bc = bias_dist[k]
        print(f"    {k:4d}      {uc:10,d}     {uc/N_SIMULATIONS*100:8.3f}%   {bc/N_SIMULATIONS*100:8.3f}%")

    # ── CONCEPT 3: Law of Large Numbers ──────────────────────────────────────
    print_section("CONCEPT 3 — Law of Large Numbers")
    print("""
  Law of Large Numbers: as n → ∞, the sample mean converges
  to the true population mean.

  We verify this by computing the running mean of 'match scores'
  as we add more simulations.  It should stabilise around the
  theoretical expected matches.

  Theoretical expected matches = 7 × (7/37) = {:.4f}
""".format(DRAW_SIZE * DRAW_SIZE / POOL))

    running_mean = np.cumsum(unif_scores) / np.arange(1, len(unif_scores) + 1)
    theoretical  = DRAW_SIZE * DRAW_SIZE / POOL
    final_mean   = running_mean[-1]
    print(f"  Theoretical expected matches : {theoretical:.4f}")
    print(f"  Simulated  running mean (end): {final_mean:.4f}")
    print(f"  Difference                   : {abs(final_mean - theoretical):.6f}  (very small ✔)")

    # ── CONCEPT 4: Confidence interval from simulation ───────────────────────
    print_section("CONCEPT 4 — Monte Carlo Confidence Intervals")
    print("""
  Monte Carlo CIs: take percentiles of the simulated distribution.
  The 2.5th–97.5th percentile range = 95% CI.
""")
    arr = np.array(unif_scores)
    ci_lo = np.percentile(arr, 2.5)
    ci_hi = np.percentile(arr, 97.5)
    print(f"  95% CI for match count : [{ci_lo:.1f}, {ci_hi:.1f}]")
    print(f"  P(≥ 3 matches)         : {(arr >= 3).mean()*100:.3f}%")
    print(f"  P(≥ 4 matches)         : {(arr >= 4).mean()*100:.4f}%")
    print(f"  P(≥ 5 matches)         : {(arr >= 5).mean()*100:.5f}%")
    print(f"  P(≥ 6 matches)         : {(arr >= 6).mean()*100:.6f}%")
    print(f"  P(= 7 matches/jackpot) : {(arr == 7).mean()*100:.6f}%")

    # ── CONCEPT 5: Number frequency from simulations ─────────────────────────
    print_section("CONCEPT 5 — Simulated Number Frequency Heatmap")
    print("""
  We count how often each number 1-37 appears across all
  simulated draws.  Comparing uniform vs biased shows how
  historical weighting shifts the predicted distribution.
""")
    def number_freq(draws):
        freq = np.zeros(POOL)
        for draw in draws:
            for n in draw:
                freq[n - 1] += 1
        return freq / freq.sum()

    freq_unif = number_freq(uniform_draws)
    freq_bias = number_freq(biased_draws)
    numbers   = np.arange(1, POOL + 1)

    top5_bias = np.argsort(freq_bias)[::-1][:5] + 1
    print(f"  Top 5 numbers in biased simulation: {sorted(top5_bias.tolist())}")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("STEP 4 — Monte Carlo Simulation (MEGA7)", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.35)

    # 4a — Match distribution comparison
    ax1 = fig.add_subplot(gs[0, :])
    match_vals = list(range(DRAW_SIZE + 1))
    unif_pct = [unif_dist[k] / N_SIMULATIONS * 100 for k in match_vals]
    bias_pct = [bias_dist[k] / N_SIMULATIONS * 100 for k in match_vals]
    x = np.arange(len(match_vals))
    w = 0.35
    ax1.bar(x - w/2, unif_pct, width=w, label="Uniform sampling",  color="#4C72B0")
    ax1.bar(x + w/2, bias_pct, width=w, label="Biased (empirical)", color="#C44E52")
    ax1.set_xticks(x); ax1.set_xticklabels([f"{k} matches" for k in match_vals])
    ax1.set_title(f"Match Distribution vs Target {target}  ({N_SIMULATIONS:,} simulations)")
    ax1.set_ylabel("Probability (%)"); ax1.legend(); ax1.grid(axis="y", alpha=0.3)
    for i, (up, bp) in enumerate(zip(unif_pct, bias_pct)):
        ax1.text(i - w/2, up + 0.3, f"{up:.2f}%", ha="center", fontsize=7, color="#4C72B0")
        ax1.text(i + w/2, bp + 0.3, f"{bp:.2f}%", ha="center", fontsize=7, color="#C44E52")

    # 4b — Law of Large Numbers (running mean)
    ax2 = fig.add_subplot(gs[1, 0])
    sample_points = np.logspace(1, np.log10(N_SIMULATIONS), 500).astype(int)
    sample_points = np.unique(sample_points)
    rm_sample = [running_mean[i - 1] for i in sample_points]
    ax2.semilogx(sample_points, rm_sample, color="#4C72B0", linewidth=1.5, label="Running mean")
    ax2.axhline(theoretical, color="red", linestyle="--", linewidth=1.5,
                label=f"Theoretical ({theoretical:.3f})")
    ax2.set_title("Law of Large Numbers — Running Mean")
    ax2.set_xlabel("Number of simulations (log scale)")
    ax2.set_ylabel("Mean matches")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # 4c — CDF of match scores
    ax3 = fig.add_subplot(gs[1, 1])
    sorted_scores = np.sort(arr)
    cdf = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    ax3.step(sorted_scores, cdf, color="#C44E52", linewidth=2)
    ax3.axhline(0.95, color="black", linestyle=":", linewidth=1, label="95%")
    ax3.set_title("CDF of Match Counts (Uniform)")
    ax3.set_xlabel("Number of matches"); ax3.set_ylabel("Cumulative probability")
    ax3.set_xticks(match_vals); ax3.legend(); ax3.grid(alpha=0.3)

    # 4d — Number frequency: uniform vs biased
    ax4 = fig.add_subplot(gs[2, :])
    w2 = 0.4
    ax4.bar(numbers - w2/2, freq_unif * 100, width=w2, label="Uniform sim",  color="#4C72B0", alpha=0.75)
    ax4.bar(numbers + w2/2, freq_bias * 100, width=w2, label="Biased sim",   color="#C44E52", alpha=0.75)
    ax4.axhline(DRAW_SIZE / POOL * 100, color="black", linestyle="--",
                linewidth=1.2, label=f"Theoretical ({DRAW_SIZE/POOL*100:.2f}%)")
    ax4.set_title("Simulated Number Frequency: Uniform vs Empirically Biased")
    ax4.set_xlabel("Number"); ax4.set_ylabel("Appearance rate (%)")
    ax4.set_xticks(numbers); ax4.tick_params(axis="x", labelsize=7)
    ax4.legend(fontsize=9); ax4.grid(axis="y", alpha=0.3)

    out = run_folder + "/step4_monte_carlo.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 4")
    print("""
  ✔  Monte Carlo: simulate thousands of outcomes to estimate probability
  ✔  Biased vs uniform sampling from historical data
  ✔  Law of Large Numbers verified through running means
  ✔  Confidence intervals derived from simulation percentiles
  ✔  How to compute "probability of jackpot" purely from simulation

  NEXT STEP → Run 05_markov_chain.py
""")


if __name__ == "__main__":
    main()
