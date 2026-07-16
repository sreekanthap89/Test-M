"""
=============================================================
 STEP 6: COMBINED PREDICTION REPORT
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
from scipy.stats import rankdata
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder

CSV_FILE   = "Emirates_Draw_EASY6.csv"
WIN_COLS   = ["Winning Number 1", "2", "3", "4", "5", "6"]
POOL       = 40
DRAW_SIZE  = 6
N_MC       = 100_000
SEED       = 42

# ── data loading ─────────────────────────────────────────────────────────────

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


# ── signal builders ───────────────────────────────────────────────────────────

def signal_frequency(df, recent_n=20) -> np.ndarray:
    """
    Signal 1: recency-weighted empirical frequency.
    Higher weight to the last 'recent_n' draws.
    """
    freq = np.zeros(POOL)
    n    = len(df)
    for i, row in enumerate(df["numbers"]):
        # Exponential decay: recent rows get full weight, older get less
        age    = n - 1 - i               # 0 = most recent
        weight = np.exp(-age / recent_n) + 1e-6
        for num in row:
            freq[num - 1] += weight
    return freq / freq.sum()


def signal_cold(df, lookback=10) -> np.ndarray:
    """
    Signal 2: 'due' numbers — those NOT seen in the last 'lookback' draws.
    The 'Gambler's fallacy' version — included to show its limitations.
    Returns a score where unseen numbers get higher score.
    """
    recent  = set()
    for row in df["numbers"].tail(lookback):
        recent.update(row)
    scores = np.array([0.0 if (i + 1) in recent else 1.0 for i in range(POOL)])
    if scores.sum() == 0:
        return np.ones(POOL) / POOL
    return scores / scores.sum()


def zone_of(n): return (n - 1) // 10


def signal_markov_zone(df) -> np.ndarray:
    """
    Signal 3: Markov zone transition probability.
    Predicts which ZONE is most likely next, then distributes
    probability uniformly within that zone.
    """
    dominant = df["numbers"].apply(lambda nums: np.argmax([
        sum(1 for n in nums if zone_of(n) == z) for z in range(4)
    ]))

    T = np.zeros((4, 4))
    for i in range(len(dominant) - 1):
        T[dominant.iloc[i], dominant.iloc[i + 1]] += 1
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T /= row_sums

    current_zone = dominant.iloc[-1]
    zone_probs   = T[current_zone]           # P over 4 zones for next draw

    # Spread zone probability uniformly over numbers in each zone
    scores = np.zeros(POOL)
    for z, zp in enumerate(zone_probs):
        zone_numbers = [i for i in range(POOL) if zone_of(i + 1) == z]
        for idx in zone_numbers:
            scores[idx] = zp / len(zone_numbers)
    return scores / scores.sum()


def signal_pair_lift(df) -> np.ndarray:
    """
    Signal 4: Pair association lift.
    Numbers that frequently co-occur with last draw's numbers
    get higher scores.
    """
    from itertools import combinations
    from math import comb

    co_count = {}
    for row in df["numbers"]:
        for a, b in combinations(row, 2):
            co_count[(a, b)] = co_count.get((a, b), 0) + 1
            co_count[(b, a)] = co_count.get((b, a), 0) + 1

    last_draw = df["numbers"].iloc[-1]
    scores    = np.zeros(POOL)
    expected  = len(df) * 6 / comb(POOL, 2)

    for num in range(1, POOL + 1):
        if num in last_draw:
            continue
        score = 0.0
        for anchor in last_draw:
            key   = (anchor, num) if anchor < num else (num, anchor)
            score += co_count.get(key, 0)
        scores[num - 1] = max(0.0, score - expected * len(last_draw))

    if scores.sum() == 0:
        return np.ones(POOL) / POOL
    return scores / scores.sum()


def ensemble(signals: dict, weights: dict) -> np.ndarray:
    """Weighted average of multiple probability signals."""
    combined = np.zeros(POOL)
    total_w  = 0.0
    for name, sig in signals.items():
        w  = weights.get(name, 1.0)
        combined += w * sig
        total_w  += w
    combined /= total_w
    return combined / combined.sum()


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def simulate_next_draw(prob_vector: np.ndarray, n_sims: int) -> list:
    rng = np.random.default_rng(SEED)
    numbers = np.arange(1, POOL + 1)
    draws   = []
    for _ in range(n_sims):
        draw = sorted(rng.choice(numbers, size=DRAW_SIZE, replace=False, p=prob_vector).tolist())
        draws.append(draw)
    return draws


def main():
    run_folder = get_run_folder()
    df         = load_data(CSV_FILE)
    last_draw  = df["numbers"].iloc[-1]
    last_date  = df["Date"].iloc[-1].date()
    n_draws    = len(df)

    print_section("COMBINED PREDICTION REPORT")
    print(f"""
  Dataset    : Emirates Draw EASY6
  Draws used : {n_draws}
  Last draw  : {last_date}  →  {last_draw}

  ─────────────────────────────────────────────────────────
  DISCLAIMER:
  This report is for EDUCATIONAL purposes only.
  Lottery draws are designed to be independent and random.
  No statistical model can reliably predict them.
  The 'predictions' below are probability-weighted suggestions
  that illustrate HOW such models would work in domains where
  patterns DO exist (markets, weather, biology, etc.)
  ─────────────────────────────────────────────────────────
""")

    # ── Build signals ─────────────────────────────────────────────────────────
    sig_freq   = signal_frequency(df, recent_n=20)
    sig_cold   = signal_cold(df, lookback=8)
    sig_markov = signal_markov_zone(df)
    sig_pair   = signal_pair_lift(df)

    signals = {
        "frequency": sig_freq,
        "cold":      sig_cold,
        "markov":    sig_markov,
        "pair_lift": sig_pair,
    }

    # Weights (reflect confidence in each signal; adjust to experiment)
    weights = {
        "frequency": 0.40,
        "cold":      0.10,   # low weight — gambler's fallacy risk
        "markov":    0.30,
        "pair_lift": 0.20,
    }

    combined = ensemble(signals, weights)

    # ── Print individual signal scores ────────────────────────────────────────
    print_section("SIGNAL SCORES (per number)")
    numbers = np.arange(1, POOL + 1)
    print(f"  {'Num':>4}  {'Freq':>8}  {'Cold':>8}  {'Markov':>8}  {'Pair':>8}  {'Combined':>10}")
    print("  " + "-" * 55)
    sorted_by_combined = np.argsort(combined)[::-1]
    for idx in sorted_by_combined:
        n = idx + 1
        print(f"  {n:4d}  {sig_freq[idx]:8.4f}  {sig_cold[idx]:8.4f}  "
              f"{sig_markov[idx]:8.4f}  {sig_pair[idx]:8.4f}  {combined[idx]:10.4f}")

    # ── Top suggestions ────────────────────────────────────────────────────────
    print_section("TOP NUMBER SUGGESTIONS")
    top_indices = sorted_by_combined[:12]
    top_numbers = sorted([int(i + 1) for i in top_indices])
    print(f"""
  Based on the ensemble of 4 signals weighted as:
    Recency-weighted frequency  : {weights['frequency']*100:.0f}%
    Cold/due numbers            : {weights['cold']*100:.0f}%
    Markov zone transition      : {weights['markov']*100:.0f}%
    Pair co-occurrence lift     : {weights['pair_lift']*100:.0f}%

  Top 12 candidates (by combined score):
  {top_numbers}

  Suggested ticket (top 6 by combined score):
""")
    top6 = sorted([int(i + 1) for i in sorted_by_combined[:6]])
    print(f"  ★  {top6}  ★")
    print()

    # ── Monte Carlo on the ensemble probability ───────────────────────────────
    print_section("MONTE CARLO VALIDATION")
    print(f"  Running {N_MC:,} simulations using the ensemble probability ...")
    mc_draws = simulate_next_draw(combined, N_MC)

    # Count number frequency across MC draws
    mc_freq = np.zeros(POOL)
    for draw in mc_draws:
        for n in draw:
            mc_freq[n - 1] += 1
    mc_freq /= mc_freq.sum()

    mc_top6 = sorted(np.argsort(mc_freq)[::-1][:6] + 1)
    print(f"  MC-based top 6 (most simulated appearance): {mc_top6}")

    # Match between direct top6 and MC top6
    overlap = set(top6) & set(mc_top6)
    print(f"  Overlap between ensemble and MC top-6    : {sorted(overlap)}")

    # ── Confidence: how often does top-6 match the last real draw? ───────────
    print_section("BACK-TEST: Signal Quality on Historical Data")
    print("""
  Back-test: how many of the 'top 6 by combined score' do we
  correctly predict for draws we already have?
  (Train on all rows BEFORE date t, predict draw t, then check.)
  Using last 30 draws as test window.
""")
    results = []
    test_start = max(30, n_draws - 30)

    for test_i in range(test_start, n_draws):
        train_df  = df.iloc[:test_i]
        actual    = df["numbers"].iloc[test_i]

        s_f  = signal_frequency(train_df)
        s_c  = signal_cold(train_df)
        s_m  = signal_markov_zone(train_df)
        s_p  = signal_pair_lift(train_df)

        comb_s = ensemble({"frequency": s_f, "cold": s_c,
                            "markov": s_m, "pair_lift": s_p}, weights)
        pred_top6 = sorted(np.argsort(comb_s)[::-1][:6] + 1)
        matches   = len(set(pred_top6) & set(actual))
        results.append(matches)

    avg_match = np.mean(results)
    print(f"  Back-test draws    : {len(results)}")
    print(f"  Avg matches (top-6 vs actual) : {avg_match:.2f} / 6")
    print(f"  Distribution of matches       : ", end="")
    for k in range(7):
        pct = results.count(k) / len(results) * 100
        print(f"{k}={pct:.1f}%", end="  ")
    print()
    print(f"\n  Random baseline (uniform) would give avg ≈ {6*6/40:.2f} matches")
    if avg_match > 6 * 6 / 40:
        print("  → Model performs slightly above chance (as expected for weak signals)")
    else:
        print("  → Model at chance level (confirms lottery randomness)")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("STEP 6 — Combined Prediction Report\nEmirates Draw EASY6", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # 6a — Signal heatmap
    ax1 = fig.add_subplot(gs[0, :])
    signal_matrix = np.array([sig_freq, sig_cold, sig_markov, sig_pair, combined])
    im = ax1.imshow(signal_matrix, aspect="auto", cmap="YlOrRd")
    ax1.set_yticks(range(5))
    ax1.set_yticklabels(["Frequency", "Cold/Due", "Markov", "Pair Lift", "COMBINED"], fontsize=9)
    ax1.set_xticks(range(POOL)); ax1.set_xticklabels(range(1, POOL + 1), fontsize=7)
    ax1.set_title("Signal Scores per Number (brighter = higher probability)")
    plt.colorbar(im, ax=ax1, fraction=0.02)

    # Mark top 6
    for n in sorted_by_combined[:6]:
        ax1.axvline(n - 0.5, color="cyan", linewidth=0.8, alpha=0.5)
        ax1.axvline(n + 0.5, color="cyan", linewidth=0.8, alpha=0.5)

    # 6b — Combined probability bar chart
    ax2 = fig.add_subplot(gs[1, :])
    bar_colors = ["#e74c3c" if i in sorted_by_combined[:6] else "#4C72B0"
                  for i in range(POOL)]
    ax2.bar(numbers, combined * 100, color=bar_colors)
    ax2.axhline(DRAW_SIZE / POOL * 100, color="black", linestyle="--",
                linewidth=1.2, label=f"Uniform baseline ({DRAW_SIZE/POOL*100:.2f}%)")
    # Annotate top 6
    for idx in sorted_by_combined[:6]:
        ax2.text(idx + 1, combined[idx] * 100 + 0.05, f"#{idx+1}",
                 ha="center", fontsize=7, color="#e74c3c", fontweight="bold")
    ax2.set_title(f"Ensemble Probability per Number  |  ★ Red = Suggested top-6: {top6}")
    ax2.set_xlabel("Number"); ax2.set_ylabel("Probability (%)")
    ax2.set_xticks(numbers); ax2.tick_params(axis="x", labelsize=7)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    # 6c — MC frequency for top 20
    ax3 = fig.add_subplot(gs[2, 0])
    top20_idx  = np.argsort(mc_freq)[::-1][:20]
    top20_nums = top20_idx + 1
    top20_pct  = mc_freq[top20_idx] * 100
    bar_c2     = ["#e74c3c" if n in top6 else "#4C72B0" for n in top20_nums]
    ax3.bar(range(20), top20_pct, color=bar_c2)
    ax3.set_xticks(range(20)); ax3.set_xticklabels(top20_nums, fontsize=8)
    ax3.axhline(DRAW_SIZE / POOL * 100, color="black", linestyle="--", linewidth=1, label="Baseline")
    ax3.set_title(f"Top 20 Numbers in MC Simulation\n({N_MC:,} draws, red = in top-6 suggestion)")
    ax3.set_xlabel("Number"); ax3.set_ylabel("Appearance rate (%)")
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    # 6d — Back-test results
    ax4 = fig.add_subplot(gs[2, 1])
    match_counts = [results.count(k) for k in range(7)]
    ax4.bar(range(7), match_counts, color="#55A868")
    ax4.axvline(avg_match, color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {avg_match:.2f}")
    ax4.axvline(6 * 6 / 40, color="orange", linestyle=":", linewidth=1.5,
                label=f"Random baseline ({6*6/40:.2f})")
    ax4.set_xticks(range(7)); ax4.set_xticklabels([f"{k} matches" for k in range(7)], fontsize=8)
    ax4.set_title(f"Back-test: Top-6 vs Actual\n(last {len(results)} draws)")
    ax4.set_ylabel("Count"); ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=0.3)

    out = run_folder + "/step6_prediction_report.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("FINAL SUMMARY — WHAT YOU LEARNED")
    print(f"""
  This project walked you through the full prediction pipeline:

  Step 1 — Data Exploration      : load, clean, describe
  Step 2 — Frequency Analysis    : hot/cold, chi-squared test
  Step 3 — Probability Theory    : empirical P, KL-div, CI, pairs
  Step 4 — Monte Carlo           : simulate 50k draws, LLN, CIs
  Step 5 — Markov Chains         : transition matrices, π, T^n
  Step 6 — Ensemble Model        : blend signals, back-test, report

  ─────────────────────────────────────────────────────────
  KEY TAKEAWAYS FOR REAL-WORLD PREDICTION:

  1. ALWAYS start with data exploration before building models.
  2. Probability distributions beat single-point predictions.
  3. Chi-squared and back-testing tell you if your model works.
  4. Monte Carlo quantifies uncertainty even for complex systems.
  5. Markov Chains are used in Google PageRank, NLP, finance.
  6. Ensemble methods (combining signals) outperform single models.
  7. HONEST UNCERTAINTY is the hallmark of good prediction.
  ─────────────────────────────────────────────────────────

  Next draw suggested ticket: ★  {top6}  ★
  (Educational output only — do not treat as investment advice)
""")


if __name__ == "__main__":
    main()
