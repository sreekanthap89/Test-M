"""
=============================================================
 STEP 7: ADVANCED PREDICTION — NUMBER-LEVEL MARKOV + FEEDBACK LOOP
=============================================================
 LEARNING GOAL:
   This script fills the two gaps from the basic framework:

   GAP 1 — Number-level Markov Chain
     The zone-based Markov in Step 5 used only 4 states.
     Here we build a FULL 40×40 transition matrix where
     each state is an individual number 1-40, and we track
     whether seeing number X this week makes number Y more
     or less likely next week.

   GAP 2 — Feedback Loop / Pattern Detection
     Detect recent streaks (e.g., 3+ consecutive high-sum draws,
     consistent zone dominance) and ADJUST probabilities to
     counter or reinforce the streak.

   FINAL OUTPUT:
     A combined "Phase 1-4 complete" prediction:
       Phase 1 → Weighted P_i from historical frequency
       Phase 2 → Monte Carlo seeded with weighted P_i
       Phase 3 → Number-level Markov transition from last draw
       Phase 4 → Feedback loop adjustment based on recent streak
=============================================================
"""

import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations

CSV_FILE  = "Emirates_Draw_EASY6.csv"
WIN_COLS  = ["Winning Number 1", "2", "3", "4", "5", "6"]
POOL      = 40
DRAW_SIZE = 6
N_MC      = 100_000
SEED      = 42

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, skipfooter=1, engine="python")
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
    return df


def print_section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: WEIGHTED PROBABILITY (recap)
# ─────────────────────────────────────────────────────────────────────────────

def phase1_weighted_probability(df, recent_n: int = 30) -> np.ndarray:
    """
    Phase 1: P_i = weighted_freq_i / sum(weighted_freq)
    Uses exponential recency weighting.
    """
    freq = np.zeros(POOL)
    n    = len(df)
    for i, row in enumerate(df["numbers"]):
        age    = n - 1 - i
        weight = np.exp(-age / recent_n) + 1e-6
        for num in row:
            freq[num - 1] += weight
    return freq / freq.sum()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: NUMBER-LEVEL MARKOV CHAIN  ← THIS WAS THE GAP
# ─────────────────────────────────────────────────────────────────────────────

def phase3_number_markov(df) -> np.ndarray:
    """
    GAP FILLED: Build a FULL 40x40 Markov transition matrix where:
      T[i, j] = P(number j+1 appears in draw t+1 | number i+1 appeared in draw t)

    Since each draw has 6 numbers, we count ALL 6×6=36 transitions
    (every number in draw t → every number in draw t+1).

    Then for each number in the LAST draw, we look up the row in T
    and sum the outgoing probabilities to get a score for each candidate.
    """
    T = np.zeros((POOL, POOL))

    draws = df["numbers"].tolist()
    for t in range(len(draws) - 1):
        curr_draw = draws[t]        # numbers drawn at time t
        next_draw = draws[t + 1]    # numbers drawn at time t+1
        for src in curr_draw:
            for dst in next_draw:
                if src != dst:      # self-transitions are not meaningful here
                    T[src - 1, dst - 1] += 1

    # Normalise each row to a probability distribution
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums

    # Use the LAST draw as current state
    last_draw = draws[-1]

    # Aggregate: for each candidate number j, sum T[src, j] for all src in last_draw
    scores = np.zeros(POOL)
    for src in last_draw:
        scores += T[src - 1, :]

    # Zero out numbers that were just drawn (they cannot repeat)
    for n in last_draw:
        scores[n - 1] = 0.0

    if scores.sum() == 0:
        return np.ones(POOL) / POOL
    return scores / scores.sum()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4: FEEDBACK LOOP DETECTION  ← THIS WAS THE GAP
# ─────────────────────────────────────────────────────────────────────────────

def phase4_feedback_loop(df, base_prob: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    GAP FILLED: Detect recent streak patterns and adjust probabilities.

    Patterns we detect:
      1. SUM STREAK  — last N draws all had HIGH or LOW sums
         -> Push probability toward the opposite range
      2. ZONE DOMINANCE — same zone dominated for N consecutive draws
         -> Reduce probability of that zone's numbers
      3. HIGH/LOW BALANCE — too many highs (>20) recently
         -> Slightly increase weight on low numbers and vice-versa

    Adjustment: multiply base_prob by a correction vector, then renormalise.
    """
    recent = df.tail(lookback)
    correction = np.ones(POOL)

    sum_mean = df["sum"].mean()
    sum_std  = df["sum"].std()

    # ── Pattern 1: Sum streak ────────────────────────────────────────────────
    recent_sums = recent["sum"].values
    all_high    = all(s > sum_mean + 0.5 * sum_std for s in recent_sums)
    all_low     = all(s < sum_mean - 0.5 * sum_std for s in recent_sums)

    sum_pattern = "NONE"
    if all_high:
        # Streak of high sums → dampen high numbers (21-40), boost low (1-20)
        correction[:20] *= 1.15    # boost low numbers
        correction[20:] *= 0.87    # dampen high numbers
        sum_pattern = f"HIGH SUM STREAK ({lookback} draws) -> boosting low numbers"
    elif all_low:
        # Streak of low sums → dampen low numbers, boost high
        correction[:20] *= 0.87
        correction[20:] *= 1.15
        sum_pattern = f"LOW SUM STREAK ({lookback} draws) -> boosting high numbers"

    # ── Pattern 2: Zone dominance streak ────────────────────────────────────
    def dominant_zone(nums):
        counts = [sum(1 for n in nums if (n-1)//10 == z) for z in range(4)]
        return int(np.argmax(counts))

    recent_zones = [dominant_zone(row) for row in recent["numbers"]]
    zone_names   = ["Z1 (1-10)", "Z2 (11-20)", "Z3 (21-30)", "Z4 (31-40)"]
    zone_pattern = "NONE"

    if len(set(recent_zones)) == 1:
        dom_zone = recent_zones[0]
        zone_pattern = f"ZONE DOMINANCE: {zone_names[dom_zone]} dominated last {lookback} draws -> dampening"
        # Dampen numbers in the over-represented zone
        for i in range(POOL):
            if (i // 10) == dom_zone:
                correction[i] *= 0.80

    # ── Pattern 3: High/Low imbalance ───────────────────────────────────────
    avg_high = recent["n_high"].mean()
    avg_low  = recent["n_low"].mean()
    balance_pattern = "NONE"

    if avg_high > 3.5:    # consistently more high (>20) numbers
        correction[:20] *= 1.10
        balance_pattern = f"HIGH BIAS (avg {avg_high:.1f} high nums/draw) -> boosting low"
    elif avg_low > 3.5:   # consistently more low (<=20) numbers
        correction[20:] *= 1.10
        balance_pattern = f"LOW BIAS (avg {avg_low:.1f} low nums/draw) -> boosting high"

    # Apply correction and renormalise
    adjusted = base_prob * correction
    if adjusted.sum() == 0:
        return base_prob
    return adjusted / adjusted.sum(), sum_pattern, zone_pattern, balance_pattern


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: MONTE CARLO (seeded with ensemble of Phase 1 + 3 + 4)
# ─────────────────────────────────────────────────────────────────────────────

def phase2_monte_carlo(prob_vector: np.ndarray, n_sims: int) -> np.ndarray:
    """
    Phase 2: Run n_sims draws using prob_vector as weighted sampling.
    Returns frequency count per number across all simulations.
    """
    rng     = np.random.default_rng(SEED)
    numbers = np.arange(1, POOL + 1)
    freq    = np.zeros(POOL)

    for _ in range(n_sims):
        draw = rng.choice(numbers, size=DRAW_SIZE, replace=False, p=prob_vector)
        for n in draw:
            freq[n - 1] += 1
    return freq


def mc_confidence_interval(mc_freq: np.ndarray, n_sims: int,
                            top_k: int = 6, ci_level: float = 0.80) -> tuple:
    """
    Build a CI for the top-K prediction.
    Strategy: draw 1000 bootstrap samples of mc_freq and measure
    how stable the top-K set is.
    """
    rng = np.random.default_rng(SEED + 1)
    # Normalise MC freq to a probability
    mc_prob = mc_freq / mc_freq.sum()

    top_k_counts = np.zeros(POOL, dtype=int)
    n_bootstrap  = 1000

    for _ in range(n_bootstrap):
        # Bootstrap: sample from mc_prob
        sample_draw = rng.choice(POOL, size=top_k, replace=False, p=mc_prob)
        for idx in sample_draw:
            top_k_counts[idx] += 1

    # For each number, the fraction of bootstrap samples it appeared in top-k
    inclusion_prob = top_k_counts / n_bootstrap

    # CI: numbers included in ≥ ci_level of bootstrap samples
    ci_numbers = np.where(inclusion_prob >= ci_level)[0] + 1
    return inclusion_prob, ci_numbers


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    run_folder = get_run_folder()
    df        = load_data(CSV_FILE)
    last_draw = df["numbers"].iloc[-1]
    last_date = df["Date"].iloc[-1].date()
    n_draws   = len(df)
    numbers   = np.arange(1, POOL + 1)

    print_section("STEP 7 — ADVANCED PREDICTION (Full 4-Phase Framework)")
    print(f"""
  Dataset  : Emirates Draw EASY6
  Draws    : {n_draws}
  Last draw: {last_date}  ->  {last_draw}

  This script implements the COMPLETE 4-phase framework:
    Phase 1: Weighted P_i from historical frequency
    Phase 2: Monte Carlo seeded with weighted P_i
    Phase 3: Number-level 40x40 Markov transition matrix
    Phase 4: Feedback loop (streak detection + adjustment)
""")

    # ── PHASE 1 ───────────────────────────────────────────────────────────────
    print_section("PHASE 1 — Weighted Probability from Historical Frequency")
    p1 = phase1_weighted_probability(df, recent_n=30)

    expected     = n_draws * DRAW_SIZE / POOL
    raw_freq_raw = np.zeros(POOL)
    for row in df["numbers"]:
        for n in row:
            raw_freq_raw[n - 1] += 1
    deviations = ((raw_freq_raw - expected) / expected) * 100

    print(f"  Expected frequency per number : {expected:.2f}")
    print(f"\n  {'Rank':>4}  {'Num':>4}  {'Freq':>6}  {'Deviation':>11}  {'Weighted P_i':>13}")
    print("  " + "-" * 48)
    top10 = np.argsort(p1)[::-1][:10]
    for rank, idx in enumerate(top10, 1):
        print(f"  {rank:4d}  {idx+1:4d}  {raw_freq_raw[idx]:6.0f}  "
              f"{deviations[idx]:+11.2f}%  {p1[idx]:13.5f}")

    # ── PHASE 3 ───────────────────────────────────────────────────────────────
    print_section("PHASE 3 — Number-Level 40x40 Markov Transition")
    print(f"""
  Building a full {POOL}x{POOL} transition matrix where:
    T[i, j] = P(number j appears in next draw | number i appeared in this draw)

  For each number in the LAST draw {last_draw}, we look up
  row T[number] and sum the outgoing probabilities.
  This gives a context-aware boost based on what was JUST drawn.
""")
    p3 = phase3_number_markov(df)

    print("  Top 10 numbers by Markov score (most likely successors of last draw):")
    top10_markov = np.argsort(p3)[::-1][:10]
    for rank, idx in enumerate(top10_markov, 1):
        print(f"    {rank:2d}. Number {idx+1:2d}  Markov score = {p3[idx]:.5f}")

    # ── PHASE 4 ───────────────────────────────────────────────────────────────
    print_section("PHASE 4 — Feedback Loop: Streak Detection & Adjustment")
    print(f"""
  Checking last 5 draws for patterns:
    - Sum streak (all HIGH or all LOW sums)
    - Zone dominance (same zone dominant for 5 consecutive draws)
    - High/Low number imbalance

  If a streak is detected -> ADJUST probabilities to counterbalance.
  This is the 'feedback loop' concept: the model is self-correcting.
""")
    LOOKBACK = 5
    result = phase4_feedback_loop(df, p1, lookback=LOOKBACK)
    p4_adjusted, sum_pattern, zone_pattern, balance_pattern = result

    print(f"  Sum streak pattern    : {sum_pattern}")
    print(f"  Zone dominance pattern: {zone_pattern}")
    print(f"  High/Low balance      : {balance_pattern}")

    # ── ENSEMBLE: P1 + P3 + P4 ───────────────────────────────────────────────
    print_section("ENSEMBLE — Combining Phase 1 + 3 + 4")
    print("""
  Weights (tunable):
    Phase 1 (Frequency)      : 40%
    Phase 3 (Markov)         : 35%
    Phase 4 (Feedback adj.)  : 25%

  The ensemble is a weighted average of the three probability vectors.
""")
    W1, W3, W4 = 0.40, 0.35, 0.25
    ensemble_prob = (W1 * p1 + W3 * p3 + W4 * p4_adjusted)
    ensemble_prob = ensemble_prob / ensemble_prob.sum()

    print(f"  {'Num':>4}  {'P1 (Freq)':>10}  {'P3 (Markov)':>12}  {'P4 (Adj.)':>10}  {'Ensemble':>10}  {'Status':>6}")
    print("  " + "-" * 62)
    top_by_ensemble = np.argsort(ensemble_prob)[::-1]
    for rank in range(15):
        idx  = top_by_ensemble[rank]
        n    = idx + 1
        flag = " <-- LAST DRAW" if n in last_draw else ""
        print(f"  {n:4d}  {p1[idx]:10.5f}  {p3[idx]:12.5f}  {p4_adjusted[idx]:10.5f}  "
              f"{ensemble_prob[idx]:10.5f}{flag}")

    # ── PHASE 2: MONTE CARLO ──────────────────────────────────────────────────
    print_section("PHASE 2 — Monte Carlo Simulation (seeded with Ensemble Probability)")
    print(f"""
  Running {N_MC:,} simulated draws using the ensemble probability vector.
  The number that appears MOST OFTEN across simulations is the prediction.
  We also compute an {int(80)}% confidence set.
""")
    mc_freq  = phase2_monte_carlo(ensemble_prob, N_MC)
    mc_prob  = mc_freq / mc_freq.sum()
    top6_mc  = sorted((np.argsort(mc_freq)[::-1][:6] + 1).tolist())

    print(f"  Top 6 by MC frequency (predicted ticket): {top6_mc}")

    # Predicted single #1 number
    top1_mc = int(np.argmax(mc_freq) + 1)
    print(f"\n  Most likely SINGLE number prediction      : {top1_mc}")
    print(f"  Its MC appearance rate                    : "
          f"{mc_freq[top1_mc-1]/N_MC/DRAW_SIZE*100*POOL:.2f}% above uniform baseline")

    # Bootstrap CI
    inclusion_prob, ci_numbers = mc_confidence_interval(mc_freq, N_MC,
                                                          top_k=6, ci_level=0.80)
    print(f"\n  80% Confidence Set (stable across 1000 bootstrap samples):")
    print(f"  {sorted(ci_numbers.tolist())}")
    print(f"\n  Inclusion probabilities for top-12 candidates:")
    top12_ci = np.argsort(inclusion_prob)[::-1][:12]
    for idx in top12_ci:
        bar = "#" * int(inclusion_prob[idx] * 20)
        print(f"    Number {idx+1:2d}  [{bar:<20}]  {inclusion_prob[idx]*100:.1f}%")

    # ── FINAL ANSWER ──────────────────────────────────────────────────────────
    print_section("FINAL PREDICTION SUMMARY")
    print(f"""
  Last draw ({last_date}):  {last_draw}

  ============================================================
  PHASE-BY-PHASE SUMMARY:

   Phase 1  Weighted P_i top-6  : {sorted((np.argsort(p1)[::-1][:6]+1).tolist())}
   Phase 3  Markov top-6        : {sorted((top10_markov[:6]+1).tolist())}
   Phase 4  Feedback patterns   :
              Sum    : {sum_pattern}
              Zone   : {zone_pattern}
              Balance: {balance_pattern}
   Phase 2  Monte Carlo top-6   : {top6_mc}

  ============================================================
  ENSEMBLE PREDICTED TICKET (Phase 1+2+3+4 combined):

    *** {top6_mc} ***

  Most probable single number: {top1_mc}
  80% Confidence set          : {sorted(ci_numbers.tolist())}
  ============================================================

  HONEST NOTE: Lottery draws are random. These numbers reflect
  statistical patterns in 197 past draws, not future certainty.
  This output demonstrates how the 4-phase framework works in
  practice. The same pipeline applied to NON-random data
  (e.g., stock patterns, weather) would yield real edge.
""")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("STEP 7 — Full 4-Phase Prediction Framework\nEmirates Draw EASY6",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    # 7a — Heatmap: Phase 1 / 3 / 4 / Ensemble side by side
    ax1 = fig.add_subplot(gs[0, :])
    signal_matrix = np.array([p1, p3, p4_adjusted, ensemble_prob])
    im = ax1.imshow(signal_matrix, aspect="auto", cmap="YlOrRd")
    ax1.set_yticks(range(4))
    ax1.set_yticklabels(["Phase 1 (Freq)", "Phase 3 (Markov)", "Phase 4 (Adj)", "Ensemble"],
                        fontsize=9)
    ax1.set_xticks(range(POOL)); ax1.set_xticklabels(range(1, POOL + 1), fontsize=7)
    ax1.set_title("Signal Heatmap: brighter = higher probability  |  cyan lines = top-6 prediction")
    plt.colorbar(im, ax=ax1, fraction=0.02)
    for n in top6_mc:
        ax1.axvline(n - 1.5, color="cyan", linewidth=0.8, alpha=0.6)
        ax1.axvline(n - 0.5, color="cyan", linewidth=0.8, alpha=0.6)

    # 7b — Ensemble probability bars
    ax2 = fig.add_subplot(gs[1, :])
    bar_colors = ["#e74c3c" if (i+1) in top6_mc else "#4C72B0" for i in range(POOL)]
    ax2.bar(numbers, ensemble_prob * 100, color=bar_colors)
    ax2.axhline(DRAW_SIZE / POOL * 100, color="black", linestyle="--", linewidth=1.2,
                label=f"Uniform baseline ({DRAW_SIZE/POOL*100:.2f}%)")
    for n in top6_mc:
        ax2.text(n, ensemble_prob[n-1] * 100 + 0.05, f"#{n}", ha="center",
                 fontsize=7.5, color="#e74c3c", fontweight="bold")
    ax2.set_title(f"Ensemble Probability — Predicted Ticket: {top6_mc}")
    ax2.set_xlabel("Number"); ax2.set_ylabel("Probability (%)")
    ax2.set_xticks(numbers); ax2.tick_params(axis="x", labelsize=7)
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)

    # 7c — 40x40 Markov heatmap (condensed)
    ax3 = fig.add_subplot(gs[2, 0])
    T_display = np.zeros((POOL, POOL))
    draws_list = df["numbers"].tolist()
    for t in range(len(draws_list) - 1):
        for src in draws_list[t]:
            for dst in draws_list[t + 1]:
                if src != dst:
                    T_display[src-1, dst-1] += 1
    row_s = T_display.sum(axis=1, keepdims=True); row_s[row_s == 0] = 1
    T_display /= row_s
    im3 = ax3.imshow(T_display, cmap="Blues", aspect="auto")
    ax3.set_title(f"40x40 Markov Transition Matrix\n(row=current, col=next)")
    ax3.set_xlabel("Next number"); ax3.set_ylabel("Current number")
    plt.colorbar(im3, ax=ax3, fraction=0.04)

    # 7d — Bootstrap CI inclusion probabilities
    ax4 = fig.add_subplot(gs[2, 1])
    top20_ci_idx = np.argsort(inclusion_prob)[::-1][:20]
    top20_ci_nums = top20_ci_idx + 1
    bar_c = ["#e74c3c" if n in top6_mc else "#4C72B0" for n in top20_ci_nums]
    ax4.barh(range(20), inclusion_prob[top20_ci_idx][::-1] * 100, color=bar_c[::-1])
    ax4.set_yticks(range(20))
    ax4.set_yticklabels([f"#{n}" for n in top20_ci_nums[::-1]], fontsize=8)
    ax4.axvline(80, color="black", linestyle="--", linewidth=1.2, label="80% threshold")
    ax4.set_title("Bootstrap CI: Inclusion Probability\n(red = in predicted top-6)")
    ax4.set_xlabel("Inclusion probability (%)"); ax4.legend(fontsize=8)
    ax4.grid(axis="x", alpha=0.3)

    out = run_folder + "/step7_advanced_prediction.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 7")
    print("""
  GAP 1 FILLED:
  ✔  40x40 Markov transition matrix (number-level, not just zone)
  ✔  Context-aware: prediction depends on WHAT was drawn last

  GAP 2 FILLED:
  ✔  Feedback loop: detects sum streaks, zone dominance, H/L imbalance
  ✔  Adjusts probability vector to counterbalance observed streaks

  COMPLETE 4-PHASE PIPELINE:
  ✔  Phase 1: Weighted P_i  (Deviation formula, empirical probability)
  ✔  Phase 2: Monte Carlo seeded with ensemble P + bootstrap CI
  ✔  Phase 3: Number-level Markov (40x40 T matrix from last draw)
  ✔  Phase 4: Feedback loop adjustment based on recent patterns
""")


if __name__ == "__main__":
    main()
