"""
=============================================================
 STEP 7: ADVANCED PREDICTION — NUMBER-LEVEL MARKOV + FEEDBACK LOOP (EASY6)
=============================================================
 LEARNING GOAL:
   This script fills the two gaps from the basic framework:

   GAP 1 — Number-level Markov Chain
     The zone-based Markov in Step 5 used only 4 states.
     Here we build a FULL 39×39 transition matrix where
     each state is an individual number 1-39, and we track
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
from utils import get_run_folder, load_data, CSV_FILE, WIN_COLS, POOL, DRAW_SIZE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations

N_MC      = 100_000
SEED      = 42

DEFAULT_WEIGHTS = {"w1": 0.30, "w3": 0.30, "w4": 0.20, "wc": 0.20}
DEFAULT_POOL_SIZE = 16


def zone_of(n: int) -> int:
    if n <= 10: return 0
    if n <= 20: return 1
    if n <= 30: return 2
    return 3


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — WEIGHTED EMPIRICAL PROBABILITY
# ─────────────────────────────────────────────────────────────────────────────

def phase1_weighted_probability(df: pd.DataFrame, recent_n: int = 15) -> np.ndarray:
    n_total = len(df)
    w_recency = np.linspace(0.5, 1.5, n_total)

    freq = np.zeros(POOL)
    for idx, row in enumerate(df["numbers"]):
        w = w_recency[idx]
        for n in row:
            freq[n - 1] += w

    recent_df = df.iloc[-recent_n:]
    recent_freq = np.zeros(POOL)
    for row in recent_df["numbers"]:
        for n in row:
            recent_freq[n - 1] += 1

    p_all    = freq / freq.sum()
    p_recent = recent_freq / (recent_freq.sum() + 1e-9)

    p_combined = 0.5 * p_all + 0.5 * p_recent
    return p_combined / p_combined.sum()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — NUMBER-LEVEL MARKOV TRANSITION MATRIX (39x39)
# ─────────────────────────────────────────────────────────────────────────────

def build_number_markov_matrix(df: pd.DataFrame) -> np.ndarray:
    T = np.zeros((POOL, POOL))
    draws = df["numbers"].tolist()

    for t in range(len(draws) - 1):
        curr_draw = draws[t]
        next_draw = draws[t + 1]
        for a in curr_draw:
            for b in next_draw:
                T[a - 1, b - 1] += 1.0

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


def phase3_number_markov(df: pd.DataFrame) -> np.ndarray:
    T = build_number_markov_matrix(df)
    last_draw = df["numbers"].iloc[-1]

    scores = np.zeros(POOL)
    for a in last_draw:
        scores += T[a - 1, :]

    if scores.sum() == 0:
        return np.ones(POOL) / POOL
    return scores / scores.sum()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — FEEDBACK LOOP & PATTERN CORRECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_streaks(df: pd.DataFrame, lookback: int = 5):
    recent = df.iloc[-lookback:]

    sums = recent["sum"].tolist()
    mean_sum = np.mean(sums)
    if mean_sum > 135:
        sum_pattern = "STREAK_HIGH_SUM"
    elif mean_sum < 105:
        sum_pattern = "STREAK_LOW_SUM"
    else:
        sum_pattern = "SUM_BALANCED"

    z_counts = [0, 0, 0, 0]
    for row in recent["numbers"]:
        for n in row:
            z_counts[zone_of(n)] += 1

    max_z = np.argmax(z_counts)
    if z_counts[max_z] / sum(z_counts) > 0.38:
        zone_pattern = f"DOMINANT_ZONE_{max_z + 1}"
    else:
        zone_pattern = "ZONE_BALANCED"

    high_counts = recent["n_high"].tolist()
    avg_high = np.mean(high_counts)
    if avg_high > 4.0:
        bal_pattern = "HIGH_DOMINANT"
    elif avg_high < 2.0:
        bal_pattern = "LOW_DOMINANT"
    else:
        bal_pattern = "BALANCED"

    return sum_pattern, zone_pattern, bal_pattern


def phase4_feedback_loop(df: pd.DataFrame, base_prob: np.ndarray, lookback: int = 5):
    sum_pat, zone_pat, bal_pat = detect_streaks(df, lookback=lookback)
    correction = np.ones(POOL)

    if sum_pat == "STREAK_HIGH_SUM":
        for i in range(POOL):
            if (i + 1) <= 19:
                correction[i] *= 1.25
    elif sum_pat == "STREAK_LOW_SUM":
        for i in range(POOL):
            if (i + 1) > 19:
                correction[i] *= 1.25

    if zone_pat.startswith("DOMINANT_ZONE_"):
        dom_z = int(zone_pat.split("_")[-1]) - 1
        for i in range(POOL):
            z = zone_of(i + 1)
            if z != dom_z:
                correction[i] *= 1.15

    if bal_pat == "HIGH_DOMINANT":
        for i in range(POOL):
            if (i + 1) <= 19:
                correction[i] *= 1.20
    elif bal_pat == "LOW_DOMINANT":
        for i in range(POOL):
            if (i + 1) > 19:
                correction[i] *= 1.20

    adjusted_prob = base_prob * correction
    adjusted_prob /= adjusted_prob.sum()

    return adjusted_prob, sum_pat, zone_pat, bal_pat


def signal_cold_due(df: pd.DataFrame, lookback: int = 8) -> np.ndarray:
    recent_draws = set()
    for row in df["numbers"].iloc[-lookback:]:
        recent_draws.update(row)

    scores = np.array([0.0 if (i + 1) in recent_draws else 1.0 for i in range(POOL)])
    if scores.sum() == 0:
        return np.ones(POOL) / POOL
    return scores / scores.sum()


# ─────────────────────────────────────────────────────────────────────────────
# DIVERSITY & SUM VALIDATION FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def diversity_select(prob_vector: np.ndarray, k: int = DRAW_SIZE, candidate_pool: int = DEFAULT_POOL_SIZE) -> list:
    top_candidates = np.argsort(prob_vector)[::-1][:candidate_pool] + 1
    selected = []

    zones_represented = set()
    for n in top_candidates:
        z = zone_of(n)
        if z not in zones_represented and len(selected) < k:
            selected.append(n)
            zones_represented.add(z)

    for n in top_candidates:
        if len(selected) >= k:
            break
        if n not in selected:
            selected.append(n)

    return sorted(selected)


def validate_sum_range(ticket: list, prob_vector: np.ndarray, historical_sums: pd.Series, candidate_pool: int = DEFAULT_POOL_SIZE) -> list:
    q25, q75 = np.percentile(historical_sums, [25, 75])
    iqr = q75 - q25
    valid_min = max(50, q25 - 0.5 * iqr)
    valid_max = min(190, q75 + 0.5 * iqr)

    current_sum = sum(ticket)
    if valid_min <= current_sum <= valid_max:
        return ticket

    top_candidates = np.argsort(prob_vector)[::-1][:candidate_pool] + 1
    all_combos = list(combinations(top_candidates, DRAW_SIZE))

    best_ticket = ticket
    best_dist = float("inf")
    target_sum = (valid_min + valid_max) / 2.0

    for combo in all_combos:
        s = sum(combo)
        if valid_min <= s <= valid_max:
            dist = abs(s - target_sum)
            if dist < best_dist:
                best_dist = dist
                best_ticket = list(combo)

    return sorted(best_ticket)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)

    print("=================================================================")
    print("  STEP 7 — ADVANCED PREDICTION (EASY6)")
    print("=================================================================")
    print(f"  Loaded dataset with {len(df)} historical draws.")
    print(f"  Latest draw: {df['Date'].iloc[-1].date()} -> {df['numbers'].iloc[-1]}\n")

    p1 = phase1_weighted_probability(df, recent_n=15)
    p3 = phase3_number_markov(df)
    p4_adj, sum_pat, zone_pat, bal_pat = phase4_feedback_loop(df, p1, lookback=5)
    p_cold = signal_cold_due(df, lookback=8)

    w = DEFAULT_WEIGHTS
    ensemble_prob = w["w1"] * p1 + w["w3"] * p3 + w["w4"] * p4_adj + w["wc"] * p_cold
    ensemble_prob /= ensemble_prob.sum()

    rng     = np.random.default_rng(SEED)
    numbers = np.arange(1, POOL + 1)
    mc_freq = np.zeros(POOL)

    for _ in range(N_MC):
        draw = rng.choice(numbers, size=DRAW_SIZE, replace=False, p=ensemble_prob)
        for n in draw:
            mc_freq[n - 1] += 1

    top6_mc = np.argsort(mc_freq)[::-1][:DRAW_SIZE] + 1
    top1_mc = top6_mc[0]

    raw_freq = np.zeros(POOL)
    for row in df["numbers"]:
        for n in row:
            raw_freq[n - 1] += 1
    expected = len(df) * DRAW_SIZE / POOL
    deviations = ((raw_freq - expected) / expected) * 100
    hot_nums  = sorted((numbers[deviations >= 20]).tolist())
    cold_nums = sorted((numbers[deviations <= -20]).tolist())

    top6_div = diversity_select(ensemble_prob, k=DRAW_SIZE, candidate_pool=DEFAULT_POOL_SIZE)
    top6_final = validate_sum_range(top6_div, ensemble_prob, df["numbers"].apply(sum), candidate_pool=DEFAULT_POOL_SIZE)

    print("── DETECTED PATTERNS & STREAKS ──")
    print(f"  Sum Pattern     : {sum_pat}")
    print(f"  Zone Pattern    : {zone_pat}")
    print(f"  High/Low Pattern: {bal_pat}\n")

    print("── PREDICTION RESULTS ──")
    print(f"  Hot Numbers (>=+20%): {hot_nums}")
    print(f"  Cold Numbers (<=-20%): {cold_nums}")
    print(f"  Top 6 Monte Carlo Ticket : {sorted(top6_mc.tolist())}")
    print(f"  Top 6 Diversity Filtered : {top6_final}")
    print(f"  Single Most Probable Ball: #{top1_mc}\n")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("STEP 7 — Advanced Prediction (39x39 Markov + Feedback Loop — EASY6)", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(range(1, POOL + 1), p1, label="P1: Weighted Freq", color="blue", alpha=0.6)
    ax1.plot(range(1, POOL + 1), p3, label="P3: Number Markov", color="green", alpha=0.6)
    ax1.plot(range(1, POOL + 1), p4_adj, label="P4: Feedback Adj", color="purple", alpha=0.6)
    ax1.plot(range(1, POOL + 1), ensemble_prob, label="Final Ensemble", color="red", linewidth=2)
    ax1.set_title("Probability Vector Components")
    ax1.set_xlabel("Number"); ax1.set_ylabel("Probability")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    colors = ["#e74c3c" if (i + 1) in top6_final else "#4C72B0" for i in range(POOL)]
    ax2.bar(range(1, POOL + 1), ensemble_prob * 100, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(DRAW_SIZE / POOL * 100, color="black", linestyle="--", label=f"Uniform ({DRAW_SIZE/POOL*100:.2f}%)")
    ax2.set_title("Final Ensemble Probability (Top 6 Highlighted)")
    ax2.set_xlabel("Number"); ax2.set_ylabel("Probability (%)")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    T_matrix = build_number_markov_matrix(df)
    im = ax3.imshow(T_matrix, cmap="Blues", aspect="auto", origin="lower")
    fig.colorbar(im, ax=ax3)
    ax3.set_title("39x39 Number-Level Markov Transition Heatmap")
    ax3.set_xlabel("Next Ball Number"); ax3.set_ylabel("Current Ball Number")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    summary_text = (
        "ADVANCED PREDICTION REPORT SUMMARY\n"
        "───────────────────────────────────────────────\n"
        f" • Primary Recommended Ticket: [ {', '.join(str(n) for n in top6_final)} ]\n"
        f" • Monte Carlo Top Ticket    : [ {', '.join(str(n) for n in sorted(top6_mc.tolist()))} ]\n"
        f" • Single Top Probability Ball: #{top1_mc}\n\n"
        "DETECTED HISTORICAL SIGNALS:\n"
        f" • Sum Trend    : {sum_pat}\n"
        f" • Zone Trend   : {zone_pat}\n"
        f" • Balance Trend: {bal_pat}\n"
        "───────────────────────────────────────────────\n"
        "KEY ADVANCEMENT:\n"
        "Combined 39x39 Markov transitions with streak-correcting\n"
        "feedback loops to optimize candidate pool coverage!"
    )
    ax4.text(0.05, 0.5, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1", alpha=0.9))

    out = run_folder + "/step7_advanced_prediction.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  Chart saved -> {out}\n")
    plt.show()


if __name__ == "__main__":
    main()
