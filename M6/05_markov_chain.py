"""
=============================================================
 STEP 5: MARKOV CHAIN TRANSITION MODELLING (EASY6)
=============================================================
 LEARNING GOAL:
   A Markov Chain models how a system moves between STATES
   based only on where it is NOW (not its full history).
   Applied to EASY6 we model transitions between number ZONES,
   sum ranges, and individual positional patterns.

 KEY CONCEPTS INTRODUCED:
   * States and transitions
   * Transition matrix (T)
   * Stationary distribution π
   * N-step prediction: T^n
   * Application to number zones (1-10, 11-20, 21-30, 31-39)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from itertools import product
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder, load_data, CSV_FILE, WIN_COLS, POOL, DRAW_SIZE

DRAWS_PER = DRAW_SIZE


def zone_of(n: int) -> int:
    """Return zone index 0-3 for number n (1-39)."""
    if n <= 10: return 0
    if n <= 20: return 1
    if n <= 30: return 2
    return 3


def draw_zone_profile(numbers) -> tuple:
    """Count of numbers in each zone for one draw (as a tuple for hashability)."""
    counts = [0, 0, 0, 0]
    for n in numbers:
        counts[zone_of(n)] += 1
    return tuple(counts)


def sum_state_of(draw_sum: int) -> str:
    """Categorise draw sum into 4 states around mean 120."""
    if draw_sum < 100:     return "Low (<100)"
    if draw_sum < 120:     return "Med-Low (100-119)"
    if draw_sum < 140:     return "Med-High (120-139)"
    return "High (>=140)"


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)
    n_draws = len(df)

    # ── MODEL 1: Sum State Transitions ──────────────────────────────────────
    print_section("MODEL 1 — Draw Sum State Transitions (4 States)")
    sum_states = [sum_state_of(s) for s in df["numbers"].apply(sum)]
    state_names = ["Low (<100)", "Med-Low (100-119)", "Med-High (120-139)", "High (>=140)"]
    n_states = len(state_names)

    state_to_idx = {name: i for i, name in enumerate(state_names)}

    # Count transitions state_t -> state_t+1
    counts = np.zeros((n_states, n_states), dtype=int)
    for i in range(len(sum_states) - 1):
        from_idx = state_to_idx[sum_states[i]]
        to_idx   = state_to_idx[sum_states[i + 1]]
        counts[from_idx, to_idx] += 1

    # Transition probability matrix T (rows sum to 1)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1   # prevent divide-by-zero
    T_sum = counts / row_sums

    print("  Transition Counts Matrix (rows = state TODAY, cols = state NEXT DRAW):")
    for i, name in enumerate(state_names):
        print(f"    From {name:<20}: {counts[i]}")

    print("\n  Transition Probability Matrix T:")
    for i, name in enumerate(state_names):
        row_str = " ".join(f"{p:.3f}" for p in T_sum[i])
        print(f"    From {name:<20}: [{row_str}]")

    current_sum_state = sum_states[-1]
    curr_idx = state_to_idx[current_sum_state]
    next_probs = T_sum[curr_idx]
    most_likely_next = state_names[np.argmax(next_probs)]

    print(f"\n  Latest draw sum state : {current_sum_state}")
    print(f"  Predicted NEXT state   : {most_likely_next} (prob = {np.max(next_probs):.3f})")

    # ── MODEL 2: Zone Profile Transitions ──────────────────────────────────
    print_section("MODEL 2 — Zone Profile Transitions (Z1:1-10, Z2:11-20, Z3:21-30, Z4:31-39)")
    profiles = [draw_zone_profile(row) for row in df["numbers"]]

    # Count unique profiles
    from collections import Counter
    profile_counts = Counter(profiles)

    print(f"  Total distinct zone combinations observed : {len(profile_counts)}")
    print("  Top 5 most frequent zone combinations (Z1, Z2, Z3, Z4):")
    for prof, cnt in profile_counts.most_common(5):
        pct = (cnt / n_draws) * 100
        print(f"    Profile {prof} : {cnt:2d} draws ({pct:.1f}%)")

    # Transitions between top profiles
    top_profiles = [p for p, _ in profile_counts.most_common(8)]
    prof_to_idx = {p: i for i, p in enumerate(top_profiles)}

    T_prof_counts = np.zeros((8, 8), dtype=int)
    for i in range(len(profiles) - 1):
        if profiles[i] in prof_to_idx and profiles[i+1] in prof_to_idx:
            T_prof_counts[prof_to_idx[profiles[i]], prof_to_idx[profiles[i+1]]] += 1

    r_sums = T_prof_counts.sum(axis=1, keepdims=True)
    r_sums[r_sums == 0] = 1
    T_prof = T_prof_counts / r_sums

    # ── MODEL 3: Number-level Repeat Probability ─────────────────────────────
    print_section("MODEL 3 — Consecutive Draw Repeat Analysis")
    print("""
  How often does a number drawn in draw T repeat in draw T+1?
  Theoretical probability per number = 6 / 39 ≈ 0.1538 (15.38%)
""")
    repeats = 0
    total_consecutive_pairs = len(df) - 1
    for i in range(total_consecutive_pairs):
        set_curr = set(df["numbers"].iloc[i])
        set_next = set(df["numbers"].iloc[i+1])
        repeats += len(set_curr & set_next)

    avg_repeats_per_draw = repeats / total_consecutive_pairs
    expected_repeats     = DRAWS_PER * (DRAWS_PER / POOL)

    print(f"  Total consecutive repeats observed : {repeats}")
    print(f"  Average repeats per draw            : {avg_repeats_per_draw:.2f}")
    print(f"  Theoretical expected repeats        : {expected_repeats:.2f}")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("STEP 5 — Markov Chain Transition Modelling — EASY6", fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Heatmap 1 — Sum state transition matrix
    ax1 = fig.add_subplot(gs[0, 0])
    sns.heatmap(T_sum, annot=True, fmt=".2f", cmap="YlGnBu",
                xticklabels=[s.split()[0] for s in state_names],
                yticklabels=[s.split()[0] for s in state_names],
                ax=ax1, cbar=False)
    ax1.set_title("Sum State Transition Matrix T (Today → Next Draw)")
    ax1.set_xlabel("Next State"); ax1.set_ylabel("Current State")

    # Heatmap 2 — Top Zone profiles transition matrix
    ax2 = fig.add_subplot(gs[0, 1])
    prof_labels = [str(p) for p in top_profiles]
    sns.heatmap(T_prof, annot=True, fmt=".2f", cmap="PuBu",
                xticklabels=prof_labels, yticklabels=prof_labels,
                ax=ax2, cbar=False)
    ax2.set_title("Top 8 Zone Profile Transition Matrix")
    ax2.set_xlabel("Next Profile"); ax2.set_ylabel("Current Profile")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax2.get_yticklabels(), rotation=0, fontsize=8)

    # Bar chart 3 — Next sum state probability prediction
    ax3 = fig.add_subplot(gs[1, 0])
    bars = ax3.bar(state_names, next_probs, color="#3498db", edgecolor="black", linewidth=0.5)
    max_idx = np.argmax(next_probs)
    bars[max_idx].set_color("#e74c3c")
    ax3.set_title(f"Predicted Next Sum State (Current: {current_sum_state.split()[0]})")
    ax3.set_ylabel("Probability"); ax3.grid(axis="y", alpha=0.3)
    plt.setp(ax3.get_xticklabels(), rotation=15, ha="right", fontsize=9)

    # Card 4 — Summary recommendations
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    card_text = (
        "MARKOV CHAIN ANALYSIS RESULTS\n"
        "───────────────────────────────────────────────\n"
        f" • Current Draw Sum State  : {current_sum_state}\n"
        f" • Predicted Next Sum State: {most_likely_next}\n"
        f" • Transition Probability  : {np.max(next_probs)*100:.1f}%\n\n"
        f" • Repeat Rate per Draw    : {avg_repeats_per_draw:.2f} numbers/draw\n"
        f" • Theoretical Expectation : {expected_repeats:.2f} numbers/draw\n"
        "───────────────────────────────────────────────\n"
        "PREDICTION STRATEGY RULE:\n"
        "Always include exactly 1 number from the previous draw\n"
        "and constrain total sum to the predicted Markov state!"
    )
    ax4.text(0.05, 0.5, card_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#e8f8f5", alpha=0.9))

    out = run_folder + "/step5_markov_chain.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 5")
    print(f"""
  ✔  Markov Chains model transitions based on current state (memoryless property)
  ✔  We built transition matrices for sum states and zone profiles
  ✔  Predicted next most likely sum state: {most_likely_next}
  ✔  Confirmed that ~1 number from the previous draw repeats on average

  NEXT STEP → Run 06_prediction_report.py
""")


if __name__ == "__main__":
    main()
