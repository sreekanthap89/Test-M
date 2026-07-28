"""
=============================================================
 STEP 5: MARKOV CHAIN TRANSITION MODELLING (MEGA7)
=============================================================
 LEARNING GOAL:
   A Markov Chain models how a system moves between STATES
   based only on where it is NOW (not its full history).
   Applied to MEGA7 we model transitions between number ZONES,
   sum ranges, and individual positional patterns.

 KEY CONCEPTS INTRODUCED:
   * States and transitions
   * Transition matrix (T)
   * Stationary distribution π
   * N-step prediction: T^n
   * Application to number zones (1-10, 11-20, 21-30, 31-37)
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder

CSV_FILE  = "Emirates_Draw_MEGA7.csv"
WIN_COLS  = ["Winning Number 1", "2", "3", "4", "5", "6", "7"]
POOL      = 37
DRAWS_PER = 7


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


def zone_of(n: int) -> int:
    """Return zone index 0-3 for number n (1-37)."""
    return (n - 1) // 10          # 0=1-10, 1=11-20, 2=21-30, 3=31-37


def draw_zone_profile(numbers) -> tuple:
    """Count of numbers in each zone for one draw (as a tuple for hashability)."""
    counts = [0, 0, 0, 0]
    for n in numbers:
        counts[zone_of(n)] += 1
    return tuple(counts)


def extract_dominant_zone(draw_numbers) -> int:
    """Return the index (0-3) of the zone with the most numbers in the draw.
    Zones: 0 = 1-10, 1 = 11-20, 2 = 21-30, 3 = 31-37."""
    counts = [sum(1 for n in draw_numbers if zone_of(n) == z) for z in range(4)]
    return int(np.argmax(counts))


def extract_sum_band(draw_numbers) -> int:
    """Return sum band category index for MEGA7 (expected ~133).
    0 = low (<110), 1 = med-low (110-134), 2 = med-high (135-159), 3 = high (>=160)."""
    total = sum(draw_numbers)
    if total < 110:
        return 0
    elif total < 135:
        return 1
    elif total < 160:
        return 2
    else:
        return 3


def print_section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ── Markov transition matrix builders ────────────────────────────────────────

def build_zone_transition_matrix(df) -> np.ndarray:
    """
    Builds a transition matrix T[i,j] = P(go to state j | currently in state i)
    based on dominant zone (0-3).
    """
    dominant = df["numbers"].apply(extract_dominant_zone)

    n_states = 4
    T = np.zeros((n_states, n_states))

    for i in range(len(dominant) - 1):
        curr = dominant.iloc[i]
        nxt  = dominant.iloc[i + 1]
        T[curr, nxt] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1       # avoid divide-by-zero
    T = T / row_sums
    return T, dominant


def build_sum_band_transition(df) -> np.ndarray:
    """
    States = draw sum discretised into bands:
      Band 0: sum < 110  (low)
      Band 1: 110 ≤ sum < 135  (medium-low)
      Band 2: 135 ≤ sum < 160  (medium-high)
      Band 3: sum ≥ 160  (high)
    """
    df = df.copy()
    df["band"] = df["numbers"].apply(extract_sum_band)

    n_states = 4
    T = np.zeros((n_states, n_states))
    for i in range(len(df) - 1):
        curr = df["band"].iloc[i]
        nxt  = df["band"].iloc[i + 1]
        T[curr, nxt] += 1

    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T = T / row_sums
    return T, df["band"]


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """
    Find the stationary distribution π where πT = π.
    This is the LEFT eigenvector of T for eigenvalue 1.
    """
    n = T.shape[0]
    A = (T.T - np.eye(n))
    A[-1, :] = 1.0
    b = np.zeros(n); b[-1] = 1.0
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        pi = np.ones(n) / n
    pi = np.abs(pi) / np.abs(pi).sum()
    return pi


def n_step_prediction(T: np.ndarray, current_state: int, n_steps: int) -> np.ndarray:
    """Return probability distribution over states after n_steps from current_state."""
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    T_norm = T / row_sums
    Tn = np.linalg.matrix_power(T_norm, n_steps)
    return Tn[current_state]


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)

    # ── CONCEPT 1: What is a Markov Chain? ───────────────────────────────────
    print_section("CONCEPT 1 — What is a Markov Chain?")
    print("""
  A Markov Chain is a sequence of states where:
    P(next state | all history) = P(next state | current state only)

  This "memoryless" property is called the MARKOV PROPERTY.

  For MEGA7 we define states as:
    • 'Dominant Zone' — which zone (1-10, 11-20, 21-30, 31-37)
      contributed the MOST numbers to a draw.
    • 'Sum Band' — whether the draw sum is low, medium-low,
      medium-high, or high.
""")

    # ── CONCEPT 2: Zone transition matrix ────────────────────────────────────
    print_section("CONCEPT 2 — Zone Transition Matrix")
    T_zone, dominant = build_zone_transition_matrix(df)
    zone_names = ["Z1 (1-10)", "Z2 (11-20)", "Z3 (21-30)", "Z4 (31-37)"]

    print("  Transition matrix T_zone[i,j] = P(go to zone j | dominant zone was i)")
    print(f"\n  {'':12}" + "  ".join(f"{n:>10}" for n in zone_names))
    for i, row in enumerate(T_zone):
        print(f"  {zone_names[i]:10}  " + "  ".join(f"{v:10.4f}" for v in row))

    last_zone = dominant.iloc[-1]
    print(f"\n  Most recent draw dominant zone : {zone_names[last_zone]}")

    # ── CONCEPT 3: Stationary distribution ───────────────────────────────────
    print_section("CONCEPT 3 — Stationary Distribution")
    print("""
  The stationary distribution π is the LONG-RUN proportion of
  time the chain spends in each state.
""")
    pi_zone = stationary_distribution(T_zone)
    print("  Long-run zone dominance probabilities (stationary π):")
    for i, (name, p) in enumerate(zip(zone_names, pi_zone)):
        print(f"    {name:12}  π = {p:.4f}  ({p*100:.1f}%)")

    # ── CONCEPT 4: N-step prediction ─────────────────────────────────────────
    print_section("CONCEPT 4 — N-Step Prediction")
    print("""
  T^n = T multiplied by itself n times gives us the probability
  of being in each state after n steps.
""")
    for steps in [1, 2, 3, 5, 10, 20]:
        pred = n_step_prediction(T_zone, last_zone, steps)
        print(f"  In {steps:2d} draw(s): " +
              "  ".join(f"{zone_names[j].split()[0]}={pred[j]:.3f}" for j in range(4)))

    print(f"\n  Stationary π (long-run): " +
          "  ".join(f"{zone_names[j].split()[0]}={pi_zone[j]:.3f}" for j in range(4)))

    # ── CONCEPT 5: Sum-band Markov chain ─────────────────────────────────────
    print_section("CONCEPT 5 — Sum-Band Transition Matrix")
    T_sum, sum_bands = build_sum_band_transition(df)
    band_names = ["Low (<110)", "Med-Low (110-134)", "Med-High (135-159)", "High (≥160)"]
    print("  Transition matrix for draw sum bands:")
    print(f"\n  {'':18}" + "  ".join(f"{n:>18}" for n in band_names))
    for i, row in enumerate(T_sum):
        print(f"  {band_names[i]:18}  " + "  ".join(f"{v:18.4f}" for v in row))

    pi_sum = stationary_distribution(T_sum)
    print("\n  Stationary distribution for sum bands:")
    for i, (name, p) in enumerate(zip(band_names, pi_sum)):
        print(f"    {name:20}  π = {p:.4f}  ({p*100:.1f}%)")

    last_band = sum_bands.iloc[-1]
    print(f"\n  Most recent draw sum band : {band_names[last_band]}")
    print(f"  1-step prediction         : ", end="")
    pred1 = n_step_prediction(T_sum, last_band, 1)
    for j, (name, p) in enumerate(zip(band_names, pred1)):
        print(f"{name.split()[0]}={p:.3f}", end="  ")
    print()

    # ── CONCEPT 6: Number-level co-transition ────────────────────────────────
    print_section("CONCEPT 6 — Pairwise Persistence: Does a Number Repeat?")
    baseline  = DRAWS_PER / POOL
    repeats   = np.zeros(POOL)
    prev_count= np.zeros(POOL)

    draws = df["numbers"].tolist()
    for i in range(len(draws) - 1):
        curr_set = set(draws[i])
        next_set = set(draws[i + 1])
        for n in curr_set:
            prev_count[n - 1] += 1
            if n in next_set:
                repeats[n - 1] += 1

    repeat_prob = np.where(prev_count > 0, repeats / prev_count, 0.0)
    numbers     = np.arange(1, POOL + 1)

    print(f"  Baseline repeat probability : {baseline:.4f}  ({DRAWS_PER}/{POOL})")
    print(f"  Average observed repeat P   : {repeat_prob[repeat_prob>0].mean():.4f}")
    above = numbers[repeat_prob > baseline * 1.5]
    below = numbers[(repeat_prob < baseline * 0.5) & (repeat_prob > 0)]
    print(f"  Numbers with HIGH persistence  (≥ 1.5× baseline): {sorted(above.tolist())}")
    print(f"  Numbers with LOW  persistence  (≤ 0.5× baseline): {sorted(below.tolist())}")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("STEP 5 — Markov Chain Transition Modelling (MEGA7)", fontsize=15, fontweight="bold")

    # 5a — Zone heatmap
    ax = axes[0, 0]
    im = ax.imshow(T_zone, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f"Z{i+1}" for i in range(4)], fontsize=9)
    ax.set_yticklabels([f"Z{i+1}" for i in range(4)], fontsize=9)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{T_zone[i,j]:.2f}", ha="center", va="center", fontsize=10,
                    color="black" if T_zone[i,j] < 0.6 else "white")
    plt.colorbar(im, ax=ax)
    ax.set_title("Zone Transition Matrix\nT[i,j] = P(next zone j | current zone i)")
    ax.set_xlabel("Next zone"); ax.set_ylabel("Current zone")

    # 5b — N-step convergence
    ax2 = axes[0, 1]
    step_range = range(1, 21)
    for state in range(4):
        probs = [n_step_prediction(T_zone, state, s)[0] for s in step_range]
        ax2.plot(step_range, probs, marker="o", markersize=3, linewidth=1.5,
                 label=f"Start: {zone_names[state].split()[0]}")
    ax2.axhline(pi_zone[0], color="black", linestyle="--", linewidth=1.2,
                label=f"Stationary π ({pi_zone[0]:.3f})")
    ax2.set_title("N-Step Convergence to Stationary\n(P of landing in Z1 after n steps)")
    ax2.set_xlabel("Steps ahead"); ax2.set_ylabel("P(Z1)")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    # 5c — Sum-band heatmap
    ax3 = axes[1, 0]
    im3 = ax3.imshow(T_sum, cmap="Greens", vmin=0, vmax=1)
    short = ["Low", "M-Lo", "M-Hi", "High"]
    ax3.set_xticks(range(4)); ax3.set_yticks(range(4))
    ax3.set_xticklabels(short, fontsize=9)
    ax3.set_yticklabels(short, fontsize=9)
    for i in range(4):
        for j in range(4):
            ax3.text(j, i, f"{T_sum[i,j]:.2f}", ha="center", va="center", fontsize=10,
                     color="black" if T_sum[i,j] < 0.6 else "white")
    plt.colorbar(im3, ax=ax3)
    ax3.set_title("Sum-Band Transition Matrix")
    ax3.set_xlabel("Next band"); ax3.set_ylabel("Current band")

    # 5d — Number-level repeat probability
    ax4 = axes[1, 1]
    ax4.bar(numbers, repeat_prob * 100, color=["#e74c3c" if p > baseline * 1.5
                                                else "#3498db" if p < baseline * 0.5 and p > 0
                                                else "#2ecc71"
                                                for p in repeat_prob])
    ax4.axhline(baseline * 100, color="black", linestyle="--", linewidth=1.5,
                label=f"Baseline ({DRAWS_PER}/{POOL} = {baseline*100:.1f}%)")
    ax4.set_title("Number Persistence: P(repeat in next draw)")
    ax4.set_xlabel("Number"); ax4.set_ylabel("Repeat probability (%)")
    ax4.set_xticks(numbers); ax4.tick_params(axis="x", labelsize=7)
    ax4.legend(fontsize=8); ax4.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = run_folder + "/step5_markov_chain.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 5")
    print("""
  ✔  States, transitions, and the Markov property
  ✔  How to build a transition matrix from real MEGA7 data
  ✔  Stationary distribution — the long-run equilibrium
  ✔  N-step prediction via matrix exponentiation T^n
  ✔  Number-level persistence (does a number tend to repeat?)

  NEXT STEP → Run 06_prediction_report.py
""")


if __name__ == "__main__":
    main()
