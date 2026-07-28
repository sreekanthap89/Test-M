"""
=============================================================
 STEP 8: DEEP LEARNING A.I. & WHEELING SYSTEM (EASY6)
=============================================================
 LEARNING GOAL:
   1. Multi-Layer Perceptron (MLP) Neural Network for multi-label
      time-series prediction.
   2. Combinatorial Wheeling (Greedy Set Cover) to generate a minimum
      set of 6-number tickets that mathematically guarantees a win
      if the predicted pool contains the winning numbers.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
import math
from sklearn.neural_network import MLPClassifier
import sys, warnings
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder, load_data, generate_covering_wheel, CSV_FILE, WIN_COLS, POOL, DRAW_SIZE


def prepare_ai_data(df: pd.DataFrame, lookback: int = 4, pool_size: int = POOL):
    """
    Format data for the Neural Network.
    X: Flattened binary vectors of the previous `lookback` draws.
    Y: Binary vector of the current draw (multi-label classification).
    """
    X, Y = [], []

    binary_draws = []
    for nums in df["numbers"]:
        b = np.zeros(pool_size)
        for n in nums:
            b[n - 1] = 1
        binary_draws.append(b)

    binary_draws = np.array(binary_draws)

    for i in range(lookback, len(binary_draws)):
        x_window = binary_draws[i - lookback : i].flatten()
        y_target = binary_draws[i]
        X.append(x_window)
        Y.append(y_target)

    return np.array(X), np.array(Y), binary_draws


def main():
    print("============================================================")
    print("  STEP 8 — DEEP LEARNING A.I. & WHEELING SYSTEM (EASY6)")
    print("============================================================")

    run_folder = get_run_folder()
    df = load_data(CSV_FILE)

    LOOKBACK = 4
    X, Y, binary_draws = prepare_ai_data(df, lookback=LOOKBACK, pool_size=POOL)

    print(f"\n[1/3] Training Neural Network (Multi-Layer Perceptron)...")
    print(f"      Features: {X.shape[1]} input nodes ({LOOKBACK} draws x {POOL} balls)")
    print(f"      Training samples: {X.shape[0]} historical draws")

    # Tuned MLP configuration
    mlp = MLPClassifier(hidden_layer_sizes=(100, 50),
                        activation="relu",
                        solver="adam",
                        alpha=0.0005,
                        max_iter=1000,
                        random_state=42)

    mlp.fit(X, Y)
    print("      Model training complete.")

    last_window = binary_draws[-LOOKBACK:].flatten().reshape(1, -1)
    next_draw_probs = mlp.predict_proba(last_window)[0]

    ai_top6_indices = np.argsort(next_draw_probs)[::-1][:DRAW_SIZE]
    ai_top6 = sorted((ai_top6_indices + 1).tolist())

    ai_top14_indices = np.argsort(next_draw_probs)[::-1][:14]
    candidate_pool   = sorted((ai_top14_indices + 1).tolist())

    print(f"\n[2/3] Deep Learning A.I. Predictions:")
    print(f"      ★ AI Recommended Top-6 Ticket: {ai_top6}")
    print(f"      Candidate Pool (Top 14 AI balls): {candidate_pool}")

    print(f"\n[3/3] Generating Mathematical Wheeling System (3-if-3 Guarantee)...")

    wheeled_tickets = generate_covering_wheel(candidates=candidate_pool,
                                               ticket_size=DRAW_SIZE,
                                               match_guarantee=3)

    print(f"      Total candidate pool size: {len(candidate_pool)} balls")
    print(f"      Generated {len(wheeled_tickets)} tickets for 3-if-3 coverage:")
    for idx, t in enumerate(wheeled_tickets[:10], 1):
        print(f"        Ticket {idx:2d}: {sorted(t)}")
    if len(wheeled_tickets) > 10:
        print(f"        ... and {len(wheeled_tickets) - 10} more tickets.")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("STEP 8 — Deep Learning Neural Network & Wheeling System (EASY6)",
                 fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(range(1, POOL + 1), next_draw_probs, color="#95a5a6", edgecolor="black", linewidth=0.5)
    for idx in ai_top6_indices:
        bars[idx].set_color("#e74c3c")
    for idx in ai_top14_indices:
        if idx not in ai_top6_indices:
            bars[idx].set_color("#f39c12")

    ax1.set_title("Deep Learning MLP Probabilities per Ball (Red = Top 6, Orange = Next 8 Pool Candidates)")
    ax1.set_xlabel("Number")
    ax1.set_ylabel("Neural Net Activation Probability")
    ax1.set_xlim(0.5, POOL + 0.5)
    ax1.set_xticks(range(1, POOL + 1))
    ax1.grid(axis="y", alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    ai_text = (
        "DEEP LEARNING NEURAL NETWORK ENGINE\n"
        "───────────────────────────────────────────────\n"
        f" • Architecture: MLP (Input={X.shape[1]} -> 100 -> 50 -> Output={POOL})\n"
        f" • Lookback Context: {LOOKBACK} draws\n"
        f" • Regularization L2 (alpha): 0.0005\n\n"
        f"★ AI Top-6 Ticket: {ai_top6}\n"
        f"  Candidate Pool (14 Balls): {candidate_pool}"
    )
    ax2.text(0.05, 0.5, ai_text, transform=ax2.transAxes, fontsize=10,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#ebf5fb", alpha=0.9))

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    wheel_text = (
        "COMBINATORIAL WHEELING SYSTEM (3-if-3 Guarantee)\n"
        "───────────────────────────────────────────────\n"
        f" • Candidate Pool: {len(candidate_pool)} numbers\n"
        f" • Guaranteed Win: Match 3+ if 3 drawn numbers are in pool\n"
        f" • Total Optimized Tickets Generated: {len(wheeled_tickets)}\n\n"
        "SAMPLE GENERATED WHEELED TICKETS:\n"
    )
    for idx, t in enumerate(wheeled_tickets[:4], 1):
        wheel_text += f"  T{idx}: {sorted(t)}\n"

    ax3.text(0.05, 0.5, wheel_text, transform=ax3.transAxes, fontsize=10,
             verticalalignment="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#fef9e7", alpha=0.9))

    out = run_folder + "/step8_deep_learning_wheel.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print("\nWHAT YOU LEARNED IN STEP 8:")
    print("  ✔  How to model multi-label sequence targets using Deep Learning MLPs")
    print("  ✔  How to construct mathematical covering wheels to guarantee wins")


if __name__ == "__main__":
    main()
