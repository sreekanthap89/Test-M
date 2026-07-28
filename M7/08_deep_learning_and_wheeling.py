"""
=============================================================
 STEP 8: DEEP LEARNING A.I. & WHEELING SYSTEM (MEGA7)
=============================================================
 LEARNING GOAL:
   1. Multi-Layer Perceptron (MLP) Neural Network for multi-label
      time-series prediction.
   2. Combinatorial Wheeling (Greedy Set Cover) to generate a minimum
      set of 7-number tickets that mathematically guarantees a win
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
from utils import get_run_folder, load_data, generate_covering_wheel

CSV_FILE  = "Emirates_Draw_MEGA7.csv"
WIN_COLS  = ["Winning Number 1", "2", "3", "4", "5", "6", "7"]
POOL      = 37
DRAW_SIZE = 7


def prepare_ai_data(df: pd.DataFrame, lookback: int = 3, pool_size: int = 37):
    """
    Format data for the Neural Network.
    X: Flattened one-hot vectors of the previous `lookback` draws.
    Y: One-hot vector of the current draw (multi-label classification).
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
    print("  STEP 8 — DEEP LEARNING A.I. & WHEELING SYSTEM (MEGA7)")
    print("============================================================")

    df = load_data(CSV_FILE)
    LOOKBACK = 4

    print(f"Preparing time-series data (Lookback = {LOOKBACK} draws)...")
    X, Y, binary_draws = prepare_ai_data(df, lookback=LOOKBACK, pool_size=POOL)

    print("\nTraining Multi-Layer Perceptron (MLP) Neural Network...")
    print(f"Architecture: Input Layer ({POOL * LOOKBACK} nodes) -> Hidden (100, 50) -> Output ({POOL} nodes)")

    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        alpha=0.0005,
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )

    model.fit(X, Y)
    print(f"Model training complete. (Score on training data: {model.score(X, Y):.4f})")
    print("Note: Perfect accuracy on lottery data usually means extreme overfitting!")

    latest_x = binary_draws[-LOOKBACK:].flatten().reshape(1, -1)

    # MLPClassifier.predict_proba() returns shape (1, 37) for multi-output binary
    # classification — each value is the probability that the corresponding ball
    # appears in the next draw.
    probas = model.predict_proba(latest_x)
    next_draw_probs = probas[0]

    # Normalize to a valid probability distribution
    if next_draw_probs.sum() > 0:
        next_draw_probs = next_draw_probs / next_draw_probs.sum()

    top_14_indices = np.argsort(next_draw_probs)[::-1][:14]
    top_14_numbers = sorted((top_14_indices + 1).tolist())

    top_7_indices = np.argsort(next_draw_probs)[::-1][:7]
    top_7_numbers = sorted((top_7_indices + 1).tolist())

    print("\n============================================================")
    print("  PHASE 1: A.I. PREDICTION")
    print("============================================================")
    print(f"A.I. Top 7 Single Predicted Ticket:  ★  {top_7_numbers}  ★")
    print(f"A.I. Top 14 Candidate Pool Size  : {top_14_numbers}")

    print("\n============================================================")
    print("  PHASE 2: COMBINATORIAL WHEELING")
    print("============================================================")
    print(f"Buying all possible combinations of these 14 numbers would require {math.comb(14, 7)} tickets.")
    print("Instead, we use a Covering Design (Wheeling).")

    tickets = generate_covering_wheel(top_14_numbers, ticket_size=DRAW_SIZE, match_guarantee=3)

    print("\nYOUR WHEELED TICKETS:")
    for i, t in enumerate(tickets, 1):
        print(f"  Ticket {i:2d}: {list(t)}")

    print(f"\nMATHEMATICAL GUARANTEE: If exactly 3 of the winning numbers fall")
    print(f"anywhere inside {top_14_numbers}, you are mathematically GUARANTEED")
    print(f"to have at least one ticket matching 3 numbers.")

    run_dir = get_run_folder()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("STEP 8 — Deep Learning Neural Network & Wheeled Ticket Predictions\nEmirates Draw MEGA7",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

    # Subplot 1: Bar Chart
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(range(1, POOL + 1), next_draw_probs * 100, color='#95a5a6')
    for idx in top_14_indices:
        bars[idx].set_color('#e74c3c')
        ax1.text(idx + 1, next_draw_probs[idx] * 100 + 0.05, f"#{idx+1}", ha="center",
                 fontsize=8, color="#e74c3c", fontweight="bold")

    ax1.set_title("A.I. Neural Network Output Probabilities (Red = Top 14 AI Candidates)")
    ax1.set_xlabel("Number"); ax1.set_ylabel("Probability (%)")
    ax1.set_xlim(0.5, POOL + 0.5)
    ax1.set_xticks(range(1, POOL + 1))
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', label='Top 14 AI Candidates'),
        Patch(facecolor='#95a5a6', label='Other Pool Numbers')
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Subplot 2: Summary Panel (Top 7 AI Single Ticket & Top 14 Pool)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    summary_text = (
        "★ A.I. SINGLE TOP-7 PREDICTED TICKET ★\n"
        f"  {top_7_numbers}\n\n"
        "★ A.I. TOP-14 CANDIDATE POOL ★\n"
        f"  {top_14_numbers}\n\n"
        "★ MATHEMATICAL GUARANTEE ★\n"
        "  3-if-3 Match Guarantee across 19 tickets\n"
        "  if 3 winning balls fall inside the 14 candidate pool."
    )
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f8f9fa", edgecolor="#34495e", linewidth=1.5))

    # Subplot 3: 19 Wheeled Tickets Panel
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ticket_lines_col1 = []
    ticket_lines_col2 = []
    for i, t in enumerate(tickets, 1):
        line = f"T{i:02d}: {list(t)}"
        if i <= 10:
            ticket_lines_col1.append(line)
        else:
            ticket_lines_col2.append(line)

    text_col1 = "\n".join(ticket_lines_col1)
    text_col2 = "\n".join(ticket_lines_col2)

    ax3.text(0.02, 0.95, "★ WHEELED TICKETS (1-10) ★\n" + text_col1, transform=ax3.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2f3", edgecolor="#7f8c8d"))

    ax3.text(0.52, 0.95, "★ WHEELED TICKETS (11-19) ★\n" + text_col2, transform=ax3.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2f3", edgecolor="#7f8c8d"))

    chart_path = f"{run_dir}/step8_deep_learning_wheel.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Chart saved -> {chart_path}")


if __name__ == "__main__":
    main()
