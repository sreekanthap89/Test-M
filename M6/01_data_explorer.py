"""
=============================================================
 STEP 1: DATA EXPLORATION — EMIRATES DRAW EASY6
=============================================================
 LEARNING GOAL:
   Before you can predict anything you must UNDERSTAND your data.
   This script teaches:
     - How to load and clean real-world CSV data
     - How to compute basic descriptive statistics
     - How to spot patterns at a glance

 KEY CONCEPTS INTRODUCED:
   * pandas DataFrame
   * Descriptive statistics (mean, median, std)
   * Data distributions
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


def print_section(title: str) -> None:
    border = "=" * 60
    print(f"\n{border}\n  {title}\n{border}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)

    print_section("DATASET OVERVIEW — EMIRATES DRAW EASY6")
    print(f"  Total draws loaded : {len(df)}")
    print(f"  Date range         : {df['Date'].min().date()}  →  {df['Date'].max().date()}")
    print(f"  Columns            : {list(df.columns)}")

    # ── CONCEPT 1: Descriptive statistics ────────────────────────────────────
    print_section("CONCEPT 1 — Descriptive Statistics")
    print(f"""
  Descriptive statistics summarise the 'shape' of your data.
  For EASY6, numbers are drawn from 1–{POOL}.
  If the draw is truly random the mean of all drawn numbers
  should be close to {(POOL+1)/2:.1f} (the midpoint of 1–{POOL}).
""")
    all_numbers = [n for row in df["numbers"] for n in row]
    arr = np.array(all_numbers)
    print(f"  Total individual draws : {len(arr)}")
    print(f"  Mean                   : {arr.mean():.2f}  (expected ≈ {(POOL+1)/2:.1f} if uniform)")
    print(f"  Median                 : {np.median(arr):.1f}")
    print(f"  Std deviation          : {arr.std():.2f}")
    print(f"  Min / Max              : {arr.min()} / {arr.max()}")

    # ── CONCEPT 2: Sum of each draw ──────────────────────────────────────────
    print_section("CONCEPT 2 — Draw Sums")
    print(f"""
  The SUM of the {DRAW_SIZE} drawn numbers per draw is useful because:
    - If numbers are truly random the sum should follow a
      bell-curve (Normal distribution) centred around {DRAW_SIZE} × {(POOL+1)/2:.1f} = {DRAW_SIZE*(POOL+1)/2:.1f}.
    - Consistent deviations hint at biases worth modelling.
""")
    print(f"  Sum  mean   : {df['sum'].mean():.1f}  (theoretical ≈ {DRAW_SIZE*(POOL+1)/2:.1f})")
    print(f"  Sum  std    : {df['sum'].std():.1f}")
    print(f"  Sum  range  : {df['sum'].min()} – {df['sum'].max()}")

    # ── CONCEPT 3: Range of each draw ────────────────────────────────────────
    print_section("CONCEPT 3 — Draw Range (spread)")
    print(f"""
  Range = highest number − lowest number in one draw.
  A small range means all {DRAW_SIZE} numbers clustered tightly.
  A large range means numbers spread across the full pool.
  Most draws should land in the middle.
""")
    df["range"] = df["numbers"].apply(lambda x: x[-1] - x[0])
    print(f"  Range  mean : {df['range'].mean():.1f}")
    print(f"  Range  std  : {df['range'].std():.1f}")
    print(f"  Range  min  : {df['range'].min()}")
    print(f"  Range  max  : {df['range'].max()}")

    # ── VISUALISATION ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("STEP 1 — Emirates Draw EASY6 — Data Exploration", fontsize=16, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1a  — Histogram of all drawn numbers
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(all_numbers, bins=POOL, range=(0.5, POOL + 0.5), color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax1.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {arr.mean():.1f}")
    ax1.axvline((POOL+1)/2, color="orange", linestyle=":", linewidth=1.5, label=f"Theoretical mean ({(POOL+1)/2:.1f})")
    ax1.set_title(f"Histogram of all drawn numbers (1–{POOL})")
    ax1.set_xlabel("Number"); ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # 1b  — Box-plot of winning numbers by position
    ax2 = fig.add_subplot(gs[0, 2])
    positions = [df[c].dropna().values for c in WIN_COLS]
    ax2.boxplot(positions, tick_labels=[f"N{i+1}" for i in range(DRAW_SIZE)], patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    ax2.set_title("Spread per draw position")
    ax2.set_ylabel("Number value"); ax2.grid(axis="y", alpha=0.3)

    # 1c  — Sum distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(df["sum"], bins=25, color="#55A868", edgecolor="white")
    ax3.axvline(df["sum"].mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {df['sum'].mean():.0f}")
    ax3.axvline(DRAW_SIZE*(POOL+1)/2, color="orange", linestyle=":", linewidth=1.5, label=f"Theoretical ({DRAW_SIZE*(POOL+1)/2:.0f})")
    ax3.set_title("Distribution of draw sums")
    ax3.set_xlabel("Sum"); ax3.set_ylabel("Count")
    ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)

    # 1d  — Range distribution
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(df["range"], bins=20, color="#C44E52", edgecolor="white")
    ax4.axvline(df["range"].mean(), color="red", linestyle="--", linewidth=1.5)
    ax4.set_title("Distribution of draw range")
    ax4.set_xlabel("Range (max − min)"); ax4.set_ylabel("Count")
    ax4.grid(axis="y", alpha=0.3)

    # 1e  — Sum over time
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(df["Date"], df["sum"], color="#8172B2", linewidth=0.8, alpha=0.7)
    ax5.axhline(DRAW_SIZE*(POOL+1)/2, color="orange", linestyle=":", linewidth=1.2, label=f"Theoretical ({DRAW_SIZE*(POOL+1)/2:.0f})")
    ax5.set_title("Sum over time")
    ax5.set_xlabel("Date"); ax5.set_ylabel("Sum")
    ax5.tick_params(axis="x", rotation=30); ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    out = run_folder + "/step1_data_exploration.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 1")
    print("""
  ✔  How to load and clean real-world EASY6 CSV data with pandas
  ✔  What descriptive statistics tell you (mean, std, range)
  ✔  How to check whether data looks 'random' vs biased
  ✔  How to visualise distributions with histograms and box-plots

  NEXT STEP → Run 02_frequency_analysis.py
""")


if __name__ == "__main__":
    main()
