"""
=============================================================
 STEP 1: DATA EXPLORATION
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
from utils import get_run_folder

# ── helpers ──────────────────────────────────────────────────────────────────

CSV_FILE = "Emirates_Draw_EASY6.csv"
WIN_COLS  = ["Winning Number 1", "2", "3", "4", "5", "6"]

def load_data(path: str) -> pd.DataFrame:
    """Load and clean the EASY6 CSV file."""
    df = pd.read_csv(path, skipfooter=1, engine="python")

    # Keep only rows that have a valid date
    df = df[df["Date"].notna() & df["Date"].str.match(r"\d{4}-\d{2}-\d{2}", na=False)].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Convert winning-number columns to numeric
    for col in WIN_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Build a flat list column of the 6 drawn numbers per row
    df["numbers"] = df[WIN_COLS].apply(
        lambda row: sorted([int(v) for v in row if pd.notna(v)]), axis=1
    )
    return df


def print_section(title: str) -> None:
    border = "=" * 60
    print(f"\n{border}\n  {title}\n{border}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    run_folder = get_run_folder()
    df = load_data(CSV_FILE)

    print_section("DATASET OVERVIEW")
    print(f"  Total draws loaded : {len(df)}")
    print(f"  Date range         : {df['Date'].min().date()}  →  {df['Date'].max().date()}")
    print(f"  Columns            : {list(df.columns)}")

    # ── CONCEPT 1: Descriptive statistics ────────────────────────────────────
    print_section("CONCEPT 1 — Descriptive Statistics")
    print("""
  Descriptive statistics summarise the 'shape' of your data.
  For lottery-style data the key measures are:

    • Mean   — the average value
    • Median — the middle value (robust to outliers)
    • Std    — how spread-out the values are
    • Min/Max — the range

  For EASY6, numbers are drawn from 1–40.
  If the draw is truly random the mean of all drawn numbers
  should be close to 20 (the midpoint of 1–40).
""")
    all_numbers = [n for row in df["numbers"] for n in row]
    arr = np.array(all_numbers)
    print(f"  Total individual draws : {len(arr)}")
    print(f"  Mean                   : {arr.mean():.2f}  (expected ≈ 20.5 if uniform)")
    print(f"  Median                 : {np.median(arr):.1f}")
    print(f"  Std deviation          : {arr.std():.2f}")
    print(f"  Min / Max              : {arr.min()} / {arr.max()}")

    # ── CONCEPT 2: Sum of each draw ──────────────────────────────────────────
    print_section("CONCEPT 2 — Draw Sums")
    print("""
  The SUM of the 6 drawn numbers per draw is useful because:
    - If numbers are truly random the sum should follow a
      bell-curve (Normal distribution) centred around 6 × 20.5 = 123.
    - Consistent deviations hint at biases worth modelling.
""")
    df["sum"] = df["numbers"].apply(sum)
    print(f"  Sum  mean   : {df['sum'].mean():.1f}  (theoretical ≈ 123)")
    print(f"  Sum  std    : {df['sum'].std():.1f}")
    print(f"  Sum  range  : {df['sum'].min()} – {df['sum'].max()}")

    # ── CONCEPT 3: Range of each draw ────────────────────────────────────────
    print_section("CONCEPT 3 — Draw Range (spread)")
    print("""
  Range = highest number − lowest number in one draw.
  A small range (e.g. 8) means all 6 numbers clustered tightly.
  A large range (e.g. 38) means numbers spread across the full pool.
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
    ax1.hist(all_numbers, bins=40, range=(0.5, 40.5), color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax1.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean = {arr.mean():.1f}")
    ax1.axvline(20.5, color="orange", linestyle=":", linewidth=1.5, label="Theoretical mean (20.5)")
    ax1.set_title("Histogram of all drawn numbers (1–40)")
    ax1.set_xlabel("Number"); ax1.set_ylabel("Frequency")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)

    # 1b  — Box-plot of winning numbers by position
    ax2 = fig.add_subplot(gs[0, 2])
    positions = [df[c].dropna().values for c in WIN_COLS]
    ax2.boxplot(positions, tick_labels=["N1", "N2", "N3", "N4", "N5", "N6"], patch_artist=True,
                boxprops=dict(facecolor="#4C72B0", alpha=0.6))
    ax2.set_title("Spread per draw position")
    ax2.set_ylabel("Number value"); ax2.grid(axis="y", alpha=0.3)

    # 1c  — Sum distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(df["sum"], bins=25, color="#55A868", edgecolor="white")
    ax3.axvline(df["sum"].mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {df['sum'].mean():.0f}")
    ax3.axvline(123, color="orange", linestyle=":", linewidth=1.5, label="Theoretical (123)")
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
    ax5.axhline(123, color="orange", linestyle=":", linewidth=1.2, label="Theoretical (123)")
    ax5.set_title("Sum over time")
    ax5.set_xlabel("Date"); ax5.set_ylabel("Sum")
    ax5.tick_params(axis="x", rotation=30); ax5.legend(fontsize=8); ax5.grid(alpha=0.3)

    out = run_folder + "/step1_data_exploration.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n  Chart saved -> {out}")
    plt.show()

    print_section("WHAT YOU LEARNED IN STEP 1")
    print("""
  ✔  How to load and clean real-world CSV data with pandas
  ✔  What descriptive statistics tell you (mean, std, range)
  ✔  How to check whether data looks 'random' vs biased
  ✔  How to visualise distributions with histograms and box-plots

  NEXT STEP → Run 02_frequency_analysis.py
""")


if __name__ == "__main__":
    main()
