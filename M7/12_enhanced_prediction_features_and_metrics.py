"""
================================================================================
12_enhanced_prediction_features_and_metrics.py — MEGA7 Feature & Metric Depth Engine
================================================================================
Implements advanced Feature Engineering, Model Tuning, and Validation Depth Metrics for MEGA7:

A. Feature Engineering Enhancements:
   1. signal_gap_analysis: Mean gap, gap std dev, and periodicity schedule regularity.
   2. signal_consecutive_streaks: Draw-to-draw consecutive streaks & in-draw clusters.
   3. signal_hot_cold_momentum: Momentum score (Freq_last_N - Freq_prev_M).

B. Model Tuning Enhancements:
   1. compute_inter_signal_correlation: Spearman rank correlation (IC) of signals vs outcomes.
   2. optimize_ensemble_weights & compute_dynamic_weights.

C. Validation Depth Metrics:
   1. chi_squared_fit_test: Chi-Squared (Chi2) goodness-of-fit test.
   2. calculate_pair_triple_match_rate: Pair match rate (C(7,2)=21) & Triple match rate (C(7,3)=35).
   3. calculate_expected_wheel_guarantee: Hypergeometric & empirical wheel safety rate.
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils import get_run_folder, load_data, CSV_FILE, POOL, DRAW_SIZE
from enhanced_features_and_metrics import (
    signal_gap_analysis,
    signal_consecutive_streaks,
    signal_hot_cold_momentum,
    compute_inter_signal_correlation,
    optimize_ensemble_weights,
    compute_dynamic_weights,
    chi_squared_fit_test,
    calculate_pair_triple_match_rate,
    calculate_expected_wheel_guarantee
)


def main():
    df = load_data(CSV_FILE)
    print("============================================================")
    print("  STEP 12 — FEATURE ENGINEERING & VALIDATION DEPTH ENGINE (MEGA7)")
    print("============================================================")
    
    gap_sig = signal_gap_analysis(df)
    streak_sig = signal_consecutive_streaks(df)
    mom_sig = signal_hot_cold_momentum(df)
    
    top3_gap = sorted((np.argsort(gap_sig)[::-1][:3] + 1).tolist())
    top3_streak = sorted((np.argsort(streak_sig)[::-1][:3] + 1).tolist())
    top3_mom = sorted((np.argsort(mom_sig)[::-1][:3] + 1).tolist())

    print(f"  [+] Gap Regularity Signal Top 3      : {top3_gap}")
    print(f"  [+] Streak Cluster Signal Top 3      : {top3_streak}")
    print(f"  [+] Hot/Cold Momentum Signal Top 3   : {top3_mom}")

    sample_pred = [3, 8, 15, 22, 29, 34, 37]
    fit_res = chi_squared_fit_test(sample_pred, df)
    print(f"  [+] Sample Chi2 Structural Fit Score : {fit_res['statistical_fit_score']}/100")

    match_res = calculate_pair_triple_match_rate(sample_pred, df["numbers"].iloc[-1])
    print(f"  [+] Pair Match Rate (C(7,2))         : {match_res['pair_match_rate_pct']}%")

    wheel_res = calculate_expected_wheel_guarantee(pool_size=14, actual_hits_in_pool=4)
    print(f"  [+] Expected Wheel Guarantee         : {wheel_res['empirical_wheel_win']}%")

    # Generate visual chart
    run_dir = get_run_folder()
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle("STEP 12 — Feature Engineering & Metric Depth Analysis\nEmirates Draw MEGA7", fontsize=14, fontweight="bold")

    axes[0].bar(range(1, POOL + 1), gap_sig * 100, color="#3498db")
    axes[0].set_title("Periodicity Gap Regularity Signal")
    axes[0].set_ylabel("Prob (%)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(1, POOL + 1), streak_sig * 100, color="#e74c3c")
    axes[1].set_title("Consecutive Streak & Cluster Propensity")
    axes[1].set_ylabel("Prob (%)")
    axes[1].grid(axis="y", alpha=0.3)

    axes[2].bar(range(1, POOL + 1), mom_sig * 100, color="#2ecc71")
    axes[2].set_title("Hot/Cold Momentum Differential")
    axes[2].set_ylabel("Prob (%)")
    axes[2].set_xlabel("Ball Number (1..37)")
    axes[2].set_xticks(range(1, POOL + 1))
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = os.path.join(run_dir, "step12_enhanced_prediction_features.png")
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Chart saved -> {chart_path}")
    print("[OK] Step 12 Feature & Metric Enhancements execution complete!")


if __name__ == "__main__":
    main()
