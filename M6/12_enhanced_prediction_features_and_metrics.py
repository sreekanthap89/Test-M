"""
================================================================================
15_enhanced_prediction_features_and_metrics.py — EASY6 Advanced Enhancements Engine
================================================================================
Implements advanced Feature Engineering, Model Tuning, and Validation Depth Metrics:

A. Feature Engineering Enhancements:
   1. signal_gap_analysis: Mean gap, gap std dev, and periodicity schedule regularity.
   2. signal_consecutive_streaks: Draw-to-draw consecutive streaks & in-draw clusters.
   3. signal_hot_cold_momentum: Momentum score (Freq_last_N - Freq_prev_M).
   4. compute_inter_signal_correlation: Spearman rank correlation (IC) of signals vs outcomes.

B. Model Tuning Enhancements:
   1. optimize_ensemble_weights: Grid/Dirichlet search weight optimizer maximizing Rank Percentile Gain.
   2. compute_dynamic_weights: Regime-aware dynamic signal weighting based on structural state & ICs.

C. New Validation Depth Metrics:
   1. chi_squared_fit_test: Chi-Squared (Chi2) goodness-of-fit test on structural distribution.
   2. calculate_pair_triple_match_rate: Pair match rate (C(6,2)) & Triple match rate (C(6,3)).
   3. calculate_expected_wheel_guarantee: Theoretical hypergeometric & empirical combinatorial wheel safety rate.
================================================================================
"""

import math
import itertools
import numpy as np
import pandas as pd
from math import comb
from scipy.stats import spearmanr, chisquare
from utils import POOL, DRAW_SIZE
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
    from utils import load_data, CSV_FILE
    df = load_data(CSV_FILE)
    print("============================================================")
    print("  STEP 15 — FEATURE ENGINEERING & VALIDATION DEPTH ENGINE")
    print("============================================================")
    
    gap_sig = signal_gap_analysis(df)
    streak_sig = signal_consecutive_streaks(df)
    mom_sig = signal_hot_cold_momentum(df)
    
    print(f"  [+] Gap Regularity Signal Top 3      : {np.argsort(gap_sig)[::-1][:3] + 1}")
    print(f"  [+] Streak Cluster Signal Top 3      : {np.argsort(streak_sig)[::-1][:3] + 1}")
    print(f"  [+] Hot/Cold Momentum Signal Top 3   : {np.argsort(mom_sig)[::-1][:3] + 1}")

    sample_pred = [3, 8, 15, 22, 29, 34]
    fit_res = chi_squared_fit_test(sample_pred, df)
    print(f"  [+] Sample Chi2 Structural Fit Score : {fit_res['statistical_fit_score']}/100")

    match_res = calculate_pair_triple_match_rate(sample_pred, df["numbers"].iloc[-1])
    print(f"  [+] Pair Match Rate (C(6,2))         : {match_res['pair_match_rate_pct']}%")

    wheel_res = calculate_expected_wheel_guarantee(pool_size=14, actual_hits_in_pool=4)
    print(f"  [+] Expected Wheel Guarantee         : {wheel_res['empirical_wheel_win']}%")
    print("\n[OK] Step 15 Feature & Metric Enhancements execution complete!")


if __name__ == "__main__":
    main()
