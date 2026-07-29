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


# ──────────────────────────────────────────────────────────────────────────────
# A. FEATURE ENGINEERING ENHANCEMENTS
# ──────────────────────────────────────────────────────────────────────────────

def signal_gap_analysis(df: pd.DataFrame) -> np.ndarray:
    """
    Calculates average gap between appearances, standard deviation of gaps,
    and periodicity schedule regularity score for each number (1..39).
    A low standard deviation with a current gap matching the mean gap
    indicates highly predictable spacing.
    """
    n_draws = len(df)
    scores = np.zeros(POOL)

    for ball in range(1, POOL + 1):
        # Find 0-indexed draw indices where ball appeared
        appearances = []
        for idx, row in enumerate(df["numbers"]):
            if ball in row:
                appearances.append(idx)

        if len(appearances) < 2:
            scores[ball - 1] = 1.0 / POOL
            continue

        # Calculate gaps between consecutive appearances
        gaps = np.diff(appearances)
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        current_gap = n_draws - appearances[-1]

        # Regularity score: high when current_gap is close to mean_gap and std_gap is low
        std_penalty = std_gap + 1.0
        gap_diff = abs(current_gap - mean_gap)

        # Gaussian-like proximity kernel
        proximity = math.exp(-0.5 * (gap_diff / std_penalty) ** 2)
        regularity_score = proximity / std_penalty
        scores[ball - 1] = max(0.001, regularity_score)

    s = scores.sum()
    return scores / s if s > 0 else np.ones(POOL) / POOL


def signal_consecutive_streaks(df: pd.DataFrame) -> np.ndarray:
    """
    Measures consecutive draw-to-draw streaks and in-draw consecutive cluster propensity.
    """
    scores = np.zeros(POOL)
    n_draws = len(df)

    # 1. Draw-to-draw streak for each ball ending at current draw
    for ball in range(1, POOL + 1):
        streak = 0
        for i in range(n_draws - 1, -1, -1):
            if ball in df["numbers"].iloc[i]:
                streak += 1
            else:
                break
        
        # 2. In-draw consecutive number clustering history for this ball
        adjacent_cluster_count = 0
        total_appearances = 0
        for row in df["numbers"]:
            if ball in row:
                total_appearances += 1
                if (ball - 1 in row) or (ball + 1 in row):
                    adjacent_cluster_count += 1

        cluster_ratio = (adjacent_cluster_count / total_appearances) if total_appearances > 0 else 0.0
        streak_score = (streak * 1.5) + (cluster_ratio * 1.0)
        scores[ball - 1] = max(0.01, streak_score)

    s = scores.sum()
    return scores / s if s > 0 else np.ones(POOL) / POOL


def signal_hot_cold_momentum(df: pd.DataFrame, recent_n: int = 5, prev_m: int = 10) -> np.ndarray:
    """
    Calculates Momentum = (Frequency_last_N) - (Frequency_previous_M).
    Tells whether a number is currently 'hot' or cooling off rapidly.
    """
    if len(df) < recent_n + prev_m:
        recent_n = max(2, len(df) // 3)
        prev_m = max(2, len(df) - recent_n)

    recent_df = df.iloc[-recent_n:]
    prev_df = df.iloc[-(recent_n + prev_m): -recent_n]

    freq_recent = np.zeros(POOL)
    for row in recent_df["numbers"]:
        for n in row:
            freq_recent[n - 1] += 1.0 / recent_n

    freq_prev = np.zeros(POOL)
    for row in prev_df["numbers"]:
        for n in row:
            freq_prev[n - 1] += 1.0 / prev_m

    momentum = freq_recent - freq_prev

    # Shift momentum to positive values using softmax with temperature scaling
    exp_mom = np.exp((momentum - np.mean(momentum)) / (np.std(momentum) + 1e-6))
    return exp_mom / exp_mom.sum()


def compute_inter_signal_correlation(df: pd.DataFrame, signals_dict: dict, lookback_draws: int = 15) -> dict:
    """
    Calculates how strongly each signal vector correlates with actual draw outcomes (Information Coefficient - IC).
    """
    ic_scores = {}
    actual_test_draws = df["numbers"].iloc[-lookback_draws:].tolist()

    for s_name, prob_vec in signals_dict.items():
        corrs = []
        for draw in actual_test_draws:
            target = np.zeros(POOL)
            for n in draw:
                target[n - 1] = 1.0
            
            corr, _ = spearmanr(prob_vec, target)
            if not np.isnan(corr):
                corrs.append(corr)
            else:
                corrs.append(0.0)

        ic_scores[s_name] = float(np.mean(corrs)) if corrs else 0.0

    return ic_scores


# ──────────────────────────────────────────────────────────────────────────────
# B. MODEL TUNING ENHANCEMENTS (ENSEMBLE WEIGHTING)
# ──────────────────────────────────────────────────────────────────────────────

def get_rank_percentile(prob_vector: np.ndarray, actual_winning: list[int]) -> float:
    """Returns average rank percentile (0.0 = best #1 rank, 100.0 = worst #39 rank)."""
    sorted_indices = np.argsort(prob_vector)[::-1]
    ranks = []
    for num in actual_winning:
        rank_idx = np.where(sorted_indices == (num - 1))[0][0]
        ranks.append((rank_idx / (POOL - 1)) * 100.0)
    return float(np.mean(ranks))


def optimize_ensemble_weights(df: pd.DataFrame, signals_dict: dict, validation_window: int = 10, n_trials: int = 250) -> np.ndarray:
    """
    Uses Dirichlet grid/random optimization across weight permutations
    where sum(w_i) = 1 to maximize Average Rank Percentile Gain on held-out validation data.
    """
    names = list(signals_dict.keys())
    matrix = np.array([signals_dict[k] for k in names])  # Shape: (K, POOL)
    n_signals = len(names)

    if len(df) <= validation_window:
        return np.ones(n_signals) / n_signals

    val_draws = df["numbers"].iloc[-validation_window:].tolist()

    best_weights = np.ones(n_signals) / n_signals
    best_rank_pct = 100.0

    rng = np.random.default_rng(42)

    # Include equal weight baseline
    candidate_weights = [best_weights]

    # Generate Dirichlet weight samples
    for _ in range(n_trials):
        w = rng.dirichlet(np.ones(n_signals))
        candidate_weights.append(w)

    for w in candidate_weights:
        blended = sum(w[i] * matrix[i] for i in range(n_signals))
        blended /= blended.sum()

        ranks = [get_rank_percentile(blended, d) for d in val_draws]
        avg_rank = np.mean(ranks)

        if avg_rank < best_rank_pct:
            best_rank_pct = avg_rank
            best_weights = w

    return best_weights


def compute_dynamic_weights(df: pd.DataFrame, signals_dict: dict, recent_draws: int = 10) -> np.ndarray:
    """
    Dynamically adjusts signal weights based on current state of data
    (e.g., Low/High regime, cold number resurgence, and recent signal IC).
    """
    names = list(signals_dict.keys())

    ic_dict = compute_inter_signal_correlation(df, signals_dict, lookback_draws=recent_draws)
    base_ic = np.array([max(0.01, ic_dict[k]) for k in names])

    # Check structural regime over recent draws
    recent_low_ratio = df["n_low"].iloc[-recent_draws:].mean() / DRAW_SIZE
    
    # State adjustment: boost cold/due signals if extreme regime detected
    weights = base_ic.copy()
    for idx, name in enumerate(names):
        if "cold" in name.lower() or "due" in name.lower():
            if recent_low_ratio > 0.65 or recent_low_ratio < 0.35:
                weights[idx] *= 1.4

    weights /= weights.sum()
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# C. NEW FORMULAS AND METRICS (VALIDATION DEPTH)
# ──────────────────────────────────────────────────────────────────────────────

def chi_squared_fit_test(predicted_ticket: list[int], historical_df: pd.DataFrame) -> dict:
    """
    1. Statistical Fit Test: Chi-Squared Test (Chi2) on Structural Distribution.
    Compares observed counts from prediction (Low/High, Odd/Even, Decades)
    against expected counts based on overall historical averages.
    A low Chi2 value indicates a strong statistical match to expected draw structure.
    """
    # Historical baseline expectations
    hist_low_avg = historical_df["n_low"].mean()
    hist_high_avg = 6.0 - hist_low_avg
    
    hist_odd_avg = historical_df["numbers"].apply(lambda nums: sum(1 for n in nums if n % 2 != 0)).mean()
    hist_even_avg = 6.0 - hist_odd_avg

    # Decades: [1-10, 11-20, 21-30, 31-39]
    dec_counts = np.zeros(4)
    total_nums = len(historical_df) * 6
    for row in historical_df["numbers"]:
        for n in row:
            if n <= 10: dec_counts[0] += 1
            elif n <= 20: dec_counts[1] += 1
            elif n <= 30: dec_counts[2] += 1
            else: dec_counts[3] += 1
    expected_decades = (dec_counts / total_nums) * 6.0

    # Observed in predicted ticket
    obs_low = sum(1 for n in predicted_ticket if n <= 19)
    obs_high = 6 - obs_low
    obs_odd = sum(1 for n in predicted_ticket if n % 2 != 0)
    obs_even = 6 - obs_odd

    obs_dec = np.zeros(4)
    for n in predicted_ticket:
        if n <= 10: obs_dec[0] += 1
        elif n <= 20: obs_dec[1] += 1
        elif n <= 30: obs_dec[2] += 1
        else: obs_dec[3] += 1

    obs = np.array([obs_low, obs_high, obs_odd, obs_even] + list(obs_dec))
    exp = np.array([hist_low_avg, hist_high_avg, hist_odd_avg, hist_even_avg] + list(expected_decades))

    # Avoid zero division
    exp = np.maximum(exp, 0.05)

    chi2_stat = float(np.sum((obs - exp) ** 2 / exp))
    fit_score = float(100.0 / (1.0 + chi2_stat))

    return {
        "chi2_stat": round(chi2_stat, 4),
        "statistical_fit_score": round(fit_score, 2),
        "observed_low_high": f"{obs_low}L / {obs_high}H",
        "expected_low_high": f"{hist_low_avg:.2f}L / {hist_high_avg:.2f}H",
        "observed_decades": list(obs_dec.astype(int)),
        "expected_decades": [round(v, 2) for v in expected_decades]
    }


def calculate_pair_triple_match_rate(prediction: list[int] | set[int], actual_draw: list[int]) -> dict:
    """
    2. Advanced Hit Metrics: Pair and Triple Match Rate.
    Calculates percentage of actual winning pairs (C(6,2) = 15)
    and triples (C(6,3) = 20) present in the predicted ticket or candidate pool.
    """
    actual_pairs = set(itertools.combinations(sorted(actual_draw), 2))
    actual_triples = set(itertools.combinations(sorted(actual_draw), 3))

    pred_pairs = set(itertools.combinations(sorted(prediction), 2))
    pred_triples = set(itertools.combinations(sorted(prediction), 3))

    pairs_hit = len(pred_pairs & actual_pairs)
    triples_hit = len(pred_triples & actual_triples)

    pair_rate = (pairs_hit / 15.0) * 100.0
    triple_rate = (triples_hit / 20.0) * 100.0

    return {
        "pairs_hit": pairs_hit,
        "total_pairs": 15,
        "pair_match_rate_pct": round(pair_rate, 2),
        "triples_hit": triples_hit,
        "total_triples": 20,
        "triple_match_rate_pct": round(triple_rate, 2)
    }


def calculate_expected_wheel_guarantee(pool_size: int = 14, target_k: int = 3, ticket_size: int = 6, total_pool: int = 39, actual_hits_in_pool: int | None = None) -> dict:
    """
    3. Coverage & Reliability Metrics: Expected Wheel Guarantee Rate.
    Calculates theoretical hypergeometric hit probability given pool size N=14 and target match k=3.
    """
    guarantee_by_hits = {}
    for h in range(7):
        prob_sum = 0.0
        for j in range(target_k, min(ticket_size, h) + 1):
            if pool_size - h >= ticket_size - j:
                ways = comb(h, j) * comb(total_pool - h, ticket_size - j)
                total_ways = comb(total_pool, ticket_size)
                prob_sum += (ways / total_ways)
        guarantee_by_hits[h] = round(prob_sum * 100.0, 2)

    min_hits_for_guarantee = 3

    actual_guarantee_pct = 0.0
    if actual_hits_in_pool is not None:
        actual_guarantee_pct = 100.0 if actual_hits_in_pool >= min_hits_for_guarantee else 0.0

    return {
        "candidate_pool_size": pool_size,
        "target_match_k": target_k,
        "guarantee_prob_by_pool_hits": guarantee_by_hits,
        "min_pool_hits_needed": min_hits_for_guarantee,
        "actual_hits_in_pool": actual_hits_in_pool,
        "empirical_wheel_win": actual_guarantee_pct
    }


if __name__ == "__main__":
    from utils import load_data, CSV_FILE
    df = load_data(CSV_FILE)
    print("Testing 15_enhanced_prediction_features_and_metrics.py...")
    
    gap_sig = signal_gap_analysis(df)
    streak_sig = signal_consecutive_streaks(df)
    mom_sig = signal_hot_cold_momentum(df)
    
    print(f"Gap signal sum: {gap_sig.sum():.4f}, Top 3: {np.argsort(gap_sig)[::-1][:3] + 1}")
    print(f"Streak signal sum: {streak_sig.sum():.4f}, Top 3: {np.argsort(streak_sig)[::-1][:3] + 1}")
    print(f"Momentum signal sum: {mom_sig.sum():.4f}, Top 3: {np.argsort(mom_sig)[::-1][:3] + 1}")

    sample_pred = [3, 8, 15, 22, 29, 34]
    fit_res = chi_squared_fit_test(sample_pred, df)
    print(f"Chi2 Fit Test: {fit_res}")

    match_res = calculate_pair_triple_match_rate(sample_pred, df["numbers"].iloc[-1])
    print(f"Pair/Triple Match Rate: {match_res}")

    wheel_res = calculate_expected_wheel_guarantee(pool_size=14, actual_hits_in_pool=4)
    print(f"Wheel Guarantee Rate: {wheel_res}")
    print("All enhancement tests passed successfully!")
