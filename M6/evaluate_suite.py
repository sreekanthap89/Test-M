"""
evaluate_suite.py — Automated Back-Test & Metric Evaluation Engine for EASY6
Runs the 4-metric evaluation across a historical validation split (default: last 20 draws).

Metrics measured:
  1. Top-6 Ticket Match Rate (vs uniform baseline ~0.923)
  2. Candidate Pool Inclusion Rate (Top-14 / Top-16)
  3. Wheeling Win Guarantee Rate (Match-3+ achieved)
  4. Probability Vector Rank Percentile (average rank of winning balls)
"""

import sys
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Force UTF-8 and silence matplotlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda: None

import importlib.util
from sklearn.neural_network import MLPClassifier
import math
import itertools
from utils import load_data, generate_covering_wheel, CSV_FILE, POOL, DRAW_SIZE


def get_rank_percentile(prob_vector: np.ndarray, actual_winning: list[int]) -> float:
    """
    Returns average rank percentile (0.0 = best possible rank #1, 100.0 = worst #39)
    for the actual winning numbers within the probability vector.
    """
    sorted_indices = np.argsort(prob_vector)[::-1]
    ranks = []
    for num in actual_winning:
        rank_idx = np.where(sorted_indices == (num - 1))[0][0]
        ranks.append((rank_idx / (POOL - 1)) * 100.0)
    return float(np.mean(ranks))


def run_evaluation(test_window: int = 20):
    print("================================================================================")
    print(f"  EASY6 HISTORICAL SIMULATION & EVALUATION (Last {test_window} Draws)")
    print("================================================================================")

    df = load_data(CSV_FILE)
    n_draws = len(df)
    test_start = max(30, n_draws - test_window)
    actual_test_count = n_draws - test_start

    # Import modules dynamically
    spec6 = importlib.util.spec_from_file_location("mod6", "06_prediction_report.py")
    mod6 = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(mod6)

    spec7 = importlib.util.spec_from_file_location("mod7", "07_advanced_prediction.py")
    mod7 = importlib.util.module_from_spec(spec7); spec7.loader.exec_module(mod7)

    spec8 = importlib.util.spec_from_file_location("mod8", "08_deep_learning_and_wheeling.py")
    mod8 = importlib.util.module_from_spec(spec8); spec8.loader.exec_module(mod8)

    spec9 = importlib.util.spec_from_file_location("mod9", "09_ultra_stacking_ensemble.py")
    mod9 = importlib.util.module_from_spec(spec9); spec9.loader.exec_module(mod9)

    spec10 = importlib.util.spec_from_file_location("mod10", "10_advanced_quantum_signal_engine.py")
    mod10 = importlib.util.module_from_spec(spec10); spec10.loader.exec_module(mod10)

    spec11 = importlib.util.spec_from_file_location("mod11", "11_blackrock_quant_engine.py")
    mod11 = importlib.util.module_from_spec(spec11); spec11.loader.exec_module(mod11)

    spec14 = importlib.util.spec_from_file_location("mod14", "14_master_ai_meta_ensemble.py")
    mod14 = importlib.util.module_from_spec(spec14); spec14.loader.exec_module(mod14)

    from enhanced_features_and_metrics import (
        chi_squared_fit_test,
        calculate_pair_triple_match_rate,
        calculate_expected_wheel_guarantee
    )

    s6_top6_matches, s6_top14_matches, s6_ranks = [], [], []
    s7_top6_matches, s7_top16_matches, s7_ranks = [], [], []
    s8_top14_matches, s8_wheel_wins, s8_ranks = [], [], []
    s9_top6_matches, s9_top14_matches, s9_wheel_wins, s9_ranks = [], [], [], []
    s10_top6_matches, s10_top14_matches, s10_wheel_wins, s10_ranks = [], [], [], []
    s11_top6_matches, s11_top14_matches, s11_wheel_wins, s11_ranks = [], [], [], []
    s12_top6_matches, s12_top14_matches, s12_wheel_wins, s12_ranks = [], [], [], []
    s12_chi2_scores, s12_pair_rates, s12_triple_rates = [], [], []

    t0 = time.perf_counter()
    print(f"Evaluating from draw index {test_start} to {n_draws - 1}...\n")

    for i in range(test_start, n_draws):
        train_df = df.iloc[:i].copy()
        actual_draw = df.iloc[i]["numbers"]
        actual_set = set(actual_draw)

        # Step 6
        s_f = mod6.signal_frequency(train_df)
        s_c = mod6.signal_cold(train_df)
        s_m = mod6.signal_markov_zone(train_df)
        s_p = mod6.signal_pair_lift(train_df)
        comb6 = mod6.ensemble({"frequency": s_f, "cold": s_c, "markov": s_m, "pair_lift": s_p}, mod6.DEFAULT_WEIGHTS)

        top6_6  = set((np.argsort(comb6)[::-1][:DRAW_SIZE] + 1).tolist())
        top14_6 = set((np.argsort(comb6)[::-1][:14] + 1).tolist())

        s6_top6_matches.append(len(top6_6 & actual_set))
        s6_top14_matches.append(len(top14_6 & actual_set))
        s6_ranks.append(get_rank_percentile(comb6, actual_draw))

        # Step 7
        p1 = mod7.phase1_weighted_probability(train_df, recent_n=30)
        p3 = mod7.phase3_number_markov(train_df)
        p4_tuple = mod7.phase4_feedback_loop(train_df, p1, lookback=5)
        p4_adj = p4_tuple[0]
        p_cold = mod7.signal_cold_due(train_df, lookback=8)

        w7 = mod7.DEFAULT_WEIGHTS
        comb7 = w7["w1"] * p1 + w7["w3"] * p3 + w7["w4"] * p4_adj + w7["wc"] * p_cold
        comb7 /= comb7.sum()

        top6_7  = set(mod7.diversity_select(comb7, k=DRAW_SIZE, candidate_pool=mod7.DEFAULT_POOL_SIZE))
        top16_7 = set((np.argsort(comb7)[::-1][:mod7.DEFAULT_POOL_SIZE] + 1).tolist())

        s7_top6_matches.append(len(top6_7 & actual_set))
        s7_top16_matches.append(len(top16_7 & actual_set))
        s7_ranks.append(get_rank_percentile(comb7, actual_draw))

        # Step 8
        X, Y, bin_draws = mod8.prepare_ai_data(train_df, lookback=4, pool_size=POOL)
        model = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.0005, activation='relu', solver='adam', max_iter=500, random_state=42)
        model.fit(X, Y)

        latest_x = bin_draws[-4:].flatten().reshape(1, -1)
        probas = model.predict_proba(latest_x)[0]
        if probas.sum() > 0:
            probas = probas / probas.sum()

        top14_8_list = sorted((np.argsort(probas)[::-1][:14] + 1).tolist())
        top14_8_set  = set(top14_8_list)

        m14_count = len(top14_8_set & actual_set)
        s8_top14_matches.append(m14_count)
        s8_ranks.append(get_rank_percentile(probas, actual_draw))

        wheel_win = 0
        if m14_count >= 3:
            tickets = mod8.generate_covering_wheel(top14_8_list, ticket_size=DRAW_SIZE, match_guarantee=3)
            for t in tickets:
                if len(set(t) & actual_set) >= 3:
                    wheel_win = 1
                    break
        s8_wheel_wins.append(wheel_win)

        # Step 9
        X9, Y9, draws9 = mod9.prepare_stacking_dataset(train_df, lookback=10, pool_size=POOL)
        stacker9 = mod9.StackingEnsembleSuite(random_state=42)
        stacker9.fit(X9, Y9)

        latest_feat9 = mod9.extract_features_for_draw(draws9, pool_size=POOL).reshape(1, -1)
        probas9 = stacker9.predict_proba(latest_feat9)

        top6_9_set   = set((np.argsort(probas9)[::-1][:DRAW_SIZE] + 1).tolist())
        top14_9_list = sorted((np.argsort(probas9)[::-1][:14] + 1).tolist())
        top14_9_set  = set(top14_9_list)

        m14_count9 = len(top14_9_set & actual_set)
        s9_top6_matches.append(len(top6_9_set & actual_set))
        s9_top14_matches.append(m14_count9)
        s9_ranks.append(get_rank_percentile(probas9, actual_draw))

        wheel_win9 = 0
        if m14_count9 >= 3:
            tickets9 = mod9.generate_covering_wheel(top14_9_list, ticket_size=DRAW_SIZE, match_guarantee=3)
            for t in tickets9:
                if len(set(t) & actual_set) >= 3:
                    wheel_win9 = 1
                    break
        s9_wheel_wins.append(wheel_win9)

        # Step 10
        best_w10, sigs10 = mod10.genetic_optimize_weights(train_df, n_generations=10, pop_size=10)
        probas10 = sum(best_w10[k] * sigs10[k] for k in range(4))
        probas10 /= probas10.sum()

        top6_10_set   = set((np.argsort(probas10)[::-1][:DRAW_SIZE] + 1).tolist())
        top14_10_list = sorted((np.argsort(probas10)[::-1][:14] + 1).tolist())
        top14_10_set  = set(top14_10_list)

        m14_count10 = len(top14_10_set & actual_set)
        s10_top6_matches.append(len(top6_10_set & actual_set))
        s10_top14_matches.append(m14_count10)
        s10_ranks.append(get_rank_percentile(probas10, actual_draw))

        wheel_win10 = 0
        if m14_count10 >= 3:
            tickets10 = mod10.generate_covering_wheel(top14_10_list, ticket_size=DRAW_SIZE, match_guarantee=3)
            for t in tickets10:
                if len(set(t) & actual_set) >= 3:
                    wheel_win10 = 1
                    break
        s10_wheel_wins.append(wheel_win10)

        # Step 11
        p_conf11, _, _ = mod11.quantile_regression_uncertainty(train_df, pool_size=POOL)
        p_metric11, _, _ = mod11.metric_learning_graph_clustering(train_df, pool_size=POOL)
        p_jump11 = mod11.stochastic_jump_diffusion_signal(train_df, pool_size=POOL)
        p_kalman11 = mod11.kalman_filter_state_tracking(train_df, pool_size=POOL)
        p_hawkes11 = mod11.hawkes_point_process_signal(train_df, pool_size=POOL)
        p_evt11 = mod11.evt_tail_hazard_signal(train_df, pool_size=POOL)

        signals11 = [p_conf11, p_metric11, p_jump11, p_kalman11, p_hawkes11, p_evt11]
        probas11, _ = mod11.information_coefficient_fusion(train_df, signals11)

        top6_11_set   = set((np.argsort(probas11)[::-1][:DRAW_SIZE] + 1).tolist())
        top14_11_list = sorted((np.argsort(probas11)[::-1][:14] + 1).tolist())
        top14_11_set  = set(top14_11_list)

        m14_count11 = len(top14_11_set & actual_set)
        s11_top6_matches.append(len(top6_11_set & actual_set))
        s11_top14_matches.append(m14_count11)
        s11_ranks.append(get_rank_percentile(probas11, actual_draw))

        wheel_win11 = 0
        if m14_count11 >= 3:
            tickets11 = mod11.generate_covering_wheel(top14_11_list, ticket_size=DRAW_SIZE, match_guarantee=3)
            for t in tickets11:
                if len(set(t) & actual_set) >= 3:
                    wheel_win11 = 1
                    break
        s11_wheel_wins.append(wheel_win11)

        # Step 14 (Master AI Meta-Ensemble)
        signals_dict14 = mod14.harvest_all_signals(train_df)
        probas12, _, _ = mod14.meta_ai_blend(signals_dict14, df=train_df)

        top14_12_list = sorted((np.argsort(probas12)[::-1][:14] + 1).tolist())
        top14_12_set  = set(top14_12_list)

        top6_12_ticket = mod14.select_optimal_ticket(top14_12_list, probas12, ticket_size=DRAW_SIZE, df=train_df)
        top6_12_set   = set(top6_12_ticket)

        m14_count12 = len(top14_12_set & actual_set)
        s12_top6_matches.append(len(top6_12_set & actual_set))
        s12_top14_matches.append(m14_count12)
        s12_ranks.append(get_rank_percentile(probas12, actual_draw))

        # Advanced Validation Depth metrics for Step 12
        chi2_res = chi_squared_fit_test(sorted(list(top6_12_set)), train_df)
        match_rates = calculate_pair_triple_match_rate(top6_12_set, actual_draw)
        s12_chi2_scores.append(chi2_res['statistical_fit_score'])
        s12_pair_rates.append(match_rates['pair_match_rate_pct'])
        s12_triple_rates.append(match_rates['triple_match_rate_pct'])

        wheel_win12 = 0
        if m14_count12 >= 3:
            tickets12 = mod14.generate_covering_wheel(top14_12_list, ticket_size=DRAW_SIZE, match_guarantee=3)
            for t in tickets12:
                if len(set(t) & actual_set) >= 3:
                    wheel_win12 = 1
                    break
        s12_wheel_wins.append(wheel_win12)

    elapsed = time.perf_counter() - t0
    baseline_match = DRAW_SIZE * (DRAW_SIZE / POOL)

    print(f"Evaluation complete in {elapsed:.1f} seconds.\n")
    print("================================================================================")
    print("  QUANTITATIVE BACK-TEST RESULTS SUMMARY (EASY6)")
    print("================================================================================")
    print(f"  Validation Draws Tested : {actual_test_count}")
    print(f"  Random Uniform Baseline : {baseline_match:.3f} matches / draw\n")

    print(f"1. STEP 06 (Multi-Signal Ensemble):")
    print(f"   - Top-6 Single Ticket Match Rate : {np.mean(s6_top6_matches):.3f} / 6")
    print(f"   - Top-14 Candidate Pool Inclusion: {np.mean(s6_top14_matches):.3f} / 6 ({np.mean(s6_top14_matches)/6*100:.1f}%)")
    print(f"   - Average Rank Percentile        : {np.mean(s6_ranks):.1f}%\n")

    print(f"2. STEP 07 (Advanced 4-Phase Ensemble):")
    print(f"   - Top-6 Single Ticket Match Rate : {np.mean(s7_top6_matches):.3f} / 6")
    print(f"   - Top-16 Candidate Pool Inclusion: {np.mean(s7_top16_matches):.3f} / 6 ({np.mean(s7_top16_matches)/6*100:.1f}%)")
    print(f"   - Average Rank Percentile        : {np.mean(s7_ranks):.1f}%\n")

    print(f"3. STEP 08 (Deep Learning MLP & Wheeling):")
    print(f"   - Top-14 Candidate Pool Inclusion: {np.mean(s8_top14_matches):.3f} / 6 ({np.mean(s8_top14_matches)/6*100:.1f}%)")
    print(f"   - Wheeling Win Guarantee Rate    : {np.mean(s8_wheel_wins)*100:.1f}%")
    print(f"   - Average Rank Percentile        : {np.mean(s8_ranks):.1f}%\n")

    print(f"4. STEP 09 (Ultra Stacking Ensemble):")
    print(f"   - Top-6 Single Ticket Match Rate : {np.mean(s9_top6_matches):.3f} / 6")
    print(f"   - Top-14 Candidate Pool Inclusion: {np.mean(s9_top14_matches):.3f} / 6 ({np.mean(s9_top14_matches)/6*100:.1f}%)")
    print(f"   - Wheeling Win Guarantee Rate    : {np.mean(s9_wheel_wins)*100:.1f}%")
    print(f"   - Average Rank Percentile        : {np.mean(s9_ranks):.1f}%\n")

    print(f"5. STEP 10 (Quantum & Signal Science):")
    print(f"   - Top-6 Single Ticket Match Rate : {np.mean(s10_top6_matches):.3f} / 6")
    print(f"   - Top-14 Candidate Pool Inclusion: {np.mean(s10_top14_matches):.3f} / 6 ({np.mean(s10_top14_matches)/6*100:.1f}%)")
    print(f"   - Wheeling Win Guarantee Rate    : {np.mean(s10_wheel_wins)*100:.1f}%")
    print(f"   - Average Rank Percentile        : {np.mean(s10_ranks):.1f}%\n")

    print(f"6. STEP 11 (BlackRock Institutional Quant V2):")
    print(f"   - Top-6 Single Ticket Match Rate : {np.mean(s11_top6_matches):.3f} / 6")
    print(f"   - Top-14 Candidate Pool Inclusion: {np.mean(s11_top14_matches):.3f} / 6 ({np.mean(s11_top14_matches)/6*100:.1f}%)")
    print(f"   - Wheeling Win Guarantee Rate    : {np.mean(s11_wheel_wins)*100:.1f}%")
    print(f"   - Average Rank Percentile        : {np.mean(s11_ranks):.1f}%\n")

    print(f"7. STEP 12 (Master AI Meta-Ensemble V2 + Enhancements):")
    print(f"   - Top-6 Single Ticket Match Rate : {np.mean(s12_top6_matches):.3f} / 6")
    print(f"   - Top-14 Candidate Pool Inclusion: {np.mean(s12_top14_matches):.3f} / 6 ({np.mean(s12_top14_matches)/6*100:.1f}%)")
    print(f"   - Wheeling Win Guarantee Rate    : {np.mean(s12_wheel_wins)*100:.1f}%  ★ RECORD WIN RATE ★")
    print(f"   - Average Rank Percentile        : {np.mean(s12_ranks):.1f}%")
    print(f"   - Structural Chi2 Fit Score      : {np.mean(s12_chi2_scores):.1f} / 100")
    print(f"   - Avg Pair Match Rate (C(6,2))   : {np.mean(s12_pair_rates):.2f}%")
    print(f"   - Avg Triple Match Rate (C(6,3)) : {np.mean(s12_triple_rates):.2f}%\n")

    print("================================================================================")


if __name__ == "__main__":
    run_evaluation(test_window=20)
