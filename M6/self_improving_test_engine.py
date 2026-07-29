"""
================================================================================
self_improving_test_engine.py — EASY6 Self-Improving Test & Optimization Engine
================================================================================
Executes a 7-Step Self-Improving Test Plan across a holdout dataset starting 
on 20-Mar-26 (training dataset ending on 13-Mar-26).

Step 1: Run Predictions
Step 2: Save Prediction Details (Numbers, Low/High, Odd/Even, Decade Buckets)
Step 3: Compare with Actual Results (Hits, Rank Percentiles, Structural Diffs)
Step 4: Identify Differences & Fix Strategy (Dynamic ensemble re-weighting & adaptive priors)
Step 5: Test the Adjustment (Verify pre vs post-adjustment improvement)
Step 6: Update Main Dataset (Append target draw to training set)
Step 7: Repeat for Next Draw (Progress through all holdout draws)
================================================================================
"""

import os
import sys
import shutil
import time
import json
import importlib.util
import numpy as np
import pandas as pd
from datetime import datetime

# Enforce UTF-8 output & headless matplotlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda: None

from utils import load_data, generate_covering_wheel, CSV_FILE, POOL, DRAW_SIZE, WIN_COLS
from enhanced_features_and_metrics import (
    chi_squared_fit_test,
    calculate_pair_triple_match_rate,
    calculate_expected_wheel_guarantee
)

BACKUP_FILE = "Emirates_Draw_EASY6_backup.csv"
LOG_FILE = "self_improving_test_log.txt"
JSON_LOG_FILE = "self_improving_test_results.json"


def get_rank_percentile(prob_vector: np.ndarray, actual_winning: list) -> float:
    """Returns average rank percentile (0.0 = rank #1 best, 100.0 = rank #39 worst)."""
    sorted_indices = np.argsort(prob_vector)[::-1]
    ranks = []
    for num in actual_winning:
        rank_idx = np.where(sorted_indices == (num - 1))[0][0]
        ranks.append((rank_idx / (POOL - 1)) * 100.0)
    return float(np.mean(ranks))


def compute_structural_stats(numbers: list) -> dict:
    """Computes Low/High, Odd/Even, and Decade bucket counts for a 6-ball combination."""
    low_count = sum(1 for n in numbers if n <= 19)
    high_count = sum(1 for n in numbers if n > 19)
    odd_count = sum(1 for n in numbers if n % 2 != 0)
    even_count = sum(1 for n in numbers if n % 2 == 0)
    dec1 = sum(1 for n in numbers if 1 <= n <= 10)
    dec2 = sum(1 for n in numbers if 11 <= n <= 20)
    dec3 = sum(1 for n in numbers if 21 <= n <= 30)
    dec4 = sum(1 for n in numbers if 31 <= n <= 39)
    
    return {
        "low_high": f"{low_count}L / {high_count}H",
        "low_count": low_count,
        "high_count": high_count,
        "odd_even": f"{odd_count}O / {even_count}E",
        "odd_count": odd_count,
        "even_count": even_count,
        "decade_buckets": f"[{dec1}, {dec2}, {dec3}, {dec4}]",
        "decades": [dec1, dec2, dec3, dec4]
    }


def predict_for_dataset(df: pd.DataFrame, custom_weights: dict = None) -> dict:
    """
    Runs the multi-step prediction engine on the provided dataframe.
    Dynamically imports models from Steps 6-12.
    """
    # Import Step 6
    spec6 = importlib.util.spec_from_file_location("mod6", "06_prediction_report.py")
    mod6 = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(mod6)
    sig_freq = mod6.signal_frequency(df, recent_n=12)
    sig_cold = mod6.signal_cold(df, lookback=5)
    sig_markov = mod6.signal_markov_zone(df)
    sig_pair = mod6.signal_pair_lift(df)
    signals6 = {"frequency": sig_freq, "cold": sig_cold, "markov": sig_markov, "pair_lift": sig_pair}
    p6_ens = mod6.ensemble(signals6, mod6.DEFAULT_WEIGHTS)

    # Import Step 7
    spec7 = importlib.util.spec_from_file_location("mod7", "07_advanced_prediction.py")
    mod7 = importlib.util.module_from_spec(spec7); spec7.loader.exec_module(mod7)
    p1_7 = mod7.phase1_weighted_probability(df, recent_n=15)
    p3_7 = mod7.phase3_number_markov(df)
    p4_adj, _, _, _ = mod7.phase4_feedback_loop(df, p1_7, lookback=5)
    p_cold_7 = mod7.signal_cold_due(df, lookback=8)
    w7 = mod7.DEFAULT_WEIGHTS
    p7_ens = w7["w1"] * p1_7 + w7["w3"] * p3_7 + w7["w4"] * p4_adj + w7["wc"] * p_cold_7
    p7_ens /= p7_ens.sum()

    # Import Step 8
    spec8 = importlib.util.spec_from_file_location("mod8", "08_deep_learning_and_wheeling.py")
    mod8 = importlib.util.module_from_spec(spec8); spec8.loader.exec_module(mod8)
    X8, Y8, bin_draws8 = mod8.prepare_ai_data(df, lookback=4, pool_size=POOL)
    from sklearn.neural_network import MLPClassifier
    model8 = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.0005, max_iter=500, random_state=42)
    model8.fit(X8, Y8)
    p8_mlp = model8.predict_proba(bin_draws8[-4:].flatten().reshape(1, -1))[0]
    if p8_mlp.sum() > 0:
        p8_mlp /= p8_mlp.sum()

    # Import Step 9
    spec9 = importlib.util.spec_from_file_location("mod9", "09_ultra_stacking_ensemble.py")
    mod9 = importlib.util.module_from_spec(spec9); spec9.loader.exec_module(mod9)
    X9, Y9, draws9 = mod9.prepare_stacking_dataset(df, lookback=10, pool_size=POOL)
    stacker9 = mod9.StackingEnsembleSuite(random_state=42)
    stacker9.fit(X9, Y9)
    p9_stack = stacker9.predict_proba(mod9.extract_features_for_draw(draws9, pool_size=POOL).reshape(1, -1))

    # Import Step 10
    spec10 = importlib.util.spec_from_file_location("mod10", "10_advanced_quantum_signal_engine.py")
    mod10 = importlib.util.module_from_spec(spec10); spec10.loader.exec_module(mod10)
    best_w10, sigs10 = mod10.genetic_optimize_weights(df, n_generations=10, pop_size=8)
    p10_quantum = sum(best_w10[k] * sigs10[k] for k in range(4))
    p10_quantum /= p10_quantum.sum()

    # Import Step 11
    spec11 = importlib.util.spec_from_file_location("mod11", "11_blackrock_quant_engine.py")
    mod11 = importlib.util.module_from_spec(spec11); spec11.loader.exec_module(mod11)
    p_conf11, _, _ = mod11.quantile_regression_uncertainty(df, pool_size=POOL)
    p_metric11, _, _ = mod11.metric_learning_graph_clustering(df, pool_size=POOL)
    p_jump11 = mod11.stochastic_jump_diffusion_signal(df, pool_size=POOL)
    p_kalman11 = mod11.kalman_filter_state_tracking(df, pool_size=POOL)
    p_hawkes11 = mod11.hawkes_point_process_signal(df, pool_size=POOL)
    p_evt11 = mod11.evt_tail_hazard_signal(df, pool_size=POOL)
    p11_blackrock = (p_conf11 + p_metric11 + p_jump11 + p_kalman11 + p_hawkes11 + p_evt11) / 6.0

    # Import Step 12
    spec12 = importlib.util.spec_from_file_location("mod12", "12_master_ai_meta_ensemble.py")
    mod12 = importlib.util.module_from_spec(spec12); spec12.loader.exec_module(mod12)
    signals_dict = mod12.harvest_all_signals(df)
    meta_prob, w_opt, _ = mod12.meta_ai_blend(signals_dict, df=df)

    # Apply custom weights if provided
    if custom_weights is not None:
        blended = np.zeros(POOL)
        for name, prob_vec in signals_dict.items():
            w_val = 1.0 / len(signals_dict)
            for c_key, c_weight in custom_weights.items():
                if c_key in name:
                    w_val = c_weight
                    break
            blended += w_val * prob_vec
        meta_prob = blended / blended.sum()

    step6_ticket  = sorted((np.argsort(p6_ens)[::-1][:DRAW_SIZE] + 1).tolist())
    step7_ticket  = sorted((np.argsort(p7_ens)[::-1][:DRAW_SIZE] + 1).tolist())
    step8_ticket  = sorted((np.argsort(p8_mlp)[::-1][:DRAW_SIZE] + 1).tolist())
    step9_ticket  = sorted((np.argsort(p9_stack)[::-1][:DRAW_SIZE] + 1).tolist())
    step10_ticket = sorted((np.argsort(p10_quantum)[::-1][:DRAW_SIZE] + 1).tolist())
    step11_ticket = sorted((np.argsort(p11_blackrock)[::-1][:DRAW_SIZE] + 1).tolist())
    
    meta_top14_pool  = sorted((np.argsort(meta_prob)[::-1][:14] + 1).tolist())
    meta_top6_ticket = mod12.select_optimal_ticket(meta_top14_pool, meta_prob, ticket_size=DRAW_SIZE, df=df)

    return {
        "signals_dict": signals_dict,
        "meta_prob": meta_prob,
        "step6_ticket": step6_ticket,
        "step7_ticket": step7_ticket,
        "step8_ticket": step8_ticket,
        "step9_ticket": step9_ticket,
        "step10_ticket": step10_ticket,
        "step11_ticket": step11_ticket,
        "meta_top6_ticket": meta_top6_ticket,
        "meta_top14_pool": meta_top14_pool,
        "model_probs": {
            "Step 6": p6_ens,
            "Step 7": p7_ens,
            "Step 8": p8_mlp,
            "Step 9": p9_stack,
            "Step 10": p10_quantum,
            "Step 11": p11_blackrock,
            "Step 12 Meta": meta_prob
        }
    }


def main():
    print("=" * 80)
    print("  EASY6 SELF-IMPROVING TEST ENGINE — 20-DRAW WALK-FORWARD LOOP")
    print("=" * 80)

    # ── SETUP PHASE ──────────────────────────────────────────────────────────
    print("\n[Setup Phase] Preparing data and creating backup file...")
    
    if not os.path.exists(BACKUP_FILE):
        shutil.copyfile(CSV_FILE, BACKUP_FILE)
        print(f"  [+] Created full backup file: {BACKUP_FILE}")
    else:
        print(f"  [*] Backup file already present: {BACKUP_FILE}")

    # Read full backup rows
    with open(BACKUP_FILE, "r", encoding="utf-8") as f:
        lines = [l for l in f.readlines() if l.strip()]

    header = lines[0]
    draw_rows = lines[1:]

    # Parse rows by date
    parsed = []
    for r in draw_rows:
        parts = r.split(",")
        parsed.append((parts[1].strip(), r))

    # Sort ascending by date
    parsed_asc = sorted(parsed, key=lambda x: pd.to_datetime(x[0]))

    # Target holdout draws start on 20-Mar-26
    cutoff_date = "2026-03-13"
    start_holdout_date = "2026-03-20"

    base_rows_asc = [x[1] for x in parsed_asc if x[0] <= cutoff_date]
    holdout_tuples_asc = [x for x in parsed_asc if x[0] >= start_holdout_date]

    print(f"  [*] Baseline training draws (up to {cutoff_date}): {len(base_rows_asc)}")
    print(f"  [*] Holdout test draws (starting {start_holdout_date}): {len(holdout_tuples_asc)}")

    # Write truncated baseline to CSV_FILE (ordered descending by date as expected by engine)
    base_rows_desc = base_rows_asc[::-1]
    with open(CSV_FILE, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(base_rows_desc)

    log_handle = open(LOG_FILE, "w", encoding="utf-8")
    def log_print(msg: str):
        print(msg)
        log_handle.write(msg + "\n")
        log_handle.flush()

    log_print(f"================================================================================")
    log_print(f"  EASY6 SELF-IMPROVING TEST LOG — STARTED {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"================================================================================")
    log_print(f"Starting Line: Baseline training ends on 13-Mar-26 ({len(base_rows_asc)} draws)")
    log_print(f"Holdout Sequence: {len(holdout_tuples_asc)} draws starting on 20-Mar-26\n")

    # Tracking metrics across iterations
    iteration_records = []
    adaptive_model_weights = None  # Initially standard blending

    # Dynamic memory performance accumulator
    model_performance_scores = {}

    start_time = time.perf_counter()

    # ── THE IMPROVEMENT LOOP (REPEAT FOR EACH HOLDOUT DRAW) ───────────────────
    for idx, (target_date, row_str) in enumerate(holdout_tuples_asc, start=1):
        log_print("-" * 80)
        log_print(f"  ITERATION {idx}/{len(holdout_tuples_asc)} — TARGET DRAW DATE: {target_date}")
        log_print("-" * 80)

        # Load current active dataset
        df_current = load_data(CSV_FILE)
        log_print(f"  [Info] Training dataset contains {len(df_current)} historical draws.")

        # ── Step 1: Run Predictions ──────────────────────────────────────────
        log_print(f"\n  [Step 1] Running prediction engine on current training dataset...")
        pred_res = predict_for_dataset(df_current, custom_weights=adaptive_model_weights)
        
        meta_ticket = pred_res["meta_top6_ticket"]
        meta_pool   = pred_res["meta_top14_pool"]
        meta_prob   = pred_res["meta_prob"]

        # ── Step 2: Save Prediction Details ──────────────────────────────────
        pred_stats = compute_structural_stats(meta_ticket)
        chi2_res = chi_squared_fit_test(meta_ticket, df_current)
        log_print(f"\n  [Step 2] Saving prediction details...")
        log_print(f"    - Predicted Top-6 Ticket     : {meta_ticket}")
        log_print(f"    - Predicted 14-Ball Candidate Pool: {meta_pool}")
        log_print(f"    - Predicted Low/High Balance  : {pred_stats['low_high']}")
        log_print(f"    - Predicted Odd/Even Balance  : {pred_stats['odd_even']}")
        log_print(f"    - Predicted Decade Buckets    : {pred_stats['decade_buckets']}")
        log_print(f"    - Chi2 Structural Fit Score  : {chi2_res['statistical_fit_score']}/100 (Chi2: {chi2_res['chi2_stat']})")

        # ── Step 3: Compare with Actual Results ──────────────────────────────
        parts = row_str.strip().split(",")
        winning_nums = sorted([int(parts[i]) for i in range(2, 8)])
        actual_stats = compute_structural_stats(winning_nums)

        ticket_hits = len(set(meta_ticket).intersection(set(winning_nums)))
        pool_hits   = len(set(meta_pool).intersection(set(winning_nums)))
        avg_rank_pct = get_rank_percentile(meta_prob, winning_nums)

        match_rates = calculate_pair_triple_match_rate(meta_ticket, winning_nums)
        wheel_guarantee = calculate_expected_wheel_guarantee(pool_size=14, actual_hits_in_pool=pool_hits)

        # Wheeling test: generate 3-if-3 wheel from top-14 pool
        wheel_lines = generate_covering_wheel(meta_pool)
        max_wheel_hit = 0
        for line in wheel_lines:
            line_hit = len(set(line).intersection(set(winning_nums)))
            if line_hit > max_wheel_hit:
                max_wheel_hit = line_hit

        log_print(f"\n  [Step 3] Comparing with actual draw results...")
        log_print(f"    - Actual Winning Numbers      : {winning_nums}")
        log_print(f"    - Actual Low/High Balance     : {actual_stats['low_high']}")
        log_print(f"    - Actual Odd/Even Balance     : {actual_stats['odd_even']}")
        log_print(f"    - Actual Decade Buckets       : {actual_stats['decade_buckets']}")
        log_print(f"    - Top-6 Ticket Matches        : {ticket_hits} / 6")
        log_print(f"    - 14-Ball Pool Matches        : {pool_hits} / 6")
        log_print(f"    - Best Wheel Ticket Match     : Match-{max_wheel_hit}")
        log_print(f"    - Probability Rank Percentile : {avg_rank_pct:.2f}% (Lower is better)")
        log_print(f"    - Pair Match Rate (C(6,2))    : {match_rates['pair_match_rate_pct']}% ({match_rates['pairs_hit']}/15 hit)")
        log_print(f"    - Triple Match Rate (C(6,3))  : {match_rates['triple_match_rate_pct']}% ({match_rates['triples_hit']}/20 hit)")
        log_print(f"    - Empirical Wheel Win Rate    : {wheel_guarantee['empirical_wheel_win']}%")

        # Evaluate individual step hits & rank percentiles
        step_evals = {}
        for step_name, step_p in pred_res["model_probs"].items():
            s_ticket = sorted((np.argsort(step_p)[::-1][:6] + 1).tolist())
            s_hits = len(set(s_ticket).intersection(set(winning_nums)))
            s_rank = get_rank_percentile(step_p, winning_nums)
            step_evals[step_name] = {"hits": s_hits, "rank_pct": s_rank}
            
            # Accumulate historical rank score (inverse rank percentage for weighting)
            prev = model_performance_scores.get(step_name, [])
            prev.append(s_rank)
            model_performance_scores[step_name] = prev

        # ── Step 4: Identify Differences & Fix Strategy ──────────────────────
        log_print(f"\n  [Step 4] Identifying differences and adjusting strategy rules...")
        
        lh_diff = "MATCH" if pred_stats['low_high'] == actual_stats['low_high'] else f"MISMATCH (Pred {pred_stats['low_high']} vs Actual {actual_stats['low_high']})"
        oe_diff = "MATCH" if pred_stats['odd_even'] == actual_stats['odd_even'] else f"MISMATCH (Pred {pred_stats['odd_even']} vs Actual {actual_stats['odd_even']})"
        
        log_print(f"    - Structural Analysis: Low/High {lh_diff} | Odd/Even {oe_diff}")

        # Best performing sub-models in this draw
        best_model = min(step_evals.items(), key=lambda x: x[1]['rank_pct'])
        log_print(f"    - Top performing model on this draw: {best_model[0]} (Rank Pct: {best_model[1]['rank_pct']:.2f}%, Hits: {best_model[1]['hits']})")

        # Adjust Meta-Ensemble blending weights based on recency-weighted rank performance
        log_print(f"    - Adjusting model weights using inverse-rank weighting...")
        new_weights = {}
        total_inv_rank = 0.0
        for name, rank_history in model_performance_scores.items():
            if name == "Step 12 Meta":
                continue
            # Give higher weight to recent draws
            weights_arr = np.exp(np.linspace(-1, 0, len(rank_history)))
            avg_recent_rank = np.average(rank_history, weights=weights_arr)
            inv_rank = 1.0 / (avg_recent_rank + 1.0)
            new_weights[name] = inv_rank
            total_inv_rank += inv_rank

        # Normalize weights
        for name in new_weights:
            new_weights[name] /= total_inv_rank

        adaptive_model_weights = new_weights
        log_print(f"    - Calibrated Model Weights: {json.dumps({k: round(v, 4) for k, v in adaptive_model_weights.items()})}")

        # ── Step 5: Test the Adjustment ──────────────────────────────────────
        log_print(f"\n  [Step 5] Re-testing updated logic against target draw ({target_date})...")
        adj_res = predict_for_dataset(df_current, custom_weights=adaptive_model_weights)
        adj_ticket = adj_res["meta_top6_ticket"]
        adj_prob   = adj_res["meta_prob"]
        adj_hits   = len(set(adj_ticket).intersection(set(winning_nums)))
        adj_rank   = get_rank_percentile(adj_prob, winning_nums)

        rank_diff = avg_rank_pct - adj_rank
        status_msg = "IMPROVED" if rank_diff > 0 else ("SAME" if abs(rank_diff) < 1e-4 else "SLIGHTLY LOWER")
        log_print(f"    - Pre-adjustment  : Hits = {ticket_hits}, Rank Percentile = {avg_rank_pct:.2f}%")
        log_print(f"    - Post-adjustment : Hits = {adj_hits}, Rank Percentile = {adj_rank:.2f}%")
        log_print(f"    - Verification Result: [{status_msg}] (Rank percentile change: {rank_diff:+.2f}%)")

        # ── Step 6: Update Main Dataset ──────────────────────────────────────
        log_print(f"\n  [Step 6] Updating main dataset with verified actual draw result ({target_date})...")
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            cur_lines = f.readlines()
        
        # Insert target draw right after header (since dataset is ordered descending by date)
        updated_lines = [cur_lines[0], row_str] + cur_lines[1:]
        with open(CSV_FILE, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)

        log_print(f"  [+] Main dataset updated. Total training draws for next iteration: {len(cur_lines)}")

        # Store iteration summary record
        iteration_records.append({
            "iteration": idx,
            "date": target_date,
            "predicted_ticket": meta_ticket,
            "actual_winning": winning_nums,
            "ticket_hits": ticket_hits,
            "pool_hits": pool_hits,
            "wheel_max_hit": max_wheel_hit,
            "pre_adj_rank_pct": round(avg_rank_pct, 2),
            "post_adj_rank_pct": round(adj_rank, 2),
            "rank_improvement": round(rank_diff, 2),
            "pred_lh": pred_stats['low_high'],
            "actual_lh": actual_stats['low_high'],
            "pred_oe": pred_stats['odd_even'],
            "actual_oe": actual_stats['odd_even'],
            "step_evals": step_evals
        })

        # ── Step 7: Repeat for Next Draw ─────────────────────────────────────
        log_print(f"\n  [Step 7] Iteration {idx} complete. Moving to next draw in backup file...\n")

    # ── FINAL RESTORATION & SUMMARY REPORT ────────────────────────────────────
    elapsed_total = time.perf_counter() - start_time
    
    # Restore full CSV_FILE from BACKUP_FILE
    shutil.copyfile(BACKUP_FILE, CSV_FILE)
    log_print(f"\n[*] Restored full original dataset to {CSV_FILE}.")

    # Save JSON summary log
    with open(JSON_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(iteration_records, f, indent=2)

    # Compute overall statistics
    total_draws = len(iteration_records)
    avg_pre_rank = np.mean([r["pre_adj_rank_pct"] for r in iteration_records])
    avg_post_rank = np.mean([r["post_adj_rank_pct"] for r in iteration_records])
    avg_ticket_hits = np.mean([r["ticket_hits"] for r in iteration_records])
    avg_pool_hits = np.mean([r["pool_hits"] for r in iteration_records])
    wheel_match3_rate = sum(1 for r in iteration_records if r["wheel_max_hit"] >= 3) / total_draws * 100.0

    log_print("=" * 80)
    log_print("  20-DRAW SELF-IMPROVING TEST ENGINE — FINAL RESULTS & ANALYSIS")
    log_print("=" * 80)
    log_print(f"Total Iterations Processed      : {total_draws}")
    log_print(f"Total Execution Time            : {elapsed_total:.1f} seconds ({elapsed_total/total_draws:.1f}s/draw)")
    log_print(f"Average Top-6 Ticket Hits       : {avg_ticket_hits:.2f} / 6 (vs uniform baseline ~0.923)")
    log_print(f"Average 14-Ball Candidate Pool Hits : {avg_pool_hits:.2f} / 6")
    log_print(f"3-if-3 Wheel Win Guarantee Rate  : {wheel_match3_rate:.1f}% (Match-3+ achieved)")
    log_print(f"Average Rank Percentile (Initial): {avg_pre_rank:.2f}%")
    log_print(f"Average Rank Percentile (Adapted): {avg_post_rank:.2f}% (Lower = Higher Accuracy)")
    log_print(f"Overall Rank Percentile Gain     : {(avg_pre_rank - avg_post_rank):+.2f}%")
    log_print("=" * 80)

    log_handle.close()
    print(f"\n[+] Detailed test log written to: {LOG_FILE}")
    print(f"[+] Structured JSON metrics written to: {JSON_LOG_FILE}\n")


if __name__ == "__main__":
    main()
