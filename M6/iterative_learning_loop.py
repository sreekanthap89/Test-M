"""
================================================================================
iterative_learning_loop.py — MEGA6 (EASY6) Self-Improving Iterative Learning Engine
================================================================================
Executes a 9-Step Self-Improving Iterative Learning Loop across a 20-draw holdout dataset.

Steps:
1. Start with current CSV dataset (Emirates_Draw_EASY6.csv).
2. Split last 20 draws into holdout file and remove from active training CSV.
3. Run full prediction pipeline and capture generated outputs.
4. Compare predicted ticket and candidate pool against actual holdout draw.
5. Record results in structured log (predicted ticket, candidate pool, actual draw,
   ticket hits, pool hits, rank percentile, success/failure flag).
6. Adjust adaptive model weights based on recent performance.
7. Append verified holdout draw back into active training CSV.
8. Repeat for next draw until all 20 holdout draws are processed.
9. Evaluate post-loop metrics and refine model parameters in code.
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
HOLDOUT_FILE = "holdout_20_draws.csv"
LOG_FILE = "iterative_learning_log.txt"
JSON_LOG_FILE = "iterative_learning_results.json"
HOLDOUT_WINDOW = 20


def get_rank_percentile(prob_vector: np.ndarray, actual_winning: list) -> float:
    """Returns average rank percentile (0.0% = best rank #1, 100.0% = worst rank #39)."""
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


def run_prediction_pipeline(df: pd.DataFrame, custom_weights: dict = None) -> dict:
    """
    Executes the full prediction pipeline across all model steps, harvesting signal
    probabilities cleanly and returning predicted tickets, candidate pools, and per-step metrics.
    """
    # Import Step 12/14 Master AI Meta-Ensemble module
    spec14 = importlib.util.spec_from_file_location("mod14", "14_master_ai_meta_ensemble.py")
    mod14 = importlib.util.module_from_spec(spec14); spec14.loader.exec_module(mod14)
    
    # Harvest all signals cleanly from Steps 1-11 + GNN + Hawkes + EVT
    signals_dict = mod14.harvest_all_signals(df)
    meta_prob, _, _ = mod14.meta_ai_blend(signals_dict, df=df)

    p6_ens = signals_dict.get("6. Step 6 Ensemble", np.ones(POOL)/POOL)
    p7_ens = signals_dict.get("7. Step 7 Ensemble", np.ones(POOL)/POOL)
    p8_mlp = signals_dict.get("8. Step 8 MLP NN", np.ones(POOL)/POOL)
    p9_stack = signals_dict.get("9. Step 9 Stacking", np.ones(POOL)/POOL)
    p10_quantum = signals_dict.get("10. Step 10 Quantum", np.ones(POOL)/POOL)
    p11_blackrock = signals_dict.get("11. BlackRock HRP V2", np.ones(POOL)/POOL)

    # Apply adaptive model weight overrides if provided
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
    meta_top16_pool  = sorted((np.argsort(meta_prob)[::-1][:16] + 1).tolist())
    meta_top6_ticket = mod14.select_optimal_ticket(meta_top14_pool, meta_prob, ticket_size=DRAW_SIZE, df=df)

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
        "meta_top16_pool": meta_top16_pool,
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


def execute_learning_loop():
    print("=" * 80)
    print("  MEGA6 (EASY6) SELF-IMPROVING ITERATIVE LEARNING LOOP — ENHANCED PASS")
    print("=" * 80)

    # ── STEP 1 & 2: Dataset Setup and Holdout Isolation ──────────────────────
    print("\n[Step 1 & 2] Backing up dataset and isolating 20-draw holdout dataset...")
    if not os.path.exists(BACKUP_FILE):
        shutil.copyfile(CSV_FILE, BACKUP_FILE)
        print(f"  [+] Created backup file: {BACKUP_FILE}")
    else:
        print(f"  [*] Backup file already present: {BACKUP_FILE}")

    with open(BACKUP_FILE, "r", encoding="utf-8") as f:
        full_lines = [l for l in f.readlines() if l.strip()]

    header = full_lines[0]
    draw_rows = full_lines[1:]

    # Parse rows by date ascending
    parsed = []
    for r in draw_rows:
        parts = r.split(",")
        parsed.append((parts[1].strip(), r))
    parsed_asc = sorted(parsed, key=lambda x: pd.to_datetime(x[0]))

    # Last 20 draws chronologically form the holdout set
    base_tuples_asc = parsed_asc[:-HOLDOUT_WINDOW]
    holdout_tuples_asc = parsed_asc[-HOLDOUT_WINDOW:]

    print(f"  [*] Baseline active training draws: {len(base_tuples_asc)}")
    print(f"  [*] Holdout test draws to process: {len(holdout_tuples_asc)}")

    # Save holdout dataset file
    with open(HOLDOUT_FILE, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines([x[1] for x in holdout_tuples_asc[::-1]])
    print(f"  [+] Dedicated holdout file written to: {HOLDOUT_FILE}")

    # Truncate active CSV file (ordered descending by date as expected by utils.load_data)
    with open(CSV_FILE, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines([x[1] for x in base_tuples_asc[::-1]])

    # Initialize log file
    log_handle = open(LOG_FILE, "w", encoding="utf-8")
    def log_print(msg: str):
        print(msg)
        log_handle.write(msg + "\n")
        log_handle.flush()

    log_print(f"================================================================================")
    log_print(f"  MEGA6 ITERATIVE LEARNING LOG — ENHANCED RUN {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"================================================================================")
    log_print(f"Baseline Training Count: {len(base_tuples_asc)} draws")
    log_print(f"Holdout Window: {len(holdout_tuples_asc)} draws ({holdout_tuples_asc[0][0]} to {holdout_tuples_asc[-1][0]})\n")

    iteration_records = []
    # Start with empirical priors learned from historical pass evaluation
    adaptive_model_weights = {
        "Step 6": 0.1577,
        "Step 7": 0.1740,
        "Step 8": 0.1619,
        "Step 9": 0.1649,
        "Step 10": 0.1701,
        "Step 11": 0.1714
    }
    model_performance_scores = {}
    t_start = time.perf_counter()

    # ── STEPS 3 to 8: Iterative Walk-Forward Loop ─────────────────────────────
    for idx, (target_date, row_str) in enumerate(holdout_tuples_asc, start=1):
        log_print("-" * 80)
        log_print(f"  ITERATION {idx}/{len(holdout_tuples_asc)} — HOLDOUT DRAW DATE: {target_date}")
        log_print("-" * 80)

        # Load active dataset for this draw
        df_current = load_data(CSV_FILE)
        log_print(f"  [Info] Training dataset size: {len(df_current)} draws.")

        # ── Step 3: Run Prediction Pipeline ──────────────────────────────────
        log_print(f"\n  [Step 3] Running full prediction pipeline...")
        pred_res = run_prediction_pipeline(df_current, custom_weights=adaptive_model_weights)
        
        meta_ticket = pred_res["meta_top6_ticket"]
        meta_pool14 = pred_res["meta_top14_pool"]
        meta_pool16 = pred_res["meta_top16_pool"]
        meta_prob   = pred_res["meta_prob"]

        # Structural characteristics of predicted ticket
        pred_stats = compute_structural_stats(meta_ticket)
        chi2_res = chi_squared_fit_test(meta_ticket, df_current)

        # ── Step 4: Compare with Actual Draw ─────────────────────────────────
        parts = row_str.strip().split(",")
        winning_nums = sorted([int(parts[i]) for i in range(2, 8)])
        actual_stats = compute_structural_stats(winning_nums)

        ticket_hits = len(set(meta_ticket).intersection(set(winning_nums)))
        pool14_hits = len(set(meta_pool14).intersection(set(winning_nums)))
        pool16_hits = len(set(meta_pool16).intersection(set(winning_nums)))
        avg_rank_pct = get_rank_percentile(meta_prob, winning_nums)
        match_rates = calculate_pair_triple_match_rate(meta_ticket, winning_nums)

        # Wheeling evaluation
        wheel_lines = generate_covering_wheel(meta_pool14)
        max_wheel_hit = max(len(set(line).intersection(set(winning_nums))) for line in wheel_lines)

        # Define Success / Failure Flag
        # SUCCESS if top-6 hits >= 2 or pool14 hits >= 4 or rank percentile <= 25.0%
        success_flag = "SUCCESS" if (ticket_hits >= 2 or pool14_hits >= 4 or avg_rank_pct <= 25.0) else "FAILURE"

        # ── Step 5: Record Results in Structured Log ────────────────────────
        log_print(f"\n  [Step 4 & 5] Verification & Structured Log Entry:")
        log_print(f"    - Predicted Ticket (Top-6)    : {meta_ticket}")
        log_print(f"    - Predicted Candidate Pool 14 : {meta_pool14}")
        log_print(f"    - Predicted Candidate Pool 16 : {meta_pool16}")
        log_print(f"    - Actual Holdout Draw         : {winning_nums}")
        log_print(f"    - Top-6 Ticket Hits           : {ticket_hits} / 6")
        log_print(f"    - 14-Ball Candidate Pool Hits : {pool14_hits} / 6")
        log_print(f"    - 16-Ball Candidate Pool Hits : {pool16_hits} / 6")
        log_print(f"    - Best Wheel Ticket Match     : Match-{max_wheel_hit}")
        log_print(f"    - Rank Percentile (Probability): {avg_rank_pct:.2f}% (Lower is better)")
        log_print(f"    - Result Status Flag          : [{success_flag}]")
        log_print(f"    - Pair Match Rate (C(6,2))    : {match_rates['pair_match_rate_pct']}% ({match_rates['pairs_hit']}/15)")
        log_print(f"    - Triple Match Rate (C(6,3))  : {match_rates['triple_match_rate_pct']}% ({match_rates['triples_hit']}/20)")
        log_print(f"    - Result Status Flag          : [{success_flag}]")
        log_print(f"    - Pair Match Rate (C(6,2))    : {match_rates['pair_match_rate_pct']}% ({match_rates['pairs_hit']}/15)")
        log_print(f"    - Triple Match Rate (C(6,3))  : {match_rates['triple_match_rate_pct']}% ({match_rates['triples_hit']}/20)")

        # Per-step sub-model evaluation
        step_evals = {}
        for step_name, step_p in pred_res["model_probs"].items():
            s_ticket = sorted((np.argsort(step_p)[::-1][:6] + 1).tolist())
            s_hits = len(set(s_ticket).intersection(set(winning_nums)))
            s_rank = get_rank_percentile(step_p, winning_nums)
            step_evals[step_name] = {"hits": s_hits, "rank_pct": round(s_rank, 2)}
            
            history = model_performance_scores.get(step_name, [])
            history.append(s_rank)
            model_performance_scores[step_name] = history

        best_model = min(step_evals.items(), key=lambda x: x[1]['rank_pct'])
        log_print(f"    - Top Sub-Model on this Draw : {best_model[0]} (Rank: {best_model[1]['rank_pct']}%, Hits: {best_model[1]['hits']})")

        # ── Step 6: Adjust Adaptive Model Weights ─────────────────────────────
        log_print(f"\n  [Step 6] Adjusting adaptive model weights based on performance history...")
        new_weights = {}
        total_inv_rank = 0.0
        for name, rank_history in model_performance_scores.items():
            if name == "Step 12 Meta":
                continue
            recency_decay = np.exp(np.linspace(-1, 0, len(rank_history)))
            avg_recent_rank = np.average(rank_history, weights=recency_decay)
            inv_rank = 1.0 / (avg_recent_rank + 1.0)
            new_weights[name] = inv_rank
            total_inv_rank += inv_rank

        for name in new_weights:
            new_weights[name] /= total_inv_rank

        adaptive_model_weights = new_weights
        log_print(f"    - Updated Adaptive Model Weights: {json.dumps({k: round(v, 4) for k, v in adaptive_model_weights.items()})}")

        # Store iteration summary
        iteration_records.append({
            "iteration": idx,
            "date": target_date,
            "predicted_ticket": meta_ticket,
            "predicted_candidate_pool_14": meta_pool14,
            "predicted_candidate_pool_16": meta_pool16,
            "actual_draw": winning_nums,
            "ticket_hits": ticket_hits,
            "pool14_hits": pool14_hits,
            "pool16_hits": pool16_hits,
            "max_wheel_hit": max_wheel_hit,
            "rank_percentile": round(avg_rank_pct, 2),
            "success_flag": success_flag,
            "pred_low_high": pred_stats['low_high'],
            "actual_low_high": actual_stats['low_high'],
            "step_evaluations": step_evals,
            "adapted_weights": {k: round(v, 4) for k, v in adaptive_model_weights.items()}
        })

        # ── Step 7: Append Verified Draw Back into Training Set ───────────────
        log_print(f"\n  [Step 7] Appending verified draw ({target_date}) back to active training CSV...")
        with open(CSV_FILE, "r", encoding="utf-8") as f:
            cur_lines = f.readlines()
        
        updated_lines = [cur_lines[0], row_str] + cur_lines[1:]
        with open(CSV_FILE, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)

        log_print(f"  [+] Active training dataset count for next draw: {len(cur_lines)}")
        log_print(f"  [Step 8] Completed Iteration {idx}/{HOLDOUT_WINDOW}.\n")

    # ── STEP 9: Evaluate Results & Refine Model Parameters ───────────────────
    elapsed_total = time.perf_counter() - t_start
    
    # Save structured JSON results
    with open(JSON_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(iteration_records, f, indent=2)

    total_draws = len(iteration_records)
    avg_ticket_hits = np.mean([r["ticket_hits"] for r in iteration_records])
    avg_pool14_hits = np.mean([r["pool14_hits"] for r in iteration_records])
    avg_pool16_hits = np.mean([r["pool16_hits"] for r in iteration_records])
    avg_rank = np.mean([r["rank_percentile"] for r in iteration_records])
    success_rate = sum(1 for r in iteration_records if r["success_flag"] == "SUCCESS") / total_draws * 100.0
    wheel_match3_rate = sum(1 for r in iteration_records if r["max_wheel_hit"] >= 3) / total_draws * 100.0

    log_print("=" * 80)
    log_print("  [Step 9] POST-LOOP EVALUATION & AGGREGATE PERFORMANCE METRICS")
    log_print("=" * 80)
    log_print(f"Total Holdout Iterations Processed : {total_draws}")
    log_print(f"Total Execution Duration           : {elapsed_total:.1f} seconds ({elapsed_total/total_draws:.1f}s / draw)")
    log_print(f"Overall Iteration Success Rate     : {success_rate:.1f}% ({sum(1 for r in iteration_records if r['success_flag'] == 'SUCCESS')}/{total_draws})")
    log_print(f"Average Top-6 Ticket Hits          : {avg_ticket_hits:.2f} / 6 (vs Uniform Baseline ~0.923)")
    log_print(f"Average 14-Ball Pool Coverage Hits : {avg_pool14_hits:.2f} / 6 (Recall: {avg_pool14_hits/6*100:.1f}%)")
    log_print(f"Average 16-Ball Pool Coverage Hits : {avg_pool16_hits:.2f} / 6 (Recall: {avg_pool16_hits/6*100:.1f}%)")
    log_print(f"3-if-3 Wheel Win Guarantee Rate    : {wheel_match3_rate:.1f}% (Match-3+ achieved)")
    log_print(f"Average Probability Rank Percentile: {avg_rank:.2f}% (Lower is better)")
    log_print("=" * 80)

    log_print("\n  PER-STEP MODEL COMPARATIVE EVALUATION:")
    for step_name in ["Step 6", "Step 7", "Step 8", "Step 9", "Step 10", "Step 11", "Step 12 Meta"]:
        step_ranks = [r["step_evaluations"][step_name]["rank_pct"] for r in iteration_records if step_name in r["step_evaluations"]]
        step_hits  = [r["step_evaluations"][step_name]["hits"] for r in iteration_records if step_name in r["step_evaluations"]]
        if step_ranks:
            log_print(f"    - {step_name:<14} : Avg Hits = {np.mean(step_hits):.2f}/6 | Avg Rank Pct = {np.mean(step_ranks):.2f}%")

    log_print("\n  FINAL CALIBRATED ADAPTIVE WEIGHTS:")
    log_print(f"    {json.dumps({k: round(v, 4) for k, v in adaptive_model_weights.items()}, indent=6)}")
    log_print("=" * 80)

    log_handle.close()

    # Restore full CSV_FILE from BACKUP_FILE
    shutil.copyfile(BACKUP_FILE, CSV_FILE)
    print(f"\n[*] Restored full original dataset to {CSV_FILE}.")
    print(f"[+] Structured JSON log saved to: {JSON_LOG_FILE}")
    print(f"[+] Human-readable text log saved to: {LOG_FILE}\n")

    return iteration_records, adaptive_model_weights


if __name__ == "__main__":
    execute_learning_loop()
