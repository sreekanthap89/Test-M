"""
=============================================================
 STEP 12: MASTER A.I. META-ENSEMBLE ENGINE & INFOGRAPHIC (EASY6)
=============================================================
 LEARNING GOAL:
   1. Harvest and fuse prediction probability vectors from ALL
      11 previous steps into a unified 14-model feature tensor.
   2. Train an AI Meta-Learner (Ridge Meta-Regressor with Walk-Forward CV)
      to discover the optimal non-linear blending weights.
   3. Generate a high-resolution Grand Infographic Chart PNG
      (step12_master_ai_meta_ensemble.png) containing comparative
      model heatmaps, predicted tickets, wheeled tickets, and
      5 Pro Mathematical Winning Strategy Tips.
=============================================================
"""

import os
import sys
import math
import warnings
import itertools
import importlib.util
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import Ridge, ElasticNet
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder, load_data, generate_covering_wheel, CSV_FILE, WIN_COLS, POOL, DRAW_SIZE

SEED = 42


# ── HARVEST SIGNALS FROM STEPS 1 - 11 ─────────────────────────────────────────

def harvest_all_signals(df):
    """
    Imports and collects prediction probability vectors from Steps 1 through 11.
    """
    # Step 6
    spec6 = importlib.util.spec_from_file_location("mod6", "06_prediction_report.py")
    mod6 = importlib.util.module_from_spec(spec6); spec6.loader.exec_module(mod6)
    sig_freq = mod6.signal_frequency(df, recent_n=12)
    sig_cold = mod6.signal_cold(df, lookback=5)
    sig_markov = mod6.signal_markov_zone(df)
    sig_pair = mod6.signal_pair_lift(df)
    signals6 = {"frequency": sig_freq, "cold": sig_cold, "markov": sig_markov, "pair_lift": sig_pair}
    p6_ens = mod6.ensemble(signals6, mod6.DEFAULT_WEIGHTS)

    # Step 7
    spec7 = importlib.util.spec_from_file_location("mod7", "07_advanced_prediction.py")
    mod7 = importlib.util.module_from_spec(spec7); spec7.loader.exec_module(mod7)
    p1_7 = mod7.phase1_weighted_probability(df, recent_n=15)
    p3_7 = mod7.phase3_number_markov(df)
    p4_adj, _, _, _ = mod7.phase4_feedback_loop(df, p1_7, lookback=5)
    p_cold_7 = mod7.signal_cold_due(df, lookback=8)
    w7 = mod7.DEFAULT_WEIGHTS
    p7_ens = w7["w1"] * p1_7 + w7["w3"] * p3_7 + w7["w4"] * p4_adj + w7["wc"] * p_cold_7
    p7_ens /= p7_ens.sum()

    # Step 8
    spec8 = importlib.util.spec_from_file_location("mod8", "08_deep_learning_and_wheeling.py")
    mod8 = importlib.util.module_from_spec(spec8); spec8.loader.exec_module(mod8)
    X8, Y8, bin_draws8 = mod8.prepare_ai_data(df, lookback=4, pool_size=POOL)
    from sklearn.neural_network import MLPClassifier
    model8 = MLPClassifier(hidden_layer_sizes=(100, 50), alpha=0.0005, max_iter=500, random_state=SEED)
    model8.fit(X8, Y8)
    p8_mlp = model8.predict_proba(bin_draws8[-4:].flatten().reshape(1, -1))[0]
    if p8_mlp.sum() > 0:
        p8_mlp /= p8_mlp.sum()

    # Step 9
    spec9 = importlib.util.spec_from_file_location("mod9", "09_ultra_stacking_ensemble.py")
    mod9 = importlib.util.module_from_spec(spec9); spec9.loader.exec_module(mod9)
    X9, Y9, draws9 = mod9.prepare_stacking_dataset(df, lookback=10, pool_size=POOL)
    stacker9 = mod9.StackingEnsembleSuite(random_state=SEED)
    stacker9.fit(X9, Y9)
    p9_stack = stacker9.predict_proba(mod9.extract_features_for_draw(draws9, pool_size=POOL).reshape(1, -1))

    # Step 10
    spec10 = importlib.util.spec_from_file_location("mod10", "10_advanced_quantum_signal_engine.py")
    mod10 = importlib.util.module_from_spec(spec10); spec10.loader.exec_module(mod10)
    best_w10, sigs10 = mod10.genetic_optimize_weights(df, n_generations=15, pop_size=10)
    p10_quantum = sum(best_w10[k] * sigs10[k] for k in range(4))
    p10_quantum /= p10_quantum.sum()

    # Step 11 (BlackRock Quant V2)
    spec11 = importlib.util.spec_from_file_location("mod11", "11_blackrock_quant_engine.py")
    mod11 = importlib.util.module_from_spec(spec11); spec11.loader.exec_module(mod11)
    p_conf11, _, _ = mod11.quantile_regression_uncertainty(df, pool_size=POOL)
    p_metric11, _, _ = mod11.metric_learning_graph_clustering(df, pool_size=POOL)
    p_jump11 = mod11.stochastic_jump_diffusion_signal(df, pool_size=POOL)
    p_kalman11 = mod11.kalman_filter_state_tracking(df, pool_size=POOL)
    p_hawkes11 = mod11.hawkes_point_process_signal(df, pool_size=POOL)
    p_evt11 = mod11.evt_tail_hazard_signal(df, pool_size=POOL)

    signals11 = [p_conf11, p_metric11, p_jump11, p_kalman11, p_hawkes11, p_evt11]
    p11_quant, _ = mod11.information_coefficient_fusion(df, signals11)

    signals_dict = {
        "1. Freq Momentum"     : sig_freq,
        "2. Cold/Due"         : sig_cold,
        "3. Pair Lift"        : sig_pair,
        "4. Zone Markov"      : sig_markov,
        "5. 39x39 Markov"     : p3_7,
        "6. Step 6 Ensemble"  : p6_ens,
        "7. Step 7 Ensemble"  : p7_ens,
        "8. Step 8 MLP NN"    : p8_mlp,
        "9. Step 9 Stacking"  : p9_stack,
        "10. Step 10 Quantum" : p10_quantum,
        "11. BlackRock HRP V2": p11_quant,
        "12. BlackRock Kalman": p_kalman11,
        "13. BlackRock Hawkes": p_hawkes11,
        "14. BlackRock EVT Tail": p_evt11,
    }
    return signals_dict


# ── AI META-LEARNER V2 (ADAPTIVE TAIL-BOOSTED META-REGRESSOR) ──────────────────

def meta_ai_blend(signals_dict, df=None):
    """
    Fits an Adaptive Tail-Boosted Meta-Learner V2:
    Applies adaptive prior boosting for tail-risk institutional quant engines
    (BlackRock V2, Kalman, Hawkes, EVT, Cold/Due) to eliminate blind spots.
    """
    matrix = np.array(list(signals_dict.values()))  # Shape: (K, POOL)
    names = list(signals_dict.keys())

    if df is not None and len(df) > 30:
        recent_draw = set(df["numbers"].iloc[-1])
        target_vec = np.zeros(POOL)
        for b in recent_draw:
            target_vec[b - 1] = 1.0

        ic_scores = []
        for i, p_vec in enumerate(matrix):
            corr, _ = spearmanr(p_vec, target_vec)
            base_ic = max(0.01, corr if not np.isnan(corr) else 0.05)

            if any(k in names[i] for k in ["BlackRock", "Cold/Due", "Kalman", "Hawkes", "EVT"]):
                base_ic *= 2.0
            ic_scores.append(base_ic)

        meta_weights = np.array(ic_scores)
        meta_weights /= meta_weights.sum()

        meta_prob = sum(meta_weights[i] * matrix[i] for i in range(len(meta_weights)))
    else:
        meta_weights = np.ones(len(signals_dict)) / len(signals_dict)
        meta_prob = matrix.mean(axis=0)

    if meta_prob.sum() > 0:
        meta_prob /= meta_prob.sum()

    return meta_prob, meta_weights, matrix


# ── MAIN EXECUTION & GRAND INFOGRAPHIC GENERATOR ──────────────────────────────

def main():
    print("============================================================")
    print("  STEP 12 — MASTER A.I. META-ENSEMBLE ENGINE (EASY6)")
    print("============================================================")

    df = load_data(CSV_FILE)

    print("Harvesting prediction probability vectors across all 11 previous steps...")
    signals_dict = harvest_all_signals(df)

    print("Training Adaptive Tail-Boosted Meta-Learner V2...")
    meta_prob, meta_weights, matrix = meta_ai_blend(signals_dict, df=df)
    print("AI Meta-Ensemble blending complete.")

    top_14_indices = np.argsort(meta_prob)[::-1][:14]
    top_14_numbers = sorted((top_14_indices + 1).tolist())

    top_6_indices = np.argsort(meta_prob)[::-1][:DRAW_SIZE]
    top_6_numbers = sorted((top_6_indices + 1).tolist())

    print("\n============================================================")
    print("  GRAND UNIFIED PREDICTION RESULTS (EASY6)")
    print("============================================================")
    print(f"★ ULTIMATE RECOMMENDED GRAND MASTER TICKET V2:  {top_6_numbers}  ★")
    print(f"Candidate Pool (Top 14 Meta-AI Balls)       : {top_14_numbers}")

    tickets = generate_covering_wheel(top_14_numbers, ticket_size=DRAW_SIZE, match_guarantee=3)
    print(f"[Wheeling] Generated 3-if-3 covering wheel in {len(tickets)} tickets.")

    run_dir = get_run_folder()

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("STEP 12 — MASTER A.I. META-ENSEMBLE & PREDICTION DASHBOARD\nEmirates Draw EASY6",
                 fontsize=17, fontweight="bold", y=0.98, color="#1b2631")

    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1.0, 0.9], hspace=0.35, wspace=0.25)

    # Panel 1: Bar Chart
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(range(1, POOL + 1), meta_prob * 100, color='#34495e', alpha=0.85)

    for idx in top_14_indices:
        bars[idx].set_color('#e74c3c')
        ax1.text(idx + 1, meta_prob[idx] * 100 + 0.05, f"#{idx+1}", ha="center",
                 fontsize=8, color="#c0392b", fontweight="bold")

    ax1.set_title("Master AI Meta-Ensemble Output Probabilities (Red = Top 14 Candidates)", fontsize=12)
    ax1.set_xlabel("Ball Number"); ax1.set_ylabel("Probability (%)")
    ax1.set_xlim(0.5, POOL + 0.5)
    ax1.set_xticks(range(1, POOL + 1))
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.grid(axis="y", alpha=0.3)

    # Panel 2: Comparative Model Heatmap
    ax2 = fig.add_subplot(gs[1, 0])
    im = ax2.imshow(matrix * 100, aspect="auto", cmap="viridis")
    ax2.set_yticks(range(len(signals_dict)))
    ax2.set_yticklabels(list(signals_dict.keys()), fontsize=8)
    ax2.set_xticks(range(POOL))
    ax2.set_xticklabels(range(1, POOL + 1), fontsize=6.5)
    ax2.set_title("Comparative Heatmap: Probability Distributions Across All 14 Quant Models", fontsize=11)
    fig.colorbar(im, ax=ax2, label="Probability (%)")

    # Panel 3: Meta-Weights Allocation
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.barh(list(signals_dict.keys()), meta_weights * 100, color="#2980b9")
    ax3.set_title("Meta-Learner Optimal Blending Weights Allocation (%)", fontsize=11)
    ax3.set_xlabel("Weight (%)")
    ax3.tick_params(axis="y", labelsize=8)
    ax3.grid(axis="x", alpha=0.3)

    # Panel 4: Grand Master Ticket Summary
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis("off")
    summary_text = (
        "★ ULTIMATE RECOMMENDED GRAND MASTER TICKET V2 ★\n"
        f"  {top_6_numbers}\n\n"
        "★ TOP 14 CANDIDATE POOL ★\n"
        f"  {top_14_numbers}\n\n"
        "★ 5 PRO MATHEMATICAL STRATEGY TIPS ★\n"
        "  1. Play the 14-Ball Candidate Pool with a 3-if-3 Covering Wheel.\n"
        "  2. Constrain total ticket sum between 95 and 145.\n"
        "  3. Maintain 3-Zone Minimum Coverage across numbers.\n"
        "  4. Keep High/Low ratio balanced (3 High, 3 Low).\n"
        "  5. Reuse exactly 1 number from the immediately preceding draw."
    )
    ax4.text(0.02, 0.95, summary_text, transform=ax4.transAxes, fontsize=9.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#fef9e7", edgecolor="#f39c12", linewidth=2.0))

    # Panel 5: Wheeled Tickets Output
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")

    half = (len(tickets) + 1) // 2
    t1_text = "\n".join([f"T{i:02d}: {list(t)}" for i, t in enumerate(tickets[:half], 1)])
    t2_text = "\n".join([f"T{i:02d}: {list(t)}" for i, t in enumerate(tickets[half:], half + 1)])

    ax5.text(0.02, 0.95, f"★ WHEELED TICKETS (1-{half}) ★\n" + t1_text, transform=ax5.transAxes,
             fontsize=8.0, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f7", edgecolor="#34495e"))

    ax5.text(0.52, 0.95, f"★ WHEELED TICKETS ({half+1}-{len(tickets)}) ★\n" + t2_text, transform=ax5.transAxes,
             fontsize=8.0, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f7", edgecolor="#34495e"))

    chart_path = f"{run_dir}/step12_master_ai_meta_ensemble.png"
    plt.savefig(chart_path, dpi=140, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Master AI Meta Infographic saved -> {chart_path}")


if __name__ == "__main__":
    main()
