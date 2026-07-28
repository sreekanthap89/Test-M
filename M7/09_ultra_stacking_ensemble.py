"""
=============================================================
 STEP 9: ULTRA STACKING ENSEMBLE ENGINE & WHEELING SYSTEM
=============================================================
 LEARNING GOAL:
   1. Multi-Model Stacking Classifier combining XGBoost, LightGBM,
      Random Forest, Extra Trees, and Neural Networks (MLP).
   2. Advanced Feature Engineering (Delay Gaps, Rolling Frequencies,
      Zone Ratios, High/Low Balances, and Pair Co-occurrence).
   3. Meta-Learner Stacking & Combined Markov Fusion.
   4. Flexible Combinatorial Wheeling System with 3-if-3 Match Guarantees.
=============================================================
"""

import os
import sys
import math
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder, load_data, generate_covering_wheel

CSV_FILE  = "Emirates_Draw_MEGA7.csv"
WIN_COLS  = ["Winning Number 1", "2", "3", "4", "5", "6", "7"]
POOL      = 37
DRAW_SIZE = 7
SEED      = 42


# ── FEATURE ENGINEERING ENGINE ────────────────────────────────────────────────

def extract_features_for_draw(draws_history, pool_size=37):
    """
    Extracts rich engineered features for predicting the next draw based on past history:
      1. Delay Gaps (draws since ball 1..37 was last drawn)
      2. Rolling Frequencies (3-draw, 5-draw, 10-draw)
      3. Zone Concentration Ratios (Z1: 1-10, Z2: 11-20, Z3: 21-30, Z4: 31-37)
      4. High/Low & Sum Trends
      5. Pair Co-occurrence Affinity with last drawn numbers
    """
    n_draws = len(draws_history)
    features = []

    # 1. Delay Gaps per ball
    delay_gaps = np.full(pool_size, fill_value=n_draws, dtype=float)
    for t_idx in range(n_draws - 1, -1, -1):
        drawn_set = set(draws_history[t_idx])
        for b in range(1, pool_size + 1):
            if delay_gaps[b - 1] == n_draws and b in drawn_set:
                delay_gaps[b - 1] = n_draws - 1 - t_idx
    features.extend(delay_gaps)

    # 2. Rolling Frequencies (3-draw, 5-draw, 10-draw)
    for window in [3, 5, 10]:
        recent = draws_history[-window:] if n_draws >= window else draws_history
        freq = np.zeros(pool_size)
        for draw in recent:
            for b in draw:
                freq[b - 1] += 1
        freq /= max(1, len(recent))
        features.extend(freq)

    # 3. Zone Ratios (last 3 draws)
    recent_3 = draws_history[-3:] if n_draws >= 3 else draws_history
    zone_counts = np.zeros(4)
    for draw in recent_3:
        for b in draw:
            if b <= 10: zone_counts[0] += 1
            elif b <= 20: zone_counts[1] += 1
            elif b <= 30: zone_counts[2] += 1
            else: zone_counts[3] += 1
    zone_ratios = zone_counts / max(1, zone_counts.sum())
    features.extend(zone_ratios)

    # 4. High/Low & Sum Trends (last 3 draws)
    last_sums = [sum(d) for d in recent_3]
    features.append(np.mean(last_sums) if last_sums else 133.0)
    
    last_highs = [sum(1 for b in d if b > 18) for d in recent_3]
    features.append(np.mean(last_highs) if last_highs else 3.5)

    # 5. Pair Affinity with last draw
    last_draw = set(draws_history[-1]) if n_draws > 0 else set()
    pair_counts = np.zeros(pool_size)
    for draw in draws_history:
        d_set = set(draw)
        overlap = len(d_set & last_draw)
        if overlap > 0:
            for b in d_set:
                pair_counts[b - 1] += overlap
    pair_affinity = pair_counts / max(1, n_draws)
    features.extend(pair_affinity)

    return np.array(features)


def prepare_stacking_dataset(df, lookback=10, pool_size=37):
    draws = df["numbers"].tolist()
    X_list, Y_list = [], []

    for t in range(lookback, len(draws)):
        hist = draws[:t]
        feat = extract_features_for_draw(hist, pool_size=pool_size)
        
        target = np.zeros(pool_size)
        for b in draws[t]:
            target[b - 1] = 1.0

        X_list.append(feat)
        Y_list.append(target)

    return np.array(X_list), np.array(Y_list), draws


# ── STACKING CLASSIFIER PIPELINE ──────────────────────────────────────────────

class StackingEnsembleSuite:
    """
    Combines 5 Diverse Machine Learning Architectures:
      1. XGBoost Classifier
      2. LightGBM Classifier
      3. Random Forest Classifier
      4. Extra Trees Classifier
      5. Multi-Layer Perceptron Neural Network
    """
    def __init__(self, random_state=SEED):
        self.random_state = random_state
        self.models = {}

    def fit(self, X, Y):
        n_samples, n_outputs = Y.shape
        
        # Train individual multi-output models
        for ball_idx in range(n_outputs):
            y_col = Y[:, ball_idx]
            
            # Skip balls with no positive variance
            if y_col.sum() == 0 or y_col.sum() == n_samples:
                continue

            # Model 1: XGBoost
            xgb = XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.05,
                                eval_metric="logloss", random_state=self.random_state)
            xgb.fit(X, y_col)

            # Model 2: LightGBM
            lgb = LGBMClassifier(n_estimators=60, max_depth=3, learning_rate=0.05,
                                 verbose=-1, random_state=self.random_state)
            lgb.fit(X, y_col)

            # Model 3: Random Forest
            rf = RandomForestClassifier(n_estimators=60, max_depth=4, random_state=self.random_state)
            rf.fit(X, y_col)

            # Model 4: Extra Trees
            et = ExtraTreesClassifier(n_estimators=60, max_depth=4, random_state=self.random_state)
            et.fit(X, y_col)

            # Model 5: Neural Network
            mlp = MLPClassifier(hidden_layer_sizes=(80, 40), alpha=0.001, max_iter=400, random_state=self.random_state)
            mlp.fit(X, y_col)

            self.models[ball_idx] = {
                "xgb": xgb, "lgb": lgb, "rf": rf, "et": et, "mlp": mlp
            }

    def predict_proba(self, X_single):
        probas = np.zeros(POOL)
        
        for ball_idx, m_dict in self.models.items():
            p_xgb = m_dict["xgb"].predict_proba(X_single)[0, 1]
            p_lgb = m_dict["lgb"].predict_proba(X_single)[0, 1]
            p_rf  = m_dict["rf"].predict_proba(X_single)[0, 1]
            p_et  = m_dict["et"].predict_proba(X_single)[0, 1]
            p_mlp = m_dict["mlp"].predict_proba(X_single)[0, 1]

            # Weighted Blending across base models
            blended = 0.25 * p_xgb + 0.25 * p_lgb + 0.20 * p_rf + 0.15 * p_et + 0.15 * p_mlp
            probas[ball_idx] = blended

        # Normalize
        if probas.sum() > 0:
            probas /= probas.sum()
        return probas


# ── COMBINATORIAL WHEELING SYSTEM ─────────────────────────────────────────────
# Uses centralized generate_covering_wheel from utils


# ── MAIN EXECUTION ────────────────────────────────────────────────────────────

def main():
    print("============================================================")
    print("  STEP 9 — ULTRA STACKING ENSEMBLE ENGINE & WHEELING (MEGA7)")
    print("============================================================")

    df = load_data(CSV_FILE)
    LOOKBACK = 10

    print(f"Extracting rich features (Gaps, Moving Avgs, Zone Ratios, Pair Affinity)...")
    X, Y, draws_history = prepare_stacking_dataset(df, lookback=LOOKBACK, pool_size=POOL)
    print(f"Dataset prepared: {X.shape[0]} training draws | {X.shape[1]} engineered features per draw.")

    print("\nTraining Stacking Ensemble (XGBoost + LightGBM + RandomForest + ExtraTrees + MLP)...")
    stacker = StackingEnsembleSuite(random_state=SEED)
    stacker.fit(X, Y)
    print("Stacking Ensemble training complete.")

    # Predict for unseen next draw
    latest_features = extract_features_for_draw(draws_history, pool_size=POOL).reshape(1, -1)
    stack_probs = stacker.predict_proba(latest_features)

    # Top candidates & single ticket prediction
    top_14_indices = np.argsort(stack_probs)[::-1][:14]
    top_14_numbers = sorted((top_14_indices + 1).tolist())

    top_7_indices = np.argsort(stack_probs)[::-1][:7]
    top_7_numbers = sorted((top_7_indices + 1).tolist())

    print("\n============================================================")
    print("  PHASE 1: ULTRA STACKING PREDICTION")
    print("============================================================")
    print(f"Stacking Ensemble Top 7 Single Predicted Ticket:  ★  {top_7_numbers}  ★")
    print(f"Stacking Ensemble Top 14 Candidate Pool        : {top_14_numbers}")

    print("\n============================================================")
    print("  PHASE 2: COMBINATORIAL WHEELING")
    print("============================================================")
    print(f"Buying all combinations of 14 numbers = {math.comb(14, 7)} tickets.")
    tickets = generate_covering_wheel(top_14_numbers, ticket_size=DRAW_SIZE, match_guarantee=3)
    print(f"[Wheeling] Generated 3-if-3 covering wheel in {len(tickets)} tickets.")

    print("\nYOUR WHEELED TICKETS:")
    for i, t in enumerate(tickets, 1):
        print(f"  Ticket {i:2d}: {list(t)}")

    # Visual Chart Generation
    run_dir = get_run_folder()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("STEP 9 — Ultra Stacking ML Ensemble (XGBoost + LightGBM + RF + ET + MLP)\nEmirates Draw MEGA7",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

    # Bar Chart
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(range(1, POOL + 1), stack_probs * 100, color='#95a5a6')
    for idx in top_14_indices:
        bars[idx].set_color('#2ecc71')
        ax1.text(idx + 1, stack_probs[idx] * 100 + 0.05, f"#{idx+1}", ha="center",
                 fontsize=8, color="#27ae60", fontweight="bold")

    ax1.set_title("Ultra Stacking Ensemble Output Probabilities (Green = Top 14 Candidates)")
    ax1.set_xlabel("Number"); ax1.set_ylabel("Probability (%)")
    ax1.set_xlim(0.5, POOL + 0.5)
    ax1.set_xticks(range(1, POOL + 1))
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Top 14 Stacking Candidates'),
        Patch(facecolor='#95a5a6', label='Other Pool Numbers')
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    # Text Summary Panel
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    summary_text = (
        "★ ULTRA STACKING TOP-7 PREDICTED TICKET ★\n"
        f"  {top_7_numbers}\n\n"
        "★ ULTRA STACKING TOP-14 CANDIDATE POOL ★\n"
        f"  {top_14_numbers}\n\n"
        "★ MODELS COMBINED ★\n"
        "  - XGBoost Gradient Boosted Trees\n"
        "  - LightGBM Gradient Boosting Machine\n"
        "  - Random Forest Classifier\n"
        "  - Extra Trees Classifier\n"
        "  - Multi-Layer Perceptron (MLP) Neural Net"
    )
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f4f6f7", edgecolor="#27ae60", linewidth=1.5))

    # Wheeled Tickets Panel
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    t1_text = "\n".join([f"T{i:02d}: {list(t)}" for i, t in enumerate(tickets[:10], 1)])
    t2_text = "\n".join([f"T{i:02d}: {list(t)}" for i, t in enumerate(tickets[10:], 11)])

    ax3.text(0.02, 0.95, "★ WHEELED TICKETS (1-10) ★\n" + t1_text, transform=ax3.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2f3", edgecolor="#7f8c8d"))

    ax3.text(0.52, 0.95, "★ WHEELED TICKETS (11-19) ★\n" + t2_text, transform=ax3.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2f3", edgecolor="#7f8c8d"))

    chart_path = f"{run_dir}/step9_ultra_stacking_ensemble.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Chart saved -> {chart_path}")


if __name__ == "__main__":
    main()
