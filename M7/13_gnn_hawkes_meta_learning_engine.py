"""
================================================================================
13_gnn_hawkes_meta_learning_engine.py — MEGA7 GNN & Process Modeling Engine
================================================================================
Implements:
1. Phase 1: 37-Node Graph Neural Network (GNN) Message Passing Layer.
2. Phase 2: Hawkes Self-Exciting Point Process & EVT Volatility Engine.
3. Phase 3: Neural Network Meta-Learning Weight Adaptation.
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
from gnn_hawkes_meta_learning import (
    signal_gnn_network,
    hawkes_jump_diffusion_process,
    predict_draw_volatility_evt,
    MetaLearningWeightPredictor
)


def main():
    df = load_data(CSV_FILE)
    print("============================================================")
    print("  STEP 13 — GNN RELATIONAL NETWORK & PROCESS MODELING ENGINE (MEGA7)")
    print("============================================================")
    
    gnn_p = signal_gnn_network(df)
    hawkes_p = hawkes_jump_diffusion_process(df)
    evt_info = predict_draw_volatility_evt(df)
    
    top1_gnn = int(gnn_p.argmax() + 1)
    top1_hawkes = int(hawkes_p.argmax() + 1)

    print(f"  [+] GNN Relational Network Top Ball  : #{top1_gnn} ({gnn_p.max()*100:.2f}%)")
    print(f"  [+] Hawkes-Jump Process Top Ball     : #{top1_hawkes} ({hawkes_p.max()*100:.2f}%)")
    print(f"  [+] EVT Draw Volatility Sum Range    : {evt_info['ideal_sum_range']}")

    # Save visual chart
    run_dir = get_run_folder()
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("STEP 13 — GNN Relational Network & Hawkes Process Modeling\nEmirates Draw MEGA7", fontsize=14, fontweight="bold")

    axes[0].bar(range(1, POOL + 1), gnn_p * 100, color="#8e44ad")
    axes[0].set_title("37-Node Graph Neural Network (GNN) Output Probabilities")
    axes[0].set_ylabel("Prob (%)")
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(range(1, POOL + 1), hawkes_p * 100, color="#d35400")
    axes[1].set_title("Hawkes Self-Exciting Point Process & Jump-Diffusion Intensity")
    axes[1].set_ylabel("Prob (%)")
    axes[1].set_xlabel("Ball Number (1..37)")
    axes[1].set_xticks(range(1, POOL + 1))
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    chart_path = os.path.join(run_dir, "step13_gnn_hawkes_meta_learning.png")
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Chart saved -> {chart_path}")
    print("[OK] Step 13 GNN & Process Modeling execution complete!")


if __name__ == "__main__":
    main()
