"""
================================================================================
16_gnn_hawkes_meta_learning_engine.py — Step 16 GNN & Process Modeling Engine
================================================================================
"""
import sys
from utils import load_data, CSV_FILE
from gnn_hawkes_meta_learning import (
    signal_gnn_network,
    hawkes_jump_diffusion_process,
    predict_draw_volatility_evt,
    MetaLearningWeightPredictor
)


def main():
    df = load_data(CSV_FILE)
    print("============================================================")
    print("  STEP 16 — GNN RELATIONAL NETWORK & PROCESS MODELING ENGINE")
    print("============================================================")
    
    gnn_p = signal_gnn_network(df)
    hawkes_p = hawkes_jump_diffusion_process(df)
    evt_info = predict_draw_volatility_evt(df)
    
    print(f"  [+] GNN Relational Network Top Ball  : #{gnn_p.argmax() + 1} ({gnn_p.max()*100:.2f}%)")
    print(f"  [+] Hawkes-Jump Process Top Ball     : #{hawkes_p.argmax() + 1} ({hawkes_p.max()*100:.2f}%)")
    print(f"  [+] EVT Draw Volatility Sum Range    : {evt_info['ideal_sum_range']}")
    print("\n[OK] Step 16 GNN & Process Modeling execution complete!")


if __name__ == "__main__":
    main()
