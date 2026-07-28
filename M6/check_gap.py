"""
check_gap.py — EASY6 Draw Gap & Property Analysis Utility
"""

import os
import sys
import numpy as np
import pandas as pd
import importlib.util
import warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from utils import load_data, POOL, DRAW_SIZE, CSV_FILE


def main():
    print("====================================================================")
    print("  EASY6 PREDICTION GAP ANALYSIS")
    print("====================================================================\n")

    df = load_data(CSV_FILE)
    print(f"Loaded {len(df)} historical draws. Latest draw in CSV: {df['Date'].iloc[-1].date()} -> {sorted(df['numbers'].iloc[-1])}\n")

    latest_win = sorted(df['numbers'].iloc[-1])
    target_set = set(latest_win)

    print("── 1. WINNING DRAW PROPERTY & GAP ANALYSIS (Latest Draw) ──")
    last_seen_gap = {}
    total_freq = {}
    recent_10_freq = {}

    for num in range(1, POOL + 1):
        total_freq[num] = sum(1 for row in df["numbers"] if num in row)
        recent_10_freq[num] = sum(1 for row in df["numbers"].iloc[-10:] if num in row)

        gap = 0
        found = False
        for i in range(len(df) - 2, -1, -1):
            if num in df["numbers"].iloc[i]:
                found = True
                break
            gap += 1
        last_seen_gap[num] = gap if found else len(df) - 1

    print(f"{'Num':>4} | {'Total Freq':>10} | {'Last 10 Freq':>12} | {'Draws Since Seen':>16} | {'Status/Note'}")
    print("─" * 65)
    for num in latest_win:
        gap = last_seen_gap[num]
        status = "REPEAT (Last draw)" if gap == 0 else f"Cold/Due ({gap} draws)" if gap >= 5 else f"Recent ({gap} draws ago)"
        print(f"{num:>4} | {total_freq[num]:>10} | {recent_10_freq[num]:>12} | {gap:>16} | {status}")
    print()

    win_sum = sum(latest_win)
    n_odd = sum(1 for n in latest_win if n % 2 != 0)
    n_even = DRAW_SIZE - n_odd
    n_low = sum(1 for n in latest_win if n <= 19)
    n_high = DRAW_SIZE - n_low
    z1 = sum(1 for n in latest_win if 1 <= n <= 10)
    z2 = sum(1 for n in latest_win if 11 <= n <= 20)
    z3 = sum(1 for n in latest_win if 21 <= n <= 30)
    z4 = sum(1 for n in latest_win if 31 <= n <= 39)

    hist_sums = df["numbers"].apply(sum)
    print(f"Sum: {win_sum} (Historical Mean: {hist_sums.mean():.1f}, Median: {hist_sums.median():.1f}, IQR: [{hist_sums.quantile(0.25):.0f}, {hist_sums.quantile(0.75):.0f}])")
    print(f"Odd/Even: {n_odd} Odd / {n_even} Even")
    print(f"Low/High (<=19 / >19): {n_low} Low / {n_high} High")
    print(f"Zone Distribution (1-10, 11-20, 21-30, 31-39): {z1}, {z2}, {z3}, {z4}\n")

    print("── 2. MODEL PREDICTION EVALUATION & RANKING OF WINNING NUMBERS ──")
    spec12 = importlib.util.spec_from_file_location("mod12", "12_master_ai_meta_ensemble.py")
    mod12 = importlib.util.module_from_spec(spec12); spec12.loader.exec_module(mod12)

    print("Harvesting prediction vectors from all 11 steps...")
    signals_dict = mod12.harvest_all_signals(df)

    p_master, _, _ = mod12.meta_ai_blend(signals_dict, df)
    signals_dict["12. Master AI Meta-Ensemble"] = p_master

    print(f"\n{'Model Name':<28} | {'Top-6 Matches':<14} | {'Top-14 Matches':<15} | {'Avg Win Rank':<12} | {'Missed in Top 14'}")
    print("─" * 105)

    for name, p_vec in signals_dict.items():
        sorted_nums = np.argsort(p_vec)[::-1] + 1
        top6 = sorted_nums[:DRAW_SIZE]
        top14 = sorted_nums[:14]

        m6 = sorted(list(set(top6).intersection(target_set)))
        m14 = sorted(list(set(top14).intersection(target_set)))
        missed14 = sorted(list(target_set - set(top14)))

        ranks = []
        for w_num in latest_win:
            r = np.where(sorted_nums == w_num)[0][0] + 1
            ranks.append(r)
        avg_rank = np.mean(ranks)

        print(f"{name:<28} | {len(m6)}: {str(m6):<11} | {len(m14)}: {str(m14):<12} | {avg_rank:<12.1f} | {missed14}")

    print("\n── 3. DETAILED RANK OF EACH WINNING NUMBER IN MASTER AI ENSEMBLE ──")
    master_sorted = np.argsort(p_master)[::-1] + 1
    for w_num in latest_win:
        r = np.where(master_sorted == w_num)[0][0] + 1
        prob = p_master[w_num - 1] * 100
        print(f"Number #{w_num:>2} -> Ranked #{r:>2} out of {POOL} (Probability: {prob:.2f}%)")

    print(f"\nMaster AI Top-6 Ticket: {sorted(list(master_sorted[:DRAW_SIZE]))}")
    print(f"Master AI Top-14 Pool : {sorted(list(master_sorted[:14]))}")


if __name__ == "__main__":
    main()
