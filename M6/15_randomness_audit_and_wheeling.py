"""
13_randomness_audit_and_wheeling.py — Step 13: Randomness Auditing & Combinatorial Wheeling Suite for EASY6.

Modules:
  1. Statistical Randomness Audit:
     - Chi-Square Goodness-of-Fit test (Uniform distribution of numbers 1-39)
     - Serial Autocorrelation (Lag-1 correlation between consecutive draws)
     - Runs Test for Randomness (Odd/Even & High/Low sequential independence)
     - Shannon Entropy Analysis vs theoretical maximum
  2. Combinatorial Wheeling Optimization:
     - Set Cover wheeling generator (Guarantees match coverage without redundant pairs)
     - Pairwise & Triplet coverage calculator for generated ticket sets
  3. Visualizations:
     - Saves diagnostic charts to step13_randomness_audit.png in the run folder.
"""

import os
import sys
import math
import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from utils import load_data, get_run_folder, POOL, DRAW_SIZE, generate_covering_wheel, CSV_FILE

def perform_chi_square_test(df: pd.DataFrame) -> dict:
    """Perform Chi-Square Goodness-of-Fit test against a uniform distribution."""
    all_numbers = [num for row in df["numbers"] for num in row]
    total_balls = len(all_numbers)
    obs_counts = np.bincount(all_numbers, minlength=POOL + 1)[1:]
    exp_counts = np.full(POOL, total_balls / POOL)
    
    chi2_stat, p_val = stats.chisquare(f_obs=obs_counts, f_exp=exp_counts)
    return {
        "chi2_stat": chi2_stat,
        "p_value": p_val,
        "obs_counts": obs_counts,
        "exp_count": total_balls / POOL,
        "total_draws": len(df),
        "total_balls": total_balls
    }

def perform_autocorrelation_test(df: pd.DataFrame) -> dict:
    """Compute lag autocorrelation across consecutive draw indices and sums."""
    sums = df["sum"].values
    autocorr_lag1 = np.corrcoef(sums[:-1], sums[1:])[0, 1] if len(sums) > 1 else 0.0
    
    # Position-wise autocorrelation across consecutive draws
    pos_autocorrs = []
    for col in ["Winning Number 1", "2", "3", "4", "5", "6"]:
        vals = df[col].values
        if len(vals) > 1:
            r = np.corrcoef(vals[:-1], vals[1:])[0, 1]
            pos_autocorrs.append(r)
        else:
            pos_autocorrs.append(0.0)
            
    return {
        "sum_autocorr_lag1": autocorr_lag1,
        "pos_autocorrs": pos_autocorrs
    }

def perform_runs_test(sequence: list) -> dict:
    """Wald-Wolfowitz Runs Test for sequence randomness."""
    binary_seq = [1 if x > np.median(sequence) else 0 for x in sequence]
    n1 = sum(binary_seq)
    n0 = len(binary_seq) - n1
    
    runs = 1
    for i in range(1, len(binary_seq)):
        if binary_seq[i] != binary_seq[i-1]:
            runs += 1
            
    mu = 1 + (2 * n0 * n1) / (n0 + n1)
    var = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / (((n0 + n1) ** 2) * (n0 + n1 - 1))
    
    z_stat = (runs - mu) / math.sqrt(var) if var > 0 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return {
        "runs": runs,
        "expected_runs": mu,
        "z_stat": z_stat,
        "p_value": p_val
    }

def compute_entropy(df: pd.DataFrame) -> dict:
    """Compute empirical Shannon entropy of drawn numbers vs theoretical max."""
    all_numbers = [num for row in df["numbers"] for num in row]
    counts = np.bincount(all_numbers, minlength=POOL + 1)[1:]
    probs = counts / sum(counts)
    probs = probs[probs > 0]
    
    empirical_entropy = -np.sum(probs * np.log2(probs))
    max_entropy = math.log2(POOL)
    entropy_ratio = empirical_entropy / max_entropy
    
    return {
        "empirical_entropy": empirical_entropy,
        "max_entropy": max_entropy,
        "entropy_ratio": entropy_ratio
    }

def calculate_wheeling_coverage(tickets: list[tuple]) -> dict:
    """Calculate pair and triplet coverage percentages for a ticket set."""
    total_pairs = set(itertools.combinations(range(1, POOL + 1), 2))
    covered_pairs = set()
    for t in tickets:
        covered_pairs.update(itertools.combinations(t, 2))
        
    pair_cov_pct = (len(covered_pairs & total_pairs) / len(total_pairs)) * 100.0
    
    total_triplets = set(itertools.combinations(range(1, POOL + 1), 3))
    covered_triplets = set()
    for t in tickets:
        covered_triplets.update(itertools.combinations(t, 3))
        
    triplet_cov_pct = (len(covered_triplets & total_triplets) / len(total_triplets)) * 100.0
    
    return {
        "num_tickets": len(tickets),
        "covered_pairs": len(covered_pairs & total_pairs),
        "total_pairs": len(total_pairs),
        "pair_coverage_pct": pair_cov_pct,
        "covered_triplets": len(covered_triplets & total_triplets),
        "total_triplets": len(total_triplets),
        "triplet_coverage_pct": triplet_cov_pct
    }

def plot_diagnostics(chi2_res: dict, autocorr_res: dict, wheel_cov: dict, wheel_tickets: list[tuple], run_folder: str):
    """Plot diagnostic charts for the audit & wheeling suite."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("EASY6 Randomness Audit & Combinatorial Wheeling Diagnostics", fontsize=14, fontweight="bold")
    
    # 1. Number Frequency vs Expected (Uniformity)
    ax1 = axes[0, 0]
    nums = np.arange(1, POOL + 1)
    obs = chi2_res["obs_counts"]
    exp = chi2_res["exp_count"]
    
    ax1.bar(nums, obs, color="skyblue", edgecolor="navy", alpha=0.7, label="Observed Count")
    ax1.axhline(exp, color="red", linestyle="--", linewidth=2, label=f"Expected Uniform ({exp:.1f})")
    ax1.set_title(f"Chi-Square Uniformity Test (p-val: {chi2_res['p_value']:.4f})")
    ax1.set_xlabel("Ball Number (1-39)")
    ax1.set_ylabel("Frequency")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.5)
    
    # 2. Position-wise Autocorrelation
    ax2 = axes[0, 1]
    positions = [f"Pos {i}" for i in range(1, 7)]
    autocorrs = autocorr_res["pos_autocorrs"]
    
    ax2.bar(positions, autocorrs, color="teal", alpha=0.7)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.axhline(0.15, color="gray", linestyle=":", label="95% Confidence Band")
    ax2.axhline(-0.15, color="gray", linestyle=":")
    ax2.set_ylim(-0.3, 0.3)
    ax2.set_title(f"Consecutive Draw Autocorrelation (Sum Lag-1: {autocorr_res['sum_autocorr_lag1']:.4f})")
    ax2.set_ylabel("Correlation Coefficient (r)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.5)
    
    # 3. Combinatorial Wheeling Pair Coverage Growth
    ax3 = axes[1, 0]
    pair_coverages = []
    accum_pairs = set()
    total_p = wheel_cov["total_pairs"]
    
    for t in wheel_tickets[:24]:
        accum_pairs.update(itertools.combinations(t, 2))
        pair_coverages.append((len(accum_pairs) / total_p) * 100.0)
        
    ax3.plot(range(1, len(pair_coverages) + 1), pair_coverages, marker="o", color="purple", linewidth=2)
    ax3.set_title("Set Cover Pairwise Coverage Curve (Combinatorial Efficiency)")
    ax3.set_xlabel("Number of Tickets in Wheel")
    ax3.set_ylabel("Pairwise Coverage (% of 741 Pairs)")
    ax3.grid(True, linestyle=":", alpha=0.5)
    
    # 4. Diagnostic Summary Text Box
    ax4 = axes[1, 1]
    ax4.axis("off")
    summary_text = (
        f"--- RANDOMNESS AUDIT SUMMARY ---\n\n"
        f"Total Draws Analyzed: {chi2_res['total_draws']}\n"
        f"Chi-Square Stat: {chi2_res['chi2_stat']:.4f} (p-value: {chi2_res['p_value']:.4f})\n"
        f"Conclusion: {'Uniform Distribution Confirmed' if chi2_res['p_value'] > 0.05 else 'Frequency Anomaly Detected'}\n\n"
        f"Sum Lag-1 Autocorr: {autocorr_res['sum_autocorr_lag1']:.4f}\n"
        f"Conclusion: {'No Temporal Dependency (Independent)' if abs(autocorr_res['sum_autocorr_lag1']) < 0.15 else 'Serial Dependence Detected'}\n\n"
        f"--- WHEELING EFFICIENCY ---\n\n"
        f"Generated Wheel Tickets: {wheel_cov['num_tickets']}\n"
        f"Pair Coverage: {wheel_cov['pair_coverage_pct']:.2f}%\n"
        f"Triplet Coverage: {wheel_cov['triplet_coverage_pct']:.2f}%\n"
    )
    ax4.text(0.05, 0.5, summary_text, fontsize=11, family="monospace", va="center", bbox=dict(boxstyle="round,pad=1", facecolor="gainsboro", alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(run_folder, "step13_randomness_audit.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[+] Saved diagnostic plot to: {out_path}")

def main():
    print("=" * 70)
    print(" STEP 13 — EASY6 RANDOMNESS AUDITING & COMBINATORIAL WHEELING SUITE ")
    print("=" * 70)
    
    df = load_data(CSV_FILE)
    run_folder = get_run_folder()
    
    chi2_res = perform_chi_square_test(df)
    print(f"\n[1] Chi-Square Goodness of Fit Test:")
    print(f"    - Total Draws: {chi2_res['total_draws']} ({chi2_res['total_balls']} drawn)")
    print(f"    - Chi2 Statistic: {chi2_res['chi2_stat']:.4f}")
    print(f"    - p-value: {chi2_res['p_value']:.4f}")
    if chi2_res['p_value'] > 0.05:
        print("    -> Result: Fail to reject H0. Distribution is statistically UNIFORM (p > 0.05).")
    else:
        print("    -> Result: Reject H0. Distribution deviates from uniform.")
        
    autocorr_res = perform_autocorrelation_test(df)
    print(f"\n[2] Serial Autocorrelation Test (Lag-1):")
    print(f"    - Sum Autocorrelation: {autocorr_res['sum_autocorr_lag1']:.4f}")
    for i, r in enumerate(autocorr_res['pos_autocorrs'], 1):
        print(f"    - Pos {i} Autocorrelation: {r:.4f}")
        
    runs_res = perform_runs_test(df["sum"].values)
    print(f"\n[3] Runs Test for Randomness (Sum Series):")
    print(f"    - Observed Runs: {runs_res['runs']} vs Expected: {runs_res['expected_runs']:.2f}")
    print(f"    - Z-statistic: {runs_res['z_stat']:.4f}, p-value: {runs_res['p_value']:.4f}")
    
    entropy_res = compute_entropy(df)
    print(f"\n[4] Shannon Entropy Metric:")
    print(f"    - Empirical Entropy: {entropy_res['empirical_entropy']:.4f} bits")
    print(f"    - Theoretical Max: {entropy_res['max_entropy']:.4f} bits")
    print(f"    - Entropy Efficiency: {entropy_res['entropy_ratio'] * 100:.2f}%")
    
    # 5. Dynamic Harvesting of Step 12 Candidate Pool for Wheeling Optimization
    candidate_pool = list(range(1, 15))
    try:
        import importlib.util
        spec14 = importlib.util.spec_from_file_location("mod14", "14_master_ai_meta_ensemble.py")
        mod14 = importlib.util.module_from_spec(spec14)
        spec14.loader.exec_module(mod14)
        signals_dict = mod14.harvest_all_signals(df)
        meta_prob, _, _ = mod14.meta_ai_blend(signals_dict, df=df)
        candidate_pool = sorted((np.argsort(meta_prob)[::-1][:14] + 1).tolist())
        print(f"\n[5] Harvested Step 14 Master Meta-AI Top 14 Candidate Pool:")
        print(f"    - Pool: {candidate_pool}")
    except Exception as err:
        print(f"\n[5] Using default candidate pool (1..14): {err}")
    
    print(f"    Generating Combinatorial Set Cover Wheel (Match-3 Guarantee)...")
    wheel_tickets = generate_covering_wheel(candidate_pool, ticket_size=DRAW_SIZE, match_guarantee=3)
    wheel_cov = calculate_wheeling_coverage(wheel_tickets)
    print(f"    - Generated Tickets: {wheel_cov['num_tickets']}")
    print(f"    - Pair Coverage: {wheel_cov['covered_pairs']}/{wheel_cov['total_pairs']} ({wheel_cov['pair_coverage_pct']:.2f}%)")
    print(f"    - Triplet Coverage: {wheel_cov['covered_triplets']}/{wheel_cov['total_triplets']} ({wheel_cov['triplet_coverage_pct']:.2f}%)")
    
    plot_diagnostics(chi2_res, autocorr_res, wheel_cov, wheel_tickets, run_folder)
    print("\n[OK] Step 13 audit and wheeling execution completed successfully.")

if __name__ == "__main__":
    main()
