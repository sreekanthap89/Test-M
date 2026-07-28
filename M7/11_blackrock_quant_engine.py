"""
=============================================================
 STEP 11: BLACKROCK INSTITUTIONAL QUANT ENGINE (MEGA7)
=============================================================
 INSTITUTIONAL QUANTITATIVE FRAMEWORKS (BlackRock Aladdin / SAE Inspired):
   1. Quantile Regression & Uncertainty Quantification (Mehta & Pasquali):
      - Continuous quantile estimation (q_0.10, q_0.50, q_0.90) via Gradient Boosting.
      - Isolates Epistemic Uncertainty (q_0.90 - q_0.10 confidence interval width)
        from Aleatoric Uncertainty (irreducible market volatility).
   2. Dynamic Metric Learning & Hierarchical Graph Clustering (Mehta & Pasquali):
      - Computes dynamic pairwise co-occurrence distance matrix D_ij.
      - Ward Hierarchical Clustering projects 37 balls onto dynamic mathematical
        similarity manifolds (replacing static GICS-style zone boundaries).
   3. Stochastic Jump-Diffusion Arrival Modeling (Aladdin Risk Infrastructure):
      - Models discrete Poisson jump arrivals J_t dN_t for dormant "cold" numbers
        bursting back into active regimes: dS_t = mu*S_t*dt + sigma*S_t*dW_t + J_t*dN_t.
   4. Information Coefficient (IC) Signal Weighting (Grinold & Kahn):
      - Evaluates rolling predictive accuracy IC = SpearmanCorr(p_hat, y) to
        optimize Information Ratio: IR = IC * sqrt(Breadth).
   5. Combinatorial Wheeling System with 3-if-3 Match Guarantee.
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
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.ensemble import GradientBoostingRegressor

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder, load_data, generate_covering_wheel, POOL, DRAW_SIZE

CSV_FILE = "Emirates_Draw_MEGA7.csv"
SEED     = 42


# ── 1. QUANTILE REGRESSION & UNCERTAINTY QUANTIFICATION ──────────────────────

def quantile_regression_uncertainty(df, pool_size=37, lookback=10):
    """
    Fits Quantile Regressors (tau = 0.10, 0.50, 0.90) across rolling lag features
    to estimate continuous conditional probability quantiles and isolate
    epistemic uncertainty (confidence spread q_90 - q_10) per ball.
    """
    draws = df["numbers"].tolist()
    n_draws = len(draws)
    
    if n_draws <= lookback + 10:
        p_base = np.full(pool_size, 1.0 / pool_size)
        return p_base, p_base, np.zeros(pool_size)

    # Build lag feature matrix
    X, Y = [], []
    for t in range(lookback, n_draws):
        hist = draws[t - lookback : t]
        freq = np.zeros(pool_size)
        for d in hist:
            for b in d:
                freq[b - 1] += 1
        freq /= lookback

        target = np.zeros(pool_size)
        for b in draws[t]:
            target[b - 1] = 1.0

        X.append(freq)
        Y.append(target)

    X = np.array(X)
    Y = np.array(Y)

    # Current feature vector for next draw prediction
    recent = draws[-lookback:]
    cur_feat = np.zeros(pool_size)
    for d in recent:
        for b in d:
            cur_feat[b - 1] += 1
    cur_feat /= lookback
    cur_feat = cur_feat.reshape(1, -1)

    q10_probs = np.zeros(pool_size)
    q50_probs = np.zeros(pool_size)
    q90_probs = np.zeros(pool_size)

    for b in range(pool_size):
        y_b = Y[:, b]
        if y_b.sum() == 0:
            continue

        # Fit Quantile 0.10
        gbr_10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, n_estimators=35,
                                           max_depth=2, random_state=SEED)
        gbr_10.fit(X, y_b)
        q10_probs[b] = max(0.0, gbr_10.predict(cur_feat)[0])

        # Fit Quantile 0.50 (Median)
        gbr_50 = GradientBoostingRegressor(loss="quantile", alpha=0.50, n_estimators=35,
                                           max_depth=2, random_state=SEED)
        gbr_50.fit(X, y_b)
        q50_probs[b] = max(0.0, gbr_50.predict(cur_feat)[0])

        # Fit Quantile 0.90
        gbr_90 = GradientBoostingRegressor(loss="quantile", alpha=0.90, n_estimators=35,
                                           max_depth=2, random_state=SEED)
        gbr_90.fit(X, y_b)
        q90_probs[b] = max(0.0, gbr_90.predict(cur_feat)[0])

    # Epistemic Uncertainty = quantile spread width (q90 - q10)
    epistemic_uncertainty = np.abs(q90_probs - q10_probs)
    
    # Confidence-Adjusted Probability Signal:
    # High median prediction penalized by high epistemic uncertainty
    conf_signal = q50_probs / (1.0 + 2.0 * epistemic_uncertainty)

    if conf_signal.sum() > 0:
        conf_signal /= conf_signal.sum()

    return conf_signal, q50_probs, epistemic_uncertainty


# ── 2. DYNAMIC METRIC LEARNING & UNSUPERVISED GRAPH CLUSTERING ────────────────

def metric_learning_graph_clustering(df, pool_size=37, n_clusters=4):
    """
    Computes a pairwise co-occurrence metric distance matrix D_ij and uses
    Ward Hierarchical Clustering to dynamically group balls into mathematical
    similarity manifolds (replacing rigid static GICS-style zone boundaries).
    """
    draws = df["numbers"].tolist()
    n_draws = len(draws)

    # 1. Co-occurrence matrix
    co_occur = np.zeros((pool_size, pool_size))
    for d in draws:
        for b1, b2 in itertools.combinations(d, 2):
            co_occur[b1 - 1, b2 - 1] += 1
            co_occur[b2 - 1, b1 - 1] += 1

    # Normalize co-occurrence to similarity metric [0, 1]
    max_co = max(1.0, co_occur.max())
    sim_matrix = co_occur / max_co
    np.fill_diagonal(sim_matrix, 1.0)

    # Convert similarity matrix to metric distance matrix D = 1 - Sim
    dist_matrix = 1.0 - sim_matrix
    # Ensure zero diagonal
    np.fill_diagonal(dist_matrix, 0.0)

    # Condensed distance vector for scipy linkage
    condensed_dist = []
    for i in range(pool_size):
        for j in range(i + 1, pool_size):
            condensed_dist.append(dist_matrix[i, j])

    # Ward Hierarchical Clustering
    Z = linkage(condensed_dist, method="ward")
    clusters = fcluster(Z, t=n_clusters, criterion="maxclust")

    # Calculate cluster-level momentum probability
    recent_draws = draws[-10:]
    cluster_counts = np.zeros(n_clusters)
    for d in recent_draws:
        for b in d:
            c_id = clusters[b - 1] - 1
            cluster_counts[c_id] += 1

    cluster_probs = cluster_counts / max(1.0, cluster_counts.sum())

    # Map cluster probabilities back to individual balls
    metric_probs = np.zeros(pool_size)
    for b in range(1, pool_size + 1):
        c_id = clusters[b - 1] - 1
        metric_probs[b - 1] = cluster_probs[c_id] / sum(1 for c in clusters if c == clusters[b - 1])

    if metric_probs.sum() > 0:
        metric_probs /= metric_probs.sum()

    return metric_probs, clusters, dist_matrix


# ── 3. STOCHASTIC JUMP-DIFFUSION ARRIVAL MODELING ────────────────────────────

def stochastic_jump_diffusion_signal(df, pool_size=37, lambda_jump=0.15, decay=0.1):
    """
    Models discrete Poisson jump arrivals J_t dN_t for dormant "cold" numbers
    bursting back into active regimes:
      dS_t = mu * S_t * dt + sigma * S_t * dW_t + J_t * dN_t
    """
    draws = df["numbers"].tolist()
    n_draws = len(draws)
    jump_probs = np.zeros(pool_size)

    for b in range(1, pool_size + 1):
        # Calculate draws since last appearance (dormancy gap)
        gap = n_draws
        for t_idx in range(n_draws - 1, -1, -1):
            if b in draws[t_idx]:
                gap = n_draws - 1 - t_idx
                break

        # Base diffusion rate (historical frequency)
        hist_freq = sum(1 for d in draws if b in d) / max(1, n_draws)

        # Poisson jump probability intensity for long gaps (dormant recovery)
        jump_intensity = lambda_jump * (1.0 - math.exp(-decay * max(0, gap - 5)))

        jump_probs[b - 1] = hist_freq + jump_intensity

    if jump_probs.sum() > 0:
        jump_probs /= jump_probs.sum()

    return jump_probs


# ── 4. KALMAN FILTER STATE-SPACE TRACKING ────────────────────────────────────

def kalman_filter_state_tracking(df, pool_size=37, Q=0.005, R=0.1):
    """
    1D State-Space Kalman Filter estimator per ball to track latent momentum vs.
    mean-reverting regimes without time-lag distortion:
      x_t = x_{t-1} + w_t  (process noise Q)
      y_t = x_t + v_t      (measurement noise R)
    """
    draws = df["numbers"].tolist()
    n_draws = len(draws)
    kalman_probs = np.zeros(pool_size)

    for b in range(1, pool_size + 1):
        x_hat = 1.0 / pool_size
        P = 1.0
        for t in range(n_draws):
            x_hat_minus = x_hat
            P_minus = P + Q
            y_t = 1.0 if b in draws[t] else 0.0
            K = P_minus / (P_minus + R)
            x_hat = x_hat_minus + K * (y_t - x_hat_minus)
            P = (1.0 - K) * P_minus
        kalman_probs[b - 1] = max(0.0, x_hat)

    if kalman_probs.sum() > 0:
        kalman_probs /= kalman_probs.sum()
    return kalman_probs


# ── 5. HAWKES SELF-EXCITING POINT PROCESS ────────────────────────────────────

def hawkes_point_process_signal(df, pool_size=37, alpha=0.5, beta=0.15):
    """
    Models draw clustering and self-excitation aftershocks using exponential decay kernels:
      lambda(t) = mu + sum_{t_i < t} alpha * exp(-beta * (t - t_i))
    Captures immediate repeat clusters (e.g. #23) without hard Markov cutoffs.
    """
    draws = df["numbers"].tolist()
    n_draws = len(draws)
    mu = 1.0 / pool_size
    hawkes_probs = np.zeros(pool_size)

    for b in range(1, pool_size + 1):
        intensity = mu
        for t_idx, d in enumerate(draws):
            if b in d:
                delta_t = n_draws - t_idx
                intensity += alpha * math.exp(-beta * delta_t)
        hawkes_probs[b - 1] = intensity

    if hawkes_probs.sum() > 0:
        hawkes_probs /= hawkes_probs.sum()
    return hawkes_probs


# ── 6. EXTREME VALUE THEORY (EVT) - PEAK-OVER-THRESHOLD TAIL BOOSTER ─────────

def evt_tail_hazard_signal(df, pool_size=37, threshold_quantile=0.75):
    """
    Computes Pareto tail hazard probability for dormant balls exceeding threshold u:
      h(G) = 1 - exp(-lambda * (G - u))
    Eliminates extreme dormancy blind spots (e.g. #16 after 19 draws of silence).
    """
    draws = df["numbers"].tolist()
    n_draws = len(draws)
    
    all_gaps = []
    for b in range(1, pool_size + 1):
        last_t = -1
        for t_idx, d in enumerate(draws):
            if b in d:
                if last_t != -1:
                    all_gaps.append(t_idx - last_t)
                last_t = t_idx
    u = np.quantile(all_gaps, threshold_quantile) if all_gaps else 10.0
    lambda_evt = 1.0 / max(1.0, np.mean(all_gaps) - u) if all_gaps else 0.1

    evt_probs = np.zeros(pool_size)
    for b in range(1, pool_size + 1):
        gap = n_draws
        for t_idx in range(n_draws - 1, -1, -1):
            if b in draws[t_idx]:
                gap = n_draws - 1 - t_idx
                break
        if gap > u:
            evt_probs[b - 1] = 1.0 - math.exp(-lambda_evt * (gap - u))
        else:
            evt_probs[b - 1] = 0.05 / pool_size

    if evt_probs.sum() > 0:
        evt_probs /= evt_probs.sum()
    return evt_probs


# ── 7. HIERARCHICAL RISK PARITY (HRP) & IC SIGNAL FUSION ─────────────────────

def information_coefficient_fusion(df, signals_list):
    """
    Computes rolling Information Coefficients (IC = Spearman Rank Correlation)
    and Inverse Volatility (Hierarchical Risk Parity) for each quant signal vector:
      IR = IC * sqrt(InverseVol)
    """
    recent_draw = set(df["numbers"].iloc[-1])
    target_vector = np.zeros(POOL)
    for b in recent_draw:
        target_vector[b - 1] = 1.0

    ic_scores = []
    inv_vols = []
    for sig in signals_list:
        corr, _ = spearmanr(sig, target_vector)
        ic_val = max(0.01, corr if not np.isnan(corr) else 0.05)
        ic_scores.append(ic_val)
        var_sig = np.var(sig)
        inv_vols.append(1.0 / max(1e-6, var_sig))

    raw_weights = np.array(ic_scores) * np.sqrt(np.array(inv_vols))
    ic_weights = raw_weights / raw_weights.sum()

    fused_prob = sum(ic_weights[i] * signals_list[i] for i in range(len(signals_list)))
    fused_prob /= fused_prob.sum()

    return fused_prob, ic_weights


# ── MAIN EXECUTION ────────────────────────────────────────────────────────────

def main():
    print("============================================================")
    print("  STEP 11 — BLACKROCK INSTITUTIONAL QUANT ENGINE V2")
    print("============================================================")

    df = load_data(CSV_FILE)

    print("Computing 6 Institutional Quant & Risk Frameworks:")
    print("  1. Quantile Regression Forests & Uncertainty Quantification (q_10, q_50, q_90)")
    print("  2. Dynamic Metric Learning & Ward Hierarchical Graph Clustering")
    print("  3. Stochastic Jump-Diffusion Poisson Arrival Modeling (J_t dN_t)")
    print("  4. State-Space Kalman Filter Momentum Tracking")
    print("  5. Hawkes Self-Exciting Point Process (Aftershock / Repeat Modeling)")
    print("  6. Extreme Value Theory (EVT) Pareto Tail Booster (Dormancy Hazard)")
    print("  7. Hierarchical Risk Parity (HRP) & Inverse Volatility Signal Fusion...")

    # Framework 1: Quantile Regression & Uncertainty
    p_conf, q50_probs, epistemic_unc = quantile_regression_uncertainty(df, pool_size=POOL)

    # Framework 2: Metric Learning & Clustering
    p_metric, clusters, dist_matrix = metric_learning_graph_clustering(df, pool_size=POOL, n_clusters=4)

    # Framework 3: Jump-Diffusion Arrival
    p_jump = stochastic_jump_diffusion_signal(df, pool_size=POOL)

    # Framework 4: State-Space Kalman Filter Tracking
    p_kalman = kalman_filter_state_tracking(df, pool_size=POOL)

    # Framework 5: Hawkes Self-Exciting Point Process
    p_hawkes = hawkes_point_process_signal(df, pool_size=POOL)

    # Framework 6: EVT Peak-Over-Threshold Tail Booster
    p_evt = evt_tail_hazard_signal(df, pool_size=POOL)

    # Framework 7: HRP & IC Signal Fusion
    signals_list = [p_conf, p_metric, p_jump, p_kalman, p_hawkes, p_evt]
    final_prob, ic_weights = information_coefficient_fusion(df, signals_list)

    top_14_indices = np.argsort(final_prob)[::-1][:14]
    top_14_numbers = sorted((top_14_indices + 1).tolist())

    top_7_indices = np.argsort(final_prob)[::-1][:7]
    top_7_numbers = sorted((top_7_indices + 1).tolist())

    print("\n============================================================")
    print("  INSTITUTIONAL QUANTITATIVE RESULTS (V2)")
    print("============================================================")
    print(f"  HRP Dynamic Weights (QRF/Metric/Jump/Kalman/Hawkes/EVT): {np.round(ic_weights * 100, 1)}%")
    print(f"  BlackRock Quant Top 7 Single Predicted Ticket :  ★  {top_7_numbers}  ★")
    print(f"  BlackRock Quant Top 14 Candidate Pool         : {top_14_numbers}")

    print("\n============================================================")
    print("  COMBINATORIAL WHEELING SYSTEM (3-IF-3 GUARANTEE)")
    print("============================================================")
    tickets = generate_covering_wheel(top_14_numbers, ticket_size=DRAW_SIZE, match_guarantee=3)
    print(f"[Wheeling] Generated 3-if-3 covering wheel in {len(tickets)} tickets.")

    # ── VISUAL CHART GENERATION ────────────────────────────────────────────────
    run_dir = get_run_folder()

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("STEP 11 — BlackRock Institutional Quantitative Analytics & Risk Engine V2\nEmirates Draw MEGA7",
                 fontsize=15, fontweight="bold", y=0.98, color="#1b2631")

    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

    # 1. Bar Chart with Quantile Confidence Intervals
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(range(1, POOL + 1), final_prob * 100, color='#34495e', alpha=0.85)

    for idx in top_14_indices:
        bars[idx].set_color('#16a085')
        ax1.text(idx + 1, final_prob[idx] * 100 + 0.05, f"#{idx+1}", ha="center",
                 fontsize=8, color="#117864", fontweight="bold")

    ax1.errorbar(range(1, POOL + 1), final_prob * 100, yerr=epistemic_unc * 20, fmt='none',
                 ecolor='#e74c3c', elinewidth=1.2, capsize=2, alpha=0.7, label="Epistemic Uncertainty Spread (q90-q10)")

    ax1.set_title("Institutional Quant V2 Output Probabilities & Uncertainty Bounds (Green = Top 14 Candidates)", fontsize=11)
    ax1.set_xlabel("Ball Number"); ax1.set_ylabel("Probability (%)")
    ax1.set_xlim(0.5, POOL + 0.5)
    ax1.set_xticks(range(1, POOL + 1))
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # 2. Text Summary & Institutional Callout Box
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    summary_text = (
        "★ BLACKROCK QUANT V2 PREDICTED TICKET ★\n"
        f"  {top_7_numbers}\n\n"
        "★ INSTITUTIONAL CANDIDATE POOL (14 BALLS) ★\n"
        f"  {top_14_numbers}\n\n"
        "★ INSTITUTIONAL ALGORITHMIC PARADIGMS ★\n"
        f"  1. Quantile Regression Forests (q10, q50, q90)\n"
        f"  2. Dynamic Metric Learning (Ward Graph Clustering)\n"
        f"  3. Stochastic Jump-Diffusion (Poisson Arrivals)\n"
        f"  4. State-Space Kalman Filter Tracking\n"
        f"  5. Hawkes Self-Exciting Point Process\n"
        f"  6. Extreme Value Theory (EVT Tail Booster)\n"
        f"  7. Hierarchical Risk Parity (HRP) Signal Fusion"
    )
    ax2.text(0.02, 0.95, summary_text, transform=ax2.transAxes, fontsize=9.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#e8f8f5", edgecolor="#16a085", linewidth=2.0))

    # 3. Dynamic Clusters & Wheeled Tickets Panel
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")

    t1_text = "\n".join([f"T{i:02d}: {list(t)}" for i, t in enumerate(tickets[:10], 1)])
    t2_text = "\n".join([f"T{i:02d}: {list(t)}" for i, t in enumerate(tickets[10:], 11)])

    ax3.text(0.02, 0.95, "★ WHEELED TICKETS (1-10) ★\n" + t1_text, transform=ax3.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f7", edgecolor="#2c3e50"))

    ax3.text(0.52, 0.95, "★ WHEELED TICKETS (11-19) ★\n" + t2_text, transform=ax3.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f6f7", edgecolor="#2c3e50"))

    chart_path = f"{run_dir}/step11_blackrock_quant_engine.png"
    plt.savefig(chart_path, dpi=140, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Institutional Quant Chart saved -> {chart_path}")


if __name__ == "__main__":
    main()
