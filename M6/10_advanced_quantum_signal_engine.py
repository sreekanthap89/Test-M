"""
=============================================================
 STEP 10: ADVANCED QUANTUM & SIGNAL SCIENCE ENGINE (EASY6)
=============================================================
 LEARNING GOAL:
   1. Fast Fourier Transform (FFT) Spectral Frequency Analysis.
   2. Hawkes Self-Exciting Point Process (Intensity Kernel).
   3. Jaynes' Maximum Entropy Principle (MaxEnt Optimization).
   4. Hidden Markov Model (HMM) Latent Regime Filter.
   5. Evolutionary Genetic Weight Optimizer (GA).
   6. Combinatorial Wheeling (Covering Design 3-if-3).
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
from scipy.optimize import minimize
from scipy.fft import fft

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
from utils import get_run_folder, load_data, generate_covering_wheel, CSV_FILE, WIN_COLS, POOL, DRAW_SIZE

SEED = 42


# ── FRAMEWORK 1: FAST FOURIER TRANSFORM (FFT) SPECTRAL ANALYSIS ─────────────

def signal_fft_spectral(df, pool_size=POOL):
    """
    Computes 1D Fast Fourier Transform over the binary arrival time-series
    of each ball to measure peak spectral energy and phase resonance.
    """
    draws = df["numbers"].tolist()
    n_draws = len(draws)
    fft_probs = np.zeros(pool_size)

    for b in range(1, pool_size + 1):
        ts = np.array([1.0 if b in draw else 0.0 for draw in draws])

        fft_vals = fft(ts - ts.mean())
        power_spectrum = np.abs(fft_vals[:n_draws // 2])**2

        peak_energy = np.max(power_spectrum) if len(power_spectrum) > 0 else 0.0
        recent_phase = ts[-3:].sum()

        fft_probs[b - 1] = peak_energy * (1.0 + 0.2 * recent_phase)

    if fft_probs.sum() > 0:
        fft_probs /= fft_probs.sum()
    return fft_probs


# ── FRAMEWORK 2: HAWKES SELF-EXCITING POINT PROCESS ───────────────────────────

def signal_hawkes_process(df, pool_size=POOL, alpha=0.4, beta=0.2):
    """
    Calculates instantaneous intensity λ_i(t) using an exponential memory decay kernel:
      λ_i(t) = μ_i + α * ∑ e^{-β * (t - t_k)}
    """
    draws = df["numbers"].tolist()
    n_draws = len(draws)
    hawkes_probs = np.zeros(pool_size)

    for b in range(1, pool_size + 1):
        base_rate = sum(1 for d in draws if b in d) / max(1, n_draws)
        self_excitation = 0.0

        for t_k, d in enumerate(draws):
            if b in d:
                delta_t = n_draws - t_k
                self_excitation += alpha * math.exp(-beta * delta_t)

        hawkes_probs[b - 1] = base_rate + self_excitation

    if hawkes_probs.sum() > 0:
        hawkes_probs /= hawkes_probs.sum()
    return hawkes_probs


# ── FRAMEWORK 3: JAYNES' MAXIMUM ENTROPY PRINCIPLE (MAXENT) ─────────────────

def signal_maxent_entropy(df, pool_size=POOL, target_mean_sum=120.0):
    """
    Finds the unique probability distribution P* that maximizes Shannon Entropy:
      H(P) = - ∑ p_i * ln(p_i)
    subject to physical constraints:
      1. ∑ p_i = 1
      2. Expected single-ball mean = target_mean_sum / 6
    """
    target_ball_mean = target_mean_sum / float(DRAW_SIZE)

    def objective(p):
        p = np.clip(p, 1e-9, 1.0)
        return np.sum(p * np.log(p))

    def constraint_sum(p):
        return np.sum(p) - 1.0

    def constraint_mean(p):
        balls = np.arange(1, pool_size + 1)
        return np.sum(p * balls) - target_ball_mean

    p0 = np.full(pool_size, 1.0 / pool_size)
    bounds = [(1e-6, 1.0) for _ in range(pool_size)]
    constraints = [{'type': 'eq', 'fun': constraint_sum}, {'type': 'eq', 'fun': constraint_mean}]

    res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints)
    maxent_probs = res.x if res.success else p0

    if maxent_probs.sum() > 0:
        maxent_probs /= maxent_probs.sum()
    return maxent_probs


# ── FRAMEWORK 4: RECENCY-WEIGHTED REGIME FILTER ──────────────────────────────

def signal_recency_regime(df, pool_size=POOL):
    """
    Models recent appearance regimes using a recency-weighted frequency filter.
    """
    draws = df["numbers"].tolist()
    n_draws = len(draws)
    hmm_probs = np.zeros(pool_size)

    recent_draws = draws[-15:] if n_draws >= 15 else draws

    for b in range(1, pool_size + 1):
        state_obs = [1.0 if b in d else 0.0 for d in recent_draws]
        p_state = np.mean(state_obs) if state_obs else 0.2

        last_seen_in_recent = state_obs[-1] if state_obs else 0.0
        regime_multiplier = 1.25 if (p_state > 0.3 and last_seen_in_recent == 1) else 0.90

        hmm_probs[b - 1] = p_state * regime_multiplier

    if hmm_probs.sum() > 0:
        hmm_probs /= hmm_probs.sum()
    return hmm_probs


# ── FRAMEWORK 5: EVOLUTIONARY GENETIC WEIGHT OPTIMIZER ───────────────────────

def genetic_optimize_weights(df, n_generations=20, pop_size=15):
    """
    Evolves optimal fusion weights for [FFT, Hawkes, MaxEnt, HMM] using a Genetic Algorithm.
    """
    n_val = min(10, len(df) - 20)
    train_df = df.iloc[:-n_val].copy() if n_val > 0 else df.copy()

    p_fft     = signal_fft_spectral(train_df)
    p_hawkes  = signal_hawkes_process(train_df)
    p_maxent  = signal_maxent_entropy(train_df)
    p_recency = signal_recency_regime(train_df)

    signals = [p_fft, p_hawkes, p_maxent, p_recency]

    rng = np.random.default_rng(SEED)
    population = rng.uniform(0.1, 1.0, size=(pop_size, 4))
    population /= population.sum(axis=1, keepdims=True)

    validation_draws = df["numbers"].iloc[-n_val:].tolist() if n_val > 0 else df["numbers"].iloc[-5:].tolist()

    best_weights = population[0]
    best_score = -1.0

    for gen in range(n_generations):
        scores = []
        for w in population:
            combined = sum(w[i] * signals[i] for i in range(4))
            combined /= combined.sum()

            top14_set = set((np.argsort(combined)[::-1][:14] + 1).tolist())
            matches = sum(len(top14_set & set(d)) for d in validation_draws)
            scores.append(matches)

        scores = np.array(scores)
        best_idx = np.argmax(scores)

        if scores[best_idx] > best_score:
            best_score = scores[best_idx]
            best_weights = population[best_idx]

        top_parents = population[np.argsort(scores)[::-1][:4]]
        new_pop = list(top_parents)

        while len(new_pop) < pop_size:
            p1_idx, p2_idx = rng.choice(len(top_parents), size=2, replace=False)
            child = 0.5 * top_parents[p1_idx] + 0.5 * top_parents[p2_idx]
            if rng.random() < 0.2:
                child += rng.normal(0, 0.05, size=4)
                child = np.clip(child, 0.05, 1.0)
            child /= child.sum()
            new_pop.append(child)

        population = np.array(new_pop)

    signals_full = [
        signal_fft_spectral(df),
        signal_hawkes_process(df),
        signal_maxent_entropy(df),
        signal_recency_regime(df),
    ]

    return best_weights, signals_full


# ── MAIN EXECUTION ────────────────────────────────────────────────────────────

def main():
    print("============================================================")
    print("  STEP 10 — ADVANCED QUANTUM & SIGNAL SCIENCE ENGINE (EASY6)")
    print("============================================================")

    df = load_data(CSV_FILE)

    print("Computing 5 Scientific & Mathematical Frameworks:")
    print("  1. Fast Fourier Transform (FFT) Spectral Analysis")
    print("  2. Hawkes Self-Exciting Point Process (Intensity Decay)")
    print("  3. Jaynes' Maximum Entropy Principle (MaxEnt Entropy)")
    print("  4. Recency-Weighted Regime Filter")
    print("  5. Evolutionary Genetic Weight Optimizer (GA)...")

    best_weights, signals = genetic_optimize_weights(df, n_generations=25, pop_size=15)

    print(f"\nEvolutionary GA Evolved Weights:")
    print(f"  FFT Spectral Weight     : {best_weights[0]*100:.1f}%")
    print(f"  Hawkes Intensity Weight : {best_weights[1]*100:.1f}%")
    print(f"  MaxEnt Entropy Weight   : {best_weights[2]*100:.1f}%")
    print(f"  Recency Regime Weight   : {best_weights[3]*100:.1f}%")

    quantum_prob = sum(best_weights[i] * signals[i] for i in range(4))
    quantum_prob /= quantum_prob.sum()

    top_14_indices = np.argsort(quantum_prob)[::-1][:14]
    top_14_numbers = sorted((top_14_indices + 1).tolist())

    top_6_indices = np.argsort(quantum_prob)[::-1][:DRAW_SIZE]
    top_6_numbers = sorted((top_6_indices + 1).tolist())

    print("\n============================================================")
    print("  PHASE 1: QUANTUM SIGNAL SCIENCE PREDICTION")
    print("============================================================")
    print(f"Quantum Science Top 6 Single Predicted Ticket:  ★  {top_6_numbers}  ★")
    print(f"Quantum Science Top 14 Candidate Pool         : {top_14_numbers}")

    print("\n============================================================")
    print("  PHASE 2: COMBINATORIAL WHEELING")
    print("============================================================")
    tickets = generate_covering_wheel(top_14_numbers, ticket_size=DRAW_SIZE, match_guarantee=3)
    print(f"[Wheeling] Generated 3-if-3 covering wheel in {len(tickets)} tickets.")

    print("\nYOUR WHEELED TICKETS:")
    for i, t in enumerate(tickets, 1):
        print(f"  Ticket {i:2d}: {list(t)}")

    run_dir = get_run_folder()

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("STEP 10 — Advanced Quantum & Signal Science Engine\nEmirates Draw EASY6",
                 fontsize=15, fontweight="bold")
    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(range(1, POOL + 1), quantum_prob * 100, color='#95a5a6')
    for idx in top_14_indices:
        bars[idx].set_color('#9b59b6')
        ax1.text(idx + 1, quantum_prob[idx] * 100 + 0.05, f"#{idx+1}", ha="center",
                 fontsize=8, color="#8e44ad", fontweight="bold")

    ax1.set_title("Quantum & Signal Science Output Probabilities (Purple = Top 14 Candidates)")
    ax1.set_xlabel("Number"); ax1.set_ylabel("Probability (%)")
    ax1.set_xlim(0.5, POOL + 0.5)
    ax1.set_xticks(range(1, POOL + 1))
    ax1.tick_params(axis="x", labelsize=7.5)
    ax1.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#9b59b6', label='Top 14 Quantum Candidates'),
        Patch(facecolor='#95a5a6', label='Other Pool Numbers')
    ]
    ax1.legend(handles=legend_elements, loc="upper right")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis("off")
    summary_text = (
        "★ QUANTUM SCIENCE TOP-6 PREDICTED TICKET ★\n"
        f"  {top_6_numbers}\n\n"
        "★ QUANTUM SCIENCE TOP-14 CANDIDATE POOL ★\n"
        f"  {top_14_numbers}\n\n"
        "★ 5 SCIENTIFIC FRAMEWORKS APPLIED ★\n"
        "  1. FFT Spectral Frequency Analysis\n"
        "  2. Hawkes Self-Exciting Point Process\n"
        "  3. Jaynes' MaxEnt Principle Entropy\n"
        "  4. Hidden Markov Model (HMM) Regimes\n"
        "  5. Evolutionary Genetic Weight Optimizer"
    )
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes, fontsize=10.5,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f4f6f7", edgecolor="#8e44ad", linewidth=1.5))

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    half = (len(tickets) + 1) // 2
    t1_text = "\n".join([f"T{i:02d}: {list(t)}" for i, t in enumerate(tickets[:half], 1)])
    t2_text = "\n".join([f"T{i:02d}: {list(t)}" for i, t in enumerate(tickets[half:], half + 1)])

    ax3.text(0.02, 0.95, f"★ WHEELED TICKETS (1-{half}) ★\n" + t1_text, transform=ax3.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2f3", edgecolor="#7f8c8d"))

    ax3.text(0.52, 0.95, f"★ WHEELED TICKETS ({half+1}-{len(tickets)}) ★\n" + t2_text, transform=ax3.transAxes,
             fontsize=8.5, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#eef2f3", edgecolor="#7f8c8d"))

    chart_path = f"{run_dir}/step10_quantum_signal_engine.png"
    plt.savefig(chart_path, dpi=130, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] Chart saved -> {chart_path}")


if __name__ == "__main__":
    main()
