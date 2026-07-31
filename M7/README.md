# Emirates Draw MEGA7 — Prediction & Probability Learning Project

A complete, step-by-step Python learning project that teaches **probabilistic thinking**, **stochastic modelling**, **deep learning**, **Graph Neural Networks (GNN)**, **Hawkes processes**, **Extreme Value Theory (EVT)**, **BlackRock HRP risk parity**, and **wheeling systems** using real Emirates Draw MEGA7 data.

---

## 🎲 About Emirates Draw MEGA7

- **Pool Size:** 37 numbers (1 to 37)
- **Draw Size:** 7 numbers per draw
- **Win Guarantee Target:** Match 3 minimum win category (Guaranteed by our mathematical wheeling system)

---

## 🗂️ Project Structure

```
M7/
├── Emirates_Draw_MEGA7.csv                      ← Real draw dataset
├── utils.py                                     ← Shared session, path, and covering wheel utilities
├── enhanced_features_and_metrics.py             ← Gap analysis, momentum & Chi2 validation engine
├── gnn_hawkes_meta_learning.py                  ← GNN network, Hawkes process & EVT volatility engine
├── self_improving_test_engine.py                ← 7-step walk-forward optimization loop
├── walkforward_backtest.py                      ← 20-draw holdout evaluation suite
├── evaluate_suite.py                            ← Metric back-test evaluator
│
├── 01_data_explorer.py                          ← Step 1: Understand your data (7/37 parameters)
├── 02_frequency_analysis.py                     ← Step 2: Hot/cold numbers, chi-squared test
├── 03_probability_distributions.py              ← Step 3: Empirical probability, KL-divergence, pair co-occurrence
├── 04_monte_carlo_simulation.py                 ← Step 4: Monte Carlo simulation & Law of Large Numbers
├── 05_markov_chain.py                           ← Step 5: Transition matrices & stationary distribution π
├── 06_prediction_report.py                      ← Step 6: Ensemble prediction & historical back-testing
├── 07_advanced_prediction.py                    ← Step 7: Number-level 37×37 Markov + feedback loop adjustment
├── 08_deep_learning_and_wheeling.py             ← Step 8: MLP Neural Network A.I. & Combinatorial Wheeling
├── 09_ultra_stacking_ensemble.py                ← Step 9: Stacking Ensemble (XGBoost + LightGBM + RF + ET + MLP)
├── 10_advanced_quantum_signal_engine.py         ← Step 10: FFT spectral, Hawkes decay, MaxEnt & GA optimizer
├── 11_blackrock_quant_engine.py                 ← Step 11: Quantile Regression, Ward Graph Clustering & HRP Fusion
├── 12_enhanced_prediction_features_and_metrics.py ← Step 12: Periodicity Gap, Momentum & Chi2 Fit Score
├── 13_gnn_hawkes_meta_learning_engine.py          ← Step 13: 37-Node GNN Graph & Hawkes-Jump Process Engine
├── 14_master_ai_meta_ensemble.py                 ← Step 14: Master A.I. Meta-Ensemble & Infographic Dashboard
├── 15_randomness_audit_and_wheeling.py          ← Step 15: NIST Randomness Audit Suite & Set Cover Wheel
├── 16_final_tabular_report_chart.py             ← Step 16: Complete 16-step visual tabular report
│
├── run_all.py                                   ← Master runner (runs all 16 steps & saves to one runs/ folder)
├── README.md                                    ← This file
├── DOCUMENTATION.md                             ← Full architectural guide
└── CHANGELOG.md                                 ← Detailed record of M7 adaptation and logic changes
```

---

## 🚀 Quick Start

### 1. Run the Complete Suite

To execute the entire 16-step pipeline in sequence and generate a consolidated prediction report:

```bash
python run_all.py
```

### 2. Run Self-Improving Walk-Forward Backtest

To evaluate performance across a 20-draw holdout window:

```bash
python walkforward_backtest.py
python self_improving_test_engine.py
```

---

## 📚 Core Learning Modules

1. **Step 01 — Data Exploration:** Visualizing frequency distributions and sum bands (~133 mean for 7 drawn balls out of 37).
2. **Step 02 — Frequency Analysis:** Defining Hot/Cold thresholds (+20% / −20%) and Chi-Squared ($df=36$) testing.
3. **Step 03 — Probability Distributions:** KL Divergence, 95% Confidence Intervals, and $C(7,2)=21$ pair co-occurrence.
4. **Step 04 — Monte Carlo Simulation:** 200,000 simulations observing convergence via Law of Large Numbers.
5. **Step 05 — Markov Chain:** Sum band state transition matrices and stationary distribution $\pi$.
6. **Step 06 — Multi-Signal Ensemble:** Blending frequency, cold due, zone Markov, and pair lift signals.
7. **Step 07 — Advanced 4-Phase Markov:** $37 \times 37$ number-level transition matrix with self-correcting feedback loops.
8. **Step 08 — Deep Learning MLP:** Neural network pattern finder and Greedy Set Cover 3-if-3 wheeling system.
9. **Step 09 — Ultra Stacking ML:** Multi-model stacking classifier (XGBoost, LightGBM, Random Forest, Extra Trees, MLP).
10. **Step 10 — Quantum Science Engine:** FFT spectral energy, Hawkes decay, Jaynes MaxEnt entropy, Genetic Algorithm optimizer.
11. **Step 11 — BlackRock Quant Engine:** Quantile Regression Forests, Ward Hierarchical Graph Clustering, Poisson Jump-Diffusion, Kalman filtering, Hierarchical Risk Parity (HRP) signal fusion.
12. **Step 12 — Feature & Metric Depth:** Periodicity gap regularity, momentum differentials, Chi-Squared structural distribution fit score.
13. **Step 13 — GNN & Hawkes Meta Engine:** 37-node Graph Neural Network message passing ($H^{(l+1)} = \text{ReLU}(\hat{A} H^{(l)} W^{(l)})$), Hawkes Jump-Diffusion hazard, EVT draw sum volatility bounds.
14. **Step 14 — Master A.I. Meta-Ensemble:** Adaptive Tail-Boosted Meta-Learner V3 fusing 19 sub-model signals, generating optimal 7-ball tickets and grand infographic dashboard.
15. **Step 15 — Randomness Audit Suite:** NIST-style randomness testing (Chi-Square uniformity, serial lag-1 autocorrelation, runs test, Shannon entropy) and Set Cover wheeling.
16. **Step 16 — Final Tabular Report:** Infographic table summarizing all 16 steps and final recommendations.
