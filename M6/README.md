# Emirates Draw EASY6 Prediction & Institutional Quant Suite (M6)

A state-of-the-art predictive learning framework, multi-model AI ensemble, and mathematical wheeling system for **Emirates Draw EASY6** (6 balls drawn out of 39).

---

## 🚀 Key Features & Pipeline Steps

This project contains a full 13-step machine learning and quantitative finance pipeline:

1. **Step 1: Data Exploration (`01_data_explorer.py`)** — Load, clean, and inspect EASY6 CSV data, distributions, and summary statistics.
2. **Step 2: Frequency Analysis (`02_frequency_analysis.py`)** — Empirical Poisson counts, Z-scores, and Chi-Squared goodness-of-fit tests.
3. **Step 3: Probability Distributions (`03_probability_distributions.py`)** — Normal bell curves for draw sums and pair co-occurrence baseline matrices.
4. **Step 4: Monte Carlo Simulator (`04_monte_carlo_simulation.py`)** — 200,000 random sampling draws for expected return and quantile boundaries.
5. **Step 5: Markov Chain Transition Modeling (`05_markov_chain.py`)** — State transition matrices across number zones and sum bands.
6. **Step 6: Multi-Signal Ensemble (`06_prediction_report.py`)** — Blended frequency, cold/due, pair lift, and Markov signals.
7. **Step 7: 4-Phase Advanced Prediction Engine (`07_advanced_prediction.py`)** — Full $39 \times 39$ number-level transition matrix with streak-correcting feedback loops and diversity filters.
8. **Step 8: Deep Learning Neural Net & Wheeling (`08_deep_learning_and_wheeling.py`)** — Multi-Layer Perceptron (MLP) classifier with a 3-if-3 greedy set cover wheeling system.
9. **Step 9: Ultra Stacking Ensemble (`09_ultra_stacking_ensemble.py`)** — XGBoost, LightGBM, Random Forest, Extra Trees, and Neural Network multi-model stacking classifier.
10. **Step 10: Quantum Science Engine (`10_advanced_quantum_signal_engine.py`)** — Fast Fourier Transform (FFT) spectral energy, Hawkes point processes, Jaynes MaxEnt entropy, and Genetic Algorithm (GA) optimization.
11. **Step 11: BlackRock Institutional Quant Engine (`11_blackrock_quant_engine.py`)** — Quantile Regression Forests ($q_{10}, q_{50}, q_{90}$), Ward Hierarchical Clustering, Jump-Diffusion Poisson arrival modeling, and Hierarchical Risk Parity (HRP) signal fusion.
12. **Step 12: Master AI Meta-Ensemble (`12_master_ai_meta_ensemble.py`)** — Adaptive Tail-Boosted Meta-Learner fusing all 14 model signals and generating the grand infographic chart (`step12_master_ai_meta_ensemble.png`).
13. **Step 13: Final Tabular Report Infographic (`13_final_tabular_report_chart.py`)** — Human-understandable visual table summary PNG chart (`step13_final_tabular_report.png`).

---

## 🛠️ Usage

Run the master pipeline script to execute all steps sequentially and harvest output predictions:

```bash
python run_all.py
```

Run the back-testing evaluation suite:

```bash
python evaluate_suite.py
```

Run draw gap analysis:

```bash
python check_gap.py
```

---

## 📊 EASY6 Constants & Rules

- **Pool Size (`POOL`)**: 39 (numbers 1 to 39)
- **Draw Size (`DRAW_SIZE`)**: 6 (6 winning numbers drawn per draw)
- **Dataset**: `Emirates_Draw_EASY6.csv`
- **Mean Expected Sum**: $6 \times 20 = 120.0$
- **High/Low Split**: Low $\le 19$, High $> 19$
