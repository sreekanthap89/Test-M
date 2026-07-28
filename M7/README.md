# Emirates Draw MEGA7 — Prediction & Probability Learning Project

A complete, step-by-step Python learning project that teaches **probabilistic thinking**, **stochastic modelling**, **deep learning**, and **wheeling systems** using real Emirates Draw MEGA7 data.

---

## 🎲 About Emirates Draw MEGA7

- **Pool Size:** 37 numbers (1 to 37)
- **Draw Size:** 7 numbers per draw
- **Win Guarantee Target:** Match 3 minimum win category (Guaranteed by our mathematical wheeling system in Step 8)

---

## 🗂️ Project Structure

```
M7/
├── Emirates_Draw_MEGA7.csv            ← Your real draw dataset (177+ draws)
├── utils.py                           ← Shared session and path utilities
│
├── 01_data_explorer.py                ← Step 1: Understand your data (7/37 parameters)
├── 02_frequency_analysis.py           ← Step 2: Hot/cold numbers, chi-squared test
├── 03_probability_distributions.py    ← Step 3: Empirical probability, KL-divergence, pair co-occurrence
├── 04_monte_carlo_simulation.py       ← Step 4: Monte Carlo simulation & Law of Large Numbers
├── 05_markov_chain.py                 ← Step 5: Transition matrices & stationary distribution π
├── 06_prediction_report.py            ← Step 6: Ensemble prediction & historical back-testing
├── 07_advanced_prediction.py          ← Step 7: Number-level 37×37 Markov + feedback loop adjustment
├── 08_deep_learning_and_wheeling.py   ← Step 8: MLP Neural Network A.I. & Combinatorial Wheeling
│
├── run_all.py                         ← Master runner (runs all steps & saves to one runs/ folder)
├── README.md                          ← This file
└── CHANGELOG.md                       ← Detailed record of M7 adaptation and logic changes
```

---

## 🚀 Quick Start

### 1. Run Individual Steps

You can run any step individually from within the `M7/` directory:

```bash
python 01_data_explorer.py
python 02_frequency_analysis.py
python 03_probability_distributions.py
python 04_monte_carlo_simulation.py
python 05_markov_chain.py
python 06_prediction_report.py
python 07_advanced_prediction.py
python 08_deep_learning_and_wheeling.py
```

Each script will output detailed educational explanations to the console and save a high-resolution PNG chart in a timestamped folder inside `runs/`.

### 2. Run the Complete Suite

To execute the entire pipeline in sequence and generate a consolidated prediction report:

```bash
python run_all.py
```

---

## 📚 Core Learning Modules

1. **Step 1 — Data Exploration:**
   - Visualizing frequency distributions for a 37-number pool.
   - Analyzing sum band distributions (expected mean ~133 for 7 drawn balls).
   - Zone breakdown across 4 zones: Z1 (1–10), Z2 (11–20), Z3 (21–30), Z4 (31–37).

2. **Step 2 — Frequency Analysis:**
   - Defining statistically significant Hot (+20% above expected) and Cold (−20% below expected) thresholds.
   - Using Chi-Squared uniformity testing to evaluate randomness.

3. **Step 3 — Probability Distributions:**
   - Measuring deviation from randomness using Kullback-Leibler (KL) Divergence.
   - Calculating 95% Confidence Intervals for ball appearance rates.
   - Pairwise co-occurrence matrix (21 pairs per draw in MEGA7).

4. **Step 4 — Monte Carlo Simulation:**
   - Simulating 50,000 draws to observe convergence via the Law of Large Numbers.
   - Comparing uniform sampling against empirically biased sampling.

5. **Step 5 — Markov Chain Transition Modelling:**
   - Modeling state transitions between draw sum bands and dominant zones.
   - Computing the long-run stationary distribution $\pi$.

6. **Step 6 — Prediction Report:**
   - Blending 4 distinct probabilistic signals into a unified ensemble score.
   - Validating predictive performance using rolling historical back-tests.

7. **Step 7 — Advanced Prediction Framework:**
   - Building a full 37×37 number-level Markov transition matrix.
   - Implementing self-correcting feedback loops (sum streak and zone dominance dampening).
   - Applying a diversity filter (enforcing zone coverage and anti-clustering).

8. **Step 8 — Deep Learning & Wheeling System:**
   - Training a Multi-Layer Perceptron (MLP) neural network on time-series windowed draws.
   - Applying a Greedy Set Cover combinatorial wheeling algorithm to turn 14 candidate numbers into a minimal set of 7-number tickets with a guaranteed 3-if-3 match.
