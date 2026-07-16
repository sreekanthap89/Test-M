# Emirates Draw EASY6 — Prediction & Probability Learning Project

A complete, step-by-step Python learning project that teaches **probabilistic thinking**,
**stochastic modelling**, and **prediction techniques** using real Emirates Draw EASY6 data.

---

## 🗂️ Project Structure

```
ESY6/
├── Emirates_Draw_EASY6.csv        ← your real draw data
│
├── 01_data_explorer.py            ← Step 1: Understand your data
├── 02_frequency_analysis.py       ← Step 2: Hot/cold numbers, chi-squared
├── 03_probability_distributions.py← Step 3: Empirical probability, KL-div
├── 04_monte_carlo_simulation.py   ← Step 4: Monte Carlo & Law of Large Numbers
├── 05_markov_chain.py             ← Step 5: Transition matrices & stationary π
├── 06_prediction_report.py        ← Step 6: Ensemble prediction & back-test
│
├── requirements.txt               ← Python dependencies
└── README.md                      ← This file
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Or with a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run the steps in order

```bash
python 01_data_explorer.py
python 02_frequency_analysis.py
python 03_probability_distributions.py
python 04_monte_carlo_simulation.py
python 05_markov_chain.py
python 06_prediction_report.py
```

Each script is **self-contained** — it explains concepts in the terminal output
AND saves a PNG chart to your project folder.

---

## 📚 Learning Path

| Step | Script | Concepts |
|------|--------|----------|
| 1 | `01_data_explorer.py` | DataFrames, descriptive statistics, histograms |
| 2 | `02_frequency_analysis.py` | Empirical frequency, hot/cold, chi-squared test |
| 3 | `03_probability_distributions.py` | Empirical P, KL-divergence, confidence intervals, pair co-occurrence |
| 4 | `04_monte_carlo_simulation.py` | Monte Carlo, Law of Large Numbers, risk estimation |
| 5 | `05_markov_chain.py` | States, transition matrix T, stationary π, T^n prediction |
| 6 | `06_prediction_report.py` | Ensemble model, back-testing, uncertainty communication |

---

## 🧠 Key Concepts You Will Learn

### Probabilistic Thinking
Instead of asking *"what number will come next?"*, you ask
*"what is the probability distribution over next numbers?"*
This mindset shift is the foundation of all modern AI and data science.

### Monte Carlo Simulation
Run a process thousands of times and measure the outcomes.
Used in: financial risk (VaR), physics, AI training, drug testing.

```
P(outcome) ≈ (number of simulations with that outcome) / (total simulations)
```

### Markov Chains
A model where the future depends only on the present, not the past.
Used in: Google PageRank, NLP language models, financial modelling.

```
T[i,j] = P(move to state j | currently in state i)
π = stationary distribution (long-run average)
```

### Ensemble Methods
Combining multiple weaker signals into one stronger prediction.
Used in: Random Forests, XGBoost, neural network ensembles.

```
Combined = w₁·Signal₁ + w₂·Signal₂ + w₃·Signal₃ + ...
```

---

## 📊 Output Charts

After running all 6 scripts you will have:

| File | Contents |
|------|----------|
| `step1_data_exploration.png` | Histograms, box-plots, sum over time |
| `step2_frequency_analysis.png` | Hot/cold bar chart, recency comparison |
| `step3_probability_distributions.png` | CDF, CI error bars, top pair co-occurrence |
| `step4_monte_carlo.png` | Match distribution, Law of Large Numbers, number freq |
| `step5_markov_chain.png` | Transition heatmaps, N-step convergence, persistence |
| `step6_prediction_report.png` | Signal heatmap, ensemble bars, back-test histogram |

---

## ⚠️ Honest Disclaimer

Lottery draws are **designed to be independent and random**.
No statistical model can reliably predict them better than chance.

The techniques in this project are **100% real and used in practice** in:
- 📈 Financial market modelling
- 🌦️ Weather forecasting  
- 🧬 Bioinformatics and genomics
- 🤖 Natural Language Processing (Markov chains → Transformers)
- 🏥 Medical risk prediction

This project uses lottery data as a **safe, harmless training ground**
because the ground truth is publicly verifiable every week.

---

## 🔧 Customising the Model

In `06_prediction_report.py`, change the `weights` dictionary to experiment:

```python
weights = {
    "frequency": 0.40,   # increase to trust recent hot numbers more
    "cold":      0.10,   # increase to trust "due" cold numbers more
    "markov":    0.30,   # increase to trust zone patterns more
    "pair_lift": 0.20,   # increase to trust pair co-occurrence more
}
```

The weights must NOT necessarily sum to 1 — the code normalises them automatically.

---

## 📖 Further Reading

- **Bishop, C.M.** — *Pattern Recognition and Machine Learning* (Bayesian methods)
- **Norris, J.R.** — *Markov Chains* (free PDF available)
- **Gelman et al.** — *Bayesian Data Analysis* (probabilistic thinking)
- **3Blue1Brown** — YouTube series on probability and statistics
