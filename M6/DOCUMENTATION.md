# Emirates Draw EASY6 — Complete Documentation
## How the Prediction Pipeline Works & How All Steps Connect

---

## Table of Contents
1. [Big Picture — The Pipeline](#1-big-picture--the-pipeline)
2. [Step-by-Step Explanation](#2-step-by-step-explanation)
3. [How the Steps Connect](#3-how-the-steps-connect)
4. [The Data Flow Diagram](#4-the-data-flow-diagram)
5. [Key Mathematical Concepts](#5-key-mathematical-concepts)
6. [The 4-Phase Framework Explained](#6-the-4-phase-framework-explained)
7. [What Each Output Chart Shows](#7-what-each-output-chart-shows)
8. [Running the Pipeline](#8-running-the-pipeline)
9. [Extending the Project](#9-extending-the-project)

---

## 1. Big Picture — The Pipeline

The project answers one question:

> **"Given 197 historical Emirates Draw EASY6 results, what can statistics and probability tell us about the next draw?"**

Instead of a single script, the work is split into **7 progressive steps** where each step builds on the previous one:

```
Raw CSV Data
     |
     v
[Step 1] Understand the data        -> descriptive statistics
     |
     v
[Step 2] Count frequencies          -> hot/cold numbers, deviation %
     |
     v
[Step 3] Build probability model    -> empirical P, confidence intervals
     |
     v
[Step 4] Simulate future draws      -> Monte Carlo (50,000 runs)
     |
     v
[Step 5] Model draw-to-draw patterns -> Markov transition matrices
     |
     v
[Step 6] Blend signals (ensemble)   -> weighted prediction + back-test
     |
     v
[Step 7] Full 4-phase framework     -> FINAL TICKET prediction
```

Each step produces:
- **Console output** — explanations + computed results
- **A PNG chart** — saved to the timestamped run folder

---

## 2. Step-by-Step Explanation

---

### Step 1 — Data Exploration (`01_data_explorer.py`)

**Purpose:** Before any prediction, you must *understand* your data.
You cannot model something you haven't looked at first.

**What it does:**

| Concept | Explanation | Result |
|---------|-------------|--------|
| Load CSV | Reads `Emirates_Draw_EASY6.csv`, parses dates, converts winning numbers to integers | Clean DataFrame |
| Descriptive stats | Mean, median, std deviation, min/max of all drawn numbers | Mean = 20.22 (theoretical 20.5) |
| Draw sums | Adds the 6 numbers in each draw | Mean sum = 121.3 (theoretical: 123) |
| Draw range | Highest - lowest number in each draw | Mean range = 28.4 |

**Key insight:** The mean (20.22) is very close to the theoretical uniform mean (20.5). This is our baseline — the draw looks approximately random.

**Output chart:** `step1_data_exploration.png`
- Histogram of all 1,182 individual drawn numbers
- Box-plot per draw position (N1 to N6)
- Distribution of draw sums
- Distribution of draw ranges
- Sum over time (checking for time trends)

---

### Step 2 — Frequency Analysis (`02_frequency_analysis.py`)

**Purpose:** Count how often each number (1-40) has appeared and measure how far each is from *expected*.

**THE FRAMEWORK FORMULA:**
```
Expected frequency = (total draws x 6) / 40
                   = (197 x 6) / 40
                   = 29.55 times per number

Deviation_i = (Actual_i - Expected) / Expected x 100%

Weighted P_i = freq_i / sum(all freq)   [used in Monte Carlo]
```

| Concept | Explanation |
|---------|-------------|
| Raw frequency | Count how many times each of 1-40 was drawn across all 197 draws |
| Deviation % | How far each number is above/below expected (+20% = HOT, -20% = COLD) |
| Weighted P_i | The empirical probability for each number |
| Chi-squared test | Is the deviation from uniform just random noise? |
| Recency weighting | Recent draws given higher weight (exponential decay) |

**Hot vs Cold:**
- **Hot numbers** appeared more than 20% above expected
- **Cold numbers** appeared more than 20% below expected
- Neither guarantees future results — but these are signals used later

**Connects to:** Step 3, Step 6, Step 7, and the final report.

**Output chart:** `step2_frequency_analysis.png`

---

### Step 3 — Probability Distributions (`03_probability_distributions.py`)

**Purpose:** Move from raw counts to proper probability theory.

| Concept | Formula | What it tells you |
|---------|---------|-------------------|
| Empirical probability | `P(n) = freq(n) / total_drawn` | Observed likelihood of each number |
| KL-Divergence | `Sum P(x) * log(P(x)/Q(x))` | How far our data is from uniform |
| 95% Confidence Interval | `P +/- 1.96 * sqrt(P*(1-P)/n)` | Range where the "true" probability lies |
| Pair co-occurrence | Count how often (A, B) appear in the same draw | Numbers that come together |

**Key finding:** KL-Divergence is very close to 0, confirming the draw is close to uniform.

**Connects to:** Step 4 (uses empirical P as sampling weights), Step 6 (`signal_pair_lift`).

**Output chart:** `step3_probability_distributions.png`

---

### Step 4 — Monte Carlo Simulation (`04_monte_carlo_simulation.py`)

**Purpose:** Instead of solving for probabilities mathematically, *simulate* them by running 50,000 fake draws.

**What Monte Carlo means:**
> Run a random process thousands of times, then measure what actually happened across all those runs.

**Two strategies compared:**
- **Uniform**: every number equally likely (pure random baseline)
- **Biased**: numbers weighted by their historical frequency

**Law of Large Numbers:** As simulations increase, the running mean stabilises. Verified visually with a convergence chart.

**Connects to:** Step 6 and Step 7 both reuse the MC engine with their ensemble probability vectors.

**Output chart:** `step4_monte_carlo.png`

---

### Step 5 — Markov Chain (`05_markov_chain.py`)

**Purpose:** Model how the system state (zone/sum band dominance) transitions from draw to draw.

**The Markov Property:**
> P(next state | all history) = P(next state | current state only)

**What it does:**

| Concept | Explanation |
|---------|-------------|
| States (Zone) | 4 zones: Z1(1-10), Z2(11-20), Z3(21-30), Z4(31-40) |
| Transition matrix T | `T[i,j]` = probability of going from zone i to zone j |
| Stationary distribution pi | Long-run proportion in each state: solve `pi x T = pi` |
| N-step prediction | `T^n` = probability distribution after n draws from current state |

**Key formula:**
```
Stationary distribution pi:   solve  pi x T = pi,  sum(pi) = 1
N-step:                        T^n = T x T x T x ... (n times)
```

**As n -> infinity:** Every row of `T^n` converges to pi. For long-term prediction, starting state stops mattering.

**Connects to:** Step 6 (`signal_markov_zone`), Step 7 (upgraded to full 40x40 number-level matrix).

**Output chart:** `step5_markov_chain.png`

---

### Step 6 — Prediction Report (`06_prediction_report.py`)

**Purpose:** Combine all signals into one **ensemble** prediction and measure its quality via back-testing.

**The four signals and their weights:**

| Signal | Source | Weight | Description |
|--------|--------|--------|-------------|
| `signal_frequency` | Steps 1-2 | 40% | Recency-weighted historical frequency |
| `signal_cold` | Step 2 | 10% | Numbers not seen in last 8 draws get bonus |
| `signal_markov_zone` | Step 5 | 30% | Zone transition probability from last draw |
| `signal_pair_lift` | Step 3 | 20% | Numbers that co-occur with last draw's numbers |

**Ensemble formula:**
```
P_combined = (0.40 x P_freq + 0.10 x P_cold + 0.30 x P_markov + 0.20 x P_pair)
           / (0.40 + 0.10 + 0.30 + 0.20)
```

**Back-testing:** For each of the last 30 draws, train on all prior data, predict that draw, measure matches. Result: ~1.03 correct per draw (vs random baseline 0.90). Confirms model is at noise level — expected for a random lottery.

**Output chart:** `step6_prediction_report.png`

---

### Step 7 — Advanced Prediction (`07_advanced_prediction.py`)

**Purpose:** Implement the complete 4-phase framework, filling 2 gaps from Step 6:
1. **Number-level Markov** (40x40 instead of 4-state zone Markov)
2. **Feedback loop** (detect streaks and adjust probabilities)

**Phase 3 — Number-Level Markov (GAP 1 FILLED):**
```
T[i, j] = P(number j+1 appears in next draw | number i+1 appeared in current draw)

For each consecutive pair of draws:
  for each src in current_draw:
    for each dst in next_draw:
      T[src-1, dst-1] += 1

Normalise rows, then:
Score for candidate = Sum of T[src, candidate] for all src in last draw
```

Why better than zone-level: We know "number 27 specifically tends to follow number 19" instead of just "zone 3 follows zone 2".

**Phase 4 — Feedback Loop (GAP 2 FILLED):**

| Pattern | Detection | Adjustment |
|---------|-----------|------------|
| Sum streak | All recent sums above/below 0.5 std from mean | Boost opposite range x1.15, dampen same x0.87 |
| Zone dominance | Same zone dominant 5 consecutive draws | Dampen that zone's numbers x0.80 |
| H/L imbalance | Avg high numbers per draw > 3.5 | Boost low numbers x1.10 (or vice versa) |

**Final ensemble:**
```
ensemble = 0.40 x P_phase1 + 0.35 x P_phase3 + 0.25 x P_phase4

Run 100,000 MC draws with ensemble as weights
Top-6 most frequent numbers = PREDICTED TICKET
```

**Output chart:** `step7_advanced_prediction.png`

---

## 3. How the Steps Connect

```
Step 1 -> Provides clean df to ALL subsequent steps
  |
  v
Step 2 -> freq[], deviation[], weighted_P[], hot, cold
  |         |
  |         +-> Step 6: signal_frequency, signal_cold
  |         +-> Step 7: Phase 1 weighted P
  |         +-> Final report: hot/cold numbers
  |
  v
Step 3 -> empirical_prob[], CI bounds, co_occurrence{}
  |         |
  |         +-> Step 6: signal_pair_lift
  |
  v
Step 4 -> MC simulation engine
  |         |
  |         +-> Step 6: simulate_next_draw()
  |         +-> Step 7: Phase 2 (100k runs)
  |
  v
Step 5 -> T_zone[4x4], pi[], persistence[]
  |         |
  |         +-> Step 6: signal_markov_zone
  |         +-> Step 7: upgraded to T_number[40x40]
  |
  v
Step 6 -> ensemble(), back_test(), combined_prob[]
  |         |
  |         +-> Step 7: same ensemble concept + more signals
  |
  v
Step 7 -> phase3_number_markov(), phase4_feedback_loop()
            |
            +-> FINAL TICKET [3, 4, 26, 27, 28, 32]
```

---

## 4. The Data Flow Diagram

```
Emirates_Draw_EASY6.csv
         |
         | pd.read_csv() + cleaning
         v
    +------------------------------------------+
    |  DataFrame (df)                           |
    |  Columns: Date, numbers[list],            |
    |           sum, n_high, n_low              |
    +------------------------------------------+
         |
    +----+--------------------------------------------+
    |                                                  |
    v                                                  v
+----------+                                    +----------+
| FREQUENCY|                                    |  MARKOV  |
| ANALYSIS |                                    |  CHAIN   |
|          |                                    |          |
| freq[40] |                                    | T[4x4]   |
| dev%     |                                    | T[40x40] |
| P_i      |                                    | pi[]     |
+----+-----+                                    +----+-----+
     |                                               |
     |    +--------------------+                    |
     +--->| PAIR CO-OCCURRENCE |                    |
          |                    |                    |
          | co_count{}         |                    |
          | pair_lift[]        |                    |
          +--------+-----------+                    |
                   |                                |
                   v                                v
         +-------------------------------------------------+
         |           ENSEMBLE                              |
         |                                                 |
         | combined_P = w1*P_freq + w2*P_cold              |
         |            + w3*P_markov + w4*P_pair            |
         |            + w5*P_feedback                      |
         |                                                 |
         |  FEEDBACK LOOP DETECTOR (Step 7 only):          |
         |    Sum streak? -> adjust P by +/-15%            |
         |    Zone dominance? -> dampen zone by -20%       |
         |    H/L imbalance? -> adjust by +/-10%           |
         +--------------------+----------------------------+
                              |
                              v
             +------------------------------------+
             |  MONTE CARLO (100,000 draws)       |
             |                                    |
             |  Use combined_P as weights         |
             |  -> draw 6 numbers 100k times     |
             |  -> count frequency of each        |
             |  -> top-6 = predicted ticket       |
             +----------------+-------------------+
                              |
                              v
                +-------------------------+
                |  BOOTSTRAP CI           |
                |  1,000 bootstrap samples|
                |  -> stable top numbers |
                +-------------+-----------+
                              |
                              v
                +---------------------------+
                |  FINAL PREDICTED TICKET   |
                |  [3, 4, 26, 27, 28, 32]  |
                |  Most likely single: #4   |
                +---------------------------+
```

---

## 5. Key Mathematical Concepts

### 5.1 Empirical Probability
```
P(number n) = count(n appeared) / count(all numbers drawn)
            = freq[n] / (197 x 6)
```

### 5.2 Deviation Formula (the Framework's Core Formula)
```
Expected = (total draws x 6) / pool_size = (197 x 6) / 40 = 29.55

Deviation_i = (Actual_i - Expected) / Expected x 100%

+20% or more  -> HOT number (drawn significantly more than expected)
-20% or less  -> COLD number (drawn significantly less than expected)
Near 0%       -> NEUTRAL (behaves as expected under uniform draw)
```

### 5.3 Chi-Squared Test
```
H0: All numbers equally likely (uniform draw)
H1: Some numbers are more likely than others (biased draw)

chi2 = Sum (Observed_i - Expected_i)^2 / Expected_i

p-value > 0.05  -> Cannot reject H0 (draw appears uniform/random)
p-value < 0.05  -> Reject H0 (real bias exists)
```

### 5.4 KL-Divergence
```
KL(P || Q) = Sum P(x) * log(P(x) / Q(x))

KL = 0     -> identical to uniform (perfectly random)
KL > 0     -> some deviation from uniform exists
```

### 5.5 Monte Carlo
```
P(event) ~= count(event in simulations) / total_simulations

As simulations -> infinity, this converges to true probability (Law of Large Numbers)
```

### 5.6 Markov Transition Matrix
```
T[i, j] = P(next state = j | current state = i)
         = count(i followed by j) / count(i appeared)

After n steps: T^n = T x T x T ... (n times)
Stationary pi: pi x T = pi,  Sum(pi) = 1
```

### 5.7 Ensemble / Weighted Average of Signals
```
P_final = Sum(w_k x P_k) / Sum(w_k)

where P_k is a different probability signal (frequency, Markov, etc.)
and   w_k is our confidence weight in that signal
```

### 5.8 Bootstrap Confidence Interval
```
1. From 100k MC results, get probability vector mc_prob[]
2. Draw 1,000 bootstrap samples of top-6 using mc_prob as weights
3. For each number: inclusion_prob = (times in top-6) / 1000
4. Numbers with inclusion_prob >= 80% -> "stable" CI set
```

---

## 6. The 4-Phase Framework Explained

| Framework Phase | Our Implementation | Script |
|----------------|-------------------|--------|
| Phase 1 — Weighted P_i from frequency | `phase1_weighted_probability()` with exponential recency decay | Step 7 |
| Phase 2 — Monte Carlo seeded with P_i | `phase2_monte_carlo()` — 100,000 draws using ensemble P | Step 7 |
| Phase 3 — Markov transition from last draw | `phase3_number_markov()` — full 40x40 T matrix | Step 7 |
| Phase 4 — Feedback loop / streak detection | `phase4_feedback_loop()` — sum streaks, zone dominance, H/L bias | Step 7 |

**Why is Phase 2 (Monte Carlo) after Phase 3 & 4?**
Because Phase 2 uses the *combined* output of all phases as its sampling weights:

```
Phase 1 --+
           +--> ensemble_P --> Phase 2 (Monte Carlo) --> TICKET
Phase 3 --+
           |
Phase 4 --+
```

---

## 7. What Each Output Chart Shows

| Chart | What to Look For |
|-------|-----------------|
| `step1_data_exploration.png` | Is the histogram roughly uniform? Is the sum near 123? Time trends? |
| `step2_frequency_analysis.png` | Which numbers are clearly hot (tall red)? Which are cold (short blue)? |
| `step3_probability_distributions.png` | Do 95% CI bars all include the uniform line? How flat is the CDF? |
| `step4_monte_carlo.png` | Does the running mean converge? Compare uniform vs biased matches |
| `step5_markov_chain.png` | Are any cells clearly brighter? Does T^n converge to stationary pi? |
| `step6_prediction_report.png` | Which signal row shows clearest pattern? Back-test vs random baseline |
| `step7_advanced_prediction.png` | 40x40 Markov structure? Clear leaders in CI inclusion probabilities? |

---

## 8. Running the Pipeline

### Run everything at once (recommended)
```bash
python run_all.py
```
Creates `runs/YYYY-MM-DD_HH-MM-SS/` with all 7 charts + final prediction summary.

### Run individual steps (best for learning)
```bash
python 01_data_explorer.py
python 02_frequency_analysis.py
python 03_probability_distributions.py
python 04_monte_carlo_simulation.py
python 05_markov_chain.py
python 06_prediction_report.py
python 07_advanced_prediction.py
```
Each standalone run creates its own timestamped folder.

### Project files overview
| File | Role |
|------|------|
| `Emirates_Draw_EASY6.csv` | Raw input data (197 draws, Oct 2022 - Jul 2026) |
| `01_data_explorer.py` | Step 1: understand the data |
| `02_frequency_analysis.py` | Step 2: frequency, hot/cold, deviation |
| `03_probability_distributions.py` | Step 3: probability theory |
| `04_monte_carlo_simulation.py` | Step 4: Monte Carlo simulation |
| `05_markov_chain.py` | Step 5: Markov chain modelling |
| `06_prediction_report.py` | Step 6: ensemble + back-test |
| `07_advanced_prediction.py` | Step 7: full 4-phase framework |
| `run_all.py` | Master runner: all steps + final result |
| `utils.py` | Shared utilities (timestamped folder management) |
| `requirements.txt` | Python dependencies |
| `runs/` | Output folder — one subfolder per run |

---

## 9. Extending the Project

### Try different weights in the ensemble
In `07_advanced_prediction.py`, change `W1, W3, W4`:
```python
W1, W3, W4 = 0.40, 0.35, 0.25   # current (balanced)
W1, W3, W4 = 0.60, 0.20, 0.20   # trust frequency more
W1, W3, W4 = 0.20, 0.60, 0.20   # trust Markov more
```

### Change the recency window
```python
p1 = phase1_weighted_probability(df, recent_n=10)   # short memory
p1 = phase1_weighted_probability(df, recent_n=50)   # long memory
```

### Change the feedback lookback
```python
LOOKBACK = 3    # detect streaks over last 3 draws (sensitive)
LOOKBACK = 10   # detect streaks over last 10 draws (stable)
```

### Add a new signal
1. Write `def signal_myidea(df) -> np.ndarray:` returning a prob vector of length 40
2. Add it to the ensemble with a weight
3. Rerun and compare back-test scores

### Apply to real-world data where patterns DO exist
The same code applies to any dataset with this structure:
```
Date, N1, N2, N3, N4, N5, N6
```
Examples: stock daily movers, weather patterns, sports results.

---

> **Honest Reminder:** Lottery draws are designed to be statistically independent and uniform.
> No model can reliably predict them better than chance.
>
> The value of this project is in learning the **techniques**:
> Monte Carlo, Markov Chains, Ensemble Methods, Back-testing, Deviation Analysis —
> which are used every day in finance, medicine, AI, and physics.
