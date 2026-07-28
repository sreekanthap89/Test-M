# Emirates Draw EASY6 — System Architecture & Prediction Guide

## Architecture Overview

The EASY6 prediction suite is an institutional-grade quantitative forecasting system designed for **Emirates Draw EASY6** (6 winning numbers drawn from a pool of 39).

```
 ┌─────────────────────────────────────────────────────────────┐
 │                      HISTORICAL DATA                        │
 │                 (Emirates_Draw_EASY6.csv)                   │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                  CENTRALIZED DATA PIPELINE                  │
 │                       (utils.load_data)                     │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │                   13-STEP QUANT PIPELINE                    │
 │ 01. Data Explorer         08. Deep Learning MLP             │
 │ 02. Frequency Analysis    09. Ultra Stacking ML Ensemble    │
 │ 03. Probability Curves    10. Quantum Science Engine        │
 │ 04. Monte Carlo Sim       11. BlackRock Institutional Quant │
 │ 05. Zone Markov           12. Master AI Meta-Ensemble     │
 │ 06. Multi-Signal Ensemble 13. Final Tabular Report Chart    │
 │ 07. 39x39 Markov Engine                                     │
 └──────────────────────────────┬──────────────────────────────┘
                                │
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │              COMBINATORIAL COVERING WHEEL SYSTEM             │
 │                (3-if-3 Match Guarantee Tickets)             │
 └─────────────────────────────────────────────────────────────┘
```

---

## Technical Specifications

| Component | EASY6 Specification |
| :--- | :--- |
| **Pool Size (`POOL`)** | 39 balls (1 to 39) |
| **Draw Size (`DRAW_SIZE`)** | 6 balls drawn per draw |
| **Theoretical Mean Sum** | $6 \times 20 = 120.0$ |
| **High/Low Threshold** | Low $\le 19$, High $> 19$ |
| **Zones** | Z1 (1–10), Z2 (11–20), Z3 (21–30), Z4 (31–39) |
| **Random Uniform Baseline** | $6 \times (6 / 39) \approx 0.92308$ matches per draw |
| **Wheeling Guarantee** | 3-if-3 Greedy Set Cover Wheel |
