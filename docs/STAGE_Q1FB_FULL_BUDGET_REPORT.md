# Scientific Report: Stage Q1-FB Full-Budget Hopper Representativeness Confirmation

**Date:** 2026-08-21  
**Target Milestone:** Q1 Journal Upgrade — Stage Q1-FB (Full-Budget Hopper Confirmation)  
**Core Scientific Question:** Does the conclusion that VaGraM lacks a robust advantage over MLE hold when MBPO is trained in a canonical locomotion regime up to 125,000 environment steps?  
**Accounting Controls:** Matched 7-model ensemble (elite 5, 4x200 ReLU), 20 SAC updates/step, retrain every 250 steps, 10 deterministic eval episodes per checkpoint on identical bitwise reset seeds.  

## 1. Primary Confirmatory Paired Analysis (Terminal Return, N=1)

|   Seed |   MLE Terminal Return |   VaGraM Terminal Return |   Paired Δ (VaGraM - MLE) | Win?   |
|-------:|----------------------:|-------------------------:|--------------------------:|:-------|
|    142 |                  7.07 |                   873.25 |                    866.18 | VaGraM |

- **Mean Paired $\Delta$:** $+866.18$ (95% BCI: $[+866.18, +866.18]$)
- **Median Paired $\Delta$:** $+866.18$
- **Paired Permutation $p$-value:** $1.0000$
- **Win Rate:** 1 / 1 (100.0%)

## 2. Secondary Mechanistic Horizon-Interaction Analysis

|   Seed |   Δ(10k) |   Δ(Terminal) |   H_i = Δ(Terminal) - Δ(10k) | Horizon Growth?   |
|-------:|---------:|--------------:|-----------------------------:|:------------------|
|    142 |   973.19 |       1023.93 |                        50.74 | YES               |

- **Mean Horizon Interaction $\bar{H}$:** $+50.74$ (95% BCI: $[+50.74, +50.74]$)
- **Median Horizon Interaction:** $+50.74$
- **Permutation $p$-value:** $1.0000$
- **Horizon Positive Fraction:** 1 / 1 (100.0%)

## 3. Individual Per-Seed Breakdown

| Cohort       | Algorithm   |   Seed | Budget   |   Terminal Return (Final 3) | Final Return (Step 125k)   |   Dyn Updates |   SAC Updates |   Imagined Steps |   Wall Clock (s) |   Peak RAM (MB) |
|:-------------|:------------|-------:|:---------|----------------------------:|:---------------------------|--------------:|--------------:|-----------------:|-----------------:|----------------:|
| Pilot        | MLE         |    101 | 25k      |                       24.13 | 4.18 ± 0.31                |          5795 |        480000 |            38400 |           3886.2 |           109.1 |
| Confirmatory | MLE         |    142 | 125k     |                        7.07 | 6.30 ± 0.14                |        151035 |       2480000 |           388800 |          14714.2 |           755.1 |
| Pilot        | VAGRAM      |    101 | 25k      |                      666.93 | 1033.83 ± 0.62             |          5795 |        480000 |            38400 |           9266.4 |           146.7 |
| Confirmatory | VAGRAM      |    142 | 125k     |                      873.25 | 1030.23 ± 0.60             |        151035 |       2480000 |           388800 |          17212.7 |           759.9 |

## 4. Epistemic Demarcation & Interpretation Rules

1. **Primary Estimand Priority:** Primary conclusion is determined strictly by the terminal return paired difference $\Delta_i^{\text{terminal}}$ on the confirmatory cohort ($[142, \dots, 151]$). Horizon interaction $H_i$ serves as a secondary mechanistic diagnostic.
2. **Separation of Pilot and Confirmatory Evidence:** Pilot Seed 101 is documented separately as exploratory/feasibility evidence and is not pooled into confirmatory inference.
