# Scientific Report: Stage Q1-FB Full-Budget Hopper Representativeness Confirmation

**Date:** 2026-08-21  
**Target Milestone:** Q1 Journal Upgrade — Stage Q1-FB (Full-Budget Hopper Confirmation)  
**Core Scientific Question:** Does the conclusion that VaGraM lacks a robust advantage over MLE hold when MBPO is trained in a canonical locomotion regime up to 125,000 environment steps?  
**Accounting Controls:** Matched 7-model ensemble (elite 5, 4x200 ReLU), 20 SAC updates/step, retrain every 250 steps, 10 deterministic eval episodes per checkpoint on identical bitwise reset seeds.  

## 1. Primary Confirmatory Paired Analysis (Terminal Return, N=6)

|   Seed |   MLE Terminal Return |   VaGraM Terminal Return |   Paired Δ (VaGraM - MLE) | Win?   |
|-------:|----------------------:|-------------------------:|--------------------------:|:-------|
|    142 |                  6.53 |                   335.04 |                    328.51 | VaGraM |
|    143 |                 10.69 |                   777.76 |                    767.07 | VaGraM |
|    144 |                 10.26 |                   566.19 |                    555.94 | VaGraM |
|    145 |                  8.77 |                  1076.3  |                   1067.54 | VaGraM |
|    146 |                238.59 |                   513.06 |                    274.47 | VaGraM |
|    147 |                 12.48 |                   456.65 |                    444.17 | VaGraM |

- **Mean Paired $\Delta$:** $+572.95$ (95% BCI: $[+376.69, +809.02]$)
- **Median Paired $\Delta$:** $+500.06$
- **Paired Permutation $p$-value:** $0.0312$
- **Win Rate:** 6 / 6 (100.0%)

## 2. Secondary Mechanistic Horizon-Interaction Analysis

|   Seed |   Δ(10k) |   Δ(Terminal) |   H_i = Δ(Terminal) - Δ(10k) | Horizon Growth?   |
|-------:|---------:|--------------:|-----------------------------:|:------------------|
|    142 |  1008.93 |        296.18 |                      -712.75 | NO                |
|    143 |    10.84 |       1024.21 |                      1013.37 | YES               |
|    144 |   218.58 |       1025.26 |                       806.69 | YES               |
|    145 |   168.42 |       1010.44 |                       842.02 | YES               |
|    146 |   111.84 |        902.67 |                       790.83 | YES               |
|    147 |    32.64 |        984.97 |                       952.33 | YES               |

- **Mean Horizon Interaction $\bar{H}$:** $+615.42$ (95% BCI: $[+65.96, +935.91]$)
- **Median Horizon Interaction:** $+824.36$
- **Permutation $p$-value:** $0.0625$
- **Horizon Positive Fraction:** 5 / 6 (83.3%)

## 3. Individual Per-Seed Breakdown

| Cohort       | Algorithm   |   Seed | Budget   |   Terminal Return (Final 3) | Final Return (Step 125k)   |   Dyn Updates |   SAC Updates |   Imagined Steps |   Wall Clock (s) |   Peak RAM (MB) |
|:-------------|:------------|-------:|:---------|----------------------------:|:---------------------------|--------------:|--------------:|-----------------:|-----------------:|----------------:|
| Confirmatory | MLE         |    142 | 125k     |                        6.53 | 5.98 ± 0.24                |        151035 |       2480000 |           388800 |          23863.5 |          2390.7 |
| Confirmatory | MLE         |    143 | 125k     |                       10.69 | 7.22 ± 0.06                |        151035 |       2480000 |           388800 |          23967.7 |          2394.8 |
| Confirmatory | MLE         |    144 | 125k     |                       10.26 | 8.93 ± 1.80                |        151035 |       2480000 |           388800 |          23658   |          2369.4 |
| Confirmatory | MLE         |    145 | 125k     |                        8.77 | 8.80 ± 0.46                |        151035 |       2480000 |           388800 |          23678.9 |          2367.1 |
| Confirmatory | MLE         |    146 | 125k     |                      238.59 | 136.70 ± 1.84              |        151035 |       2480000 |           388800 |          23401.5 |          2388.2 |
| Confirmatory | MLE         |    147 | 125k     |                       12.48 | 10.99 ± 0.16               |        151035 |       2480000 |           388800 |          23463.9 |          2385.6 |
| Confirmatory | VAGRAM      |    142 | 125k     |                      335.04 | 302.16 ± 1.64              |        151035 |       2480000 |           388800 |          25397.9 |          2529   |
| Confirmatory | VAGRAM      |    143 | 125k     |                      777.76 | 1031.43 ± 0.62             |        151035 |       2480000 |           388800 |          25488.9 |          2532   |
| Confirmatory | VAGRAM      |    144 | 125k     |                      566.19 | 1034.20 ± 0.31             |        151035 |       2480000 |           388800 |          25233.8 |          2501.6 |
| Confirmatory | VAGRAM      |    145 | 125k     |                     1076.3  | 1019.25 ± 0.73             |        151035 |       2480000 |           388800 |          25172.4 |          2495.3 |
| Confirmatory | VAGRAM      |    146 | 125k     |                      513.06 | 1039.37 ± 0.57             |        151035 |       2480000 |           388800 |          25144.7 |          2525.2 |
| Confirmatory | VAGRAM      |    147 | 125k     |                      456.65 | 995.97 ± 3.55              |        151035 |       2480000 |           388800 |          25055.3 |          2525.2 |

## 4. Epistemic Demarcation & Interpretation Rules

1. **Primary Estimand Priority:** Primary conclusion is determined strictly by the terminal return paired difference $\Delta_i^{\text{terminal}}$ on the confirmatory cohort ($[142, \dots, 151]$). Horizon interaction $H_i$ serves as a secondary mechanistic diagnostic.
2. **Separation of Pilot and Confirmatory Evidence:** Pilot Seed 101 is documented separately as exploratory/feasibility evidence and is not pooled into confirmatory inference.
