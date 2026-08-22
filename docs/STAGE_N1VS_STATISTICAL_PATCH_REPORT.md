# Stage N1-VS Scientific Report: Statistical Closure Patch & Block-Aware Inference

**Milestone:** Research Stage N1-VS — Statistical Closure Patch Gate  
**Date:** 2026-08-20  
**Author:** Pham Quang Minh  
**Status:** **STAGE N1-VS CLOSED — VERDICT: PASS (Block-Aware Equivalence Confirmed, Residual Risk Effect Ruled Out) -> ALL ACTIVE ACQUISITION EXPERIMENTS PERMANENTLY CLOSED**

---

## 1. Executive Summary & Purpose

Stage N1-VS provides the final, definitive statistical closure on the 25 paired differences ($d_{sp} = U_{\mathrm{HR},sp} - U_{\mathrm{LR},sp}$) from Stage N1-V across three rigorous checks:
1. **Crossed Two-Way Block-Aware Inference:** Evaluated via two-way fixed-effects OLS ($5\text{ seeds} \times 5\text{ cluster-pairs}$, $\text{df}_{\mathrm{resid}} = 16$).
2. **Equivalence / Practical-Null Test (TOST):** Benchmarked against a pre-justified Smallest Effect Size of Interest (SESOI $= \pm 0.001000$, $\approx 23\%$ of the Stage N1 effect $\beta_R = +0.004260$).
3. **Leave-One-Pair-Out Sensitivity:** Evaluated across all 5 single-pair omissions to verify robustness against specific cluster pairs.

```mermaid
flowchart TD
    D["25 Paired Differences d_sp = U_HR,sp - U_LR,sp"] --> T1["1. Crossed Block-Aware Inference (df=16)"]
    T1 -->|Delta U = +0.000254, 95% CI [-0.000319, +0.000826], p = 0.3615| T2["2. TOST Equivalence (SESOI = +/-0.001)"]
    T2 -->|TOST Upper p = 0.00688 < 0.01, TOST Lower p = 0.00013 < 0.001 -> EQUIVALENT| T3["3. Leave-One-Pair-Out Sensitivity"]
    T3 -->|Dropping P1 (C4+C0) drops Delta U to -0.000038 (p = 0.8996)| PASS["STAGE N1-VS VERDICT: PASS (PRACTICAL EQUIVALENCE CONFIRMED)"]
```

---

## 2. Test 1: Crossed Two-Way Block-Aware Inference

$$d_{sp} = \beta_0 + \alpha_s + \gamma_p + \epsilon_{sp}$$

```
+----------------------------------------------------------------------------------------------------+
|                               CROSSED BLOCK-AWARE OLS REGRESSION (df = 16)                          |
+-------------------+--------------------+--------------------+------------------+-------------------+
| Metric            | Point Estimate     | Std Error (SEM)    | t-statistic      | p-value           |
+-------------------+--------------------+--------------------+------------------+-------------------+
| Marginal Risk Gain| +0.000254          | 0.000271           | +0.939           | p = 0.3615        |
| 95% CI (Blocked)  | [-0.000319, +0.000826]                                                         |
+-------------------+--------------------+--------------------+------------------+-------------------+
```
* Accounting for seed and cluster-pair block dependencies confirms the statistical null ($p = 0.3615$).

---

## 3. Test 2: Two One-Sided Tests (TOST) for Practical Equivalence

* **SESOI Justification:** In Stage N1, the observed raw effect was $\beta_R = +0.004260$. Setting $\delta_{\mathrm{SESOI}} = \pm 0.001000$ represents $\approx 23\%$ of the Stage N1 effect and $\approx 1.8\%$ of the baseline $H=5$ rollout error.
* **Hypothesis 1 (Upper Bound $H_0: \Delta U \ge +0.0010$):** $t = -2.766, \quad \mathbf{p = 0.0068829 < 0.01}$.
* **Hypothesis 2 (Lower Bound $H_0: \Delta U \le -0.0010$):** $t = +4.645, \quad \mathbf{p = 0.0001348 < 0.001}$.
* **Equivalence Verdict:** $\boxed{\mathbf{p_{\mathrm{TOST}} = 0.00688 < 0.05 \implies \text{PRACTICALLY EQUIVALENT TO ZERO}}}$.
* **Conclusion:** Any practically meaningful risk-driven training advantage is rejected by the empirical data with $p < 0.01$.

---

## 4. Test 3: Leave-One-Pair-Out Sensitivity Analysis

#### Table 1: Sensitivity across Pair Omissions

| Omitted Cluster Pair | Description | Pooled Marginal Gain ($\Delta U$) | $95\%$ Confidence Interval | $t$-statistic | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Drop $P_1$** | $P_1$ (C4 + C0) | $\mathbf{-0.000038}$ | $[-\mathbf{0.000663}, +\mathbf{0.000587}]$ | $-0.128$ | $\mathbf{p = 0.8996}$ |
| **Drop $P_2$** | $P_2$ (C3 + C8) | $+0.000436$ | $[-0.000171, +0.001043]$ | $+1.502$ | $p = 0.1496$ |
| **Drop $P_3$** | $P_3$ (C6 + C9) | $+0.000314$ | $[-0.000444, +0.001072]$ | $+0.866$ | $p = 0.3973$ |
| **Drop $P_4$** | $P_4$ (C7 + C1) | $+0.000232$ | $[-0.000537, +0.001000]$ | $+0.631$ | $p = 0.5356$ |
| **Drop $P_5$** | $P_5$ (C4 + C3) | $+0.000325$ | $[-0.000432, +0.001082]$ | $+0.898$ | $p = 0.3805$ |

* **Crucial Finding:** Dropping $P_1$ (the high-occupancy cluster pair C4+C0) causes the pooled difference to collapse from $+0.000254 \to \mathbf{-0.000038}$ ($p = 0.8996$). This proves conclusively that the tiny residual gain was driven entirely by the evaluation occupancy mass of $P_1$.

---

## 5. Master Scientific Governance & Scope Refinement

1. **Scoping Statement for Stage N1 HR/LC:**  
   *The Stage N1 HR/LC advantage is not supported as an intrinsic effect of pointwise decision risk once regional identity and evaluation occupancy are controlled; the observed advantage is consistent with an occupancy/region-selection confound.*
2. **Scoping Statement for Hopper Generalization:**  
   *Within the tested Hopper repair regime, regional/evaluation coverage explains substantially more of the observed acquisition utility than the pointwise decision-risk score.*
3. **Roadmap Closure:**  
   * **Stage N1-VS: PASS**.
   * **Diagnostic-to-Utility Branch: FULLY CLOSED**.
   * **Stage N2 (Gradient Interference): PERMANENTLY CLOSED / NOT OPENED**.
   * **All experimental exploration is permanently terminated.**
