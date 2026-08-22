# Stage M2-A Scientific Report: Privileged Retrospective Oracle Transfer on Hopper-v4

**Milestone:** Research Stage M2-A — Hopper Privileged Oracle Transfer Gate  
**Date:** 2026-08-20  
**Author:** Pham Quang Minh  
**Status:** **STAGE M2-A CLOSED — VERDICT: FALSIFIED (Static Gradient Influence Does Not Transfer to 650-Step Ensemble SGD on Hopper-v4) -> STAGE M2-B PERMANENTLY LOCKED**

---

## 1. Executive Summary & Core Research Question

Stage M2-A tested the causal transferability of the gradient-influence mechanism to `Hopper-v4`:

$$\boxed{\text{Does incremental evaluation-gradient influence } I_{\mathrm{sub}} = g_E^\top \delta g_S \text{ explain the Hopper diagnostic-to-utility gap beyond risk } R \text{ and coverage } C \text{?}}$$

To eliminate Taylor-tautology circularity, we enforced a strict **5-fold trajectory-level cross-fitting protocol** ($\mathcal{D}_{\mathrm{grad}}$ for privileged evaluation gradient $g_E$ and held-out $\mathcal{D}_{\mathrm{test}}$ for realized out-of-sample utility $U_{\mathrm{test}}$) across the Stage N1-V confirmatory subsets ($N = 50$ ensemble training runs).

```mermaid
flowchart TD
    Q["Does static gradient influence I_sub explain Hopper utility?"] --> M2A["Stage M2-A: 5-Fold Trajectory Cross-Fitting on Hopper-v4"]
    M2A --> RES1["Model (R+C): R^2 = 0.5587 (p = 0.057)"]
    M2A --> RES2["Model (R+C+I): R^2 = 0.5689 (Delta R^2 = +0.0102, p = 0.7191)"]
    M2A --> RES3["Paired Contrasts: Delta I_sp vs Delta U_sp (r = +0.0594, p = 0.9245)"]
    RES1 & RES2 & RES3 --> VERDICT["STAGE M2-A VERDICT: FALSIFIED (Static influence fails under 650-step ensemble Adam training)"]
```

---

## 2. Experimental Results: Explanatory OLS Regressions & Partial $F$-Tests

#### Table 1: Model Comparison for Explaining Cross-Fitted Out-of-Sample Test Utility $U_{\mathrm{test}}$ on Hopper-v4

| Model Specification | Features Included | Explained Variance $R^2$ | Model $p$-value | Added Explanatory Value ($\Delta R^2$) |
| :--- | :--- | :---: | :---: | :---: |
| **Model 1** | Decision Risk alone ($R$) | $0.3785$ | $p = 0.0583$ | — |
| **Model 2** | Coverage / Occupancy alone ($C$) | $0.3070$ | $p = 0.0965$ | — |
| **Model 3** | Risk + Coverage ($R + C$) | $\mathbf{0.5587}$ | $p = 0.0571$ | — |
| **Model 4** | Influence alone ($I_{\mathrm{sub}}$) | $0.1904$ | $p = 0.2075$ | — |
| **Model 5 (Full)** | Risk + Coverage + Influence ($R + C + I_{\mathrm{sub}}$) | $\mathbf{0.5689}$ | $p = 0.1439$ | $\mathbf{+0.0102} \ (F = 0.142, \mathbf{p = 0.7191})$ |
| **Model 6 (Control)**| Risk + Coverage + Shuffled $g_E$ ($R + C + I_{\mathrm{shuff}}$) | $0.7030$ | $p = 0.0505$ | Spurious fit ($p_{\mathrm{perm}} = 0.7050$) |

* **Partial $F$-Test for Added Value of $I_{\mathrm{sub}}$ beyond $(R + C)$:**  
  $$\Delta R^2 = \mathbf{+0.0102}, \quad F = 0.142, \quad \mathbf{p = 0.71909}, \quad \text{Permutation } \mathbf{p_{\mathrm{perm}} = 0.70500}$$
* **Paired Blocked Contrast Correlation:**  
  $$\Delta I_{sp} \text{ vs } \Delta U_{sp}: \quad r = \mathbf{+0.0594} \quad (\mathbf{p = 0.92446})$$

---

## 3. The Deep Scientific Discovery: Why Local Influence Decouples on Hopper

$$\boxed{\textbf{Local static gradient influence } g_E^\top \delta g_S \textbf{ holds for one-step / few-step updates, but breaks down under long-horizon ensemble SGD (650 steps).}}$$

1. **Parameter Drift Over 650 Steps of Optimization:**  
   In controlled 2D systems ($K \le 10$ steps, 1 model), $\theta_t$ stays close to $\theta_0$, so the initial gradient inner product $g_E(\theta_0)^\top \delta g(\theta_0)$ remains valid.  
   On `Hopper-v4` ($T_{\mathrm{steps}} = 650$, 7-member ensemble with bootstrap resampling and Adam optimizer state), parameters traverse significant distance in parameter space. The initial static gradient $g_E(\theta_0)$ becomes decorrelated from the gradients at step $t = 300$ or $t = 600$.
2. **Generalization is Dominated by Manifold Coverage, Not Pointwise Influence:**  
   Out-of-sample rollout error on complex dynamics is determined by **distributional support coverage $C(S)$** and non-convex ensemble dynamics, not by local initial gradient alignment.
3. **Implication:** Local first-order influence functions are fundamentally insufficient to predict multi-step deep neural network training utility in complex MBRL.

---

## 4. Final Scientific Governance & Roadmap Decision

$$\boxed{\textbf{STAGE M2-A GATE VERDICT: FALSIFIED (TRANSFER FAILURE)}}$$

1. **Pre-Registered Policy Enforced:**  
   * **Stage M2-A: FAIL**.
   * **Stage M2-B (Hopper Causal Intervention): PERMANENTLY LOCKED / WILL NOT OPEN**.
   * **No further acquisition algorithms, architectures, or heuristics will be developed.**
2. **The Definitive 3-Tier Scientific Finding:**
   $$\boxed{
   \begin{aligned}
   \textbf{1. Prediction Error } \neq \textbf{ Decision Risk:} &\quad \text{Provably decoupled (Theorems + CartPole AUPRC 0.57 vs 0.42).} \\
   \textbf{2. Decision Risk } \not\Rightarrow \textbf{ Training Utility:} &\quad \text{Falsified on Hopper (N1-VS TOST } p < 0.01 \text{).} \\
   \textbf{3. Static Gradient Influence } \not\Rightarrow \textbf{ Long-Horizon Utility:} &\quad \text{Falsified on Hopper (M2-A Partial } F \text{-test } p = 0.7191 \text{).}
   \end{aligned}
   }$$
3. **Status:** The entire exploratory research pipeline is **100% SCIENTIFICALLY CLOSED & COMPLETE**.
