# EXPERIMENT STAGE 0 SPECIFICATION: FORK-MDP ORACLE FALSIFICATION

**Project:** Decision-Margin Geometry of World-Model Errors in Model-Based Reinforcement Learning  
**Milestone:** Stage 0 Oracle Falsification & Matched-Intervention Protocol  
**Goal:** Empirically falsify or validate Hypothesis C1 using exact dynamic programming on a parameterized stochastic Fork-MDP.

---

## 1. Environment: Parameterized Stochastic Fork-MDP

```
               ┌── [ s_L1 ] ──> [ s_L2 ] ──> [ Terminal L ]  (Reward R_L, prob p_L)
   (Action a_L)│
[ s_0 (Fork) ] ┤
   (Action a_R)│
               └── [ s_R1 ] ──> [ s_R2 ] ──> [ Terminal R ]  (Reward R_R, prob p_R)
```

### 1.1 State Space $\mathcal{S}$
* $\mathcal{S} = \{s_0, s_{L1}, s_{L2}, s_{R1}, s_{R2}, s_{\text{term}}\}$ ($|\mathcal{S}| = 6$)
* Initial state: $\mu_0(s_0) = 1.0$.

### 1.2 Action Space $\mathcal{A}$
* At $s_0$: $\mathcal{A} = \{a_L, a_R\}$ (or extended to 3+ actions $\{a_L, a_M, a_R\}$ for multi-action controls).
* At intermediate states: $\mathcal{A} = \{a_{\text{fwd}}\}$.

### 1.3 Transition Dynamics $P(s' \mid s, a)$
* At $s_0, a_L$:
  $$P(s_{L1} \mid s_0, a_L) = p_L, \quad P(s_{\text{term}} \mid s_0, a_L) = 1 - p_L$$
* At $s_0, a_R$:
  $$P(s_{R1} \mid s_0, a_R) = p_R, \quad P(s_{\text{term}} \mid s_0, a_R) = 1 - p_R$$
* Along branches (e.g. $s_{L1} \rightarrow s_{L2}$): stochastic or deterministic progression with success probability $p_{\text{branch}}$.
* At terminal states: absorbing state $s_{\text{term}}$.

### 1.4 Reward Structure $R(s, a)$
* Step costs along branches: $R(s, a) = -c$ (e.g. $c = 0.01$).
* Terminal outcomes:
  $$R(s_{L2}, a) = R_L, \quad R(s_{R2}, a) = R_R$$
* Default values: $R_L = 1.0, R_R = 0.6, \gamma = 0.95$.
* True optimal action: $a^*(s_0) = a_L$.
* True action gap:
  $$m_P(s_0, a_R) = Q_P^*(s_0, a_L) - Q_P^*(s_0, a_R) > 0$$

---

## 2. Controlled Error Injection & Matched-Pair Generator

To directly test whether **margin deformation direction** matters independently of prediction error and value sensitivity:

### 2.1 Paired Perturbation Algorithm
For a chosen state-action $(s, a)$:
1. Select perturbation magnitude $\epsilon > 0$.
2. **Compressive Perturbation ($\delta P^+$):**
   Perturb transitions such that $\hat{P}(s_{R1} \mid s_0, a_R)$ increases or $\hat{P}(s_{L1} \mid s_0, a_L)$ decreases, causing $\Delta m(s_0, a_R) < 0 \implies B > 0$ (Boundary Pressure toward decision flip).
3. **Expansive Perturbation ($\delta P^-$):**
   Perturb transitions such that $\hat{P}(s_{L1} \mid s_0, a_L)$ increases or $\hat{P}(s_{R1} \mid s_0, a_R)$ decreases, causing $\Delta m(s_0, a_R) > 0 \implies B < 0$ (Margin Expansion / stabilization).
4. **Matching Constraints:**
   $$\|\delta P^+\|_1 = \|\delta P^-\|_1 \pm 10^{-6}$$
   $$D_{\mathrm{KL}}(P \parallel \hat{P}^+) \approx D_{\mathrm{KL}}(P \parallel \hat{P}^-)$$
   $$|G^+| = |\delta P^{+\top} V_P^*| \approx |\delta P^{-\top} V_P^*| = |G^-|$$

---

## 3. Ground-Truth Computation & Metrics

For every model error component $i = (s_i, a_i)$ within a multi-error model $\hat{P}$:
1. **True Optimal Policy & Values:**
   Compute $V_P^*, Q_P^*, \pi_P^*$ via exact Value Iteration ($|V_{k+1} - V_k|_\infty < 10^{-12}$).
2. **Corrupted Planning:**
   Compute $V_{\hat{P}}^*, Q_{\hat{P}}^*, \pi_{\hat{P}}^*$ on $\hat{P}$.
3. **Counterfactual Single-Component Restoration:**
   Construct $\hat{P}^{, i \leftarrow P}$ by restoring ONLY transition $(s_i, a_i)$ to $P(s' \mid s_i, a_i)$.
   Compute $V_{\hat{P}^{, i \leftarrow P}}^*, \pi^*_{\hat{P}^{, i \leftarrow P}}$.
4. **Exact Environment Evaluation:**
   Compute $J(\pi; P) = \sum_{s} \mu_0(s) V^\pi_P(s)$ via exact Policy Evaluation ($I - \gamma P^\pi)^{-1} R^\pi$.
5. **Exact Counterfactual Value:**
   $$C_i = J(\pi^*_{\hat{P}^{, i \leftarrow P}}; P) - J(\pi^*_{\hat{P}}; P)$$
6. **Diagnostic Metrics:**
   * $B_i = -\frac{m_{\hat{P}}(s_i, a) - m_P(s_i, a)}{m_P(s_i, a) + \varepsilon}$
   * $G_i^\pm = (\hat{P}(s_i, a_i) - P(s_i, a_i))^\top V_P^*$
   * $A_i^\pm = \Delta Q_i(a^*) - \Delta Q_i(a_{\text{competitor}})$

---

## 4. Analytical & Statistical Testing Plan

### 4.1 Incremental Explanatory Value ($\Delta R^2$)
Fit hierarchical Ordinary Least Squares (OLS) models:
* **Model 0 (Standard Controls):**
  $$C_i = \beta_0 + \beta_1 E_i + \beta_2 d_i + \beta_3 m_i^{-1} + \beta_4 |G_i| + \epsilon$$
* **Model 1 (Signed First-Order Baseline):**
  $$C_i = \beta_0 + \dots + \beta_5 G_i^\pm + \beta_6 A_i^\pm + \epsilon$$
* **Model 2 (Full Decision-Margin Model):**
  $$C_i = \beta_0 + \dots + \beta_5 G_i^\pm + \beta_6 A_i^\pm + \beta_7 B_i + \epsilon$$

Compute:
$$\Delta R^2 = R^2(\text{Model 2}) - R^2(\text{Model 1})$$
$$F\text{-test } p\text{-value for } \beta_7 \neq 0$$

### 4.2 Matched-Pair Hypothesis Test
For matched error pairs $(e_i^+, e_i^-)$ with $B^+ > 0$ and $B^- < 0$:
* Two-sided paired t-test: $H_0: \mathbb{E}[C_i^+ - C_i^-] = 0$.
* Wilcoxon signed-rank test (non-parametric).
* Cohen's $d$ paired effect size:
  $$d = \frac{\bar{D}}{\sigma_D}, \quad D_k = C_k^+ - C_k^-$$

### 4.3 Budgeted Model Correction Protocol ($\operatorname{Recovery@K}$)
For an MDP with $N = 10$ simultaneous corrupted components:
* Rank components by ranking functions $f_j(i) \in \{E_i, d_i \cdot E_i, |G_i|, G_i^\pm, B_i, C_i\}$.
* For budget $K \in \{1, 2, \dots, N\}$, select top-$K$ components $\mathcal{S}_K$, restore them in $\hat{P}$, re-plan, and compute:
  $$\operatorname{Recovery@K} = \frac{J(\pi^*_{\hat{P}_{\text{corrected}(\mathcal{S}_K)}}; P) - J(\pi^*_{\hat{P}}; P)}{J(\pi^*_P; P) - J(\pi^*_{\hat{P}}; P)}$$
