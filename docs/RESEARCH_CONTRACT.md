# RESEARCH CONTRACT

## Project: Decision-Margin Geometry of World-Model Errors in Model-Based Reinforcement Learning

**Document Version:** 1.0.0  
**Project Status:** `B2 = SELECTED, NOVELTY YELLOW`  
**Target Milestone:** Bachelor Thesis (Core Mechanism) $\rightarrow$ Q1 Journal / Top-tier AI/RL Conference (Full Extension)

---

### 1. Executive Research Question

> **Among localized world-model errors that are matched in predictive magnitude, model uncertainty, visitation, true action gap, and conventional value sensitivity, does the direction in which an error deforms competing-action margins predict the counterfactual value of correcting that error?**

The proposed causal mechanism is:
$$\boxed{\text{Localized Model Error } \delta P_i \longrightarrow \text{Signed Action-Margin Deformation } \Delta m_i \longrightarrow \text{Counterfactual Correction Value } C_i}$$

---

### 2. Formal Mathematical Definitions

Let the true Markov Decision Process (MDP) be:
$$M = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \mu_0)$$
where:
* $\mathcal{S}$ is the finite state space,
* $\mathcal{A}$ is the finite discrete action space,
* $P(s' \mid s, a)$ is the true transition dynamics,
* $R(s, a)$ is the reward function,
* $\gamma \in [0, 1)$ is the discount factor,
* $\mu_0(s)$ is the initial state distribution.

Let the imperfect world model be:
$$\hat{M} = (\mathcal{S}, \mathcal{A}, \hat{P}, \hat{R}, \gamma, \mu_0)$$
Throughout the foundational stages (Stages 0–3), we isolate dynamics misspecification by setting $\hat{R} = R$.

#### 2.1 Optimal Action-Value and Action Margin
* **True optimal action-value function:**
  $$Q_P^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V_P^*(s')$$
  $$V_P^*(s) = \max_{a \in \mathcal{A}} Q_P^*(s, a)$$
* **True optimal action:**
  $$a^*(s) = \arg\max_{a \in \mathcal{A}} Q_P^*(s, a)$$
* **True action margin against competitor $a \neq a^*(s)$:**
  $$m_P(s, a) = Q_P^*(s, a^*(s)) - Q_P^*(s, a) > 0$$
* **Model-induced action margin under planning with $\hat{P}$:**
  $$m_{\hat{P}}(s, a) = Q_{\hat{P}}^*(s, a^*(s)) - Q_{\hat{P}}^*(s, a)$$
* **Signed Action-Margin Deformation:**
  $$\Delta m(s, a) = m_{\hat{P}}(s, a) - m_P(s, a)$$
* **Normalized Boundary Pressure:**
  $$B(s, a) = -\frac{\Delta m(s, a)}{m_P(s, a) + \varepsilon}$$
  * $B < 0$: Error expands/stabilizes the true action ranking.
  * $0 < B < 1$: Error compresses the action margin toward the decision boundary.
  * $B \ge 1$: Error crosses the decision boundary, reversing the planned action ranking.

#### 2.2 Counterfactual Correction Value ($C_i$)
For a corrupted model $\hat{P}$ with $N$ localized errors, let component $i = (s_i, a_i)$ be restored to ground truth:
$$\hat{P}^{, i \leftarrow P}(s' \mid s, a) = \begin{cases} P(s' \mid s, a) & \text{if } (s, a) = (s_i, a_i) \\ \hat{P}(s' \mid s, a) & \text{otherwise} \end{cases}$$
The **Counterfactual Correction Value** of error $i$ evaluated strictly in the true environment $P$ is:
$$\boxed{C_i = J(\pi^*_{\hat{P}^{, i \leftarrow P}}; P) - J(\pi^*_{\hat{P}}; P)}$$
where $J(\pi; P) = \mathbb{E}_{s_0 \sim \mu_0}\left[ V^\pi_P(s_0) \right]$ is the true expected return.

---

### 3. Hypotheses & Contribution Decomposition

```
[ C1: Scientific Finding ] (Primary Thesis Core)
       │
       ├── Must pass Gate G1 (Incremental Explanatory Value)
       ▼
[ C2: Diagnostic Metric ] (Boundary Pressure B_i)
       │
       ├── Must pass Gate G2 (Budgeted Allocation Superiority)
       ▼
[ C3: Matched-Intervention Protocol ] (Causal Falsification Benchmark)
       │
       ├── Must pass Gate G3 (Replication in Multi-action GridWorld)
       ▼
[ C4: Decision-Margin-Aware Budgeted Model Correction ] (Paper Method)
```

#### Contribution 1 (C1 - Scientific Finding)
**Hypothesis:** Among localized world-model errors matched for predictive loss ($E_i$), uncertainty ($U_i$), occupancy ($d_i$), true action gap ($m_i$), and signed first-order value sensitivity ($G_i^\pm$), the signed action-margin deformation $B_i$ contains significant non-zero incremental mutual information regarding true correction value $C_i$:
$$I(B_i; C_i \mid E_i, U_i, d_i, m_i, G_i^\pm, A_i^\pm) > 0$$

#### Contribution 2 (C2 - Diagnostic Quantity)
**Hypothesis:** The continuous normalized boundary pressure $B_i$ predicts error ranking and severe decision degradation significantly better than binary argmax disagreement indicators $\mathbf{1}[\arg\max Q_{\hat{P}} \neq \arg\max Q_P]$ and unsigned value-gradient norms $\|\nabla V^\top \delta P\|$.

#### Contribution 3 (C3 - Matched Causal Intervention Protocol)
**Deliverable:** A transparent, reproducible benchmark protocol in exact MDPs that generates perfectly matched counterfactual error pairs $(\delta P_i^+, \delta P_i^-)$ with identical predictive severity but opposite margin deformation directions, providing ground-truth causal validation.

#### Contribution 4 (C4 - Budgeted Model Correction Algorithm)
**Hypothesis:** Under a strict correction budget $K \ll N$, allocating corrections via the decision-margin diagnostic yields higher policy return recovery than error-only, uncertainty-only, or unsigned value-aware baselines:
$$\operatorname{Recovery@K}(B) > \operatorname{Recovery@K}(\text{Baselines})$$
where $\operatorname{Recovery@K} = \frac{J_K - J_{\text{corrupt}}}{J_{\text{true}} - J_{\text{corrupt}}}$.

---

### 4. Controlled Variables & Baselines

| Signal | Symbol | Formulation | Role |
| :--- | :--- | :--- | :--- |
| **Prediction Error (TV/L1)** | $E_i^{\text{L1}}$ | $\frac{1}{2} \sum_{s'} \|\hat{P}(s' \mid s, a) - P(s' \mid s, a)\|_1$ | Baseline / Control |
| **Prediction Error (KL)** | $E_i^{\text{KL}}$ | $D_{\mathrm{KL}}(P(\cdot \mid s, a) \parallel \hat{P}(\cdot \mid s, a))$ | Baseline / Control |
| **State-Action Occupancy** | $d_i$ | $d^{\pi^*_P}(s, a) = (1-\gamma) \sum_{t=0}^\infty \gamma^t \mathbb{P}(s_t=s, a_t=a \mid \pi^*_P)$ | Baseline / Control |
| **True Action Gap** | $m_i$ | $\min_{a \neq a^*(s)} [Q_P^*(s, a^*(s)) - Q_P^*(s, a)]$ | Baseline / Control |
| **Unsigned Value Sensitivity** | $\|G_i\|$ | $\|\delta P_i^\top V_P^*\|_2$ (VaGraM-style) | Baseline / Control |
| **Signed Value Sensitivity** | $G_i^\pm$ | $\delta P_i^\top V_P^*$ | **Critical Negative Control** |
| **Signed Advantage Perturbation** | $A_i^\pm$ | $\Delta Q_i(a^*(s)) - \Delta Q_i(a_{\text{competitor}})$ | **Critical Negative Control** |
| **Proposed Boundary Pressure** | $B_i$ | $-\frac{\Delta m(s, a)}{m_P(s, a) + \varepsilon}$ | **Proposed Signal** |
| **Theoretical Boundary Pressure** | $B_i^{\text{exact}}$ | $-\frac{\Delta m(s_i, a_i)}{m_P(s_i)}$ | Theoretical Oracle Margin Metric |
| **Numerical Boundary Pressure** | $B_i^{(\varepsilon)}$ | $-\frac{\Delta m(s_i, a_i)}{m_P(s_i) + \varepsilon}$ | Numerically Stabilized Metric |
| **Boundary Crossing Indicator** | $Z_{\text{cross}}$ | $\mathbf{1}[m_{\hat{P}}(s_i, \cdot) < 0]$ | Strict Action Reversal ($m_{\hat{P}} < 0$) vs Tie ($m_{\hat{P}}=0$) vs No-Cross ($m_{\hat{P}}>0$) |
| **Ground Truth Target** | $C_i$ | $J(\pi^*_{\hat{P}^{, i \leftarrow P}}; P) - J(\pi^*_{\hat{P}}; P)$ | Target Ground Truth Counterfactual Value |

---

### 5. Explicit Novelty Boundaries (What We Do NOT Claim)

1. $\times$ We do **NOT** claim that "lower world-model prediction error does not mean better control" (established by VAML/PAML/Value Equivalence).
2. $\times$ We do **NOT** claim that "errors near small action gaps matter more" (established by action gap regularity literature).
3. $\times$ We do **NOT** claim that "model misspecification can reverse policy rankings" (established by ordinal ranking / model exploitation work).
4. $\times$ We do **NOT** claim to be the "first to use action margins with world models" (policy-margin gating and argmax agreement signals exist).
5. $\checkmark$ **Narrow Defensible Scientific Claim:** We establish that the **signed direction of action-margin deformation retains measurable incremental information about counterfactual correction value ($C_i$)** even when predictive severity ($E_{L1}, E_{\mathrm{KL}}$), occupancy ($d$), true action gap ($m$), and conventional first-order value sensitivity ($|G|, G^\pm, A^\pm$) are matched.

---

### 6. Kill Criteria & Hard Falsification Rules

* **Kill Rule 1 (Mechanism Collapse):** If in controlled regression on matched error pairs, the partial $R^2$ increase of $d_i B_i$ satisfies $\Delta R^2(d_i B_i \mid d E, d |G|, d G^\pm, d A^\pm) < 0.01$ with $p > 0.05$, the decision-margin hypothesis is falsified. *(PASSED in Stage 0B: $\Delta R^2 = +0.1836, p < 10^{-14}$)*.
* **Kill Rule 2 (First-Order Redundancy):** If $d_i B_i$ is algebraically or empirically collinear with occupancy value sensitivity ($|r| > 0.98$), $B_i$ is rejected as a trivial rewrite of policy-loss gradient expansions. *(PASSED in Stage 0B)*.
* **Kill Rule 3 (Stage 3 Multi-Action Headroom Quality Gate & Budget Kill Rule):**
  - **Headroom Pre-condition:** The Stage 3 environment MUST satisfy discriminative headroom:
    $$\text{AUC}_{\text{Oracle}} - \text{AUC}_{\text{Random}} \ge 0.15$$
  - **Stage 3 Hard Kill Criterion:** If under discriminative multi-action headroom, boundary-aware correction satisfies:
    $$\text{AUC}(dB) \le \max(\text{AUC}(dE), \text{AUC}(d|G|))$$
    then the **correction algorithm claim is killed**, and the thesis contribution is formally downgraded from a *model correction algorithm* to a *world-model error diagnostic/evaluation framework*.
* **Kill Rule 4 (Scaling Disappearance):** If the phenomenon is observed in the 2-action Fork MDP but vanishes completely in multi-action stochastic GridWorld (NC4), scale-up to learned models will NOT proceed.

---

### 7. Current Progression Gates & Scientific Status

$$\boxed{\text{Stage 0: Mechanism}} \xrightarrow{\text{G1, G2: PASS}} \boxed{\text{Stage 3: Multi-Action}} \xrightarrow{\text{G3: FALSIFIED}} \boxed{\text{Stage 4: Learned Models}} \xrightarrow{\text{G4A: PASS, G4B: NEGATIVE}} \boxed{\text{Stage 5: CartPole Ext. Validity (PASS)}}$$

| Gate | Scientific Question | Status | Empirical Verdict |
| :--- | :--- | :---: | :--- |
| **Gate G1** | Is there causal separation in repair value when predictive and value severity are matched? | **PASS** | $N=1,782$ matched pairs; Cluster-level 95% BCI: $[0.0064, 0.0117]$ (excludes 0). |
| **Gate G2** | Does boundary geometry provide incremental explanatory power over all occupancy-weighted baselines? | **PASS** | $\Delta R^2(d \cdot B_i \mid d E, d |G|, d G^\pm, d A^\pm) = \mathbf{+0.1836}$ ($F = 70.94, p = 8.1 \times 10^{-15}$). |
| **Gate G3** | Does boundary pressure universally outperform value sensitivity under budgeted model correction? | **FALSIFIED / RESOLVED** | Multi-action GridWorld: unweighted $B$ beats predictive error ($E_{L1}, d \cdot E, p < 0.01$), but $\|G\|$ achieves higher aggregate full-trajectory AUC ($0.694 > 0.670, p = 0.0059$). Algorithm claim killed; re-aligned to diagnostic boundary geometry. |
| **Gate G4-A** | Does a finite-data neural world model recover action margins and boundary crossings? | **PASS** | Neural Categorical MLP: $\text{AUROC}(Z_{\text{cross}}) = \mathbf{0.8699}$, Margin rank correlation $\rho = \mathbf{0.7839}$, Policy action agreement $= 73.2\%$. |
| **Gate G4-B** | Does continuous estimated boundary pressure retain incremental regression power under estimation noise? | **NEGATIVE / BOUNDED** | $R_0^2 = 0.0739 \rightarrow R_1^2 = 0.0775 \implies \Delta R^2 = \mathbf{+0.0036}$ ($F = 1.35, p = 0.246$). Crossing classification survives, but point-wise continuous $B$ collapses under estimation noise. |
| **Gate G4-C** | Does approximate boundary prioritization retain scarce-budget advantages under learned models? | **CHARACTERIZED** | Observed early recovery at $K/N=14\%$: $\text{Rec}(\hat{B}) = 34.2\%$ vs $\text{Rec}(\hat{E}) = 26.0\%$, but full-trajectory AUC tracks $\|G\|$ ($\text{AUC} = 0.606$). |
| **Gate G5** | Does the boundary proximity phenomenon replicate in a continuous-state domain (CartPole)? | **PASS** | Matched error deciles: Damage difference $= \mathbf{+0.3796}$ ($95\%$ BCI: $[0.2795, 0.5188]$). Incremental $\Delta R^2 = \mathbf{+0.2400}$ ($p = 1.11 \times 10^{-16}$). Flip AUROC $= \mathbf{0.9836}$. |

---

### 8. Paper-Level Four Contribution Decomposition

1. **C1 (Controlled Causal Separation):** Under matched predictive loss ($E_{L1}, E_{\text{KL}}$), occupancy ($d$), and first-order value sensitivity ($|G|, G^\pm, A^\pm$), the signed direction of action-margin deformation produces non-zero differences in counterfactual correction value ($\Delta C = +0.0088$, cluster bootstrap 95% BCI: $[0.0064, 0.0117]$).
2. **C2 (Incremental Explanatory Information):** Decision-boundary geometry explains substantial variance that predictive error and conventional first-order value sensitivity do not capture ($\Delta R^2 = \mathbf{+0.1836}, p < 10^{-14}$).
3. **C3 (Controlled Multi-Action Falsification & Algorithmic Limits):** In non-saturated stochastic GridWorld, boundary pressure substantially outperforms prediction error heuristics ($B > E_{L1}, d \cdot E, p < 0.01$, achieving $1.7\times$ to $2.2\times$ recovery under scarce budgets $K/N \le 15\%$), while first-order value sensitivity ($\|G\|$) bounds the achievable algorithmic claim ($0.694 > 0.670$). Thus: $\boxed{\text{Mechanistic Relevance } \neq \text{ Universal Prioritization Superiority}}$.
4. **C4 (Oracle-to-Learned Model Degradation & Continuous-State Validity):** Finite-data neural world models successfully recover boundary-crossing structure ($\text{AUROC} = 0.8699, \rho = 0.7839$), while continuous explanatory statistics attenuate ($\Delta R^2 \approx 0.0036$). The core matched-error boundary damage phenomenon replicates in continuous-state CartPole ($\Delta R^2 = +0.2400, p < 10^{-16}$, Flip $\text{AUROC} = 0.9836$).

