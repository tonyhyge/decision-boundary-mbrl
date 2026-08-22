# EXPERIMENT SPECIFICATION: STAGE 3 — STOCHASTIC MULTI-ACTION GRIDWORLD

**Document:** Stage 3 Pre-Registration & Methodological Blueprint  
**Goal:** Decisive falsification test for the practical utility of decision-boundary-aware world-model correction ($G_3$) in a multi-action, multi-route stochastic environment with discriminative headroom.

---

## 1. Motivation & Failure Mode of Stage 0

In Stage 0 (Fork MDP):
* The single root state ($s_0$) had deterministic unit visitation ($d(s_0) = 1.0$) and dominated total return loss.
* Random ranking achieved $\text{AUC} = 0.978$ and all heuristics reached $1.0$ recovery at $K/N \le 25\%$, resulting in **severe ceiling saturation** ($\Delta \text{AUC}_{\text{Oracle}-\text{Random}} \approx 0.015$).
* **Stage 3 Mission:** Break benchmark saturation by introducing multi-action choice geometry, non-root bottlenecks, alternative detour routes, and heterogeneous occupancy distributions.

---

## 2. Environment Architecture: Choice GridWorld ($5 \times 5$, $|\mathcal{A}| = 4$)

### 2.1 Grid Topology & Actions
* **State Space:** $\mathcal{S} = \{0, 1, \dots, 24\}$ corresponding to coordinates $(r, c) \in \{0, 1, 2, 3, 4\}^2$, plus an absorbing Goal state $s_{\text{goal}} = (4, 4)$.
* **Action Space:** $|\mathcal{A}| = 4$ corresponding to $\{\text{UP (0)}, \text{DOWN (1)}, \text{LEFT (2)}, \text{RIGHT (3)}\}$.
* **Start Distribution:** $\mu_0(0, 0) = 1.0$.
* **Goal Reward:** $R(s_{\text{goal}}) = +10.0$. Step cost $c_{\text{step}} = -0.05$.
* **Discount Factor:** $\gamma = 0.95$.

### 2.2 Competing Alternative Routes
1. **Route 1 (Northern Safe Detour):**
   * Path: $(0,0) \to (0,4) \to (4,4)$ along the top and right perimeter.
   * Dynamics: High transition reliability ($p_{\text{succ}} = 0.92$), low slip.
2. **Route 2 (Central Risky Shortcut):**
   * Path: $(0,0) \to (2,2) \to (4,4)$ cutting through the center.
   * Dynamics: Higher stochastic slip ($p_{\text{succ}} = 0.70$) with hazard penalty cells at $(1, 2)$ and $(2, 3)$ ($R_{\text{hazard}} = -2.0$).
3. **Route 3 (Southern Intermediate Detour):**
   * Path: $(0,0) \to (4,0) \to (4,4)$ along the left and bottom perimeter.
   * Dynamics: Intermediate transition reliability ($p_{\text{succ}} = 0.85$) with a narrow bottleneck at $(3, 1)$.

### 2.3 Key Decision Geometry Features
* **No Single Root Dominance:** Decisions at $(0, 0)$, $(0, 2)$, $(2, 0)$, $(2, 2)$, $(3, 1)$, and $(1, 3)$ dynamically route probability mass across paths.
* **Near-Tie Decision States:** States where the top two actions have small true action gaps ($m_P(s) \in [0.01, 0.06]$).
* **Large-Gap Decision States:** States along corridors where moving toward walls or backwards is heavily suboptimal ($m_P(s) > 0.50$).
* **Stochastic Transitions:** Agent moves in intended direction with probability $p_{\text{succ}}$, slips orthogonally with probability $\frac{1 - p_{\text{succ}}}{2}$, or remains in place upon hitting grid boundaries.

---

## 3. Corruption Design & Multi-Error Injection

In each MDP instance, $N = 12$ to $16$ simultaneous localized transition corruptions are injected:

1. **Category 1 (Unvisited / Policy-Irrelevant Errors - NC3):**
   * Large prediction errors ($E_{L1} \in [0.6, 0.9]$) injected at states off the optimal support ($d(s) < 0.01$).
2. **Category 2 (Large-Gap Visited Errors - NC1/NC2):**
   * Moderate-to-large errors ($E_{L1} \in [0.4, 0.7]$) at visited states with large margin gaps ($m_P > 0.50$). $Z_{\text{cross}} = 0$, $C_i = 0$.
3. **Category 3 (Near-Tie Visited Bottleneck Errors - Proposed Target):**
   * Small errors ($E_{L1} \in [0.10, 0.25]$) at visited bottleneck choice points with small margin gaps ($m_P \in [0.02, 0.08]$). These cross the action boundary ($Z_{\text{cross}} = 1$), causing route abandonment and severe return degradation ($C_i > 0$).
4. **Category 4 (Interacting Detour Errors):**
   * Interacting corruptions along alternative routes where repairing one error restores partial route viability.

---

## 4. Pre-Registered Headroom Quality Gate (Pre-condition)

Before evaluating ranker competition, the GridWorld benchmark must verify:

$$\boxed{\text{Discriminative Headroom Pre-condition: } \text{AUC}_{\text{Oracle}} - \text{AUC}_{\text{Random}} \ge 0.15}$$

If $\text{AUC}_{\text{Oracle}} - \text{AUC}_{\text{Random}} < 0.15$, the environment configuration is flagged as saturated and must be re-calibrated before ranking evaluation.

---

## 5. Formal Hard Kill Criteria (Pre-Registered)

1. **Gate G3 Pass Condition:**
   Under discriminative headroom ($\ge 0.15$), occupancy boundary pressure $d \cdot B_i$ must achieve:
   $$\text{AUC}(d \cdot B_i) > \max\left(\text{AUC}(d \cdot E_{L1}), \; \text{AUC}(d \cdot |G|), \; \text{AUC}(d \cdot G^\pm)\right)$$
   with paired bootstrap $p < 0.01$ across $\ge 50$ random seeds.
2. **Kill Rule 3 (Formal Downgrade):**
   If $\text{AUC}(d \cdot B_i) \le \max(\text{AUC}(d \cdot E_{L1}), \text{AUC}(d \cdot |G|))$ across budget fractions $K/N \in [0.1, 0.5]$, the **correction algorithm claim is falsified and killed**. The thesis contribution is formally restricted to **world-model error diagnostics and evaluation**.
