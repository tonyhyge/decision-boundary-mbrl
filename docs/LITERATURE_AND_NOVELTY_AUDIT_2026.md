# Saturated Literature & Novelty Audit (2023–2026): Learning Utility of World-Model Data

**Milestone:** Next-Phase Strategic Research Foundation & Novelty Audit  
**Date:** 2026-08-20  
**Author:** Pham Quang Minh  
**Scope:** Exhaustive audit of exact neighbours, competitive landscape, threat levels, and defensible white space around *Learning Utility of World-Model Data* and *Intervention-Aware Data Valuation for MBRL*.

---

## 1. Competitive Landscape & Exact-Neighbour Matrix (2023–2026)

```
+-----------------------------------------------------------------------------------------------------------------------+
| Paper / System            | Venue / Year | Core Mechanism & Scope                      | Threat Level | Differentiation |
+-----------------------------------------------------------------------------------------------------------------------+
| Action Shapley            | ICLR-W 2026  | Shapley data valuation of trajectories for  | HIGH         | Computes empirical |
| (Ghosh & Dutta)           |              | world models via failure-mode cut-offs      |              | coalition values; does |
|                           |              | for downstream RL cumulative reward.        |              | not explain diagnostic |
|                           |              |                                             |              | to utility decoupling. |
+-----------------------------------------------------------------------------------------------------------------------+
| RENEW                     | alphaXiv     | Active querying in high epistemic-          | HIGH         | Focuses on preference- |
| (2026)                    | 2026         | uncertainty regions + dynamics fine-tuning  |              | guided repair; naive   |
|                           |              | to mitigate model exploitation.             |              | uncertainty acquisition.|
+-----------------------------------------------------------------------------------------------------------------------+
| Offline vs Online MBRL    | RLJ 2025     | Proves that limited state-action coverage   | MEDIUM       | Empirical benchmark on |
| (Chen et al.)             |              | and data mixing dominate generalization     |              | data mixing; no        |
|                           |              | across 31 continuous control benchmarks.    |              | decision geometry.     |
+-----------------------------------------------------------------------------------------------------------------------+
| ROMI                      | arXiv 2026   | Value-aware adaptive loss weighting to      | MEDIUM       | Objective weighting in |
|                           |              | stabilize world-model updates in offline RL.|              | offline RL; no data    |
|                           |              |                                             |              | acquisition / utility. |
+-----------------------------------------------------------------------------------------------------------------------+
| Influence Functions for   | ICML/NeurIPS | Data attribution via first-order Hessian    | MEDIUM-HIGH  | Standard influence     |
| MBRL / LLM World Models   | 2024-2026    | vector products for filtering noisy data.   |              | filtering; no decision |
|                           |              |                                             |              | boundary coupling.     |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Occupied Territory (Do NOT Re-Invent)

The following scientific claims and formulations are **ALREADY OCCUPIED** in the 2024–2026 literature and must NOT be presented as primary contributions:

1. **"Not all world-model transitions have equal downstream training utility."**  
   *Occupied by Action Shapley (2026).*
2. **"Uncertainty-guided active querying repairs pretrained world models."**  
   *Occupied by RENEW (2026).*
3. **"State-action coverage and replay mixing ratios dominate world-model rollout error over raw data volume."**  
   *Occupied by Chen et al. (RLJ 2025).*
4. **"First-order influence functions / gradient matching can filter training transitions."**  
   *Occupied by standard ML data valuation literature (LearnAlign, PRISM, GCR).*
5. **"Heuristic weighted sum of uncertainty + coverage diversity."**  
   *Saturated across classical and modern active learning.*

---

## 3. The Defensible Novelty White Space

The unresolved scientific gap uniquely exposed by our empirical findings (from CartPole Stage 4A to Hopper Stage N1-VS) is:

$$\boxed{\textbf{Why does an analytical signal that is provably predictive of policy decision vulnerability fail as a batch training criterion under finite neural network optimization?}}$$

### The 3-Component Latent Utility Model

```
Pointwise Decision Risk:        R(x) = nabla_a Q(s, a)  or  m(s)^-1   (Evaluation-side vulnerability)
Distributional Coverage:        C(S) = Entropy / Spatial Support      (Manifold representation stability)
Parameter Update Influence:     I(x) = nabla_theta l_i^T g_E          (Optimization-side gradient alignment)
```

$$\boxed{U(S) = f(R(S), \ C(S), \ I(S)) \quad \neq \quad \sum_{x \in S} R(x)}$$

* **The Core Differentiation from Action Shapley:**  
  Action Shapley treats the world model as a black box and computes empirical marginal coalition values via Monte Carlo / cut-off sampling. It does **not** explain *why* analytical decision-risk geometry decouples from gradient update mechanics.
* **The Core Theoretical Contribution:**  
  Decision relevance $R(x)$ is an **evaluation-side property** of policy sensitivity, whereas neural network training utility $U(S)$ is an **optimization-side property** governed by gradient alignment $I(x)$ and representation support $C(S)$.

---

## 4. Controlled Next-Phase Roadmap (Falsification-First Protocol)

To develop this into a standalone research project without premature implementation:

```mermaid
flowchart TD
    M1["1. Synthetic 2D Controlled MDP"] -->|Independently Manipulate Risk R, Coverage C, and Gradient Alignment I| ORACLE["2. Ground-Truth Oracle Training Utility U_K(x)"]
    ORACLE --> TEST["3. Causal Mechanism Falsification"]
    TEST -->|If I(x) and C(S) fully explain U(S)| EST["4. Deployable First-Order Estimator U_i^(1) = eta g_E^T g_i"]
    TEST -->|If Gradient Alignment Fails| REP["5. Empirical Decomposition of Non-Additive Data Value"]
```

1. **Step 1: Controlled 2D Dynamic System (Toy MDP):**  
   Construct a minimal continuous system where decision boundary curvature $R(x)$, state distribution occupancy $C(x)$, and neural parameter alignment $I(x) = \nabla_\theta \ell_i^\top g_E$ can be manipulated independently.
2. **Step 2: Ground-Truth Oracle Utility Metric:**  
   Define $U_K(x) = L_{\mathrm{eval}}(\theta) - L_{\mathrm{eval}}(\mathcal{A}_K(\theta, \mathcal{D} \cup x))$ under exact $K$-step SGD updates.
3. **Step 3: First-Order Estimator Formulation:**  
   $$U_i^{(1)} \approx \eta g_E^\top g_i - \frac{\eta^2}{2} g_i^\top H_E g_i$$
   Directly test whether gradient alignment predicts realized multi-step training utility across synthetic and continuous control domains.
