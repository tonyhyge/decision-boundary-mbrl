# Stage Q1-W2: Preregistered Protocol for Walker2d-v4 Full-Budget Replication

**Target Milestone:** Q1 Journal Publication (IEEE T-NNLS / JMLR / Machine Learning)  
**Study Type:** Cross-Environment External Validity & Transfer Replication  
**Environment:** `Walker2d-v4` (MuJoCo Continuous Locomotion, $\mathcal{S} \subseteq \mathbb{R}^{17}, \mathcal{A} \subseteq \mathbb{R}^6$)  
**Status:** Protocol Frozen Prior to Observation of Comparative Results

---

## 1. Research Question & Scientific Rationale

### 1.1 Core Research Question
> **"Does the observed full-budget objective advantage of VaGraM over MLE on Hopper-v4 transfer robustly to Walker2d-v4 when evaluated under the identical model-based reinforcement learning architecture without environment-specific tuning?"**

### 1.2 Intentional Architectural Reuse (Scientific Design Choice)
We intentionally reuse the exact `Hopper-v4` configuration without environment-specific hyperparameter search to directly test the transferability of the objective advantage across continuous locomotion domains.

---

## 2. Experimental Specifications & Symmetry Invariants

| Specification Parameter | Value / Setting | Matched Control Branch |
| :--- | :--- | :--- |
| **Environment** | `Walker2d-v4` (Gymnasium v1.0 / MuJoCo 3.x) | Identical |
| **Interaction Budget** | $125,000$ environment steps | Identical |
| **Dynamics Model** | Ensemble of 7 MLPs (5 elites), 4 layers $\times$ 200 units, ReLU activations | Identical |
| **Dynamics Optimizer** | Adam ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-5}$), batch size 256 | Identical |
| **Model Retrain Schedule**| Every 250 environment steps (patience 50, valid ratio 0.2) | Identical |
| **SAC Policy Agent** | Double-Q Critic ($2\times 256$, $\text{lr}=3\times 10^{-4}$), Actor ($2\times 256$, $\text{lr}=3\times 10^{-5}$) | Identical |
| **SAC Updates Ratio** | $G = 20$ updates per environment step | Identical |
| **Rollout Horizon** | Scheduled expansion $[20\mathrm{k}, 150\mathrm{k}, 1, 5]$ | Identical |
| **Evaluation Protocol** | Every 5,000 steps ($25$ checkpoints total), $10$ deterministic test episodes | Identical |
| **Confirmatory Cohort** | $N=10$ paired seeds: $[142, 143, 144, 145, 146, 147, 148, 149, 150, 151]$ | Identical |
| **Engineering Smoke Seed** | Seed $101$ (Single engineering smoke run to verify pipeline integrity) | Out-of-cohort |

---

## 3. Four Potential Contingency Outcomes & Scientific Interpretation

| Outcome on Walker2d-v4 | Primary Estimand Condition | Scientific Interpretation & Narrative Role |
| :--- | :--- | :--- |
| **1. Robust Transfer** | $\bar{\Delta} > 0$ on majority ($8\text{--}10/10$) seeds, $p < 0.05$ | **Strong cross-environment external validity**: Value-gradient objective advantage is general across distinct continuous locomotion dynamics. |
| **2. Moderate / Heterogeneous** | $\bar{\Delta} > 0$ but high variance ($5\text{--}7/10$ wins) | **Regime-dependent advantage**: Value-gradient utility is positive but sensitive to domain-specific contact and posture properties. |
| **3. Invariant / Neutral** | $\bar{\Delta} \approx 0$, $p \ge 0.05$ | **Domain-bounded transfer**: The Hopper effect does not transfer uniformly, illustrating environment-specific boundary conditions. |
| **4. Inversion / MLE Advantage**| $\bar{\Delta} < 0$, $p < 0.05$ | **Strong boundary condition**: Demonstrates that decision-aware weighting interacts negatively with Walker2d's higher-dimensional contact space under default hyperparameters, directly supporting the central thesis that *$\text{Decision Relevance} \not\Rightarrow \text{Universal Utility}$*. |

*All four outcomes yield rigorous, falsifiable scientific insights for the final manuscript.*

---

## 4. Failure & Replacement Policy

$$\boxed{
\text{Invalid Run} \implies \text{Rerun exactly the same seed/method from scratch under identical frozen commit/environment}
}$$

- **Invalidity Triggers:** Premature termination/OOM, missing terminal checkpoints ($115\mathrm{k}, 120\mathrm{k}, 125\mathrm{k}$), NaN/Inf, incomplete updates ($< 2.4\text{M}$ SAC updates), corrupted artifacts.
- **Strict Prohibition:** Never replace a valid run due to low return; never replace seeds; never patch individual checkpoints.

---

## 5. Statistical Inference Plan

1. **Primary Estimand:** $\bar{\Delta} = \frac{1}{N} \sum_{i=1}^N (\bar{J}_i^{\mathrm{V}} - \bar{J}_i^{\mathrm{M}})$, where $\bar{J}_i = \frac{1}{3} (J_i(115\mathrm{k}) + J_i(120\mathrm{k}) + J_i(125\mathrm{k}))$.
2. **Exact Permutation Sign-Flip Test:** Exact enumeration of all $2^{10} = 1024$ sign permutations across observed delta magnitudes.
3. **Paired Percentile Bootstrap 95% CI:** $B = 10,000$ paired bootstrap resamples.
4. **Execution Sequence:** `audit_q1w2_integrity.py` (100% PASS) $\to$ `W2_COHORT_VALIDITY_PASS.txt` $\to$ `statistical_closure_q1w2.py` (executed exactly once).
