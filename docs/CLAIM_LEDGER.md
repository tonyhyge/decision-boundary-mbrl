# Project Claim Ledger & Scientific Governance

**Milestone:** Phase M1 — Claim Ledger Audit  
**Date:** 2026-08-20  
**Governance Standard:** Every scientific statement made in the thesis must map 1-to-1 to an entry in this ledger with an explicit evidential status label (`SUPPORTED`, `SUPPORTED-BUT-BOUNDED`, `DESCRIPTIVE ONLY`, `NOT SUPPORTED`).

---

## 1. Claim Taxonomy & Master Ledger

```
+-----------------------------------------------------------------------------------------------------------------------+
| ID   | Scientific Claim Statement                                                | Status Label         | Pointer     |
+-----------------------------------------------------------------------------------------------------------------------+
| C1.1 | Prediction error magnitude is decoupled from downstream control damage.    | SUPPORTED            | Step 1 / 3A |
| C1.2 | Decision vulnerability scales inversely with the discrete action margin.   | SUPPORTED            | Step 1 / 3A |
| C2.1 | Unprivileged boundary signals rank rare decision flips better than null.  | SUPPORTED            | Stage 3A    |
| C2.2 | Unprivileged ranker beats action-disagreement comparator (AUPRC 0.57 vs 0.42).| SUPPORTED        | Stage 3A    |
| C2.3 | Unprivileged detector provides superior calibration over baselines.        | NOT SUPPORTED        | Stage 3A    |
| C2.4 | Standardized regression coefficients prove causal feature dominance.      | NOT SUPPORTED        | Stage 3A    |
| C3.1 | Rank-based acquisition improves world-model repair efficiency on CartPole. | SUPPORTED-BUT-BOUNDED| Stage 4A    |
| C3.2 | Combined ranker is universally optimal across all acquisition budgets.    | NOT SUPPORTED        | Stage 4A    |
| C3.3 | Targeted repair produces major downstream return jumps on CartPole.        | DESCRIPTIVE ONLY     | Stage 4A    |
| C4.1 | Continuous decision displacement follows implicit perturbation law delta_a.| SUPPORTED           | Stage 3B    |
| C4.2 | Action curvature H_a modulates decision displacement magnitude under error.| SUPPORTED           | Stage 3B    |
| C4.3 | Estimated Hessian curvature improves deployed ranking score in practice.   | NOT SUPPORTED        | Stage 3B/4B0|
| C4.4 | Action-gradient disagreement reliably ranks continuous decision regret.    | SUPPORTED            | Stage 3B    |
| C5.1 | One-shot active acquisition improves MBPO rollout error on Hopper-v4.      | NOT SUPPORTED        | Stage 4B.1  |
| C5.2 | Extra optimization on sparse data produces measurable parameter drift.     | SUPPORTED-BUT-BOUNDED| Stage 4B.1-R|
| C5.3 | Dynamics uncertainty selects exclusively off-manifold/falling states.      | DESCRIPTIVE ONLY     | Stage 4B.1-R|
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Detailed Audit and Textual Guidance for Every Claim

### Claim C1.1 & C1.2: Discrete Decision Geometry & Action Margin
* **Statement:** Model prediction error alone does not determine policy loss; vulnerability to ranking reversal scales inversely with the local action-value margin $m(s) = |Q(s, a_1) - Q(s, a_2)|$.
* **Status:** `SUPPORTED`
* **Evidence:** Analytical proofs on Tabular MDPs (Step 1); empirical validation on CartPole (Stage 3A) where interaction ratio $\sigma_{\mathrm{dyn}} / (\hat{m} + \varepsilon)$ achieves standardized regression coefficient $\beta = +1.3204$ vs $\beta = +0.0611$ for raw dynamics error.
* **Manuscript Phrasing Rule:** State as core theoretical framework; do not claim prediction error is irrelevant, but that it must be geometrically scaled by distance to the decision boundary.

---

### Claim C2.1 & C2.2: Unprivileged Decision-Risk Ranking
* **Statement:** An unprivileged model-internal risk score ($S(s) = \sigma_{\mathrm{dyn}} / (\hat{m} + \varepsilon)$) identifies rare true-environment decision flips ($1.97\%$ natural test prevalence) on held-out trajectories with $\text{AUPRC} = 0.5678$, significantly outperforming the frozen action-disagreement baseline ($\text{AUPRC} = 0.4168$, $\Delta\text{AUPRC} = +0.1510$, $p = 0.0010$) and the prevalence base rate ($\text{AUPRC}_{\mathrm{null}} = 0.0197, 28.8\times$ enrichment).
* **Status:** `SUPPORTED`
* **Evidence:** Held-out test evaluation on $N=1,881$ states across $24$ trajectories with $B=2,000$ clustered bootstrap replicates ($95\%\text{ BCI}: [+0.0617, +0.2380]$) and full pipeline permutation null $p_{\mathrm{perm}} = 0.004975$.
* **Manuscript Phrasing Rule:** Report both the natural prevalence base rate ($0.0197$) and the strong action-disagreement comparator ($0.4168$). Do not conflate the two.

---

### Claim C2.3: Calibration Superiority
* **Statement:** The proposed method produces calibrated failure probabilities that are strictly superior in Brier score to competing baselines.
* **Status:** `NOT SUPPORTED`
* **Evidence:** Brier score parity ($\Delta\text{Brier} = -0.00001, p = 0.9600$ vs Action Disagreement).
* **Manuscript Phrasing Rule:** Explicitly state: *"The method functions as a decision-risk prioritization ranker rather than a well-calibrated posterior probability estimator; calibration superiority was not established."*

---

### Claim C2.4: Causal Feature Dominance from Regression
* **Statement:** The interaction feature causally drives decision flips because its regression coefficient is largest.
* **Status:** `NOT SUPPORTED`
* **Evidence:** Cross-sectional multivariate logistic regression establishes statistical association, not causal intervention.
* **Manuscript Phrasing Rule:** Use *"standardized multivariable association"* or *"explanatory variance contribution"*, never *"causal driver"*.

---

### Claim C3.1: Targeted Active World-Model Repair on CartPole
* **Statement:** Active data acquisition prioritized by the unprivileged decision-risk ranker eliminates held-out decision flips with significantly higher sample efficiency than uniform random exploration under constrained budgets ($B = 5\% - 10\%$).
* **Status:** `SUPPORTED-BUT-BOUNDED`
* **Evidence:** At $B=5\%$ budget, Combined Ranker achieves $-16.5\%$ flip reduction ($\eta = 0.0346$, a $2.8\times$ efficiency gain over Random $\eta = 0.0123$). Confirmatory $10$-seed comparison at $B=10\%$ confirms paired efficiency gain $+0.0093$ flips/query ($t(9) = 2.764, p = 0.0220$).
* **Manuscript Phrasing Rule:** Strictly bound this claim to the continuous CartPole benchmark under warm-start retraining. Do not claim universal optimality across all continuous control benchmarks.

---

### Claim C3.2: Universal Optimality across All Budget Regimes
* **Statement:** The Combined Ranker is universally optimal across all query budget fractions from $1\%$ to $20\%$.
* **Status:** `NOT SUPPORTED`
* **Evidence:** At $B=1\%$, pure dynamics uncertainty achieves higher initial repair ($\eta = 0.1875$ vs $0.1000$); at $B=20\%$, marginal returns decay non-monotonically.
* **Manuscript Phrasing Rule:** Present this as an empirical finding: *"Decision-risk ranking provides maximum relative sample-efficiency advantage in intermediate scarce-budget regimes ($5\%-10\%$), while pure dynamics uncertainty captures gross errors at micro-budgets ($1\%$) and updates begin to saturate at larger budgets ($20\%$)."*

---

### Claim C3.3: Downstream Policy Return Improvement
* **Statement:** Active model repair produces major downstream policy return gains.
* **Status:** `DESCRIPTIVE ONLY` (Secondary Consistency)
* **Evidence:** Return post-repair is $115.8 \pm 4.0$ vs $114.2 \pm 4.1$ for Random ($\Delta J \approx +1.6$ steps out of $120$).
* **Manuscript Phrasing Rule:** Frame as *"bounded consistency / non-degradation of downstream control in a near-saturated CartPole baseline"*, not as primary evidence of efficacy.

---

### Claim C4.1 & C4.2: Continuous Action Geometry & Implicit Perturbation Law
* **Statement:** In continuous action spaces, decision displacement obeys the first-order implicit-function law $\delta a \approx - H_a^{-1} \nabla_a \delta Q(s, a^*)$, and decision regret scales with $R_{\mathrm{dec}}(s) \approx \frac{1}{2} g(s)^\top (-H_a)^{-1} g(s)$.
* **Status:** `SUPPORTED`
* **Evidence:** Phase 3B.0 confirms $5.04\times$ displacement amplification in flat vs sharp optima under matched prediction error ($p = 2.56 \times 10^{-90}$). Phase 3B.1 confirms $99.9\%$ directional sign match, $\cos = 0.8747$, and Spearman $\rho = 0.9048$ ($p < 10^{-5}$).
* **Manuscript Phrasing Rule:** State as the continuous local analogue of the discrete boundary-margin principle. Use the multi-dimensional vector formulation $g(s)^\top (-H_a)^{-1} g(s)$.

---

### Claim C4.3: Practical Deployment Utility of Estimated Curvature
* **Statement:** Estimated Hessian curvature $|\hat{H}_a|^{-1}$ provides additive ranking utility when deployed in the unprivileged score.
* **Status:** `NOT SUPPORTED`
* **Evidence:** Stage 3B ranking achieves identical $\text{AUPRC} = 0.9965$ with or without curvature, and $0.9966$ with shuffled curvature. Stage 4B.0 repair benchmark confirms $S_{\mathrm{cont}}$ vs $\sigma_{\nabla_a Q}$ paired $p = 0.5858$.
* **Manuscript Phrasing Rule:** Explicitly declare: *"Curvature governs local decision sensitivity mechanistically, but in practical unprivileged detection, action-gradient epistemic disagreement ($\sigma_{\nabla_a Q}$) provides the actionable ranking information."*

---

### Claim C4.4: Action-Gradient Disagreement as Continuous Risk Ranker
* **Statement:** Action-gradient epistemic disagreement $\sigma_{\nabla_a Q}$ reliably ranks continuous decision regret on held-out trajectories, outperforming pure dynamics uncertainty.
* **Status:** `SUPPORTED`
* **Evidence:** Stage 3B held-out test evaluation: $\text{AUPRC} = 0.9965$ vs $0.4971$ for $\sigma_{\mathrm{dyn}}$ ($\Delta\text{AUPRC} = +0.5038, p_{\mathrm{boot}} < 0.0005$).
* **Manuscript Phrasing Rule:** Designate $\sigma_{\nabla_a Q}$ as the primary unprivileged continuous decision-risk ranker.

---

### Claim C5.1: One-Shot Active MBPO Repair on Hopper-v4
* **Statement:** Active data acquisition based on action-gradient disagreement improves world-model repair over uniform random sampling in one-shot Hopper retraining.
* **Status:** `NOT SUPPORTED`
* **Evidence:** In Stage 4B.1, all active strategies and random acquisition degraded $H=5$ rollout error ($0.0681 \to 0.0734$ for $\sigma_{\nabla_a Q}$ vs $0.0731$ for Random, $p = 0.6770$).
* **Manuscript Phrasing Rule:** State honestly as a negative finding: one-shot active query acquisition under this protocol did not beat uniform random sampling on `Hopper-v4`.

---

### Claim C5.2: Optimization Drift under Parameter Retraining
* **Statement:** Under the tested one-shot retraining protocol on Hopper-v4, parameter optimization on sparse exploratory transitions induces update drift that exceeds the local benefit of $B=200$ acquired transitions.
* **Status:** `SUPPORTED-BUT-BOUNDED`
* **Evidence:** Stage 4B.1-R establishes that $\mathcal{D}_0$-only extra SGD (zero new data) increases rollout error by $+8.79\%$ ($p = 3.73 \times 10^{-6}$). Privileged Residual Acquisition also degrades by $+9.71\%$ ($p = 1.69 \times 10^{-7}$). Only full pool coverage ($N=807$) achieves net error reduction ($-1.43\%$).
* **Manuscript Phrasing Rule:** Do not claim a universal law for all MBRL systems. State strictly: *"Under the tested one-shot repair operator, training budget, checkpoint, and protocol on Hopper-v4, additional optimization steps produce parameter drift large enough to outweigh the local corrective benefit of $B=200$ acquired transitions."*
