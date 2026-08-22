# NOVELTY MATRIX & EXACT-NEIGHBOUR AUDIT

**Project:** Decision-Margin Geometry of World-Model Errors in Model-Based Reinforcement Learning  
**Last Updated:** August 2026  
**Status:** `NOVELTY YELLOW` (Active Defense & Boundary Monitoring)

---

## 1. Audit Framework & Methodology

To ensure absolute scientific integrity and avoid hollow novelty claims, each candidate neighbouring work is audited across 14 rigorous dimensions:
1. **Paper & Authors / Year**
2. **Exact Core Claim**
3. **Assumptions**
4. **Predictive Signal Used** (e.g., L2/KL, TV loss)
5. **Decision Signal Used** (e.g., value gradient $\|\nabla V\|$, advantage, policy margin)
6. **Are Action/Policy Rankings Explicitly Studied?**
7. **Is True Model Ground Truth Available / Explored?**
8. **Are Matched Interventions Performed?** (Holding error magnitude constant while varying decision direction)
9. **Downstream Objective / Metric** (e.g., policy return, test MSE, ranking regret)
10. **Intervention Mechanism** (e.g., loss reweighting, dynamics masking, budget allocation)
11. **Experimental Benchmark** (e.g., tabular, MuJoCo, Atari, robotics)
12. **Exact Overlap with Our Hypothesis**
13. **Remaining Distinction / Defensible Gap**
14. **Novelty Threat Level** (`HIGH`, `MEDIUM`, `LOW`)

---

## 2. Literature Family Audits

### Family A: Predictive Error vs. Control Relevance (Value-Aware / Policy-Aware MBRL)

| Paper | Year | Core Claim & Signal | Matched Interventions? | True Ground Truth? | Overlap with Our Work | Defensible Distinction | Threat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VAML** (Farahmand et al.) | 2017 | Minimizing $(P - \hat{P}) V^*$ is better for control than standard MLE. Uses $\|(P - \hat{P})V\|_2$. | **No** (Only global loss minimization) | Tabular / Linear | Replaces pure MLE with value-aware loss. | Studies global training loss; does not examine signed deformation of competing action margins or per-error causal correction value ($C_i$). | `MEDIUM` |
| **Iterative VAML / Calibrated VAML** (Voelcker et al.) | 2024–2025 | Iterative value weighting prevents distribution drift in model learning. | **No** (Global model fitting) | Continuous (Inverted Pendulum, etc.) | Evaluates model quality based on value function consistency. | Focuses on training dynamics of world models; does not analyze individual error geometry or counterfactual single-error repair under budget constraints. | `MEDIUM` |
| **VaGraM** (Voelcker & Farahmand) | 2022 | Weights transition loss by $\|\nabla_s V(s')\|^2$. | **No** | Continuous RL | Recognizes that transitions into high-gradient regions matter more. | Uses **unsigned first-order gradient magnitude** $|\nabla V^\top \delta P|$. Our critical negative control directly tests if signed margin deformation $B_i$ adds independent value over $|G_i|$ and $G_i^\pm$. | `HIGH` |
| **Proper Value Equivalence** (Grimm et al.) | 2020–2021 | Models need only make valid value predictions across a policy class $\Pi$. | **No** | Exact MDPs & Deep RL | Replaces environmental fidelity with value equivalence. | Focuses on set-theoretic equivalence classes of models rather than individual localized errors deforming action gaps. | `MEDIUM` |
| **ROMI** (2026) | 2026 | Robust value-aware model learning using adaptive state uncertainty sets. | **No** | Offline RL | Avoids gradient explosion in value-weighted models. | Focuses on offline RL pessimism and scale-adjustable uncertainty, not action-margin boundary pressure. | `LOW` |

---

### Family B: Model Trust, Selective Expansion & Error Correction

| Paper | Year | Core Claim & Signal | Matched Interventions? | True Ground Truth? | Overlap with Our Work | Defensible Distinction | Threat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Selective Dyna / STEVE** (Buckman et al.) | 2018 | Adapt rollout horizons based on ensemble variance/uncertainty. | **No** | Continuous Control | Adjusts model reliance based on predicted reliability. | Uses model uncertainty ($U_i$) as a scalar confidence; ignores the geometric direction of error relative to the optimal action gap. | `MEDIUM` |
| **Policy-Margin Gated Rollouts** | 2023–2024 | Gates model rollouts when policy entropy or margin $Q(a_1) - Q(a_2)$ is low. | **No** | Classic Control | Uses action margin as a threshold for model trust. | Uses margin as a scalar safety filter on the agent's policy; does NOT study how model errors *cause* margin deformation ($\Delta m$) or allocate *corrections* to dynamics errors. | `HIGH` |
| **MOPO / MOREL** (Yu et al., Kidambi et al.) | 2020 | Penalize reward by model uncertainty $\hat{R} - \lambda U(s, a)$. | **No** | Offline MuJoCo | Conservative planning under model error. | Uniform/heuristic penalty proportional to variance; does not isolate which individual errors reverse rankings versus those that widen margins. | `LOW` |

---

### Family C: Action-Gap Regularity & Ranking Invariance

| Paper | Year | Core Claim & Signal | Matched Interventions? | True Ground Truth? | Overlap with Our Work | Defensible Distinction | Threat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Increasing the Action Gap** (Bellemare et al.) | 2016 | Consistent Bellman operators expand action gaps to improve policy stability. | **No** | Atari / Discrete MDPs | Highlights the fundamental importance of action gaps $Q(s, a^*) - Q(s, a)$. | Studies Bellman operator design in model-free RL; does not study world-model dynamics errors deforming these gaps. | `LOW` |
| **Policy Invariance under Model Shifts** | 2022–2024 | Conditions under which $\arg\max_a Q_{\hat{P}} = \arg\max_a Q_P$. | **No** (Mostly worst-case bounds) | Tabular / Theoretical | Connects model error to policy ranking stability. | Provides uniform $L_\infty$ bounds (e.g., $\|\delta P\| \le \frac{\Delta Q}{2\gamma}$); does not investigate continuous signed deformation $B_i$, counterfactual correction value $C_i$, or budgeted allocation. | `HIGH` |

---

### Family D: Recent High-Threat Neighbours (2024–2026)

| Paper | Year | Core Claim & Signal | Matched Interventions? | True Ground Truth? | Overlap with Our Work | Defensible Distinction | Threat |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ordinal Policy-Ranking Preservation in MBRL** | 2024–2025 | World models can suffer catastrophic failure when policy rankings flip despite low loss. | **No** (Observational) | Continuous benchmarks | Identifies ranking flips as a failure mode of predictive world models. | Observes aggregate policy ranking degradation across candidate policies; does NOT isolate per-transition error attribution, signed action-margin pressure $B_i$, or budgeted single-component error repairs. | `HIGH` |
| **ContactGuard / World-Action Models** | 2025–2026 | Predicts failure trajectories and action gaps in robotic manipulation. | **No** | Robotics simulation | Evaluates rollouts by whether planned actions match grounded execution. | Domain-specific application to embodied robotics; no causal matched-pair interventions on dynamics parameters or formal margin geometry. | `LOW` |

---

## 3. Explicit Differentiation Summary

```
Prior Literature Landscape:
[Global MLE Loss] ──(VAML / VaGraM)──> [Unsigned Value Weighting: |∇V·δP|]
                                             │
                                             ▼
                                  [Scalar Policy Trust / Gating]
                                             │
                                             ▼
Our Unique Contribution:
[Matched Error Pairs (δP⁺, δP⁻)] ──> [Signed Margin Deformation: Δm / B_i] ──> [Counterfactual Repair Value: C_i]
                                                                                     │
                                                                                     ▼
                                                                     [Budgeted Correction @ K]
```

### Key Differences at a Glance:
1. **Causal Matched Pairs vs Global Correlations:** Prior work correlates aggregate model MSE with policy return. We design matched error pairs with identical $E_i, U_i, d_i, m_i, |G_i|$ and prove that signed margin deformation $B_i$ dictates the true causal correction utility $C_i$.
2. **Signed Action-Margin Geometry vs Unsigned Value Gradients:** VaGraM weights by unsigned gradient norm $\|\nabla V\|$. We prove that an error pushing towards a competitor ($B > 0$) has radically different repair value than an error pushing away from a competitor ($B < 0$), even when $\|\nabla V^\top \delta P\|$ is identical.
3. **Component-Level Budgeted Repair:** Prior work either trusts or rejects the entire model / rollout. We formulate the budgeted error repair problem ($\operatorname{Recovery@K}$), identifying which localized transition parameters to collect real data for.
