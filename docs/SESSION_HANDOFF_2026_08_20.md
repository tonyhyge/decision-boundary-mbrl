# Comprehensive Session Handoff: Decision Boundaries & Learning Utility in Model-Based RL

**Date:** 2026-08-20  
**Author:** Pham Quang Minh  
**Canonical Project Title:** *From Decision Relevance to Learning Utility: Operator-Conditional Limits of World-Model Error Prioritization*  
**Short Title:** *Decision Relevance Is Not Learning Utility in Neural World Models*  
**Repository State:** **35/35 pytest unit tests PASSING**  
**Core Mechanism/Falsification Program (M0->M2-A):** **CLOSED & PERMANENTLY LOCKED**  
**Q1 Comparative-Validation & Manuscript-Closure Program:** **OPEN & ACTIVELY EXECUTING**

```latex
\boxed{\text{Scientific falsification branch closed} \neq \text{Q1 revision checklist closed}.}
```

---

## 1. Executive Master Hierarchy & The Three Canonical Boundaries

```latex
\boxed{
\begin{aligned}
\textbf{Boundary 1: }& \text{Prediction error magnitude alone does not determine decision risk.} \\
\textbf{Boundary 2: }& \text{Decision risk is not sufficient to predict marginal training utility} \\
&\text{under the tested Hopper repair operator after regional controls.} \\
\textbf{Boundary 3: }& \text{Static initial-point gradient influence is not sufficient to predict} \\
&\text{650-step Adam utility in the tested Hopper neural-world-model regime.}
\end{aligned}
}
```

---

## 2. Definitive Status of All Experimental Stages

```
+-----------------------------------------------------------------------------------------------------------------------+
| Stage ID  | Domain / Benchmark      | Core Research Question                      | Final Determination & Status      |
+-----------------------------------------------------------------------------------------------------------------------+
| Step 1    | Tabular MDP & Theory    | Prediction error vs policy loss             | PASS — Decoupling proved.         |
| Stage 3A  | CartPole (Continuous S) | Unprivileged discrete risk ranking          | PASS — AUPRC 0.5678 vs 0.4168.    |
| Stage 4A  | CartPole (Continuous S) | Active targeted world-model repair          | PASS — 2.8x-11.4x over Random.    |
| Stage 3B  | Continuous CartPole LQR | Continuous perturbation geometry            | PASS — Perturbation law verified. |
| Stage 4B.0| Continuous CartPole LQR | Deployed estimated curvature contribution   | NOT ESTABLISHED (p = 0.5858).     |
| Stage 4B.1| Hopper-v4 (MuJoCo)      | One-shot continuous active repair           | CLOSED — Repair operator shock.   |
| Stage R0  | Hopper-v4 (MuJoCo)      | Repair operator falsification benchmark     | PASS — Monotonic SGD convergence. |
| Stage R0-V| Hopper-v4 (MuJoCo)      | Operator validity closure & normalization   | PASS — Non-degrading operator.    |
| Stage 4B1C| Hopper-v4 (MuJoCo)      | Operator-normalized acquisition re-test     | NEGATIVE TRANSFER (p = 0.6026).   |
| Stage N0  | Hopper-v4 (MuJoCo)      | Pointwise risk vs set utility gap           | PASS — Risk != Utility (p=0.5215).|
| Stage N1  | Hopper-v4 (MuJoCo)      | Orthogonalized risk x coverage interaction  | PASS — Crossover beta_RC=-0.0056. |
| Stage N1-V| Hopper-v4 (MuJoCo)      | Cluster-blocked validation & occupancy audit| FALSIFIED — HR/LC was an artifact.|
| Stage N1VS| Hopper-v4 (MuJoCo)      | Statistical patch & TOST equivalence        | PASS — TOST p_equiv = 0.00688.    |
| Stage M0  | 2D Controlled Dynamics  | Local singleton operator falsification      | PASS — First-order Taylor holds.  |
| Stage M1-B| 2D Controlled Dynamics  | Fixed-budget substitution across 10 ckpts   | PASS — rho = +0.9993, p < 1e-8.   |
| Stage M1CV| 2D Controlled Dynamics  | Set-utility validity & operator null        | PASS — Algebraic null established.|
| Stage M2-A| Hopper-v4 (MuJoCo)      | Privileged cross-fitted oracle transfer     | FALSIFIED — Delta R^2 p = 0.7191.  |
| Stage M2-B| Hopper-v4 (MuJoCo)      | Orthogonal Hopper causal intervention       | PERMANENTLY LOCKED / NOT OPENED.  |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 3. Key Findings & Empirical Determinations

### A. Stage N1-V & N1-VS (Cluster-Blocked Validity Closure on Hopper-v4)
* **Occupancy Confound Exposed:** The Stage N1 HR/LC advantage was confounded by Clusters 4 and 0 containing **$42.1\%$ of the evaluation trajectory distribution**.
* **Strict Cluster-Blocked Control ($N = 25$ runs: $5\text{ seeds} \times 5\text{ cluster pairs}$):** Comparing High-Risk vs Low-Risk within the *exact same cluster pairs* yielded a pooled marginal difference of $\Delta U = \mathbf{+0.000254}$ ($t = 0.921, \mathbf{p = 0.3664}$).
* **TOST Practical Equivalence:** Tested against SESOI $= \pm 0.001000$ ($\approx 23\%$ of the Stage N1 effect $\beta_R = +0.004260$). TOST passed with $\mathbf{p_{\mathrm{TOST}} = 0.00688 < 0.01}$, proving practical equivalence to zero.
* **Leave-One-Pair-Out:** Omitting $P_1$ (C4+C0) caused pooled $\Delta U$ to collapse to $\mathbf{-0.000038} \ (p = 0.8996)$.

### B. Stage M0, M1-B, & M1-CV (Operator-Conditional Mechanics on 2D Dynamics)
* **Local Substitution Formula:** $\hat{U}_{\mathrm{sub}}^{(1)} = \eta g_E^\top \delta g_x = \eta g_E^\top (g_{B,x} - g_B)$.
* **Checkpoint-Level Robustness ($S = 10$ independent checkpoints):**
  * Incremental Influence $\rho(I_{\mathrm{sub}}, U_{\mathrm{sub}}) = \mathbf{+0.9993} \ [95\%\text{ CI: } +0.9985, +1.0000]$ (Wilcoxon $W = 55.0, \mathbf{p = 9.77 \times 10^{-4}}$, Permutation $\mathbf{p_{\mathrm{perm}} = 0.0012}$).
  * Decision Risk $\rho(R, U_{\mathrm{sub}}) = \mathbf{-0.3879} \ [95\%\text{ CI: } -0.5182, -0.2576]$.
* **Mechanistic Discovery:** A high-risk sample ($R(x) \gg 0$) produces severe model degradation ($U < 0$) if its parameter gradient opposes the evaluation objective ($g_E^\top \delta g_x < 0$).
* **Operator Additive Null (M1-CV):** The apparent "$4\times$ dilution" in mini-batch sets is the algebraic consequence of optimizer gradient averaging ($\delta g_S = \frac{1}{|S|} \sum \delta g_i$). Evaluating against the operator-consistent null $\eta g_E^\top \delta g_S$ holds with an error of $< 4.3 \times 10^{-9}$.

### C. Stage M2-A (Privileged Retrospective Oracle Transfer on Hopper-v4)
* **Cross-Fitted Protocol:** 5-fold trajectory cross-fitting on $\mathcal{D}_{\mathrm{eval}}$ ($\mathcal{D}_{\mathrm{grad}}$ for privileged $g_E$, held-out $\mathcal{D}_{\mathrm{test}}$ for realized out-of-sample $U_{\mathrm{test}}$) across $50$ ensemble training runs.
* **Empirical Outcome:**
  * Model 3 ($U \sim R + C$): $R^2 = 0.5587, p = 0.0571$.
  * Model 5 ($U \sim R + C + I_{\mathrm{sub}}$): $R^2 = 0.5689, p = 0.1439$.
  * Incremental Added Value: $\Delta R^2 = \mathbf{+0.0102} \ (F = 0.142, \mathbf{p_F = 0.7191}, \text{Permutation } \mathbf{p_{\mathrm{perm}} = 0.7050})$.
  * Paired Contrast: $\Delta I_{sp}$ vs $\Delta U_{sp} \implies r = \mathbf{+0.0594} \ (\mathbf{p = 0.9245})$.
* **Scientific Interpretation:** Static initial-point gradient influence $g_E(\theta_0)^\top \delta g(\theta_0)$ fails under $650$ steps of multi-batch Adam ensemble optimization. Long-horizon neural generalization is governed by trajectory distribution coverage $C(S)$ and non-convex optimization dynamics, not by local initial gradient alignment.

---

## 4. Master Claim-Evidence Governance & Narrative Architecture

```
+-----------------------------------------------------------------------------------------------------------------------------------------------+
| ID    | Formal Scientific Claim             | Supported Evidence & Regimes        | Tested Assumptions & Scope      | Prohibited / Non-Supported Claims |
+-----------------------------------------------------------------------------------------------------------------------------------------------+
| C1    | Prediction error magnitude alone    | • Tabular MDP analytical theorems   | Valid across finite tabular     | Cannot claim that prediction error|
|       | does not determine downstream       | • CartPole test (AUPRC 0.5678 vs    | MDPs and single-step continuous | is statistically independent of   |
|       | decision consequence.               |   0.4168, Delta = +0.1510, p=0.0010)| control benchmarks tested.      | policy loss in all environments.  |
|       |                                     | • Permutation null p = 0.004975.    |                                 |                                   |
+-----------------------------------------------------------------------------------------------------------------------------------------------+
| C2    | Decision-sensitive geometry         | • Continuous CartPole LQR dynamics  | Assumes local quadratic Taylor  | Cannot claim that estimated       |
|       | diagnoses local policy vulnerability| • Perturbation law (cosine = 0.8747,| approximation of Q-function      | curvature adds unprivileged       |
|       | in continuous action spaces.        |   Spearman rho = 0.9048, p < 10^-5) | around optimal continuous action| ranking value in deployed models  |
|       |                                     | • Test AUPRC = 0.9965 vs 0.4971.    | (Hessian invertible).           | (Stage 4B.0 showed p = 0.5858).   |
+-----------------------------------------------------------------------------------------------------------------------------------------------+
| C3    | Decision risk ranking enables       | • CartPole acquisition benchmark    | Low-dimensional (4D state, 1D   | Cannot claim that decision-risk   |
|       | sample-efficient world-model repair | • 5% budget: 2.8x efficiency (p<0.05| action), single-joint dynamics,  | acquisition generalises to high-  |
|       | in controlled low-D regimes.        | • 10% budget: 11.4x over Random     | discrete action boundary.       | dimensional continuous control.   |
|       |                                     |   (Paired t(9) = 2.764, p = 0.0220).|                                 |                                   |
+-----------------------------------------------------------------------------------------------------------------------------------------------+
| C4    | Decision risk is not sufficient to  | • Hopper-v4 Stage 4B.1-C (p=0.6026) | Multi-joint continuous dynamics | Cannot claim that decision risk   |
|       | predict training utility under      | • Stage N0 benchmark (p=0.5215)     | (11D state, 3D action), fixed   | has zero causal effect in every   |
|       | neural repair after regional control| • Stage N1-V paired Delta U=0.000254| compute (T=650), Two-Stream     | continuous control setting.       |
|       |                                     | • Stage N1-VS TOST p_equiv = 0.00688| replay mixture (80/20).         |                                   |
+-----------------------------------------------------------------------------------------------------------------------------------------------+
| C5    | In one-step / few-step regimes,     | • 2D toy system Stage M0 (r=0.9989) | Single-step / few-step (K<=10)  | Cannot claim that first-order     |
|       | incremental gradient alignment      | • Stage M1-B across 10 checkpoints  | SGD on small neural networks;   | gradient alignment governs        |
|       | governs substitution utility.       |   (rho=0.9993, Wilcoxon p < 0.001). | fixed-budget substitution.      | multi-step non-convex training.   |
+-----------------------------------------------------------------------------------------------------------------------------------------------+
| C6    | Static initial-point gradient       | • Hopper-v4 Stage M2-A 5-fold cross-| 650-step Adam optimizer with    | Cannot claim that parameter drift |
|       | influence does not transfer to      |   fitting (Delta R^2 = +0.0102,     | momentum on 7-member ensemble   | was causally proven without       |
|       | 650-step ensemble Adam on Hopper.   |   p_F = 0.7191, p_perm = 0.7050)   | of deep neural networks.        | trajectory-level cosine logs.     |
|       |                                     | • Paired contrast r = 0.0594.       |                                 |                                   |
+-----------------------------------------------------------------------------------------------------------------------------------------------+
```

---

## 5. Master Artifacts & Code Manifest

### Core Scientific Reports
- [`docs/MASTER_CLAIM_EVIDENCE_MATRIX.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/MASTER_CLAIM_EVIDENCE_MATRIX.md): Master matrix of claims, empirical evidence, scope, assumptions, and prohibited claims.
- [`docs/MASTER_CAUSAL_EVIDENCE_MAP.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/MASTER_CAUSAL_EVIDENCE_MAP.md): Comprehensive causal evidence flowchart and master hierarchy.
- [`docs/LITERATURE_AND_NOVELTY_AUDIT_2026.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/LITERATURE_AND_NOVELTY_AUDIT_2026.md): Exhaustive 2023–2026 literature audit (*Action Shapley, RENEW, GRAD-MATCH, RFT-Inf*).
- [`docs/STAGE_M2A_HOPPER_ORACLE_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE_M2A_HOPPER_ORACLE_REPORT.md): Stage M2-A Hopper Oracle Transfer report.
- [`docs/STAGE_M1CV_SET_VALIDITY_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE_M1CV_SET_VALIDITY_REPORT.md): Stage M1-CV Set-Utility validity patch report.
- [`docs/STAGE_M1_FIXED_BUDGET_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE_M1_FIXED_BUDGET_REPORT.md): Stage M1 Fixed-Budget Substitution report.
- [`docs/STAGE_M0_OPERATOR_UTILITY_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE_M0_OPERATOR_UTILITY_REPORT.md): Stage M0 Operator Falsification report.
- [`docs/STAGE_N1VS_STATISTICAL_PATCH_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE_N1VS_STATISTICAL_PATCH_REPORT.md): Stage N1-VS Statistical Closure report.
- [`docs/STAGE_N1V_CLUSTER_BLOCKED_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE_N1V_CLUSTER_BLOCKED_REPORT.md): Stage N1-V Cluster-Blocked Validation report.
- [`docs/STAGE_N1_ORTHOGONALIZED_UTILITY_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE_N1_ORTHOGONALIZED_UTILITY_REPORT.md): Stage N1 Orthogonalized Utility report.
- [`docs/STAGE_N0_GAP_FALSIFICATION_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE_N0_GAP_FALSIFICATION_REPORT.md): Stage N0 Gap Falsification report.

### Executable Python Benchmark Scripts
- [`experiments/learning_utility/run_stage_m2a_hopper_oracle_transfer.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/learning_utility/run_stage_m2a_hopper_oracle_transfer.py)
- [`experiments/learning_utility/run_stage_m1cv_set_validity_patch.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/learning_utility/run_stage_m1cv_set_validity_patch.py)
- [`experiments/learning_utility/run_stage_m1_fixed_budget_utility.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/learning_utility/run_stage_m1_fixed_budget_utility.py)
- [`experiments/learning_utility/run_stage_m0_operator_utility_falsification.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/learning_utility/run_stage_m0_operator_utility_falsification.py)
- [`experiments/diagnostic_utility_gap/run_stage_n1vs_statistical_patch.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/diagnostic_utility_gap/run_stage_n1vs_statistical_patch.py)
- [`experiments/diagnostic_utility_gap/run_stage_n1v_cluster_blocked_validation.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/diagnostic_utility_gap/run_stage_n1v_cluster_blocked_validation.py)
- [`experiments/diagnostic_utility_gap/run_stage_n1_orthogonalized_utility.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/diagnostic_utility_gap/run_stage_n1_orthogonalized_utility.py)
- [`experiments/diagnostic_utility_gap/run_stage_n0_falsification.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/diagnostic_utility_gap/run_stage_n0_falsification.py)
- [`experiments/baselines/run_value_aware_baselines_benchmark.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/baselines/run_value_aware_baselines_benchmark.py)
- [`experiments/unprivileged_detection/report_auroc_calibration_audit.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/unprivileged_detection/report_auroc_calibration_audit.py)

---

## 6. Defensible 8-Feedback Q1 Journal Upgrade Checklist Audit

```latex
\boxed{
\text{Scientific falsification branch (M0}\to\text{M2-A) closed} \neq \text{Q1 comparative-validation program closed.}
}
```

| # | Advisor Feedback Item | Status | Defensible Assessment & Scientific Takeaway |
| :--- | :--- | :---: | :--- |
| **1** | **Scale to MuJoCo / modern MBRL backbone** | 🟡 **Substantially Addressed** | Evaluated on Hopper-v4 with neural ensemble (7 MLPs, elite 5, SAC) plus cross-env pilots on Walker2d-v4 and HalfCheetah-v4. Explicitly demarcated as a **matched low-budget MBPO-style stress test** ($10\mathrm{k}$ env steps) rather than an asymptotic locomotion benchmark. |
| **2** | **"Negative $\to$ Positive", validate $\mathcal{L}_{\mathrm{margin}}$** | 🟢 **Closed by Reframing** | $\mathcal{L}_{\mathrm{margin}}$ is permanently demoted from central contributions to Discussion/Future Work. The paper is strictly framed as a **boundary/limits paper** (validating that local mechanisms hold in controlled regimes but fail under non-convex neural training), not an algorithmic rescue paper. |
| **3** | **Supplementary Theory / Regret Bound** | 🟡$\to$🟢 **Formally Audited** | Proved **Margin Stability Lemma** ($2e_Q(s) < m_P(s) \implies \hat{\pi}(s) = a_P^*(s)$) and established non-tautological **Margin-Vulnerability Regret Bound**: $\operatorname{Regret}(\hat{\pi}; P) \le \frac{\Delta_{\max}}{1-\gamma} F_m(2\epsilon_Q)$ linking local boundary crossings to global policy regret. |
| **4** | **Replace oracle repair with practical acquisition** | 🟢 **Closed** | Practical acquisition verified in CartPole (2.8x--11.4x over Random in low-D); closed-loop active repair tested and cleanly demarcated on Hopper. |
| **5** | **Unprivileged Detection AUROC + Calibration** | 🟡$\to$🟢 **Closed as Risk Ranker** | Held-out test split ($N=1,881$, prevalence $1.97\%$): Proposed Combined Detector achieves **$\mathrm{AUROC} = 0.9083$** ($95\%$ BCI: $[0.7983, 0.9860]$) and **$\mathrm{AUPRC} = 0.5678$** vs $0.0197$ null. **Calibration superiority is NOT established** (Brier $0.01218$ vs $0.01220$, $p=0.96$); contribution is strictly bounded as an **unprivileged risk ranker**. |
| **6** | **Value-Aware Baselines (VaGraM, VAML, $\lambda$-VAML vs MLE)** | 🟢 **Experiment Complete** | Completed matched $N=5$ benchmark on Hopper-v4. **Canonical Finding:** Value-aware objectives exhibit substantially greater seed sensitivity (Sample SD $40.2$--$63.9$ vs $16.5$) and do NOT provide a robust advantage over MLE (wins on only 1--2 of 5 paired seeds, median $\Delta \in [-14.24, -10.09]$). **No claim of universal failure or statistical equivalence is made**; utility is environment- and regime-conditional. |
| **7** | **Continuous Action Extension** | 🟢 **Closed (Exceeded)** | Continuous optimum geometry and implicit perturbation law ($\delta a \simeq -H^{-1}g$, $\cos \approx 0.875, \rho \approx 0.905$, AUPRC $0.9965$) fully verified on continuous control. |
| **8** | **Abstract, Code, Seeds, Presentation** | 🟢 **Closed (Engineering)** | Repository unit tests passing (35/35), deterministic seeds logged across all stages. Abstract revised to 150--200 words reflecting the 3 Canonical Boundaries. |

---

## 7. Strategic Governance Rules & Prohibitions

1. **NO ALGORITHMIC RESCUE BRANCHES:** Do not reopen Stage M2-B or invent ad-hoc reweighting heuristics.
2. **NO OVERREACHING CLAIMS:**
   - Do **NOT** claim "statistical equivalence" without formal preregistered TOST/SESOI equivalence testing.
   - Do **NOT** claim "universal failure across all objective families"; the evidence shows environment and operator conditionality (e.g. Walker2d/HalfCheetah pilot differences).
   - Do **NOT** claim a common causal failure mechanism between value-gradient weighting and gradient influence decay without direct causal proof.
3. **REGIME TRANSPARENCY:** Formally characterize the 10k-step continuous experiments as a matched low-budget MBPO-style stress test / early-training regime.

