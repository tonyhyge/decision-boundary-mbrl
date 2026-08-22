# Project Evidence Registry: Single Source of Truth

**Milestone:** Phase M1 — Evidence Registry Freeze  
**Date:** 2026-08-20  
**Scope:** Exhaustive registry of all experimental protocols, dataset sizes, seed counts, numerical baselines, primary metrics, confidence intervals, $p$-values, and artifact pointers across the entire thesis project.

---

## 1. Project Global Master Index

```
+-------------------------------------------------------------------------------------------------------------------+
| Stage        | Domain / Task        | Pipeline Purpose                         | Sample Budget / Seeds | Verdict  |
+-------------------------------------------------------------------------------------------------------------------+
| Step 1       | Tabular MDP & Theory | Boundary Sensitivity Theorems            | Exact Analytical DP   | PASS     |
| Step 2       | Hopper-v4 (MBPO)     | Historical Reference Reproduction        | 25k Env Steps, N=5    | PASS     |
| Stage 3A     | CartPole Continuous  | Unprivileged Decision-Risk Ranking       | N=1,881 Eval, B=2,000 | PASS     |
| Stage 4A     | CartPole Continuous  | Targeted World-Model Active Acquisition  | Sweep N=5, Conf N=10  | PASS     |
| Stage 3B     | Continuous CartPole  | Action-Geometry & Perturbation Law       | N=1,500 Eval, B=2,000 | PASS     |
| Stage 4B.0   | Continuous CartPole  | Curvature Closure Pre-Gate               | N=1,250 Eval, N=10    | NOT EST. |
| Stage 4B.1   | Hopper-v4 (MBPO)     | One-Shot Model Repair Benchmark          | B=200, N=10 Seeds     | NEGATIVE |
| Stage 4B.1-R | Hopper-v4 (MBPO)     | Repairability Closure & Decomposition    | 5 Controls, N=10 Seeds| CLOSED   |
| Stage 4B.2   | Hopper-v4 (MBPO)     | Closed-Loop Active MBPO                  | —                     | LOCKED   |
+-------------------------------------------------------------------------------------------------------------------+
```

---

## 2. Granular Evidence Registry by Stage

### Stage 1: Mathematical Theory of Decision Boundaries
* **Artifacts:** [`src/envs/tabular_mdp.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/src/envs/tabular_mdp.py), [`tests/test_tabular_dp.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/tests/test_tabular_dp.py)
* **Domain:** Exact Tabular Markov Decision Processes with analytical Dynamic Programming.
* **Key Theoretical Findings:**
  1. *Prediction Error vs Control Regret Decoupling:* There exist model perturbations $\delta P_1, \delta P_2$ with $\|\delta P_1\| < \|\delta P_2\|$ where control performance loss under $\delta P_1$ strictly exceeds $\delta P_2$ due to decision boundary crossing ($m(s) < \|\delta Q\|$).
  2. *Action Margin Invariance:* The decision boundary is the zero-level set $\{s : Q(s, a_1) = Q(s, a_2)\}$. Vulnerability to ranking reversal scales inversely with the action margin $m(s)^{-1}$.
* **Status:** CLOSED / PASS.

---

### Stage 2: Historical Reference MBPO Reproduction
* **Artifacts:** [`src/baselines/adapters/vagram_mbpo.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/src/baselines/adapters/vagram_mbpo.py), [`results/step2_historical_mbpo_reproduction.json`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/results/step2_historical_mbpo_reproduction.json)
* **Domain:** `Hopper-v4` (MuJoCo continuous control benchmark, 11D observation, 3D action).
* **Protocol & Sample Counts:** Truncated reference reproduction run of $25,000$ environment steps ($100,000$ model updates, rollout horizon $H=1-15$, ensemble size $K=7$), evaluated across $N=5$ independent seeds (`[42, 43, 44, 45, 46]`).
* **Numerical Results:**
  * Mean Return: $1,241.8 \pm 386.4$ (Seed Returns: `[1420.5, 1180.2, 1610.4, 690.1, 1307.8]`).
  * Demonstrates qualitative learning dynamics and unstable/falling modes characteristic of early-phase MBPO without claiming full $500\text{k}$-step asymptotic convergence.
* **Status:** CLOSED / PASS.

---

### Stage 3A: Unprivileged Decision-Risk Ranking on CartPole
* **Artifacts:** [`src/detection/signal_extractor.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/src/detection/signal_extractor.py), [`docs/STAGE3A_DETECTOR_FALSIFICATION_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE3A_DETECTOR_FALSIFICATION_REPORT.md), [`results/stage3a_closure_patch_report.json`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/results/stage3a_closure_patch_report.json)
* **Domain:** Continuous CartPole (4D state, 2 discrete actions $\pm 10$ N).
* **Dataset Splits:**
  * Initial dynamics training $\mathcal{D}_0$: $N_0 = 1,416$ transitions ($25$ exploratory rollouts).
  * Held-out calibration split: $N_{\mathrm{cal}} = 1,600$ states ($25$ trajectories).
  * Held-out test split: $N_{\mathrm{test}} = 1,881$ states ($24$ independent trajectories, $37$ positive flip states, natural test prevalence $= 1.97\%$).
* **Statistical Resampling:** $B = 2,000$ clustered trajectory bootstrap replicates. Permutation null test with $M = 200$ full pipeline permutations.
* **Key Numerical Metrics (Held-Out Test Set):**
  * **Prevalence Base Rate (Null Baseline):** $\text{AUPRC}_{\mathrm{null}} = 0.0197$, $\text{Recall@10\%}_{\mathrm{null}} = 10.20\% \pm 4.99\%$.
  * **Frozen Action Disagreement Comparator ($H_{\mathrm{vote}}$):** $\text{AUPRC} = 0.4168$, $\text{Recall@10\%} = 59.46\%$, $\text{Brier} = 0.0122$.
  * **Combined Decision-Risk Ranker:** $\text{AUPRC} = \mathbf{0.5678}$, $\text{Recall@10\%} = \mathbf{89.19\%}$ ($33 / 37$ flips captured at $10\%$ budget), $\text{Brier} = 0.0122$.
  * **Paired Comparison vs Action Disagreement Comparator:**
    * $\Delta\text{AUPRC} = \mathbf{+0.1510}$ ($95\%\text{ BCI}: [+0.0617, +0.2380], \mathbf{p = 0.0010}$).
    * $\Delta\text{Recall@10\%} = \mathbf{+29.73\%}$ ($95\%\text{ BCI}: [+10.81\%, +48.65\%], \mathbf{p = 0.0020}$).
    * $\Delta\text{Brier} = -0.00001$ ($p = 0.9600 \implies$ Calibration Parity / Superiority Not Established).
  * **Full Permutation Test:** $p_{\mathrm{perm}} = 1 / (200 + 1) = 1/201 = \mathbf{0.004975}$.
  * **Mechanistic Subpopulation ($H_{\mathrm{vote}} = 0$, $N=1,845$, $16$ flips):** Margin-only achieves $\text{AUPRC} = 0.2967, \text{AUROC} = 0.8826$ vs Combined $\text{AUPRC} = 0.2042$, demonstrating continuous boundary proximity retains risk resolution when ensemble voting is unanimous.
* **Standardized Logistic Regression Association:** Standardized coefficient on interaction $\sigma_{\mathrm{dyn}} / (\hat{m} + \varepsilon)$ is $\beta_{\mathrm{ratio}} = +1.3204$ vs $\beta_{\mathrm{margin}} = +0.7133$ and raw dynamics uncertainty $\beta_{\mathrm{dyn}} = +0.0611$.
* **Status:** CLOSED / PASS with narrowed claim: **Unprivileged Decision-Risk Ranker**.

---

### Stage 4A: Targeted World-Model Active Acquisition on CartPole
* **Artifacts:** [`src/acquisition/repair_trainer.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/src/acquisition/repair_trainer.py), [`docs/STAGE4A_ACTIVE_ACQUISITION_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE4A_ACTIVE_ACQUISITION_REPORT.md), [`results/stage4a_acquisition_sweep.csv`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/results/stage4a_acquisition_sweep.csv)
* **Domain:** Continuous CartPole (4D state, 2 actions).
* **Datasets:** Base $\mathcal{D}_0$ ($N_0 = 1,389$), Candidate pool $\mathcal{D}_{\mathrm{pool}}$ ($N_{\mathrm{pool}} = 1,612$, $26$ flips), Held-out evaluation set $\mathcal{D}_{\mathrm{eval}}$ ($N_{\mathrm{eval}} = 1,444$, baseline flips before repair $= 17$).
* **Model Retraining:** Continual warm-start fine-tuning ($30$ epochs, Adam lr $0.001$, batch size $64$, strictly matched gradient budgets).
* **Confirmatory 10-Seed Benchmark at $B=10\%$ ($161$ queries, $N=10$ seeds):**
  * Baseline Flips: $17 / 1,444$ ($1.18\%$ prevalence).
  * Flip-Oracle Acquisition: $16.2 \pm 0.7$ flips ($\Delta = -0.8, \eta = +0.0050$, Gate G4A.1 PASS).
  * Random Acquisition: $15.9 \pm 1.1$ flips ($\eta = +0.0068$).
  * Combined Decision-Risk Ranker: $\mathbf{14.4 \pm 0.8}$ flips ($\Delta = \mathbf{-2.6}, \eta = \mathbf{+0.0161}$).
  * Paired Gain ($\eta_{\mathrm{Combined}} - \eta_{\mathrm{Random}}$): $\mathbf{+0.0093}$ flips/query ($t(9) = 2.764, \mathbf{p = 0.0220 < 0.05}$, Gate G4A.2 PASS).
* **4-Budget Sweep ($N=5$ Seeds/Configuration):**
  * At $B=1\%$ ($16$ queries): Dynamics achieves highest initial efficiency $\eta = +0.1875$ ($14.0 \pm 0.6$ flips) vs Combined $\eta = +0.1000$ ($15.4 \pm 0.5$ flips).
  * At $B=5\%$ ($81$ queries): Combined achieves highest flip reduction ($-16.5\%$, $14.2 \pm 1.2$ flips, $\eta = \mathbf{0.0346}$, **$2.8\times$ over Random** $\eta = 0.0123$).
  * At $B=10\%$ ($161$ queries): Combined achieves $-12.9\%$ reduction ($14.8 \pm 1.3$ flips, $\eta = \mathbf{0.0137}$, **$11.4\times$ over Random** $\eta = 0.0012$).
  * At $B=20\%$ ($322$ queries): Combined achieves $-11.8\%$ reduction ($15.0 \pm 0.0$ flips, $\eta = 0.0062$ vs Random $\eta = 0.0025$).
* **Downstream Return Audit (G4A.4):** Policy return post-repair is $115.8 \pm 4.0$ (Combined) vs $114.2 \pm 4.1$ (Random). Characterized as bounded non-degradation / secondary consistency in a near-saturated CartPole baseline ($118.5 \pm 42.9$ steps).
* **Status:** CLOSED / PASS — **Rank-Based Targeted Acquisition Supported on CartPole**.

---

### Stage 3B: Continuous-Action Geometry & Implicit Perturbation Law
* **Artifacts:** [`src/continuous_geometry/implicit_perturbation.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/src/continuous_geometry/implicit_perturbation.py), [`docs/STAGE3B_CONTINUOUS_GEOMETRY_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE3B_CONTINUOUS_GEOMETRY_REPORT.md)
* **Domain:** Continuous CartPole with continuous force $a \in [-1.0, 1.0]$ and exact LQR / Riccati solver.
* **Mathematical Law:** First-order implicit displacement $\delta a \approx - H_a^{-1} g(s)$ with $g = \nabla_a \delta Q$ and Hessian $H_a = \nabla_a^2 Q_P(s, a^*)$. Decision regret $R_{\mathrm{dec}}(s) \approx \frac{1}{2} g(s)^\top (-H_a)^{-1} g(s)$.
* **Phase 3B.0 (Matched Error Sanity, $N=500$ State Pairs):** Under identical $\|\delta f\|$, displacement in flat optimum ($R_{\mathrm{cost}} = 0.01$) is $0.4561$ rad vs sharp optimum ($R_{\mathrm{cost}} = 0.50$) $0.0905$ rad (**$5.04\times$ Flatness Amplification**, $t = 25.058, \mathbf{p = 2.56 \times 10^{-90}}$).
* **Phase 3B.1 (Perturbation Law Audit, $N=1,000$ States):**
  * Directional Sign Match: $\mathbf{99.9\%}$.
  * Vector Cosine Alignment: $\mathbf{0.8747}$.
  * Spearman Rank Correlation: $\rho = \mathbf{0.9048} \quad (p < 10^{-5})$.
  * Median Relative Error: $\mathbf{5.82\%}$.
* **Phase 3B.2 & 3B.3 (Held-Out Test Ranking, $N=1,500$ States, Severe Regret Prevalence $10.0\%$):**
  * Dynamics Uncertainty $\sigma_{\mathrm{dyn}}$: $\text{AUPRC} = 0.4971$, $\text{AUROC} = 0.4973$, $\text{Recall@10\%} = 46.0\%$.
  * Action Disagreement $\sigma_a$: $\text{AUPRC} = 0.9957$, $\text{AUROC} = 0.9995$, $\text{Recall@10\%} = 96.7\%$.
  * Action-Gradient Disagreement $\sigma_{\nabla_a Q}$: $\text{AUPRC} = \mathbf{0.9965}$, $\text{AUROC} = \mathbf{0.9996}$, $\text{Recall@10\%} = \mathbf{96.0\%}$.
  * Proposed Continuous Score $S_{\mathrm{cont}}$: $\text{AUPRC} = \mathbf{0.9965}$, $\text{AUROC} = \mathbf{0.9996}$, $\text{Recall@10\%} = \mathbf{96.0\%}$.
  * Shuffled Curvature Control: $\text{AUPRC} = 0.9966$, $\text{AUROC} = 0.9996$.
  * Permuted Signal Null: $\text{AUPRC} = 0.1012$, $\text{AUROC} = 0.5046$.
  * Paired Bootstrap ($B=2,000$ replicates): $\Delta\text{AUPRC}(S_{\mathrm{cont}} - \sigma_{\mathrm{dyn}}) = \mathbf{+0.5038}$ ($95\%\text{ BCI}: [+0.1687, +0.8272], \mathbf{p_{\mathrm{boot}} = 1/2001 \approx 4.998 \times 10^{-4} < 0.0005}$).
* **Status:** CLOSED / PASS (Continuous geometric mechanism and ranking confirmed; added value of curvature in deployed score: **NOT ESTABLISHED / UNPROVEN**).

---

### Stage 4B.0: Curvature Contribution Closure Pre-Gate
* **Artifacts:** [`experiments/active_mbpo/run_stage4b0_curvature_closure.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/active_mbpo/run_stage4b0_curvature_closure.py), [`results/stage4b0_curvature_closure_report.json`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/results/stage4b0_curvature_closure_report.json)
* **Domain:** Continuous CartPole ($N_{\mathrm{pool}} = 1,500, N_{\mathrm{eval}} = 1,250, B = 10\%, N = 10\text{ seeds}$).
* **Empirical Regret Comparison Post-Repair:**
  * Action-Gradient Disagreement $\sigma_{\nabla_a Q}$: $0.168707 \pm 0.012669$.
  * $S_{\mathrm{cont}}$: $0.168427 \pm 0.013267$ (Paired $t = -0.565, \mathbf{p = 0.5858}$).
  * Shuffled-Curvature Control: $0.169263 \pm 0.011542$ (Paired $t = -1.016, \mathbf{p = 0.3362}$).
* **Gate Verdict:** **NOT ESTABLISHED**. Canonical intervention score permanently frozen to **Action-Gradient Epistemic Disagreement ($\sigma_{\nabla_a Q}$)**.

---

### Stage 4B.1 & Stage 4B.1-R: Model Repairability Closure on Hopper-v4
* **Artifacts:** [`experiments/active_mbpo/run_stage4b1r_repairability_closure.py`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/experiments/active_mbpo/run_stage4b1r_repairability_closure.py), [`docs/STAGE4B_REPAIRABILITY_CLOSURE_REPORT.md`](file:///Users/minhpham/Documents/Tony/Giấy tờ cá nhân/on my own /Research/DRL/drl-thesis/docs/STAGE4B_REPAIRABILITY_CLOSURE_REPORT.md)
* **Domain:** `Hopper-v4` (MuJoCo, 11D observation, 3D action).
* **Protocol Reconciliation (Stage 4B.1 vs Stage 4B.1-R):**
  * *Stage 4B.1 Run:* Trained on base $\mathcal{D}_0$ ($N_0 = 714$ transitions from $30$ rollouts), candidate pool $N_{\mathrm{pool}} = 709$, held-out test $N_{\mathrm{eval}} = 453$ transitions from $20$ rollouts. Initial baseline $H=5$ rollout error $= 0.068075$.
  * *Stage 4B.1-R Run:* Trained on independently generated base $\mathcal{D}_0$ ($N_0 = 830$ transitions from $35$ rollouts), candidate pool $N_{\mathrm{pool}} = 807$, held-out test $N_{\mathrm{eval}} = 555$ transitions from $25$ rollouts. Initial baseline $M_0$ $H=5$ rollout error $= \mathbf{0.063825}$. Both runs are distinct, independently logged experiments.
* **Stage 4B.1-R Causal Decomposition ($N=10$ Seeds, $B=200$, 40 Matched Epochs):**
  * **1. No-Update Baseline ($M_0$):** $H=5$ Error $= \mathbf{0.063825 \pm 0.000000}$, 1-Step MSE $= 0.021881$.
  * **2. $\mathcal{D}_0$-Only Extra-SGD Control (0 new data):** $H=5$ Error $= \mathbf{0.069437 \pm 0.001692}$ ($\mathbf{+8.79\%}$ degradation, $t = 9.949, \mathbf{p = 3.73 \times 10^{-6}}$). Confirms pure parameter update drift.
  * **3. Random Acquisition ($B=200$):** $H=5$ Error $= 0.069983 \pm 0.001412$ ($+9.65\%$).
  * **4. Action-Gradient Disagreement ($\sigma_{\nabla_a Q}$):** $H=5$ Error $= 0.072617 \pm 0.000781$ ($+13.78\%$).
  * **5. Dynamics Uncertainty ($\sigma_{\mathrm{dyn}}$):** $H=5$ Error $= 0.072870 \pm 0.001484$ ($+14.17\%$).
  * **6. Privileged Residual Acquisition ($B=200$):** $H=5$ Error $= \mathbf{0.070022 \pm 0.001298}$ ($\mathbf{+9.71\%}$ degradation, $t = 15.097, \mathbf{p = 1.69 \times 10^{-7}}$). Even ground-truth 1-step residual selection degrades under this update operator.
  * **7. Full-Pool Retraining Control ($N=807$):** $H=5$ Error $= \mathbf{0.062913 \pm 0.001752}$ ($\mathbf{-1.43\%}$ net improvement, 1-step MSE $= 0.021016$).
* **Two-Component Decomposition Relative to $\mathcal{D}_0$-Only Drift:**
  $$\Delta_{\mathrm{total}} = \underbrace{\Delta_{\mathrm{update}}}_{\text{D0-only vs M0 (+0.005612)}} + \underbrace{\Delta_{\mathrm{acquisition}}}_{\text{Acquisition vs D0-only}}$$
  * Random: $\Delta_{\mathrm{acq}} = +0.000546$.
  * Privileged Residual: $\Delta_{\mathrm{acq}} = +0.000585$.
  * Action-Gradient Disagreement: $\Delta_{\mathrm{acq}} = +0.003180$.
  * Dynamics Uncertainty: $\Delta_{\mathrm{acq}} = +0.003433$.
  * Full-Pool Retraining: $\Delta_{\mathrm{acq}} = \mathbf{-0.006524}$ (overcomes drift barrier).
* **State Distribution Diagnostic:** $100\%$ of evaluated states in non-terminal exploration rollouts have $z \ge 1.0$ (mean $z \approx 1.22$), explaining identical full vs in-distribution 1-step MSE ($0.021881$).
* **Final Verdict:** **STAGE 4B CLOSED (Case C: Optimization drift dominates one-shot Hopper repair under this protocol) -> STAGE 4B.2 PERMANENTLY LOCKED -> EXPERIMENTS FROZEN.**
