# Stage Q1-FB Manuscript: Methods, Experimental Protocol & Reproducibility Package

**Authors:** Deep Reinforcement Learning Research Group  
**Target Milestone:** Q1 Journal Publication (IEEE T-NNLS / JMLR / Machine Learning)  
**Status:** Pre-Closure Analysis Freeze (Results set to `TBD` pending completion of cohort $N=10$)

---

## 1. Methodology & Algorithm Formulation

### 1.1 Problem Setup
We consider a Markov Decision Process (MDP) defined by the tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$, where $\mathcal{S} \subseteq \mathbb{R}^d$ is the continuous state space, $\mathcal{A} \subseteq \mathbb{R}^m$ is the continuous action space, $P(s' \mid s, a)$ is the unknown transition dynamics, $R(s, a)$ is the reward function, $\gamma \in [0, 1)$ is the discount factor, and $\rho_0$ is the initial state distribution. The objective is to learn a policy $\pi(a \mid s)$ maximizing expected discounted return $J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \right]$.

### 1.2 Model-Based Policy Optimization (MBPO) Baseline
In canonical MBPO (Janner et al., 2019), an ensemble of $E=7$ probabilistic neural networks $\{\hat{P}_{\theta_e}\}_{e=1}^E$ (with $E_{\mathrm{elite}}=5$) models the transitions as Gaussian distributions $\hat{P}_\theta(s', r \mid s, a) = \mathcal{N}(\mu_\theta(s, a), \Sigma_\theta(s, a))$. Standard Maximum Likelihood Estimation (MLE) minimizes the Gaussian negative log-likelihood (NLL):
$$\mathcal{L}_{\mathrm{MLE}}(\theta) = \sum_{e=1}^E \sum_{i=1}^B \left[ (\mu_{\theta_e}(s_i, a_i) - y_i)^\top \Sigma_{\theta_e}^{-1}(s_i, a_i) (\mu_{\theta_e}(s_i, a_i) - y_i) + \log \det \Sigma_{\theta_e}(s_i, a_i) \right]$$
where $y_i = [s'_i - s_i, r_i]$ is the target transition vector.

### 1.3 Value-Gradient Weighted Model Learning (VaGraM)
Value-aware model learning reformulates the dynamics fitting objective to weight state prediction errors by their downstream task criticality. Following Voelcker et al. (ICLR 2022), VaGraM computes the value gradient $g_i = \nabla_{s'} V_\psi(s'_i)$ of the current Soft Actor-Critic (SAC) value function $V_\psi(s') = \mathbb{E}_{a' \sim \pi}[Q_\psi(s', a') - \alpha \log \pi(a' \mid s')]$. To prevent gradient explosion, a 95% quantile bound clipping is applied:
$$\tilde{g}_i = \begin{cases} g_i & \text{if } \|g_i\|_2 \le q_{0.95}(\{\|g_k\|_2\}) \\ \frac{q_{0.95}(\{\|g_k\|_2\})}{\|g_i\|_2} g_i & \text{otherwise} \end{cases}$$
The coordinate-weighted dynamics loss is then defined as:
$$\mathcal{L}_{\mathrm{VaGraM}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left[ \sum_{j=1}^d \tilde{g}_{i, j}^2 \left( \hat{\mu}_{\theta, j}(s_i, a_i) - (s'_{i, j} - s_{i, j}) \right)^2 + \left( \hat{\mu}_{\theta, r}(s_i, a_i) - r_i \right)^2 \right]$$

---

## 2. Experimental Protocol & Implementation Symmetry

### 2.1 Benchmark Environment & Budget
All experiments are conducted on `Hopper-v4` (Gymnasium v1.0 / MuJoCo 3.x) over a canonical interaction budget of $T = 125,000$ environment steps.

### 2.2 Pipeline Symmetry & Experimental Control
The baseline (MLE) and treatment (VaGraM) branches are **strictly matched on all 21 audited architectural, optimization, data-flow, and evaluation parameters**:
- **Dynamics Model:** 4 hidden layers $\times$ 200 units, ReLU activations, ensemble size $E=7$, elite size $E_{\mathrm{elite}}=5$, Adam optimizer ($\text{lr}=10^{-3}$, weight decay $10^{-5}$), trained every 250 environment steps.
- **SAC Agent:** Double-Q Critic (2 layers $\times$ 256 units, $\text{lr}=3\times 10^{-4}$), Diagonal Gaussian Actor (2 layers $\times$ 256 units, $\text{lr}=3\times 10^{-5}$), temperature $\alpha$ auto-tuned ($\text{lr}=3\times 10^{-4}$), update ratio $G=20$ SAC updates per environment step.
- **Rollout Schedule:** Linear expansion $[20\mathrm{k}, 150\mathrm{k}, 1, 5]$, retain epochs $=1$, synthetic batch size $=100,000$.
- **Evaluation:** Evaluated every 5,000 environment steps (25 checkpoints total) across 10 deterministic test episodes ($\text{eval\_mode}=\text{True}$).
- **Total Operations:** Exactly 151,035 dynamics gradient updates, 2,480,000 SAC gradient updates, and 388,800 imagined rollouts per 125k run.

### 2.3 Paired-Seed Cohort Design
To eliminate inter-seed environmental variance, evaluation is conducted under a **pre-specified paired-seed design** ($N=10$, seeds $[142, \dots, 151]$). For each seed $s$, identical pseudo-random number generator initializations are supplied to environment dynamics, model parameter initialization, and policy exploration buffers.

### 2.4 Explicit Failure and Replacement Policy
To prevent selective attrition or survival bias:
1. **Invalidity Criteria:** A run is designated invalid if and only if it incurs:
   - Premature hardware / process termination or CUDA Out-Of-Memory (OOM).
   - Missing terminal evaluation checkpoints (115k, 120k, 125k).
   - Numerical instability (NaN or Inf in losses, predictions, or evaluations).
   - Configuration / code checksum mismatch.
   - Incomplete update counts ($<151,035$ dynamics updates or $<2,480,000$ SAC updates).
   - Truncated or corrupted output artifacts.
2. **Replacement Action:** Any invalid run must be **rerun from scratch using the exact same seed and algorithm configuration** under the frozen environment.
3. **Prohibition on Substitution:** Under no circumstances will an invalid run be replaced with a different seed (e.g. seed 152), nor will individual checkpoints be patched ad-hoc.
4. **Validity of Low-Return Runs:** A run that executes completely and passes all integrity checks but achieves low return (e.g. baseline collapse) is **fully valid confirmatory data** and is strictly retained without rerunning.

---

## 3. Pre-Specified Statistical Analysis Plan

> **Governance Notice:** The final analysis code, estimands, statistical tests, figure-generation scripts, and integrity verification gates were frozen under immutable hash anchors prior to completing the pre-specified $N=10$ confirmatory cohort.

### 3.1 Primary Confirmatory Estimand: Mean Paired Terminal Return Delta
Let $J_i^{\mathrm{M}}(t)$ and $J_i^{\mathrm{V}}(t)$ denote the deterministic evaluation returns of MLE and VaGraM on seed $i$ at interaction step $t$. The terminal return is defined as the mean performance across the final three checkpoints:
$$\bar{J}_i = \frac{1}{3} \sum_{t \in \{115\mathrm{k}, 120\mathrm{k}, 125\mathrm{k}\}} J_i(t)$$
The primary confirmatory estimand is the mean paired terminal return difference:
$$\bar{\Delta} = \frac{1}{N} \sum_{i=1}^N \Delta_i^{\mathrm{terminal}} = \frac{1}{N} \sum_{i=1}^N \left( \bar{J}_i^{\mathrm{V}} - \bar{J}_i^{\mathrm{M}} \right)$$

### 3.2 Hypothesis Testing & Confidence Bounds
1. **Exact Two-Sided Permutation Sign-Flip Test:** Under the sharp null hypothesis $H_0: \mathbb{E}[\Delta_i] = 0$, we compute the exact permutation $p$-value by evaluating all $2^N = 1024$ possible sign flips across the exact sample magnitudes:
   $$p = \frac{1}{2^N} \sum_{\mathbf{s} \in \{-1, +1\}^N} \mathbb{I}\left( \left| \frac{1}{N} \sum_{i=1}^N s_i \Delta_i \right| \ge |\bar{\Delta}| \right)$$
2. **Paired Percentile Bootstrap CI:** A two-sided 95% confidence interval for $\bar{\Delta}$ is computed via $B = 10,000$ paired bootstrap resamples.
3. **Learning Curve Uncertainty:** Error bands displayed on longitudinal learning curves $J(t)$ represent **pointwise percentile bootstrap 95% confidence intervals** across seeds at each checkpoint $t$.

### 3.3 Secondary Endpoint: Training-Budget Advantage Shift
To evaluate how the paired advantage shifts between early exploration ($10\mathrm{k}$ steps) and canonical maturity ($125\mathrm{k}$ steps), we define the secondary interaction endpoint:
$$H_i = \Delta_i(125\mathrm{k}) - \Delta_i(10\mathrm{k}) = \left( J_i^{\mathrm{V}}(125\mathrm{k}) - J_i^{\mathrm{M}}(125\mathrm{k}) \right) - \left( J_i^{\mathrm{V}}(10\mathrm{k}) - J_i^{\mathrm{M}}(10\mathrm{k}) \right)$$
- **Multiplicity Hierarchy:** The primary terminal return difference $\bar{\Delta}$ is the sole confirmatory endpoint for statistical inference. $H_i$ serves strictly as a secondary descriptive endpoint to characterize training-budget interaction and cannot override primary endpoint conclusions. No formal multiplicity adjustments are applied across primary and secondary endpoints.

---

## 4. Hardware, Provenance & Reproducibility Package

- **Hardware Platform:** NVIDIA GeForce RTX 5090 (Blackwell architecture, compute capability `sm_120`), 32 GB VRAM, Driver 580.167.08.
- **Software Stack:** Linux OS, Python 3.11.15, PyTorch 2.13.0+cu130, CUDA 13.0, cuDNN 91002.
- **Numerical Determinism:** `TF32 = False` (forcing pure IEEE 754 float32 precision on Tensor Cores), `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`.
- **Cross-Device Numerical Invariant Audit:** Verified that mathematical operators between CPU and CUDA backends under pure FP32 exhibit gradient norm-relative error $< 1.5 \times 10^{-7}$ and cosine similarity $> 0.999999$.
- **Execution Gate:** Statistical closure scripts require mandatory verification of the full run integrity audit across all 20 JSON artifacts (`FB3_COHORT_VALIDITY_PASS.txt`) prior to final inference generation.

---

## 5. Confirmatory Experimental Results (Placeholders for Closure)

*Note: The following table and summary will be populated upon completion of the entire 10-paired cohort ($N=10$).*

### Table 1: Final Confirmatory Results on Hopper-v4 ($125,000$ Steps)

| Seed $i$ | MLE Terminal $\bar{J}^{\mathrm{M}}_i$ | VaGraM Terminal $\bar{J}^{\mathrm{V}}_i$ | Paired Delta $\Delta_i$ | VaGraM Win? | Early Delta $\Delta_i(10\mathrm{k})$ | Late Delta $\Delta_i(125\mathrm{k})$ | Budget Shift $H_i$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **142** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **143** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **144** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **145** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **146** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **147** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **148** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **149** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **150** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |
| **151** | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` | `TBD` |

### Primary Statistical Inferences ($N=10$ Confirmatory Cohort):
- **Mean Paired Terminal Return Delta $\bar{\Delta}$:** `TBD`
- **95% Paired Percentile Bootstrap CI:** `[TBD, TBD]`
- **Exact Permutation Sign-Flip $p$-value ($2^{10} = 1024$):** `p = TBD`
- **Confirmatory Win Rate:** `TBD / 10 (TBD%)`
- **Training-Budget Advantage Shift Mean $\bar{H}$:** `TBD` (95% CI: `[TBD, TBD]`, `p = TBD`)
- **Descriptive Runtime Overhead:** `+6.7%` under audited concurrent CUDA execution.
