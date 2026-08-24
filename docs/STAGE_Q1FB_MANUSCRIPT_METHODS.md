# Stage Q1-FB Manuscript: Methods, Experimental Protocol & Reproducibility Package

**Authors:** Deep Reinforcement Learning Research Group  
**Target Milestone:** Q1 Journal Publication (IEEE T-NNLS / JMLR / Machine Learning)  
**Status:** Confirmatory Cohort Closed Across 3 Environments ($N=10$ paired seeds each, 60 runs total, 100% Integrity Validated)

---

## 1. Methodology & Algorithm Formulation

### 1.1 Problem Setup & Theoretical Context
We consider a continuous-state Markov Decision Process (MDP) defined by $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$, where $\mathcal{S} \subseteq \mathbb{R}^d$, $\mathcal{A} \subseteq \mathbb{R}^m$, $P(s' \mid s, a)$ is unknown transition dynamics, $R(s, a)$ is the reward function, and $\gamma \in [0, 1)$ is the discount factor. The objective is learning a policy $\pi(a \mid s)$ maximizing expected discounted return $J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t R(s_t, a_t) \right]$.

A central conceptual question in model-based reinforcement learning (MBRL) is whether minimizing statistical dynamics error translates to downstream policy improvement. Our framework formalizes the distinction across three decoupled tiers:
$$\boxed{\text{Prediction Error (MLE Loss)} \neq \text{Decision Relevance (Value Sensitivity)} \neq \text{Downstream Learning Utility (Policy Improvement)}}$$

### 1.2 Model-Based Policy Optimization (MBPO) Baseline
In canonical MBPO (Janner et al., 2019), an ensemble of $E=7$ neural networks $\{\hat{P}_{\theta_e}\}_{e=1}^E$ (with $E_{\mathrm{elite}}=5$) models transition dynamics $\hat{P}_\theta(s', r \mid s, a) = \mathcal{N}(\mu_\theta(s, a), \Sigma_\theta(s, a))$. Standard Maximum Likelihood Estimation (MLE) minimizes Gaussian negative log-likelihood (NLL):
$$\mathcal{L}_{\mathrm{MLE}}(\theta) = \sum_{e=1}^E \sum_{i=1}^B \left[ (\mu_{\theta_e}(s_i, a_i) - y_i)^\top \Sigma_{\theta_e}^{-1}(s_i, a_i) (\mu_{\theta_e}(s_i, a_i) - y_i) + \log \det \Sigma_{\theta_e}(s_i, a_i) \right]$$
where $y_i = [s'_i - s_i, r_i]$ is the target transition delta.

### 1.3 Value-Gradient Weighted Model Learning (VaGraM)
Value-aware model learning reformulates the dynamics objective to weight state prediction errors by downstream task criticality. Following Voelcker et al. (ICLR 2022), VaGraM computes the value gradient $g_i = \nabla_{s'} V_\psi(s'_i)$ of the current Soft Actor-Critic (SAC) value function $V_\psi(s') = \mathbb{E}_{a' \sim \pi}[Q_\psi(s', a') - \alpha \log \pi(a' \mid s')]$. To stabilize optimization, a 95% quantile bound clipping is applied:
$$\tilde{g}_i = \begin{cases} g_i & \text{if } \|g_i\|_2 \le q_{0.95}(\{\|g_k\|_2\}) \\ \frac{q_{0.95}(\{\|g_k\|_2\})}{\|g_i\|_2} g_i & \text{otherwise} \end{cases}$$
The coordinate-weighted dynamics loss is then defined as:
$$\mathcal{L}_{\mathrm{VaGraM}}(\theta) = \frac{1}{B} \sum_{i=1}^B \left[ \sum_{j=1}^d \tilde{g}_{i, j}^2 \left( \hat{\mu}_{\theta, j}(s_i, a_i) - (s'_{i, j} - s_{i, j}) \right)^2 + \left( \hat{\mu}_{\theta, r}(s_i, a_i) - r_i \right)^2 \right]$$

---

## 2. Experimental Protocol & Implementation Symmetry

### 2.1 Benchmark Environments & Budget
Experiments evaluate three canonical continuous-control environments with distinct dynamical and contact/termination characteristics:
1. **`Hopper-v4`:** Single-leg contact-truncated locomotion (early termination if torso height $z < 0.7$ or pitch $|\theta| > 0.2$).
2. **`Walker2d-v4`:** Bi-pedal multi-contact truncated locomotion (early termination if torso height $z \notin [0.8, 2.0]$ or pitch $|\theta| > 1.0$).
3. **`HalfCheetah-v4`:** Planar non-terminating continuous contact locomotion (fixed 1,000 interaction steps per episode, testing velocity tracking without early failure truncation).

All environments operate under a fixed canonical interaction budget of $T = 125,000$ real environment steps.

### 2.2 Strict Pipeline Symmetry & Experimental Control
The baseline (MLE) and treatment (VaGraM) branches are **strictly matched on all 21 audited architectural, optimization, data-flow, and evaluation parameters**:
- **Dynamics Model:** 4 hidden layers $\times$ 200 units, ReLU activations, ensemble size $E=7$, elite size $E_{\mathrm{elite}}=5$, Adam optimizer ($\text{lr}=10^{-3}$, weight decay $10^{-5}$), retrained every 250 environment steps.
- **SAC Agent:** Double-Q Critic (2 layers $\times$ 256 units, $\text{lr}=3\times 10^{-4}$), Diagonal Gaussian Actor (2 layers $\times$ 256 units, $\text{lr}=3\times 10^{-5}$), temperature $\alpha$ auto-tuned ($\text{lr}=3\times 10^{-4}$), update ratio $G=20$ SAC updates per environment step.
- **Rollout Schedule:** Linear expansion $[20\mathrm{k}, 150\mathrm{k}, 1, 5]$, retain epochs $=1$, synthetic batch size $=100,000$.
- **Evaluation:** Evaluated every 5,000 environment steps (25 checkpoints total) across 10 deterministic test episodes ($\text{eval\_mode}=\text{True}$).
- **Total Operations:** Exactly 151,035 dynamics gradient updates, 2,480,000 SAC gradient updates, and 388,800 imagined rollouts per 125k run.

### 2.3 Paired-Seed Cohort Design & Replacement Policy
- **Paired Cohorts:** Evaluation is conducted under a pre-specified paired-seed design ($N=10$, seeds $[142, \dots, 151]$) per environment. Identical pseudo-random number generator seeds initialize environment dynamics, network weights, and exploration buffers.
- **Replacement Policy:** A run is designated invalid only upon premature hardware failure/OOM, missing checkpoints, or NaN/Inf. Invalid runs must be rerun from scratch on the identical seed. Low-return runs that pass all integrity checks represent fully valid confirmatory data and are strictly retained. Across all 60 runs, 100% completed validly.

---

## 3. Pre-Specified Statistical Analysis Plan

### 3.1 Primary Confirmatory Estimand: Mean Paired Terminal Return Delta
Let $J_i^{\mathrm{M}}(t)$ and $J_i^{\mathrm{V}}(t)$ denote the deterministic evaluation returns of MLE and VaGraM on seed $i$ at interaction step $t$. The terminal return is defined as the mean performance across the final three checkpoints:
$$\bar{J}_i = \frac{1}{3} \sum_{t \in \{115\mathrm{k}, 120\mathrm{k}, 125\mathrm{k}\}} J_i(t)$$
The primary confirmatory estimand is the mean paired terminal return difference:
$$\bar{\Delta} = \frac{1}{N} \sum_{i=1}^N \Delta_i^{\mathrm{terminal}} = \frac{1}{N} \sum_{i=1}^N \left( \bar{J}_i^{\mathrm{V}} - \bar{J}_i^{\mathrm{M}} \right)$$

### 3.2 Hypothesis Testing & Confidence Bounds
1. **Exact Two-Sided Permutation Sign-Flip Test:** Under the sharp null hypothesis $H_0: \mathbb{E}[\Delta_i] = 0$, we compute the exact permutation $p$-value by evaluating all $2^N = 1024$ possible sign flips across the exact sample magnitudes:
   $$p = \frac{1}{2^N} \sum_{\mathbf{s} \in \{-1, +1\}^N} \mathbb{I}\left( \left| \frac{1}{N} \sum_{i=1}^N s_i \Delta_i \right| \ge |\bar{\Delta}| \right)$$
2. **Paired Percentile Bootstrap CI:** Two-sided 95% confidence intervals for $\bar{\Delta}$ are computed via $B = 10,000$ paired bootstrap resamples.
3. **Budget Interaction Estimand:** $H_i = \Delta_i(125\mathrm{k}) - \Delta_i(10\mathrm{k})$ evaluates the longitudinal shift in paired advantage between early exploration and canonical maturity. Statistical testing on $\bar{H}$ uses exact sign-flip permutation tests, while the number of positive seeds ($H_i > 0$) is reported as a descriptive count.

---

## 4. Confirmatory Tri-Environment Experimental Results (60 Runs Total)

### Table 1: Tri-Environment Confirmatory Results Summary ($N=10$ Paired Seeds per Environment)

| Environment | Dynamic / Contact Character | Mean Terminal $\bar{\Delta}$ (95% CI) | Paired Wins | $p_{\text{exact}}(\bar{\Delta})$ | Budget Shift $\bar{H}$ (95% CI) | $p_{\text{exact}}(\bar{H})$ | Positive Shifts ($H_i > 0$) | Compute Overhead |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hopper-v4** | Contact-Truncated ($z < 0.7$) | **$+620.09$** ($[+485.66, +756.46]$) | **$10 / 10$** | $0.001953$ | $+609.32$ ($[+265.39, +843.58]$) | $0.009766$ | $9 / 10$ | $+5.1\%$ |
| **Walker2d-v4** | Multi-Contact Truncated ($z \notin [0.8, 2.0]$) | **$+239.51$** ($[+133.51, +356.58]$) | **$10 / 10$** | $0.001953$ | $+172.24$ ($[-3.20, +346.35]$) | $0.101562$ | $7 / 10$ | $+4.7\%$ |
| **HalfCheetah-v4** | Non-Terminating Locomotion | **$+3248.94$** ($[+1381.44, +5115.20]$) | **$10 / 10$** | $0.001953$ | $+3108.80$ ($[+1274.87, +4930.62]$) | $0.015625$ | $7 / 10$ | $+4.7\%$ |

*Statistical Reporting Standard:* Across three independently closed confirmatory cohorts, all 30 paired seed–environment comparisons favored VaGraM at the preregistered terminal endpoint. Each environment achieves the minimum attainable two-sided exact permutation $p$-value under $N=10$ ($p = 2/1024 = 0.001953$). $p$-values for budget shifts ($p_{\text{exact}}(\bar{H})$) correspond strictly to continuous permutation tests on $\bar{H}$, with positive shift counts reported as descriptive cohort metrics.

### Table 2: Complete 30-Seed Confirmatory Ledger ($\Delta_i = \bar{J}_i^{\mathrm{V}} - \bar{J}_i^{\mathrm{M}}$)

| Seed $i$ | Hopper-v4 $\Delta_i$ | Walker2d-v4 $\Delta_i$ | HalfCheetah-v4 $\Delta_i$ | Cumulative Verdict |
| :---: | :---: | :---: | :---: | :---: |
| **142** | $+328.51$ | $+263.81$ | $+58.94$ | $3 / 3$ VaGraM Wins |
| **143** | $+767.07$ | $+2.97$ | $+175.73$ | $3 / 3$ VaGraM Wins |
| **144** | $+555.94$ | $+290.42$ | $+6355.07$ | $3 / 3$ VaGraM Wins |
| **145** | $+1067.54$ | $+526.19$ | $+6162.55$ | $3 / 3$ VaGraM Wins |
| **146** | $+274.47$ | $+54.42$ | $+6405.49$ | $3 / 3$ VaGraM Wins |
| **147** | $+444.17$ | $+594.90$ | $+6605.47$ | $3 / 3$ VaGraM Wins |
| **148** | $+640.27$ | $+168.37$ | $+329.50$ | $3 / 3$ VaGraM Wins |
| **149** | $+724.46$ | $+115.70$ | $+302.02$ | $3 / 3$ VaGraM Wins |
| **150** | $+673.41$ | $+244.25$ | $+6087.16$ | $3 / 3$ VaGraM Wins |
| **151** | $+725.04$ | $+134.05$ | $+7.41$ | $3 / 3$ VaGraM Wins |

---

## 5. Mechanistic Synthesis & Core Scientific Claims

### 5.1 Primary Claim: Regime-Conditional Learning Utility
> **Across three independently closed MuJoCo confirmatory cohorts, VaGraM produced positive paired terminal-return differences over MLE for all 30 seed–environment comparisons under a single fixed MBPO configuration without environment-specific retuning. Effect magnitudes varied substantially across environments and seeds, and the advantage was generally stronger at the mature training budget, supporting the conclusion that downstream learning utility is regime- and operator-conditional rather than an intrinsic property of model-error magnitude or decision relevance alone.**

### 5.2 Mechanistic Interpretation & Heterogeneity Across Environments
1. **Regime-Conditional Downstream Policy Returns:** At the mature training budget, value-aware dynamics learning yielded substantially larger downstream policy returns across all three environments. Together with the weaker and more heterogeneous low-budget results, this supports a regime-conditional interpretation of learning utility.
2. **Contact-Truncated vs. Non-Terminating Tasks:** In contact-truncated tasks (Hopper, Walker2d), catastrophic termination penalizes boundary errors heavily, producing consistent positive terminal shifts ($\bar{\Delta} = +620.09$ and $+239.51$). In non-terminating tasks (HalfCheetah), seed effects are strongly bimodal:
   - HalfCheetah exhibited strongly heterogeneous seed-level effects, ranging from $+7.41$ to $+6,605.47$, while all ten paired terminal differences remained positive.
   - *High-Velocity Breakout Cluster ($5/10$ seeds: 144–147, 150):* VaGraM achieved high-speed forward locomotion ($\bar{J}^{\mathrm{V}} \approx +5,700 \to +6,107$), while standard MLE remained trapped in sub-zero returns ($\bar{J}^{\mathrm{M}} \approx -317 \to -498$).
   - *Sub-Zero Advantage Cluster ($5/10$ seeds: 142, 143, 148, 149, 151):* VaGraM consistently attained higher return than MLE ($\Delta_i \in [+7.41, +329.50]$), though neither policy achieved runaway forward velocity by 125k steps.

---

## 6. Publication Figures & Artifact Package

1. **`figures/stage_q1fb_fig1_learning_curves.pdf`:** Hopper-v4 longitudinal learning curves and pointwise 95% bootstrap CIs.
2. **`figures/stage_q1w2_fig1_learning_curves.pdf`:** Walker2d-v4 longitudinal learning curves.
3. **`figures/stage_q1hc_fig1_learning_curves.pdf`:** HalfCheetah-v4 longitudinal learning curves.
4. **`figures/stage_q1_3env_cross_environment_synthesis.pdf`:** Tri-environment paired difference forest plot synthesis (Figure 5).
