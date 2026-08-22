# Exact-Neighbour Literature Matrix & Plausible Mechanisms (2017–2026)

**Document Version:** 2.1.0 (Comprehensive Comparator Edition)  
**Milestone:** Q1 Journal Upgrade — Stage Q1-FB  
**Scope:** Value-Aware Model Learning, Decision-Aware World Models, Dyna-style Co-adaptation, and Training-Budget Interaction.

---

## 1. Structured Literature Comparator Table

| Paper & Citation | Core Model Objective | Benchmark Domain | Training Budget Evaluated | MLE Baseline Included? | Value-Aware Formulation | Full Learning Curves Shown? | Key Mechanism Claimed / Analyzed |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| **Farahmand (2017, 2018)**<br>*(NeurIPS/arXiv)* | Theoretical VAML / Iterative VAML | Inverted Pendulum, Synthetic Linear MDPs | $1\mathrm{k}$–$10\mathrm{k}$ steps | Yes | Value-error upper bound minimization | Partial | Objective mismatch: MLE minimizes loose upper bound on value error. |
| **Janner et al. (2019)**<br>*(NeurIPS)* | MBPO (Standard Gaussian NLL) | MuJoCo Continuous Locomotion (Hopper, Walker, Ant) | $125\mathrm{k}$–$300\mathrm{k}$ steps | Yes (Canonical) | Pure MLE (Reconstruction) | Yes | Compounding model error increases with rollout horizon $h$; branching rollouts ($h=1\to5$). |
| **Grimm et al. (2020, 2021)**<br>*(NeurIPS)* | Value Equivalence Principle | Discrete Gridworlds & Control | $\le 50\mathrm{k}$ steps | Yes | Value-operator equivalence | Yes | Model equivalence classes w.r.t. value functions and policy sets. |
| **Modhe et al. (2021)**<br>*(arXiv:2106.14080)* | Model-Advantage VAML | Continuous Control (SLBO / Dyna) | $50\mathrm{k}$–$100\mathrm{k}$ steps | Yes | Model-advantage bounded VAML | Yes | **Stale value estimates:** Dyna-style co-training causes value gradients to lag policy updates. |
| **Voelcker et al. (2022)**<br>*(ICLR 2022)* | VaGraM (Value-Gradient Weighted MSE) | MuJoCo Locomotion (Hopper, Ant, Walker2d) | $125\mathrm{k}$–$300\mathrm{k}$ steps | Yes | Coordinate-weighted MSE + $q_{0.95}$ bound clipping | Yes | Value-gradient weighting focuses model capacity; quantile clipping prevents gradient explosion. |
| **Efroni et al. (2022)**<br>*(ICML)* | Decision-Aware Model Estimation | Linear Bandits & Bilinear MDPs | Theoretical / Small scale | Yes | Pessimistic / optimistic decision-aware | No (Theory) | Sample efficiency bounds under partial coverage and function approximation. |
| **Lyle et al. (2023)**<br>*(ICLR)* | Capacity Loss in Off-Policy Value Learning | Atari / Deep RL Baselines | Extended horizons | N/A | Critic plasticity & feature rank | Yes | Non-stationary target networks suffer plasticity loss and gradient noise over long horizons. |
| **Our Study (Stage Q1-FB, 2026)** | Canonical VaGraM vs Standard MLE | Gymnasium Hopper-v4 (MuJoCo) | Paired $10\mathrm{k}$ vs $125\mathrm{k}$ interactions | Yes (Direct Paired) | Frozen FB2 VaGraM vs Standard Gaussian NLL | Yes (10 paired seeds) | **Empirical Training-Budget Dependence:** Short-budget parity ($\bar{\Delta}_{10\mathrm{k}} \approx 0$) vs mature-budget separation ($\bar{\Delta}_{125\mathrm{k}} > 0$). |

---

## 2. Plausible Theoretical Mechanisms (Hypothesis Demarcation)

The empirical comparison across 125,000 steps demonstrates clear training-budget dependence. Prior literature identifies four interconnected mechanisms that provide plausible explanations for why value-aware advantages emerge in mature training regimes:

1. **Critic Calibration Dynamics (Modhe et al., 2021):**  
   Early in training ($T \le 25\mathrm{k}$), policy exploration is high and critic estimates $Q(s, a)$ exhibit significant approximation variance. Value gradients $\nabla_{s'} V(s')$ computed from an uncalibrated critic inject noise into dynamics training, preventing statistical separation from standard MLE.
2. **Expanding Rollout Horizon $h: 1 \to 5$ (Janner et al., 2019):**  
   MBPO dynamically increases synthetic rollout length from $h=1$ (at step 20k) to $h=5$ (at step 125k). Under $h=5$, compounding errors along value-irrelevant dimensions degrade policy updates in MLE, whereas VaGraM's inductive bias selectively penalizes drift along value-sensitive state dimensions.
3. **Capacity Allocation under Asymmetric State Coordinates (Farahmand, 2018; Voelcker et al., 2022):**  
   In locomotion dynamics (e.g. Hopper-v4 vertical torso position $z$, pitch angle $\theta$), certain coordinates dictate episode termination and future reward much more heavily than velocity fluctuations. VaGraM concentrates representational capacity on these critical dimensions.

---

## 3. Methodological Governance Rule
In the final journal manuscript, these mechanisms will be framed strictly as **empirically supported plausible hypotheses**, rather than claimed as causally isolated proof, maintaining total integrity against reviewer scrutiny.
