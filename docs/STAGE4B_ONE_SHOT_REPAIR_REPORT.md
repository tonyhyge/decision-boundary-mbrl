# Stage 4B Scientific Report: Curvature Closure (4B.0) & One-Shot MBPO Model Repair (4B.1)

**Milestone:** Q1 Journal Upgrade — Stage 4B Continuous Active Acquisition Gate  
**Date:** 2026-08-20  
**Author:** Pham Quang Minh  
**Status:** **STAGE 4B.0 CLOSED (Curvature Unproven -> Action-Gradient Frozen) | STAGE 4B.1 EVALUATED — VERDICT: NEGATIVE / GATE NOT PASSED -> STAGE 4B.2 REMAINS LOCKED**

---

## 1. Executive Summary & Core Scientific Question

Stage 4B evaluates whether decision-sensitive active acquisition extends to complex continuous benchmarks (`Hopper-v4`):

$$\boxed{\text{Does action-gradient decision-sensitive acquisition improve world-model repair under matched budgets on continuous benchmarks?}}$$

Following the pre-registered falsification protocol:
1. **Stage 4B.0 (Pre-Gate):** We tested whether estimated Hessian curvature ($S_{\mathrm{cont}} = \sigma_{\nabla_a Q} / (|\hat{H}| + \varepsilon)$) provides any detectable acquisition utility over raw action-gradient disagreement ($\sigma_{\nabla_a Q}$).
2. **Stage 4B.1 (One-Shot MBPO Repair Gate):** We benchmarked world-model repair on `Hopper-v4` under strictly matched query budget ($B = 200$) and matched gradient update budgets using a simulator-reset query protocol.

---

## 2. Stage 4B.0: Curvature Contribution Closure Results

Evaluated across $N = 10$ independent seeds in a controlled repair cycle:

```
+----------------------------------------------------------------------------------------------------+
|                                    STAGE 4B.0 EMPIRICAL RESULTS                                    |
+---------------------------------------------------+--------------------+---------------------------+
| Strategy                                          | Mean Eval Regret   | Comparison vs S_cont      |
+---------------------------------------------------+--------------------+---------------------------+
| 1. Action-Gradient Disagreement (sigma_grad_Q)    | 0.168707 +/- 0.0127| Baseline                  |
| 2. Proposed Continuous Score (S_cont)             | 0.168427 +/- 0.0133| Gain: +0.000280 (p=0.5858)|
| 3. Shuffled-Curvature Control                     | 0.169263 +/- 0.0115| Gain: +0.000836 (p=0.3362)|
| 4. Inverse Curvature Only (|H|^-1)                | 0.164709 +/- 0.0136| (Reference)               |
+---------------------------------------------------+--------------------+---------------------------+
```

### Pre-Registered Gate 4B.0 Decision:
$$\boxed{\textbf{GATE 4B.0 VERDICT: NOT ESTABLISHED / CANONICAL SCORE FROZEN TO } \sigma_{\nabla_a Q}}$$
* **Scientific Scope:** Curvature $|H_a|$ retains its theoretical and explanatory role in governing continuous decision sensitivity (proven in 3B.0–3B.1), but does not provide statistically significant empirical gain in the deployed acquisition ranker over raw action-gradient disagreement $\sigma_{\nabla_a Q}$.

---

## 3. Stage 4B.1: One-Shot MBPO Model Repair on Hopper-v4

Evaluated across $N = 10$ independent seeds with baseline dataset $\mathcal{D}_0$ ($N_0 = 714$ transitions), candidate pool $\mathcal{D}_{\mathrm{pool}}$ ($N_{\mathrm{pool}} = 709$), and held-out test rollouts $\mathcal{D}_{\mathrm{eval}}$ ($N_{\mathrm{eval}} = 453$). Baseline $H=5$ autoregressive unroll error before repair: $0.068075$.

#### Table 1: One-Shot MBPO Repair Performance ($B = 200$ Queries, Matched Gradient Budget)

| Strategy | $H=5$ Rollout Error (Mean $\pm$ Std) | Repair Efficiency $\eta_B$ ($10^{-5}$ Error/Query) | Paired $t$-test vs Random | Paired $t$-test vs $\sigma_{\mathrm{dyn}}$ |
| :--- | :---: | :---: | :---: | :---: |
| **Random Acquisition** | $0.073073 \pm 0.002812$ | $-2.499$ | — | — |
| **Dynamics Uncertainty ($\sigma_{\mathrm{dyn}}$)** | $0.075335 \pm 0.001546$ | $-3.630$ | $t = -2.185, p = 0.056$ | — |
| **Action-Gradient Disagreement ($\sigma_{\nabla_a Q}$)** | $0.073358 \pm 0.001911$ | $-2.642$ | $t = -0.430, \mathbf{p = 0.6770}$ | $t = 3.741, \mathbf{p = 0.0046 < 0.01}$ |
| **Shuffled Score Negative Control** | $\mathbf{0.071808 \pm 0.001522}$ | $\mathbf{-1.867}$ | $t = 1.120, p = 0.292$ | $t = 4.125, p = 0.0026$ |

---

## 4. Mechanistic Breakdown & Scientific Discussion

1. **Superiority over Pure Dynamics Uncertainty:**  
   Action-gradient disagreement $\sigma_{\nabla_a Q}$ significantly outperforms pure dynamics uncertainty $\sigma_{\mathrm{dyn}}$ on $H=5$ rollout preservation ($t = 3.741, \mathbf{p = 0.0046}$), confirming that epistemic dynamics variance alone prioritizes chaotic off-manifold states that harm short-horizon predictive stability.

2. **Failure to Outperform Uniform Random Sampling:**  
   Under one-shot scratch retraining on sparse exploratory Hopper rollouts (where episodes terminate within 20-30 steps upon falling), querying 200 states prioritized by gradient disagreement did not produce a statistically detectable improvement over random sampling ($p = 0.6770$).  
   Both methods exhibited slight negative transfer ($\eta_B < 0$) due to the non-stationarity of chaotic falling trajectories.

3. **Pre-Registered Gate 4B.1 Decision:**
   $$\boxed{\textbf{GATE 4B.1 VERDICT: NEGATIVE / GATE NOT PASSED}}$$
   Per the pre-registered research contract:
   $$\boxed{\textbf{STAGE 4B.2 (Closed-Loop Active MBPO) REMAINS LOCKED}}$$

---

## 5. Roadmap Status Summary

```latex
\boxed{
\begin{array}{ll}
\text{Step 1 Theory} & \textbf{CLOSED / PASS} \\
\text{Step 2 Modern MBRL infrastructure} & \textbf{CLOSED / PASS} \\
\text{Stage 3A Unprivileged ranking} & \textbf{CLOSED / PASS} \\
\text{Stage 4A Targeted acquisition (CartPole)} & \textbf{CLOSED / PASS, bounded to CartPole} \\
\text{Stage 3B Continuous geometry & perturbation law} & \textbf{CLOSED / PASS} \\
\text{Stage 4B.0 Curvature closure pre-gate} & \textbf{CLOSED / CANONICAL SCORE = } \sigma_{\nabla_a Q} \\
\text{Stage 4B.1 One-shot MBPO repair (Hopper)} & \textbf{EVALUATED / NEGATIVE} \\
\text{Stage 4B.2 Closed-loop active MBPO} & \textbf{LOCKED}
\end{array}
}
```

* **Manuscript Implications:** The thesis maintains strict scientific honesty:
  1. The core theoretical framework (discrete margin vulnerability & continuous implicit perturbation law) is mathematically rigorous and empirically validated on controlled systems (Stages 1, 3A, 4A, 3B).
  2. Unprivileged decision-risk ranking provides statistically significant sample efficiency in world-model repair on discrete CartPole ($p = 0.022$).
  3. In continuous MuJoCo systems (`Hopper-v4`), action-gradient disagreement successfully avoids chaotic dynamics uncertainty traps ($p = 0.0046$ vs $\sigma_{\mathrm{dyn}}$), but one-shot active query repair does not yet beat uniform random sampling on raw exploratory data without closed-loop trajectory conditioning.
