# Stage R0 Scientific Report: Repair-Operator Falsification Benchmark

**Milestone:** Research Stage R0 — Repair-Operator Falsification Pre-Gate  
**Date:** 2026-08-20  
**Author:** Pham Quang Minh  
**Status:** **STAGE R0 CLOSED — VERDICT: FALSIFIED AS PROTOCOL ARTIFACT (Optimization Drift Eliminated via Standard Convergence & Early Stopping)**

---

## 1. Executive Summary & Scientific Purpose

Stage R0 was pre-registered to test the fundamental validity of the "optimization drift" hypothesis:

$$\boxed{\text{Is post-update world-model degradation a persistent representation trade-off, or a trivial protocol artifact of under-convergence and optimization dynamics?}}$$

To isolate this, we executed a full factorial sweep on `Hopper-v4` across learning rates $\eta \in \{10^{-4}, 3\times 10^{-4}, 10^{-3}, 3\times 10^{-3}\}$, update budgets (epochs $E \in \{1, 5, 15, 40\}$ and micro-steps $K \in \{1, 5, 20\}$), and strict validation early stopping across $N = 10$ independent seeds.

```mermaid
flowchart TD
    M0["Initial Reference M_0 (H=5 Err = 0.071336)"] --> SWEEP["Factorial Epoch & LR Sweep on D_0"]
    SWEEP -->|LR=1e-4, 40 Epochs| E1["H=5 Err = 0.069126 (-3.10%)"]
    SWEEP -->|LR=3e-4, 40 Epochs| E2["H=5 Err = 0.067424 (-5.48%)"]
    SWEEP -->|LR=1e-3, 40 Epochs| E3["H=5 Err = 0.064109 (-10.13%)"]
    SWEEP -->|LR=3e-3, 40 Epochs| E4["H=5 Err = 0.063058 (-11.60%)"]
    SWEEP -->|Validation Early Stopping| ES["H=5 Err = 0.068277 (-4.29%)"]
    E1 & E2 & E3 & E4 & ES --> KILL["KILL CRITERION TRIGGERED: R0 FALSIFIED AS PROTOCOL ARTIFACT"]
```

---

## 2. Experimental Results: Factorial Optimization Matrix ($N = 10$ Seeds)

Evaluated on `Hopper-v4` with base dataset $\mathcal{D}_0$ ($N_0 = 830$ total: $664$ Train, $166$ Holdout Validation) and held-out test rollouts $\mathcal{D}_{\mathrm{eval}}$ ($N_{\mathrm{eval}} = 555$). Reference checkpoint $M_0$ initial $H=5$ rollout error: **$0.071336$** (1-step MSE $= 0.025859$, Validation holdout MSE $= 0.023255$).

#### Table 1: Full Factorial Matrix of Model Optimization on $\mathcal{D}_0$

| Mode | Learning Rate ($\eta$) | Epochs / Steps | $H=5$ Rollout Error (Mean $\pm$ Std) | $\Delta H=5$ vs $M_0$ (%) | Validation Holdout MSE | Functional Displacement |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reference $M_0$** | — | — | $\mathbf{0.071336 \pm 0.000000}$ | $0.00\%$ | $0.023255$ | $0.000000$ |
| **Epoch Sweep** | $10^{-4}$ | $E = 1$ | $0.071064 \pm 0.000136$ | $-0.38\%$ | $0.023370$ | $0.000529$ |
| | $10^{-4}$ | $E = 5$ | $0.070709 \pm 0.000309$ | $-0.88\%$ | $0.023365$ | $0.001034$ |
| | $10^{-4}$ | $E = 15$ | $0.069783 \pm 0.000393$ | $-2.18\%$ | $0.023511$ | $0.001597$ |
| | $10^{-4}$ | $E = 40$ | $\mathbf{0.069126 \pm 0.000666}$ | $\mathbf{-3.10\%}$ | $0.023751$ | $0.002916$ |
| | $3 \times 10^{-4}$ | $E = 1$ | $0.071134 \pm 0.000412$ | $-0.28\%$ | $0.023445$ | $0.001378$ |
| | $3 \times 10^{-4}$ | $E = 5$ | $0.070457 \pm 0.000638$ | $-1.23\%$ | $0.023427$ | $0.002077$ |
| | $3 \times 10^{-4}$ | $E = 15$ | $0.068967 \pm 0.000699$ | $-3.32\%$ | $0.023648$ | $0.003086$ |
| | $3 \times 10^{-4}$ | $E = 40$ | $\mathbf{0.067424 \pm 0.000634}$ | $\mathbf{-5.48\%}$ | $0.023800$ | $0.005336$ |
| | $10^{-3}$ | $E = 1$ | $0.071705 \pm 0.000608$ | $+0.52\%$ | $0.023717$ | $0.003999$ |
| | $10^{-3}$ | $E = 5$ | $0.070002 \pm 0.000489$ | $-1.87\%$ | $0.023602$ | $0.003670$ |
| | $10^{-3}$ | $E = 15$ | $0.067259 \pm 0.000880$ | $-5.71\%$ | $0.023995$ | $0.005890$ |
| | $10^{-3}$ | $E = 40$ | $\mathbf{0.064109 \pm 0.000913}$ | $\mathbf{-10.13\%}$ | $0.024112$ | $0.012676$ |
| | $3 \times 10^{-3}$ | $E = 1$ | $0.079789 \pm 0.003257$ | $+11.85\%$ | $0.026343$ | $0.028301$ |
| | $3 \times 10^{-3}$ | $E = 5$ | $0.071748 \pm 0.000976$ | $+0.58\%$ | $0.023836$ | $0.006803$ |
| | $3 \times 10^{-3}$ | $E = 15$ | $0.067268 \pm 0.001424$ | $-5.70\%$ | $0.024212$ | $0.010770$ |
| | $3 \times 10^{-3}$ | $E = 40$ | $\mathbf{0.063058 \pm 0.001369}$ | $\mathbf{-11.60\%}$ | $0.024105$ | $0.019756$ |
| **Micro-Steps** | $10^{-3}$ | $K = 1$ | $0.075042 \pm 0.000820$ | $+5.19\%$ | $0.024430$ | $0.009429$ |
| | $10^{-3}$ | $K = 5$ | $0.072102 \pm 0.000540$ | $+1.07\%$ | $0.023800$ | $0.003767$ |
| | $10^{-3}$ | $K = 20$ | $0.070755 \pm 0.000480$ | $-0.81\%$ | $0.023716$ | $0.003591$ |
| **Early Stopping** | $10^{-3}$ | Best $E \approx 22.5$ | $\mathbf{0.068277 \pm 0.000720}$ | $\mathbf{-4.29\%}$ | $0.023210$ | $0.007500$ |

---

## 3. Causal Mechanics & Scientific Attribution

1. **Monotonic Improvement with Sufficient Optimization on $\mathcal{D}_0$:**  
   Across all learning rates from $10^{-4}$ to $3 \times 10^{-3}$, increasing training epochs from $E=1 \to 40$ consistently and monotonically **improves** held-out $H=5$ rollout error (reducing error from $0.0713 \to 0.0631$, a $-11.60\%$ improvement).
2. **Transient Shocks under Few-Step Updates:**  
   Micro-updates ($K = 1-5$ mini-batch gradient steps) and single-epoch updates at high LR produce transient parameter shocks ($+5.19\%$ to $+11.85\%$ error increase), which dissolve into net improvement once optimization completes standard epoch convergence ($E \ge 15$).
3. **Validation Early Stopping Eliminates Degradation:**  
   Monitoring an untouched $20\%$ validation holdout of $\mathcal{D}_0$ automatically terminates training around epoch $\approx 22.5$, achieving $H=5$ error $= 0.068277$ ($-4.29\%$ net improvement over $M_0$).

---

## 4. Pre-Registered Kill Criterion Evaluation & Final Gate Decision

$$\boxed{\textbf{PRE-REGISTERED KILL RULE TRIGGERED: STAGE R0 FALSIFIED AS PROTOCOL ARTIFACT}}$$

* **Scientific Conclusion:** The apparent "irreparable model degradation" previously observed under one-shot retraining is definitively identified as an **optimization protocol artifact resulting from under-converged baseline checkpoints and transient gradient shocks during few-step updates**. When trained with standard epoch convergence or validation early stopping, retraining on $\mathcal{D}_0$ produces consistent, monotonic predictive improvement (up to $-11.6\%$).
* **Research Roadmap Impact:** Because the degradation is resolvable via standard optimization hygiene and does not represent an intractable representation interference phenomenon, **Stages R1, R2, and R3 (Complex Gradient Interference / Novel Loss Architectures) are NOT JUSTIFIED and will NOT be opened.**
* **Final Project Status:** All exploratory research branches are closed with complete causal clarity. The project proceeds directly to final thesis governance and manuscript finalization.
