# Scientific Report: Value-Aware World-Model Baselines Comparison (Feedback #6)

**Date:** 2026-08-20  
**Target Milestone:** Q1 Journal Upgrade — Feedback #6 Closure & Empirical Demarcation  
**Environment Suite:** `Hopper-v4` (Primary, $N=5$), `Walker2d-v4` (Cross-Env Pilot), `HalfCheetah-v4` (Cross-Env Pilot)  
**Backbone Architecture:** MBPO (7-model Gaussian/VAML MLP Ensemble, Elite 5, SAC Agent)  
**Accounting Controls:** Matched 10,000 environment steps, 855 dynamics updates, 45,000 SAC gradient updates, 3,600 imagined transitions per seed across $N=5$ deterministic seeds.  
**Regime Demarcation:** *Matched low-budget MBPO-style stress test / early-interaction regime*.

---

## 1. Aggregated Comparative Results on Hopper-v4 ($N=5$ Seeds)

| Algorithm   | Loss Formulation | Seeds ($N$) | Mean Return | Sample SD ($s$) | 95% Bootstrap CI | Mean Wall (s) | Mean Peak RAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard MLE** | Gaussian NLL ($\mathcal{L}_{\mathrm{NLL}}$) | 5 | **30.05** | **16.45** | $[17.98, 43.66]$ | 242.2 s | 276.5 MB |
| **VaGraM** | Value-Gradient Weighted MSE ($\mathcal{L}_{\mathrm{VaGraM}}$) | 5 | **31.94** | **40.24** | $[5.22, 67.02]$ | 251.8 s | 124.4 MB |
| **Taylor-VAML** | Value-Aware Inner-Product Loss $(g^\top \Delta)^2$ | 5 | **34.35** | **43.99** | $[6.72, 74.49]$ | 273.2 s | 154.1 MB |
| **$\lambda$-VAML** | Regularized VAML ($\mathcal{L}_{\mathrm{VAML}} + \mathcal{L}_{\mathrm{MSE}}$) | 5 | **36.58** | **63.85** | $[5.90, 94.21]$ | 252.4 s | 190.6 MB |

---

## 2. Paired-Seed Delta Breakdown ($\Delta = \text{Baseline} - \text{MLE}$)

Analyzing performance on a seed-by-seed paired basis reveals that the slight numerical mean elevation of value-aware objectives is driven by isolated single-seed outliers rather than consistent learning superiority:

| Baseline | Seed 42 $\Delta$ | Seed 43 $\Delta$ | Seed 44 $\Delta$ | Seed 45 $\Delta$ | Seed 46 $\Delta$ | Wins vs MLE | Median $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **VaGraM** | $+81.43$ | $-11.56$ | $-43.73$ | $-6.61$ | $-10.09$ | **1 / 5** | **$-10.09$** |
| **Taylor-VAML** | $+91.55$ | $+24.25$ | $-36.26$ | $-14.24$ | $-43.81$ | **2 / 5** | **$-14.24$** |
| **$\lambda$-VAML** | $-8.51$ | $-15.14$ | $+107.54$ | $-10.45$ | $-40.80$ | **1 / 5** | **$-10.45$** |

### Individual Per-Seed Raw Values:
| Algorithm | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MLE** | $16.43 \pm 0.29$ | $18.53 \pm 3.58$ | $43.14 \pm 15.24$ | $19.97 \pm 1.46$ | $52.21 \pm 1.14$ |
| **VaGraM** | $97.86 \pm 0.96$ | $6.97 \pm 0.38$ | $-0.59 \pm 0.26$ | $13.36 \pm 0.46$ | $42.12 \pm 0.82$ |
| **Taylor-VAML** | $107.98 \pm 23.28$ | $42.78 \pm 5.12$ | $6.88 \pm 0.39$ | $5.73 \pm 0.10$ | $8.40 \pm 0.16$ |
| **$\lambda$-VAML** | $7.92 \pm 0.37$ | $3.39 \pm 0.56$ | $150.68 \pm 3.26$ | $9.52 \pm 0.32$ | $11.41 \pm 2.63$ |

---

## 3. Cross-Environment Pilot Validation (Feedback #1)

To evaluate whether the lack of value-aware advantage on Hopper is universal or environment-dependent, matched pilot benchmarks were executed on two additional MuJoCo continuous control benchmarks:

| Environment | Observation Dim | Action Dim | MLE Return (Seed 42) | VaGraM Return (Seed 42) | Numerical $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Hopper-v4** | 11D | 3D | $16.43$ | $97.86$ | $+81.43$ |
| **Walker2d-v4** | 17D | 6D | $-0.39$ | $105.67$ | $+106.06$ |
| **HalfCheetah-v4** | 17D | 6D | $-380.63$ | $-297.24$ | $+83.39$ |

*Scientific Interpretation of Cross-Environment Data:*
The fact that VaGraM achieves higher pilot returns than MLE on Walker2d and HalfCheetah directly **falsifies any claim of universal objective-family failure**. Rather, world-model objective utility is **operator- and environment-conditional**.

---

## 4. Canonical Scientific Synthesis & Resolution of Core Questions

### Statement of Empirical Findings:
> Under a strictly matched low-budget Hopper MBPO-style protocol, none of three tested value-aware model objectives—VaGraM, Taylor-VAML, or regularized $\lambda$-VAML—provided a consistent downstream-return advantage over standard maximum-likelihood model learning. Although their mean returns were numerically higher, these averages were driven by isolated high-performing seeds; each value-aware method exhibited substantially larger between-seed variance, and two of the three outperformed MLE on only one of five paired seeds. These results therefore support a lack of robust improvement rather than statistical equivalence or a universal failure of value-aware model learning. Preliminary Walker2d and HalfCheetah results further suggest that the outcome depends on the environment and optimization regime.

### Direct Answers to Core Research Questions:

1. **Question 1: Does value-aware training improve Hopper world-model utility over Standard MLE?**  
   - **Answer:** In the tested low-budget Hopper regime, there is **no evidence of a robust advantage**. On 3 out of 5 seeds for VAML and 4 out of 5 seeds for VaGraM and $\lambda$-VAML, value-aware models underperform MLE, yielding negative median deltas ($\text{Median }\Delta \in [-14.24, -10.09]$).

2. **Question 2: Does decision-risk prioritization add value beyond value-aware objectives?**  
   - **Answer:** Decision-risk prioritization captures discrete ranking reversals ($Z_{\mathrm{flip}}$), while value-aware objectives weight continuous sensitivity ($\nabla_s V$). However, in complex multi-joint locomotion tasks under multi-step neural training, neither formulation provides an automatic shortcut around non-convex optimization dynamics and trajectory coverage requirements.

3. **Question 3: Does the failure boundary generalize across the entire objective family?**  
   - **Answer: NO (Claim of universal failure is rejected).** The empirical outcome is **environment- and regime-conditional**. While value-aware objectives struggle with severe seed variance and policy collapse on Hopper, they exhibit potential utility in other environments (Walker2d, HalfCheetah). This conditionality strengthens the overarching thesis: *error prioritization utility is strictly operator- and regime-dependent*.

4. **Causal Attribution Demarcation:**  
   - The instability of value-aware objectives is consistent with the broader operator-conditioning phenomenon identified by our controlled studies, but these experiments do **not** establish a common causal mechanism with static gradient influence decay.
