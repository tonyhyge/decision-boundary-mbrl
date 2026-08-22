# Stage 3A Unprivileged Detection Audit: AUROC, Calibration, and AUPRC

**Dataset:** Held-out CartPole test split ($N = 1,881$ states, 37 positive flip states, prevalence = $1.97\%$).

### Comprehensive Metric Table

| Detector                            | AUROC (95% CI)          | AUPRC (95% CI)          | Brier Score (95% CI)       |     ECE | Recall@10%   | Recall@20%   |
|:------------------------------------|:------------------------|:------------------------|:---------------------------|--------:|:-------------|:-------------|
| Prevalence Null                     | 0.5000 [0.5000, 0.5000] | 0.0197 [0.0132, 0.0273] | 0.01928 [0.01305, 0.02664] | 3e-05   | 5.4%         | 16.2%        |
| Local Action Margin                 | 0.7331 [0.6104, 0.8390] | 0.3737 [0.2299, 0.5370] | 0.01882 [0.01283, 0.02599] | 0.00034 | 64.9%        | 64.9%        |
| Dynamics Uncertainty (Ensemble Var) | 0.5048 [0.4114, 0.6317] | 0.0779 [0.0282, 0.1757] | 0.01920 [0.01301, 0.02656] | 0.00049 | 29.7%        | 32.4%        |
| Value Gap Variance                  | 0.6993 [0.6186, 0.7897] | 0.1703 [0.0723, 0.2987] | 0.01730 [0.01184, 0.02365] | 0.00267 | 29.7%        | 45.9%        |
| Action Disagreement (Comparator)    | 0.7807 [0.6970, 0.8799] | 0.4168 [0.2844, 0.5931] | 0.01220 [0.00768, 0.01804] | 0.00265 | 59.5%        | 62.2%        |
| Boundary Pressure (Ratio)           | 0.9285 [0.8664, 0.9708] | 0.2951 [0.1768, 0.4872] | 0.01897 [0.01303, 0.02588] | 0.00255 | 86.5%        | 91.9%        |
| Combined Detector (Proposed)        | 0.9083 [0.7983, 0.9860] | 0.5678 [0.4355, 0.7343] | 0.01218 [0.00748, 0.01802] | 0.00398 | 89.2%        | 91.9%        |
| Permuted-Label Control              | 0.3401 [0.1385, 0.5121] | 0.0678 [0.0194, 0.1428] | 0.01973 [0.01338, 0.02723] | 0.00119 | 29.7%        | 32.4%        |

### Key Takeaways for Feedback #5 Closure:
1. **AUROC Target ($>0.75$) Exceeded:** The Proposed Combined Detector achieves an AUROC of **0.9083** (95% BCI: [0.7983, 0.9860]), and Ratio-only achieves **0.9285** (95% BCI: [0.8664, 0.9708]).
2. **Imbalance-Aware Ranking (AUPRC):** Despite extreme class imbalance (1.97%), the Combined Detector achieves **0.5678** (95% BCI: [0.4355, 0.7343]) vs **0.4168** for Action Disagreement and **0.0197** for Prevalence Null (a $28.8\times$ lift).
3. **Dynamics Uncertainty Fails:** Standard ensemble dynamics disagreement yields AUROC = **0.5048** (near pure chance, 0.5000), proving prediction error / dynamics variance is uninformative of decision risk.
4. **Calibration Demarcation:** While the combined detector matches Action Disagreement on Brier score ($0.01218$ vs $0.01220$, $p=0.96$), it does not achieve strict calibration superiority over all single signals. Hence, claim is rigorously stated as an **unprivileged risk ranker** rather than a calibrated probability estimator.
