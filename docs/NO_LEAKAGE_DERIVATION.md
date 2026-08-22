# NO-LEAKAGE MATHEMATICAL DERIVATION & STRUCTURAL AUDIT

**Document:** Formal Algebraic Relationship between Boundary Pressure ($B_i$), Occupancy-Weighted Margin Deformation ($d_i B_i$), and Counterfactual Correction Value ($C_i$)  
**Goal:** Address the reviewer objection: *"Is $d_i B_i$ merely an algebraic reconstruction or trivial first-order approximation of $C_i$?"*

---

## 1. Mathematical Formulations

### 1.1 Ground-Truth Counterfactual Value ($C_i$)
Let $P$ be the true transition matrix, $\hat{P}$ be the corrupted world model, and $\hat{P}^{, i \leftarrow P}$ be the world model with localized component $i = (s_i, a_i)$ restored to true dynamics $P(\cdot \mid s_i, a_i)$.

By the **Performance Difference Lemma** (Kakade & Langford, 2002), the true expected return difference between the policy planned on the restored model $\pi_{\text{restored}} = \pi^*_{\hat{P}^{, i \leftarrow P}}$ and the policy planned on the corrupt model $\pi_{\text{corrupt}} = \pi^*_{\hat{P}}$ evaluated strictly in $P$ is:
$$C_i = J(\pi_{\text{restored}}; P) - J(\pi_{\text{corrupt}}; P) = \frac{1}{1-\gamma} \sum_{s \in \mathcal{S}} d_P^{\pi_{\text{restored}}}(s) \sum_{a \in \mathcal{A}} \pi_{\text{restored}}(a \mid s) A_P^{\pi_{\text{corrupt}}}(s, a)$$

For deterministic greedy policies $\pi_{\text{restored}}(s) = a_{\text{res}}(s)$ and $\pi_{\text{corrupt}}(s) = a_{\text{corr}}(s)$:
$$C_i = \frac{1}{1-\gamma} \sum_{s \in \mathcal{S}} d_P^{\pi_{\text{restored}}}(s) \left[ Q_P^{\pi_{\text{corrupt}}}(s, a_{\text{res}}(s)) - V_P^{\pi_{\text{corrupt}}}(s) \right]$$

Key properties of $C_i$:
1. **Global Support:** $C_i$ depends on all states where the policy actions differ: $\mathcal{S}_{\text{diff}} = \{s \in \mathcal{S} \mid a_{\text{res}}(s) \neq a_{\text{corr}}(s)\}$.
2. **Exact Non-Linearity:** $C_i$ is piecewise-constant with respect to local perturbation magnitude $\delta$, because policy choices $a_{\text{res}}(s) = \arg\max_a Q_{\hat{P}^{, i \leftarrow P}}(s, a)$ are discrete $\arg\max$ operations. If the action selection does not flip, $C_i = 0$ identically.

---

### 1.2 Proposed Normalized Boundary Pressure ($B_i$)
At state $s_i$ and action $a_i$, let the true action margin against runner-up competitor $a_{\text{comp}}$ be:
$$m_P(s_i) = Q_P^*(s_i, a^*(s_i)) - Q_P^*(s_i, a_{\text{comp}}(s_i)) > 0$$
Under corrupted planning, the model-induced margin is:
$$m_{\hat{P}}(s_i) = Q_{\hat{P}}^*(s_i, a^*(s_i)) - Q_{\hat{P}}^*(s_i, a_{\text{comp}}(s_i))$$
The margin deformation is:
$$\Delta m(s_i) = m_{\hat{P}}(s_i) - m_P(s_i)$$
The normalized boundary pressure is defined as:
$$B_i = -\frac{\Delta m(s_i)}{m_P(s_i) + \varepsilon}$$
The occupancy-weighted boundary pressure is:
$$B_i^{\text{occ}} = d_P^{\pi^*}(s_i, a_i) \cdot B_i = -d_P^{\pi^*}(s_i, a_i) \frac{\Delta m(s_i)}{m_P(s_i) + \varepsilon}$$

---

### 1.3 First-Order Taylor Approximation of Policy Loss
Under a first-order value-gradient expansion (e.g. VaGraM / Value-Aware Loss), the return sensitivity to a single transition perturbation $\delta P_i = \hat{P}(\cdot \mid s_i, a_i) - P(\cdot \mid s_i, a_i)$ is:
$$G_i^\pm = \delta P_i^\top V_P^*$$
$$G_i^{\text{occ},\pm} = d_P^{\pi^*}(s_i, a_i) \left( \delta P_i^\top V_P^* \right)$$
$$G_i^{\text{occ,abs}} = d_P^{\pi^*}(s_i, a_i) \left| \delta P_i^\top V_P^* \right|$$

---

## 2. Structural Divergence: Why $d_i B_i \neq C_i$ and $d_i B_i \neq d_i G_i^\pm$

We establish the exact points of mathematical divergence between $d_i B_i$, $C_i$, and $d_i G_i^\pm$:

| Dimension | First-Order Sensitivity ($d_i G_i^\pm$) | Occupancy Boundary Pressure ($d_i B_i$) | Counterfactual Value ($C_i$) |
| :--- | :--- | :--- | :--- |
| **Mathematical Nature** | Linear scalar projection $d_i (\delta P_i^\top V^*)$ | Normalized relative ratio $-d_i \frac{\Delta m_i}{m_i + \varepsilon}$ | Non-linear discrete functional $J(\pi^*_{\hat{P}^{, i \leftarrow P}}) - J(\pi^*_{\hat{P}})$ |
| **Normalization / Scale** | Unbounded linear in $\delta$ | Normalized by true action gap $m_P(s_i)$ | Bounded by $\frac{R_{\max}}{1-\gamma}$ |
| **Response to Margin Expansion ($B < 0$)** | Predicts non-zero linear value drift | $B < 0$ (Explicitly signals zero decision damage) | $C_i = 0$ identically |
| **Response to Sub-threshold Compression ($0 < B < 1$)** | Predicts non-zero linear value drift | $0 < B < 1$ (Signals margin compression towards boundary) | $C_i = 0$ identically |
| **Response to Boundary Crossing ($B \ge 1$)** | Same linear slope as sub-threshold | $B \ge 1$ (Signals discrete boundary flip) | $C_i > 0$ (Step-function onset) |

### Formal Separation:
1. **$d_i B_i$ is NOT an algebraic identity of $C_i$:**  
   $C_i$ is a discontinuous step function ($C_i = 0$ for all $B < 1$, and $C_i > 0$ when $B \ge 1$). In contrast, $B_i$ is a continuous geometric indicator tracking distance and direction relative to the decision manifold.
2. **$d_i B_i$ is NOT redundant to $d_i G_i^\pm$ or $d_i |G_i|$:**  
   $G_i^\pm$ tracks absolute value change of the perturbed action $\Delta Q(s_i, a_i)$, but **completely ignores the distance to the competitor action** $m_P(s_i)$. An error with $\Delta Q = -0.1$ at a state with gap $m_P = 1.0$ has $B = 0.1 \ll 1$ (negligible decision threat), whereas the exact same $\Delta Q = -0.1$ at a state with gap $m_P = 0.05$ has $B = 2.0 > 1$ (catastrophic decision reversal).

---

## 3. Empirical No-Leakage Verification Protocol

To empirically prove non-redundancy, we construct a full nested regression controlling for all occupancy-weighted first-order terms:

$$C_i = \beta_0 + \beta_1 (d_i E_{L1, i}) + \beta_2 (d_i |G_i|) + \beta_3 (d_i G_i^\pm) + \beta_4 (d_i A_i^\pm) + \beta_5 (d_i B_i) + \epsilon_i$$

If $\Delta R^2(d_i B_i \mid d_i E, d_i |G|, d_i G^\pm, d_i A^\pm) > 0$ with $p < 0.01$, then $d_i B_i$ contains statistically significant, non-redundant decision-margin geometry beyond any first-order policy-loss reconstruction.
