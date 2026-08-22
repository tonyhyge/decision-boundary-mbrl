# EXPERIMENT SPECIFICATION: STAGE 4 — LEARNED WORLD MODELS & APPROXIMATE DECISION GEOMETRY

**Document:** Stage 4 Protocol & Sub-Gate Verification  
**Core Scientific Question:** *Does the decision-boundary phenomenon survive when dynamics ($P$), values ($V$), and margins ($m$) must be estimated from finite trajectory data rather than provided by an exact oracle?*

---

## 1. Scope & Isolation Principle

To avoid confounding the scientific investigation with neural value approximation, actor learning divergence, or latent rollout truncation, Stage 4 uses the **minimal isolated neural world model**:

$$\boxed{\text{GridWorld Trajectories } \mathcal{D} = \{(s, a, s')\} \longrightarrow \text{MLP Model } \hat{P}_\theta(s' \mid s, a) \longrightarrow \text{Exact DP in } \hat{P}_\theta \longrightarrow \hat{V}, \hat{Q}, \hat{m}, \hat{B}}$$

This strictly isolates **world-model estimation error** without confounding architectural noise.

---

## 2. Three Sub-Gates Protocol

### Sub-Gate G4-A: Estimation Fidelity
* **Learned Model:** Multi-Layer Perceptron (MLP) parameterized with softmax categorical output over $\mathcal{S}$ ($|\mathcal{S}|=25, |\mathcal{A}|=4$).
* **Metrics:**
  1. Margin Estimation Error: $\text{MAE}(\hat{m}, m_{\text{oracle}}) = \frac{1}{|\mathcal{S}|} \sum_{s} |\hat{m}(s) - m(s)|$.
  2. Crossing Classification: Precision, Recall, F1, and AUROC for $\hat{Z}_{\text{cross}} = \mathbf{1}[\hat{m}(s) < 0]$ vs ground-truth $Z_{\text{cross}}$.
  3. Boundary Pressure Rank Correlation: Spearman $\rho(\hat{B}, B_{\text{oracle}})$.
* **Fidelity Kill Rule:** If $\text{AUROC}(\hat{Z}_{\text{cross}}) \le 0.55$ or $\rho(\hat{B}, B_{\text{oracle}}) \le 0.10$, the learned representation fails to capture decision geometry $\implies$ halt.

---

### Sub-Gate G4-B: Incremental Information Under Estimation Noise
* **Nested Regression:**
  $$\text{Model 0: } C_i = \beta_0 + \beta_1 \hat{E}_{L1, i} + \beta_2 |\hat{G}_i| + \epsilon$$
  $$\text{Model 1: } C_i = \beta_0 + \beta_1 \hat{E}_{L1, i} + \beta_2 |\hat{G}_i| + \beta_3 \hat{B}_i + \beta_4 \hat{Z}_{\text{cross}} + \epsilon$$
* **Success Condition:** Measure whether $\Delta R^2(\hat{B}, \hat{Z}_{\text{cross}} \mid \hat{E}, |\hat{G}|) > 0$ with $p < 0.05$.

---

### Sub-Gate G4-C: Practical Recovery Under Approximate Prioritization
* **Budget Benchmark:** Prioritize model repairs using estimated rankers ($\hat{B}$, $|\hat{G}|$, $\hat{E}$, Random) across 50 seeds.
* **Goal:** Test whether the early-budget advantage of boundary geometry ($B$ vs $E$) survives finite sample estimation noise.
