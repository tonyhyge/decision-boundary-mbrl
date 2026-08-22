# Stage Q1-FB Implementation Symmetry Audit

**Objective:** Formally prove that the experimental pipeline maintains complete structural and algorithmic symmetry between the baseline (MLE) and the proposed method (VaGraM), isolating the dynamics loss function as the sole treatment variable.

| Pipeline Dimension | Standard MLE Branch | VaGraM Branch | Symmetry Status |
|:---|:---|:---|:---|
| Environment Benchmark | Gymnasium Hopper-v4 (MuJoCo) | Gymnasium Hopper-v4 (MuJoCo) | MATCHED |
| State / Action Space | Obs Dim = 11, Act Dim = 3 | Obs Dim = 11, Act Dim = 3 | MATCHED |
| Interaction Budget | 125,000 Environment Steps | 125,000 Environment Steps | MATCHED |
| Model Class | GaussianMLP | VAMLMLP | SYMMETRIC (Subclassed) |
| Ensemble Architecture | 7 MLP Members, Elite = 5 | 7 MLP Members, Elite = 5 | MATCHED |
| Network Depth & Width | 4 Hidden Layers x 200 Units | 4 Hidden Layers x 200 Units | MATCHED |
| Activation Function | ReLU (use_silu=False) | ReLU (use_silu=False) | MATCHED |
| Weight Initialization | Truncated Normal Init | Truncated Normal Init | MATCHED |
| Model Optimizer | Adam (lr=1e-3, weight_decay=1e-5) | Adam (lr=1e-3, weight_decay=1e-5) | MATCHED |
| Dynamics Training Cadence | Every 250 Env Steps | Every 250 Env Steps | MATCHED |
| Max Dynamics Epochs / Patience | max_epochs=None, patience=None | max_epochs=None, patience=None | MATCHED |
| Critic Architecture | DoubleQCritic (2 layers x 256 units) | DoubleQCritic (2 layers x 256 units) | MATCHED |
| Actor Architecture | DiagGaussianActor (2 layers x 256 units) | DiagGaussianActor (2 layers x 256 units) | MATCHED |
| SAC Optimizer & Learning Rates | Critic lr=3e-4, Actor lr=3e-5, Alpha lr=3e-4 | Critic lr=3e-4, Actor lr=3e-5, Alpha lr=3e-4 | MATCHED |
| SAC Update Frequency | 20 Updates per Environment Step | 20 Updates per Environment Step | MATCHED |
| Discount Factor (gamma) | 0.99 | 0.99 | MATCHED |
| Critic Polyak Target Tau | 0.005 (Target Update Freq = 4) | 0.005 (Target Update Freq = 4) | MATCHED |
| Rollout Schedule | [20000, 150000, 1, 5] | [20000, 150000, 1, 5] | MATCHED |
| Rollout Batch Size | 100,000 Transitions | 100,000 Transitions | MATCHED |
| Model Buffer Retain Epochs | 1 Epoch | 1 Epoch | MATCHED |
| Real Data Batch Size for SAC | 256 Transitions | 256 Transitions | MATCHED |
| Evaluation Cadence | Every 5,000 Environment Steps (25 total) | Every 5,000 Environment Steps (25 total) | MATCHED |
| Eval Episodes per Checkpoint | 10 Deterministic Episodes (eval_mode=True) | 10 Deterministic Episodes (eval_mode=True) | MATCHED |
| Primary Estimand | Mean of Checkpoints 115k, 120k, 125k | Mean of Checkpoints 115k, 120k, 125k | MATCHED |
| Dynamics Loss Objective | L_MLE = -log P(s', r \| s, a) (Gaussian NLL) | L_VaGraM = \|\|∇_s' V ⊙ (s^_hat - s')\|\|^2 + L_r (q_0.95 clipped) | ISOLATED TREATMENT VARIABLE |

---

### Formal Scientific Statement of Control:
The two pipelines were **matched on all 21 audited architectural, optimization, data-flow, and evaluation fields**; the intended experimental difference was the world-model training objective ($\mathcal{L}_{\mathrm{MLE}}$ vs $\mathcal{L}_{\mathrm{VaGraM}}$).
