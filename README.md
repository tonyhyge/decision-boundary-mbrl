# Decision-Boundary Geometry of World-Model Errors in Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-22%20passed-success.svg)]()

Official open-source research implementation for the paper:
**"Decision-Boundary Geometry of World-Model Errors in Reinforcement Learning"**

---

## Overview

Model-based reinforcement learning (MBRL) fundamentally relies on world models to plan actions and synthesize policies; however, minimizing transition prediction error frequently fails to improve downstream control due to the **objective mismatch** problem. 

This repository provides the official diagnostic framework, experimental benchmarks, and verification test suites establishing:
1. **Decision-Dependent Error Consequence**: Predictive loss ($E^{L1}, E^{\mathrm{MSE}}$) and unsigned value sensitivity ($|G|$) are blind to the signed direction of margin deformation, whereas compressive perturbations alone drive policy degradation ($\Delta C_{\mathrm{matched}} = +0.00885, p = 2.15 \times 10^{-9}$), while expansive perturbations leave optimal actions invariant ($C^{\mathrm{away}} = 0$).
2. **Categorical Crossing Mechanism**: Explanatory gain in discrete control is dominated by categorical ranking reversal ($Z_{\mathrm{cross}}$), yielding significant incremental explanatory power beyond first-order signed margins ($\Delta R^2 = +0.0452, p = 4.58 \times 10^{-5}$), with no positive graded depth within crossing errors ($r = -0.0337$).
3. **Diagnostic vs. Intervention Boundary**: While boundary metrics substantially outperform predictive error under scarce repair budgets ($54.1\%$ vs. $27.2\%$ at $K=1$), first-order value sensitivity maintains an aggregate rank advantage across full repair trajectories ($p = 0.0042$). Converting crossing signals into empirical risk sample weights yields no downstream control improvements due to non-local value propagation bottlenecks.

---

## Repository Structure

```
decision-boundary-mbrl/
├── src/                          # Core diagnostic and algorithmic library
│   ├── envs/                     # Fork-MDP, Stochastic GridWorld, and CartPole environments
│   ├── metrics/                  # Diagnostic estimators (B, G, Z_cross, action margins)
│   ├── planning/                 # Exact Dynamic Programming & Policy Evaluation
│   ├── corruptions/              # Controlled error injectors & matched-pair generators
│   ├── models/                   # Tabular learned models & neural ensemble dynamics
│   ├── correction/               # Budgeted model repair & empirical loss reweighting
│   ├── baselines/                # Priority rankers (Boundary, Value, Predictive, Gap)
│   └── generate_all_figures.py   # Publication figure generation pipeline
├── experiments/                  # Modular experiment reproduction pipelines
│   ├── causal_geometry/          # Controlled causal identification in Fork MDPs
│   ├── budgeted_repair/          # Multi-error budgeted repair benchmark (static & adaptive)
│   ├── finite_sample_scaling/    # Finite-sample estimation & data scaling
│   ├── continuous_control/       # Continuous-state reference-policy validation (CartPole-v1)
│   └── empirical_reweighting/    # Empirical risk reweighting benchmark & negative controls
├── tests/                        # Automated unit & integration test suite (22 tests)
├── requirements.txt              # Python dependencies
├── pytest.ini                    # Pytest path configuration
├── conftest.py                   # Root path fixture
├── LICENSE                       # MIT License
└── README.md
```

---

## Quick Start

### 1. Installation
```bash
git clone https://github.com/tonyhyge/decision-boundary-mbrl.git
cd decision-boundary-mbrl

# Optional: create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite
```bash
pytest tests/ -v
```

Expected output:
```
============================== 22 passed in 1.91s ==============================
```

---

## Reproducing Empirical Experiments

All experiments can be executed directly via Python:

### A. Controlled Causal Geometry (Fork MDPs)
```bash
python3 experiments/causal_geometry/run_fork_interventions.py
python3 experiments/causal_geometry/run_geometry_audits.py
python3 experiments/causal_geometry/run_crossing_nonlinearity.py
```

### B. Multi-Error Budgeted Repair Benchmark (Stochastic GridWorld)
```bash
python3 experiments/budgeted_repair/run_repair_benchmark.py
python3 experiments/budgeted_repair/run_adaptive_repair.py
python3 experiments/budgeted_repair/run_sparse_budget_sweep.py
```

### C. Finite-Sample Estimation & Data Scaling
```bash
python3 experiments/finite_sample_scaling/run_finite_sample_benchmark.py
python3 experiments/finite_sample_scaling/run_data_sweep.py
```

### D. Continuous-State Reference-Policy Validation (CartPole-v1)
```bash
python3 experiments/continuous_control/run_cartpole_evaluation.py
python3 experiments/continuous_control/run_cartpole_audits.py
```

### E. Empirical Risk Reweighting Benchmark & Negative Controls
```bash
python3 experiments/empirical_reweighting/run_reweighting_experiment.py
python3 experiments/empirical_reweighting/run_nonconservative_benchmark.py
python3 experiments/empirical_reweighting/run_reweighting_audit.py
```

---

## Citation

*Citation details and BibTeX entry will be updated upon formal preprint / publication.*

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
