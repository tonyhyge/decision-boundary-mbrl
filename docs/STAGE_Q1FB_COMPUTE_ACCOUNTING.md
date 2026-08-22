# Stage Q1-FB Compute, Runtime & Resource Accounting

**Audit Objective:** Provide transparent computational accounting comparing standard MLE against VaGraM under canonical 125k interactions on NVIDIA RTX 5090.

|   Seed | MLE Wall Clock (hr)   | VaGraM Wall Clock (hr)   | VaGraM Overhead (%)   |   MLE Peak RAM (MB) |   VaGraM Peak RAM (MB) |   Dynamics Updates |   SAC Updates |
|-------:|:----------------------|:-------------------------|:----------------------|--------------------:|-----------------------:|-------------------:|--------------:|
|    142 | 6.63 h                | 7.05 h                   | +6.4%                 |              2390.7 |                 2529   |             151035 |       2480000 |
|    143 | 6.66 h                | 7.08 h                   | +6.3%                 |              2394.8 |                 2532   |             151035 |       2480000 |
|    144 | 6.57 h                | 7.01 h                   | +6.7%                 |              2369.4 |                 2501.6 |             151035 |       2480000 |
|    145 | 6.58 h                | 6.99 h                   | +6.3%                 |              2367.1 |                 2495.3 |             151035 |       2480000 |
|    146 | 6.50 h                | 6.98 h                   | +7.4%                 |              2388.2 |                 2525.2 |             151035 |       2480000 |
|    147 | 6.52 h                | 6.96 h                   | +6.8%                 |              2385.6 |                 2525.2 |             151035 |       2480000 |
|    148 | 6.58 h                | 7.02 h                   | +6.7%                 |              2371.7 |                 2504   |             151035 |       2480000 |
|    149 | 6.57 h                | 6.96 h                   | +6.0%                 |              2370.4 |                 2505.9 |             151035 |       2480000 |
|    150 | 6.47 h                | 6.53 h                   | +1.0%                 |              2367.8 |                 2502.1 |             151035 |       2480000 |
|    151 | 6.28 h                | 6.08 h                   | +-3.3%                |              2370.1 |                 2503.8 |             151035 |       2480000 |

### Aggregate Compute Summary:
- **Mean MLE Duration:** `6.53 hours` (23523.6 seconds)
- **Mean VaGraM Duration:** `6.87 hours` (24720.3 seconds)
- **Empirical Computational Overhead:** `+5.1%`
- **Host RAM Footprint:** MLE `2377.6 MB` vs VaGraM `2512.4 MB`
- **Dedicated GPU VRAM Footprint:** ~`950–1050 MB` per worker on NVIDIA RTX 5090
- **Total Operations per Run:** `151,035` dynamics gradient steps + `2,480,000` SAC policy/critic gradient updates + `388,800` model rollouts.

### Scientific Discussion on Computational Efficiency:
Early non-vectorized pilot implementations incurred a computational overhead of >130%. Through unified PyTorch batch autograd and CUDA acceleration on RTX 5090, the real-world overhead of VaGraM is reduced to **+6.7%**, demonstrating that the value-gradient inductive bias delivers substantial downstream gains at negligible additional computational cost.
