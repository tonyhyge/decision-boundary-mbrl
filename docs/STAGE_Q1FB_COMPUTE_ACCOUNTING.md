# Stage Q1-FB Compute, Runtime & Resource Accounting

**Audit Objective:** Provide transparent computational accounting comparing standard MLE against VaGraM under canonical 125k interactions on NVIDIA RTX 5090.

| Seed | MLE Wall Clock (hr) | VaGraM Wall Clock (hr) | VaGraM Overhead (%) | MLE Peak RAM (MB) | VaGraM Peak RAM (MB) | Dynamics Updates | SAC Updates |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **142** | 6.63 h | 7.05 h | +6.4% | 2390.7 | 2529.0 | 151035 | 2480000 |
| **143** | 6.66 h | 7.08 h | +6.3% | 2394.8 | 2532.0 | 151035 | 2480000 |
| **144** | 6.57 h | 7.01 h | +6.7% | 2369.4 | 2501.6 | 151035 | 2480000 |
| **145** | 6.58 h | 6.99 h | +6.3% | 2367.1 | 2495.3 | 151035 | 2480000 |
| **146** | 6.50 h | 6.98 h | +7.4% | 2388.2 | 2525.2 | 151035 | 2480000 |
| **147** | 6.52 h | 6.96 h | +6.8% | 2385.6 | 2525.2 | 151035 | 2480000 |

---

### Aggregate Compute Summary:
- **Mean MLE Duration:** `6.58 hours` ($23,671.2\text{ s}$)
- **Mean VaGraM Duration:** `7.01 hours` ($25,248.8\text{ s}$)
- **Empirical Computational Overhead:** `+6.7%`
- **Host RAM Footprint:** MLE `2382.6 MB` vs VaGraM `2518.0 MB`
- **Dedicated GPU VRAM Footprint:** ~`950–1050 MB` per worker on NVIDIA RTX 5090
- **Total Operations per Run:** `151,035` dynamics gradient steps + `2,480,000` SAC policy/critic gradient updates + `388,800` model rollouts.

---

### Methodological Note on Runtime Measurement:
The reported `+6.7%` overhead is a **descriptive runtime overhead under the audited concurrent CUDA execution regime** (4 workers sharing an NVIDIA RTX 5090 with balanced process scheduling). It is **not a hardware-isolated micro-benchmark** or intrinsic algorithmic complexity proof. However, it empirically demonstrates that in practical deep MBRL workflows with unified PyTorch vectorized autograd, the value-gradient calculation introduces modest wall-clock overhead.
