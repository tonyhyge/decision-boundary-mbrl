# Stage Q1-FB Environment & Provenance Manifest

- **Stage:** `STAGE_Q1FB_CUDA_FULL_BUDGET`
- **Specification:** `FB2 (Hopper-v4, 125k steps, N=10 paired seeds 142-151)`
- **Git Commit:** `2b83588141afa35aa17fc9d1b88340f9dd71cc48` (Branch: `main`)
- **GPU Hardware:** `NVIDIA GeForce RTX 5090` (`sm_120`)
- **Software Stack:** Python `3.11.15` | PyTorch `2.13.0+cu130` | CUDA `13.0` | cuDNN `92000`
- **Determinism:** `TF32 = False` (Disabled for pure FP32), `Deterministic = False`

### SHA-256 Code Checksums:

| Source File | SHA-256 Checksum |
| :--- | :--- |
| `experiments/q1_upgrade/run_stage_q1fb_full_budget_hopper.py` | `98944b392e00b8527f8d0c36357612f5470eab93fd99a167b5f86b31785e7c78` |
| `experiments/q1_upgrade/run_stage_q1fb_cuda_parallel.py` | `6a4e645fcbd0c62694b904de74c0eb64b589d3c4aeb79cbf7e26d545a81a06fa` |
| `experiments/q1_upgrade/statistical_closure_pipeline.py` | `851703eef7cdc1dd8a6a3c3f1abe862b2460091eca5c642499f32aa836025c57` |
| `experiments/q1_upgrade/audit_cross_device_invariants.py` | `bdb5a3417100968f225243aafa95ed7edffac3e528b35af71bcd70c0152a7097` |
| `experiments/q1_upgrade/audit_run_integrity.py` | `ce434c2ac23469fa58689e949d1aa3e5263d366a2b1ee2ba19ed52a9e3ce5307` |
| `experiments/q1_upgrade/audit_implementation_symmetry.py` | `2a0a5ffb98dddf8f27c24c062cc4ed9b2fef0039c535334da6992c0e55e5925c` |
| `src/baselines/external/vagram/mbrl/models/gaussian_mlp.py` | `2d93836332166a63b41be785c098f049840a62eba68b461e5c5bd8ce632f7104` |
| `src/baselines/external/vagram/mbrl/models/vaml_mlp.py` | `6b1af8659613de6a4ccedcee94b3e8e449b622041579fcbcb25ff8fc186c69ee` |
| `src/baselines/external/vagram/mbrl/third_party/pytorch_sac/agent/sac.py` | `b7131381503f5d9647fec80dfc920324354907d6635451e03d499d57c28aa2ed` |
