# FB3 Confirmatory Run Integrity & Provenance Audit

**Audit Objective:** Verify that 100% of completed confirmatory runs strictly adhere to the frozen FB2 specification without truncation, missing checkpoints, or numerical corruption.

|   Seed | Method   |   Steps | Checkpoints   | Terminal Evals   |   Dyn Updates |   SAC Updates |   Imagined Rollouts | No NaN/Inf   |   Wall Clock (s) |   Peak RAM (MB) | Git Commit   | Integrity Verdict   |
|-------:|:---------|--------:|:--------------|:-----------------|--------------:|--------------:|--------------------:|:-------------|-----------------:|----------------:|:-------------|:--------------------|
|    142 | MLE      |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          23863.5 |          2390.7 | 2b835881     | PASS                |
|    142 | VAGRAM   |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          25397.9 |          2529   | 2b835881     | PASS                |
|    143 | MLE      |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          23967.7 |          2394.8 | 2b835881     | PASS                |
|    143 | VAGRAM   |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          25488.9 |          2532   | 2b835881     | PASS                |
|    144 | MLE      |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          23658   |          2369.4 | 2b835881     | PASS                |
|    144 | VAGRAM   |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          25233.8 |          2501.6 | 2b835881     | PASS                |
|    145 | MLE      |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          23678.9 |          2367.1 | 2b835881     | PASS                |
|    145 | VAGRAM   |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          25172.4 |          2495.3 | 2b835881     | PASS                |
|    146 | MLE      |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          23401.5 |          2388.2 | 2b835881     | PASS                |
|    146 | VAGRAM   |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          25144.7 |          2525.2 | 2b835881     | PASS                |
|    147 | MLE      |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          23463.9 |          2385.6 | 2b835881     | PASS                |
|    147 | VAGRAM   |  125000 | 25/25         | PASS             |        151035 |       2480000 |              388800 | PASS         |          25055.3 |          2525.2 | 2b835881     | PASS                |

### Integrity Verification Checklist:
-  **Target Environment:** All runs executed strictly on `Hopper-v4`.
-  **Interaction Budget:** Exactly 125,000 environment steps per run.
-  **Evaluation Cadence:** Exactly 25 deterministic evaluation checkpoints (every 5,000 steps) across 10 evaluation episodes per checkpoint.
-  **Terminal Checkpoints:** 115k, 120k, and 125k evaluations verified intact for all runs.
-  **Update Accounting:** Exactly 151,035 dynamics updates, 2,480,000 SAC updates, and 388,800 imagined rollouts per run.
-  **Numerical Cleanliness:** Zero NaN, Inf, or exception tracebacks detected across all runs.
-  **Provenance Lock:** Single unified codebase commit hash recorded across all runs.
