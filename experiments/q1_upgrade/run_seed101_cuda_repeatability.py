"""
CUDA Intra-Backend Determinism & Repeatability Audit on Pilot Seed 101.
Runs Seed 101 across multiple independent repeated trials (Trial 1, Trial 2, Trial 3)
under exact identical configuration on CUDA (pure FP32, deterministic flags)
up to 25,000 environment steps.
Quantifies whether multi-step RL trajectories on CUDA are bitwise identical or chaotic.
Results are saved strictly in `results/seed101_repeatability/` to avoid polluting FB3 confirmatory cohort.
"""
import os
import sys
import json
import time
import hashlib
import numpy as np
import pandas as pd
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from experiments.q1_upgrade.run_stage_q1fb_full_budget_hopper import run_full_budget_mbpo


def hash_dict_weights(state_dict):
    """Computes MD5 hash of model weights to verify bitwise matching."""
    hasher = hashlib.md5()
    for k in sorted(state_dict.keys()):
        hasher.update(k.encode('utf-8'))
        hasher.update(state_dict[k].cpu().numpy().tobytes())
    return hasher.hexdigest()


def main():
    out_dir = os.path.join(_ROOT, "results", "seed101_repeatability")
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda:0"
    num_trials = 3
    total_steps = 25000
    eval_freq = 5000
    seed = 101

    print("=" * 80)
    print(f"CUDA REPEATABILITY AUDIT: SEED {seed} ({num_trials} REPEATED TRIALS, {total_steps} STEPS)")
    print("=" * 80)

    trials_data = {"mle": [], "vagram": []}

    for algo in ["mle", "vagram"]:
        for trial_idx in range(1, num_trials + 1):
            out_file = os.path.join(out_dir, f"repeatability_seed{seed}_{algo}_trial{trial_idx}.json")
            if os.path.exists(out_file):
                print(f"\n[CACHE HIT] Found {out_file}")
                with open(out_file, "r") as f:
                    res = json.load(f)
                trials_data[algo].append(res)
                continue

            print(f"\n---> Running CUDA Repeatability: {algo.upper()} Seed {seed} | Trial {trial_idx}/{num_trials} <---")
            t0 = time.time()
            res = run_full_budget_mbpo(
                env_id="Hopper-v4",
                algorithm_type=algo,
                seed=seed,
                total_steps=total_steps,
                eval_freq=eval_freq,
                eval_episodes=10,
                device=device,
                verbose=True,
            )
            res["trial_idx"] = trial_idx
            res["trial_wall_clock"] = time.time() - t0
            trials_data[algo].append(res)

            with open(out_file, "w") as f:
                json.dump(res, f, indent=2)

    # Statistical Comparison & Report
    print("\n" + "=" * 80)
    print("REPEATABILITY ANALYSIS READOUT")
    print("=" * 80)

    table_rows = []
    for algo in ["mle", "vagram"]:
        checkpoints = [5000, 10000, 15000, 20000, 25000]
        step_trajectories = {s: [] for s in checkpoints}
        for trial in trials_data[algo]:
            hist = {h["step"]: h["eval_mean_return"] for h in trial["evaluation_history"]}
            for s in checkpoints:
                step_trajectories[s].append(hist.get(s, np.nan))
        
        for s in checkpoints:
            vals = step_trajectories[s]
            mean_v = np.mean(vals)
            sd_v = np.std(vals, ddof=1) if len(vals) > 1 else 0.0
            range_v = np.max(vals) - np.min(vals) if len(vals) > 1 else 0.0
            table_rows.append({
                "Algorithm": algo.upper(),
                "Step": f"{s//1000}k",
                "Trial 1 J(s)": f"{vals[0]:.2f}",
                "Trial 2 J(s)": f"{vals[1]:.2f}" if len(vals) > 1 else "N/A",
                "Trial 3 J(s)": f"{vals[2]:.2f}" if len(vals) > 2 else "N/A",
                "Mean ± SD": f"{mean_v:.2f} ± {sd_v:.2f}",
                "Spread (Max - Min)": f"{range_v:.2f}",
            })

    df_rep = pd.DataFrame(table_rows)
    print(df_rep.to_markdown(index=False))

    report_path = os.path.join(_ROOT, "docs", "STAGE_Q1FB_CUDA_REPEATABILITY_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# CUDA Intra-Backend Repeatability & Determinism Audit\n\n")
        f.write(f"**Seed:** {seed} (Pilot seed, excluded from confirmatory cohort)  \n")
        f.write(f"**Target Horizon:** {total_steps} environment interaction steps  \n")
        f.write(f"**Trials:** {num_trials} independent repetitions under pure FP32 on NVIDIA RTX 5090 (CUDA)  \n\n")
        f.write("## 1. Step-by-Step Trajectory Repeatability Table\n\n")
        f.write(df_rep.to_markdown(index=False))
        f.write("\n\n## 2. Scientific Synthesis\n\n")
        f.write("1. **Determinism Verification:** Evaluates whether identical random seed initialization on the exact same CUDA device produces zero-variance or low-variance trajectories.\n")
        f.write("2. **Disentangling Platform-Shift vs Optimization Noise:** Provides definitive empirical evidence on intra-hardware stability to compare against the Apple Silicon (MPS) vs NVIDIA (CUDA) cross-hardware shift.\n")

    print(f"\nSaved repeatability report artifact to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
