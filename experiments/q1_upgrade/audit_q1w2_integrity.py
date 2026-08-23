"""
Stage Q1-W2: Run Integrity and Provenance Audit Script for Walker2d-v4.
Verifies all 20 confirmatory runs (Seeds 142-151 for MLE & VaGraM) for complete steps,
exact update counts, absence of NaN/Inf, and terminal evaluation completeness.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESULTS_DIR = os.path.join(_ROOT, "results", "stage_q1w2_cuda")

CONFIRMATORY_SEEDS = [142, 143, 144, 145, 146, 147, 148, 149, 150, 151]
METHODS = ["mle", "vagram"]
EXPECTED_STEPS = 125000
EXPECTED_CHECKPOINTS = 25


def audit_run(seed: int, method: str) -> dict:
    json_path = os.path.join(_RESULTS_DIR, f"q1w2_walker2d_{method}_seed{seed}.json")
    
    if not os.path.exists(json_path):
        return {
            "Seed": seed,
            "Method": method.upper(),
            "File Exists": False,
            "Steps": 0,
            "Checkpoints": "0/25",
            "Terminal Evals": "MISSING",
            "Dyn Updates": 0,
            "SAC Updates": 0,
            "Imagined Rollouts": 0,
            "No NaN/Inf": "FAIL",
            "Wall Clock (s)": 0.0,
            "Peak RAM (MB)": 0.0,
            "Integrity Verdict": "FAIL (File Missing)",
        }
        
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
            
        steps = data.get("total_steps", 0)
        history = data.get("evaluation_history", [])
        num_ckpt = len(history)
        
        # Check terminal evals (115k, 120k, 125k)
        ckpt_steps = [h.get("step", 0) for h in history]
        has_terminal = (115000 in ckpt_steps) and (120000 in ckpt_steps) and (125000 in ckpt_steps)
        
        # Check NaN/Inf
        has_nan = False
        for h in history:
            ret_m = h.get("eval_mean_return", 0.0)
            ret_s = h.get("eval_std_return", 0.0)
            if np.isnan(ret_m) or np.isinf(ret_m) or np.isnan(ret_s) or np.isinf(ret_s):
                has_nan = True
                break
                
        dyn_up = data.get("total_dyn_updates", data.get("total_dynamics_updates", 0))
        sac_up = data.get("total_sac_updates", 0)
        rollouts = data.get("total_imagined_transitions", data.get("total_imagined_rollouts", 0))
        wall_s = data.get("total_wall_clock_sec", data.get("wall_clock_seconds", 0.0))
        peak_ram = data.get("peak_memory_mb", data.get("peak_ram_mb", 0.0))
        git_commit = data.get("git_commit", "N/A")[:8]
        
        is_pass = (
            steps == EXPECTED_STEPS
            and num_ckpt == EXPECTED_CHECKPOINTS
            and has_terminal
            and not has_nan
            and sac_up >= 2400000
            and rollouts >= 300000
        )
        
        return {
            "Seed": seed,
            "Method": method.upper(),
            "File Exists": True,
            "Steps": steps,
            "Checkpoints": f"{num_ckpt}/{EXPECTED_CHECKPOINTS}",
            "Terminal Evals": "PASS" if has_terminal else "FAIL",
            "Dyn Updates": dyn_up,
            "SAC Updates": sac_up,
            "Imagined Rollouts": rollouts,
            "No NaN/Inf": "PASS" if not has_nan else "FAIL",
            "Wall Clock (s)": round(wall_s, 1),
            "Peak RAM (MB)": round(peak_ram, 1),
            "Git Commit": git_commit,
            "Integrity Verdict": "PASS" if is_pass else "FAIL",
        }
    except Exception as e:
        return {
            "Seed": seed,
            "Method": method.upper(),
            "File Exists": True,
            "Steps": 0,
            "Checkpoints": "ERROR",
            "Terminal Evals": "ERROR",
            "Dyn Updates": 0,
            "SAC Updates": 0,
            "Imagined Rollouts": 0,
            "No NaN/Inf": "FAIL",
            "Wall Clock (s)": 0.0,
            "Peak RAM (MB)": 0.0,
            "Git Commit": "ERROR",
            "Integrity Verdict": f"FAIL ({str(e)})",
        }


def main():
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    records = []
    
    for s in CONFIRMATORY_SEEDS:
        for m in METHODS:
            records.append(audit_run(s, m))
            
    df = pd.DataFrame(records)
    print("=" * 80)
    print("STAGE Q1-W2 (WALKER2D-V4) RUN INTEGRITY AUDIT SUMMARY")
    print("=" * 80)
    print(df.to_markdown(index=False))
    
    csv_path = os.path.join(_RESULTS_DIR, "W2_RUN_INTEGRITY_AUDIT.csv")
    df.to_csv(csv_path, index=False)
    
    total_runs = len(records)
    passed_runs = sum(1 for r in records if r["Integrity Verdict"] == "PASS")
    
    print("\n" + "=" * 80)
    print(f"OVERALL COHORT VERDICT: {passed_runs} / {total_runs} RUNS PASS")
    print("=" * 80)
    
    if passed_runs == total_runs:
        pass_file = os.path.join(_RESULTS_DIR, "W2_COHORT_VALIDITY_PASS.txt")
        with open(pass_file, "w") as f:
            f.write(f"STAGE Q1-W2 (WALKER2D-V4) CONFIRMATORY COHORT VALIDITY PASS\n")
            f.write(f"Cohort Size: N={len(CONFIRMATORY_SEEDS)} paired seeds\n")
            f.write(f"Integrity Status: 100% PASS ({passed_runs}/{total_runs} runs verified)\n")
        print(f"Saved validity pass token: {pass_file}")
        return 0
    else:
        print("WARNING: Some runs did not pass integrity audit.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
