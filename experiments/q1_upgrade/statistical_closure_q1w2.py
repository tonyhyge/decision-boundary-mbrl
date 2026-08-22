"""
Stage Q1-W2: Frozen Statistical Closure Pipeline for Walker2d-v4 Replication.
Encodes pre-specified estimands, exact two-sided sign-flip permutation tests (2^10=1024),
and paired percentile bootstrap CIs (B=10,000) under mandatory validity gate enforcement.
"""
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESULTS_DIR = os.path.join(_ROOT, "results", "stage_q1w2_cuda")
_VALIDITY_TOKEN_FILE = os.path.join(_RESULTS_DIR, "W2_COHORT_VALIDITY_PASS.txt")

CONFIRMATORY_SEEDS = [142, 143, 144, 145, 146, 147, 148, 149, 150, 151]


def compute_exact_sign_flip_p_value(deltas: np.ndarray) -> float:
    """Exact two-sided permutation test over all 2^N sign flips."""
    N = len(deltas)
    assert N <= 20, "Exact enumeration intended for N <= 20"
    t_obs = abs(float(np.mean(deltas)))
    count_extreme = 0
    total_permutations = 2 ** N
    for signs in itertools.product([-1.0, 1.0], repeat=N):
        s_arr = np.array(signs, dtype=np.float64)
        t_perm = abs(float(np.mean(s_arr * deltas)))
        if t_perm >= t_obs - 1e-12:
            count_extreme += 1
    return float(count_extreme / total_permutations)


def compute_paired_bootstrap_ci(deltas: np.ndarray, B: int = 10000, alpha: float = 0.05, seed: int = 2026) -> Tuple[float, float]:
    """Paired percentile bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    N = len(deltas)
    boot_means = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.choice(N, size=N, replace=True)
        boot_means[b] = np.mean(deltas[idx])
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return ci_lower, ci_upper


def main():
    if not os.path.exists(_VALIDITY_TOKEN_FILE):
        print(f"[GATE ERROR] Cohort validity token missing: {_VALIDITY_TOKEN_FILE}")
        print("You must execute 'audit_q1w2_integrity.py' and pass 100% before statistical closure.")
        return 1

    per_seed_records = []
    deltas = []
    h_shifts = []

    for s in CONFIRMATORY_SEEDS:
        mf = os.path.join(_RESULTS_DIR, f"q1w2_walker2d_mle_seed{s}.json")
        vf = os.path.join(_RESULTS_DIR, f"q1w2_walker2d_vagram_seed{s}.json")
        
        with open(mf) as f: m_data = json.load(f)
        with open(vf) as f: v_data = json.load(f)
        
        m_term = float(m_data["terminal_return_mean"])
        v_term = float(v_data["terminal_return_mean"])
        d_term = v_term - m_term
        
        # 10k returns
        m_10k = float(m_data["evaluation_history"][1]["eval_mean_return"]) if len(m_data["evaluation_history"]) > 1 else m_term
        v_10k = float(v_data["evaluation_history"][1]["eval_mean_return"]) if len(v_data["evaluation_history"]) > 1 else v_term
        d_10k = v_10k - m_10k
        
        m_125k = float(m_data["evaluation_history"][-1]["eval_mean_return"])
        v_125k = float(v_data["evaluation_history"][-1]["eval_mean_return"])
        d_125k = v_125k - m_125k
        h_shift = d_125k - d_10k
        
        deltas.append(d_term)
        h_shifts.append(h_shift)
        
        per_seed_records.append({
            "seed": s,
            "mle_terminal": m_term,
            "vagram_terminal": v_term,
            "paired_delta_terminal": d_term,
            "vagram_wins": bool(d_term > 0),
            "delta_10k": d_10k,
            "delta_125k": d_125k,
            "training_budget_shift_H": h_shift
        })

    deltas_arr = np.array(deltas, dtype=np.float64)
    h_arr = np.array(h_shifts, dtype=np.float64)

    mean_delta = float(np.mean(deltas_arr))
    median_delta = float(np.median(deltas_arr))
    std_delta = float(np.std(deltas_arr, ddof=1))
    ci_lower, ci_upper = compute_paired_bootstrap_ci(deltas_arr, B=10000, alpha=0.05, seed=2026)
    p_val = compute_exact_sign_flip_p_value(deltas_arr)
    win_count = int(np.sum(deltas_arr > 0))
    win_rate = float(win_count / len(deltas_arr))

    mean_h = float(np.mean(h_arr))
    median_h = float(np.median(h_arr))
    ci_h_lower, ci_h_upper = compute_paired_bootstrap_ci(h_arr, B=10000, alpha=0.05, seed=2026)
    p_val_h = compute_exact_sign_flip_p_value(h_arr)
    pos_h = int(np.sum(h_arr > 0))

    summary = {
        "benchmark": "Stage Q1-W2: Walker2d-v4 Full-Budget MBPO Replication",
        "cohort_size_N": len(CONFIRMATORY_SEEDS),
        "primary_estimand": {
            "name": "Mean Paired Terminal Return Delta (115k-125k)",
            "mean_paired_delta": mean_delta,
            "median_paired_delta": median_delta,
            "std_paired_delta": std_delta,
            "bootstrap_95_ci": [ci_lower, ci_upper],
            "exact_permutation_p_value": p_val,
            "win_rate": win_rate,
            "win_count": win_count,
        },
        "secondary_training_budget_endpoint": {
            "name": "Training-Budget Interaction Endpoint H_i = Delta(125k) - Delta(10k)",
            "mean_H": mean_h,
            "median_H": median_h,
            "bootstrap_95_ci": [ci_h_lower, ci_h_upper],
            "exact_permutation_p_value": p_val_h,
            "positive_fraction": float(pos_h / len(h_arr)),
        },
        "per_seed_breakdown": per_seed_records
    }

    out_json = os.path.join(_RESULTS_DIR, "stage_q1w2_confirmatory_closure.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    df = pd.DataFrame(per_seed_records)
    out_csv = os.path.join(_RESULTS_DIR, "stage_q1w2_confirmatory_per_seed.csv")
    df.to_csv(out_csv, index=False)

    print("=" * 80)
    print("STAGE Q1-W2 (WALKER2D-V4) CONFIRMATORY STATISTICAL CLOSURE")
    print("=" * 80)
    print(df.to_markdown(index=False))
    print("\n--- Primary Terminal Return Estimand (N=10) ---")
    print(f"Mean Paired Delta:              {mean_delta:+.2f}")
    print(f"95% Paired Bootstrap CI:        [{ci_lower:+.2f}, {ci_upper:+.2f}]")
    print(f"Median Paired Delta:            {median_delta:+.2f}")
    print(f"Exact Permutation p-value:      {p_val:.6f}")
    print(f"VaGraM Wins vs MLE:             {win_count} / {len(deltas_arr)} ({win_rate*100:.1f}%)")
    print(f"\nSaved closure record to: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
