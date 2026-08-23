"""
Stage Q1-HC: Preregistered Statistical Closure Script for HalfCheetah-v4.
Requires valid token: results/stage_q1hc_cuda/HC_COHORT_VALIDITY_PASS.txt
Computes Primary Paired Estimand (mean terminal return delta), 95% bootstrap CI,
exact permutation sign-flip p-value, and Secondary Training-Budget Interaction.
"""
import os
import sys
import json
import numpy as np
import pandas as pd

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESULTS_DIR = os.path.join(_ROOT, "results", "stage_q1hc_cuda")
_VALIDITY_TOKEN = os.path.join(_RESULTS_DIR, "HC_COHORT_VALIDITY_PASS.txt")

CONFIRMATORY_SEEDS = [142, 143, 144, 145, 146, 147, 148, 149, 150, 151]


def exact_permutation_sign_flip_test(differences: np.ndarray) -> float:
    N = len(differences)
    obs_mean = np.abs(np.mean(differences))
    num_combos = 1 << N  # 2^N = 1024
    more_extreme = 0
    
    for mask in range(num_combos):
        signs = np.array([1 if (mask & (1 << i)) else -1 for i in range(N)])
        perm_mean = np.abs(np.mean(differences * signs))
        if perm_mean >= obs_mean - 1e-9:
            more_extreme += 1
            
    return float(more_extreme / num_combos)


def bootstrap_ci_paired(differences: np.ndarray, num_bootstrap: int = 10000, alpha: float = 0.05, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    N = len(differences)
    indices = rng.integers(0, N, size=(num_bootstrap, N))
    sample_means = np.mean(differences[indices], axis=1)
    low = np.percentile(sample_means, 100 * (alpha / 2.0))
    high = np.percentile(sample_means, 100 * (1.0 - alpha / 2.0))
    return float(low), float(high)


def main():
    if not os.path.exists(_VALIDITY_TOKEN):
        print(f"ERROR: Validity token not found at {_VALIDITY_TOKEN}.")
        print("You must successfully run experiments/q1_upgrade/audit_q1hc_integrity.py first.")
        sys.exit(1)
        
    records = []
    paired_deltas_terminal = []
    paired_budget_shifts = []
    
    for s in CONFIRMATORY_SEEDS:
        mle_file = os.path.join(_RESULTS_DIR, f"q1hc_halfcheetah_mle_seed{s}.json")
        vag_file = os.path.join(_RESULTS_DIR, f"q1hc_halfcheetah_vagram_seed{s}.json")
        
        with open(mle_file, "r") as f: m_data = json.load(f)
        with open(vag_file, "r") as f: v_data = json.load(f)
        
        m_term = m_data["terminal_return_mean"]
        v_term = v_data["terminal_return_mean"]
        delta_term = v_term - m_term
        paired_deltas_terminal.append(delta_term)
        
        # 10k delta vs 125k delta
        m_10k = next(h["eval_mean_return"] for h in m_data["evaluation_history"] if h["step"] == 10000)
        v_10k = next(h["eval_mean_return"] for h in v_data["evaluation_history"] if h["step"] == 10000)
        delta_10k = v_10k - m_10k
        
        m_125k = next(h["eval_mean_return"] for h in m_data["evaluation_history"] if h["step"] == 125000)
        v_125k = next(h["eval_mean_return"] for h in v_data["evaluation_history"] if h["step"] == 125000)
        delta_125k = v_125k - m_125k
        
        H_i = delta_125k - delta_10k
        paired_budget_shifts.append(H_i)
        
        records.append({
            "seed": s,
            "mle_terminal": m_term,
            "vagram_terminal": v_term,
            "paired_delta_terminal": delta_term,
            "vagram_wins": delta_term > 0,
            "delta_10k": delta_10k,
            "delta_125k": delta_125k,
            "training_budget_shift_H": H_i,
        })
        
    df = pd.DataFrame(records)
    deltas = np.array(paired_deltas_terminal)
    shifts = np.array(paired_budget_shifts)
    
    mean_delta = float(np.mean(deltas))
    ci_low, ci_high = bootstrap_ci_paired(deltas)
    p_sign_flip = exact_permutation_sign_flip_test(deltas)
    wins = int(np.sum(deltas > 0))
    
    mean_h = float(np.mean(shifts))
    h_ci_low, h_ci_high = bootstrap_ci_paired(shifts)
    p_h_sign_flip = exact_permutation_sign_flip_test(shifts)
    h_pos = int(np.sum(shifts > 0))
    
    print("=" * 80)
    print("STAGE Q1-HC (HALFCHEETAH-V4) CONFIRMATORY STATISTICAL CLOSURE")
    print("=" * 80)
    print(df.to_markdown(index=False))
    
    print(f"\n--- Primary Terminal Return Estimand (N={len(CONFIRMATORY_SEEDS)}) ---")
    print(f"Mean Paired Delta:              {mean_delta:+.2f}")
    print(f"95% Paired Bootstrap CI:        [{ci_low:+.2f}, {ci_high:+.2f}]")
    print(f"Median Paired Delta:            {float(np.median(deltas)):+.2f}")
    print(f"Exact Permutation p-value:      {p_sign_flip:.6f}")
    print(f"VaGraM Wins vs MLE:             {wins} / {len(CONFIRMATORY_SEEDS)} ({wins/len(CONFIRMATORY_SEEDS)*100:.1f}%)")
    
    closure_summary = {
        "protocol": "STAGE_Q1HC_HALFCHEETAH_CONFIRMATORY",
        "cohort_size": len(CONFIRMATORY_SEEDS),
        "seeds": CONFIRMATORY_SEEDS,
        "primary_estimand": {
            "name": "Mean Paired Terminal Return Delta (115k-125k window)",
            "mean_paired_delta": mean_delta,
            "ci_95_bootstrap": [ci_low, ci_high],
            "median_paired_delta": float(np.median(deltas)),
            "exact_permutation_p_value": p_sign_flip,
            "wins": wins,
            "win_rate": wins / len(CONFIRMATORY_SEEDS),
        },
        "secondary_budget_interaction": {
            "name": "Training-Budget Shift Estimand H = Delta(125k) - Delta(10k)",
            "mean_shift_H": mean_h,
            "ci_95_bootstrap": [h_ci_low, h_ci_high],
            "exact_permutation_p_value": p_h_sign_flip,
            "positive_shifts": h_pos,
        },
        "per_seed_breakdown": records,
    }
    
    closure_file = os.path.join(_RESULTS_DIR, "stage_q1hc_confirmatory_closure.json")
    with open(closure_file, "w") as f:
        json.dump(closure_summary, f, indent=2)
    print(f"\nSaved closure record to: {closure_file}")


if __name__ == "__main__":
    main()
