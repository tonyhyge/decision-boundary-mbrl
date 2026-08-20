"""
Stage 4B.0: Curvature Contribution Closure Pre-Gate Benchmark.

Evaluates whether dividing by estimated Hessian curvature S_cont = sigma_grad_Q / (|H_hat| + eps)
yields any statistically detectable acquisition utility over raw action-gradient disagreement sigma_grad_Q
and shuffled-curvature control S_cont^(shuffled-H) in a model repair cycle.

Pre-Registered Gate Rule:
  Require S_cont > sigma_grad_Q AND S_cont > S_cont^(shuffled-H) with 95% bootstrap CI > 0.
  If not established -> Canonical intervention score for Stage 4B is strictly Action-Gradient Disagreement (sigma_grad_Q).
"""
import os
import sys
import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.continuous_control_geometry import ContinuousCartPoleEnv, ExactLQRGeometrySolver
from src.continuous_geometry.continuous_risk_detector import (
    ContinuousEnsembleDynamics,
    ContinuousUnprivilegedSignalExtractor,
)
from experiments.continuous_geometry.run_stage3b_continuous_geometry import collect_continuous_training_data


def run_stage4b0_curvature_closure(
    output_dir: str = "results",
    seed: int = 42,
    budget_pcts: List[float] = [0.05, 0.10],
    n_seeds: int = 10,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    print("================================================================================")
    print("   STAGE 4B.0: CURVATURE CONTRIBUTION CLOSURE BENCHMARK                        ")
    print("================================================================================")

    env = ContinuousCartPoleEnv()
    solver = ExactLQRGeometrySolver(env=env, r_cost=0.1)

    # 1. Base dataset and initial ensemble
    print("\n[Step 1] Initializing base ensemble and candidate pool...")
    x_base, y_base = collect_continuous_training_data(env, solver, num_episodes=30, max_steps=60, seed=seed)
    base_ensemble = ContinuousEnsembleDynamics(num_models=7, hidden_dim=64, seed=seed)
    base_ensemble.train_ensemble(x_base, y_base, epochs=60, batch_size=64, seed=seed)

    # 2. Candidate pool and held-out evaluation set
    extractor = ContinuousUnprivilegedSignalExtractor(ensemble=base_ensemble, solver=solver, env=env)
    rng_pool = np.random.default_rng(seed + 100)

    pool_records = []
    for _ in range(30):
        s = rng_pool.uniform(low=[-0.25, -0.25, -0.10, -0.12], high=[0.25, 0.25, 0.10, 0.12])
        for _ in range(50):
            sig = extractor.extract_state_signals(s)
            pool_records.append({"s": s, **sig})
            ns, _, done = env.step(s, sig["a_hat"])
            s = ns
            if done:
                break
    df_pool = pd.DataFrame(pool_records)

    eval_records = []
    rng_eval = np.random.default_rng(seed + 200)
    for _ in range(25):
        s = rng_eval.uniform(low=[-0.25, -0.25, -0.10, -0.12], high=[0.25, 0.25, 0.10, 0.12])
        for _ in range(50):
            sig = extractor.extract_state_signals(s)
            eval_records.append({"s": s, **sig})
            ns, _, done = env.step(s, sig["a_hat"])
            s = ns
            if done:
                break
    df_eval = pd.DataFrame(eval_records)

    n_pool = len(df_pool)
    n_eval = len(df_eval)
    base_eval_regret = float(df_eval["regret"].mean())
    print(f"-> Candidate Pool: {n_pool} states | Evaluation Set: {n_eval} states.")
    print(f"-> Base Ensemble Mean Decision Regret on Eval Set: {base_eval_regret:.6f}")

    # Strategies compared for Gate 4B.0
    # 1. sigma_grad_Q
    # 2. S_cont = sigma_grad_Q / (curvature + eps)
    # 3. S_cont^(shuffled-H) = sigma_grad_Q * permuted(inv_H)
    # 4. inv_H only

    b_10pct = max(1, int(np.round(0.10 * n_pool)))
    print(f"\n[Step 2] Running 10-seed repair comparison at B = 10% ({b_10pct} queries)...")

    regrets_grad_q = []
    regrets_s_cont = []
    regrets_shuffled_h = []
    regrets_inv_h_only = []

    for s_idx in range(n_seeds):
        rng_s = np.random.default_rng(seed + s_idx * 31)

        # Strategy 1: sigma_grad_Q
        idx_grad_q = np.argsort(-df_pool["sigma_grad_q"].to_numpy())[:b_10pct]
        # Strategy 2: S_cont
        idx_s_cont = np.argsort(-df_pool["s_cont"].to_numpy())[:b_10pct]
        # Strategy 3: S_cont^(shuffled-H)
        shuffled_inv_h = rng_s.permutation(df_pool["inv_h_hat"].to_numpy())
        score_shuffled = df_pool["sigma_grad_q"].to_numpy() * shuffled_inv_h
        idx_shuffled_h = np.argsort(-score_shuffled)[:b_10pct]
        # Strategy 4: inv_H only
        idx_inv_h = np.argsort(-df_pool["inv_h_hat"].to_numpy())[:b_10pct]

        for name, q_idx, reg_list in [
            ("sigma_grad_q", idx_grad_q, regrets_grad_q),
            ("s_cont", idx_s_cont, regrets_s_cont),
            ("shuffled_h", idx_shuffled_h, regrets_shuffled_h),
            ("inv_h_only", idx_inv_h, regrets_inv_h_only),
        ]:
            # Collect transitions (s, a_hat, f_P(s, a_hat))
            x_q, y_q = [], []
            for i in q_idx:
                row = df_pool.iloc[i]
                s_val = row["s"]
                a_val = row["a_hat"]
                ns_val, _, _ = env.step(s_val, a_val)
                x_q.append(np.concatenate([s_val, [a_val]]))
                y_q.append(ns_val - s_val)

            x_aug = np.concatenate([x_base, np.array(x_q, dtype=np.float32)], axis=0)
            y_aug = np.concatenate([y_base, np.array(y_q, dtype=np.float32)], axis=0)

            # Fine-tune ensemble clone
            ens_repaired = ContinuousEnsembleDynamics(num_models=7, hidden_dim=64, seed=seed + s_idx)
            ens_repaired.train_ensemble(x_aug, y_aug, epochs=40, batch_size=64, seed=seed + s_idx)

            # Evaluate mean decision regret on eval set
            ext_rep = ContinuousUnprivilegedSignalExtractor(ensemble=ens_repaired, solver=solver, env=env)
            reg_evals = []
            for _, r_ev in df_eval.iterrows():
                sig_ev = ext_rep.extract_state_signals(r_ev["s"])
                reg_evals.append(sig_ev["regret"])
            reg_list.append(float(np.mean(reg_evals)))

    mean_reg_grad = float(np.mean(regrets_grad_q))
    mean_reg_scont = float(np.mean(regrets_s_cont))
    mean_reg_shuf = float(np.mean(regrets_shuffled_h))
    mean_reg_invh = float(np.mean(regrets_inv_h_only))

    paired_diff_scont_vs_grad = np.array(regrets_grad_q) - np.array(regrets_s_cont)  # Positive if S_cont has lower regret
    paired_diff_scont_vs_shuf = np.array(regrets_shuffled_h) - np.array(regrets_s_cont)

    t_grad, p_grad = stats.ttest_rel(regrets_s_cont, regrets_grad_q)
    t_shuf, p_shuf = stats.ttest_rel(regrets_s_cont, regrets_shuffled_h)

    print("\n--- STAGE 4B.0 EMPIRICAL RESULTS (N = 10 Seeds) ---")
    print(f"  1. Action-Gradient Disagreement (sigma_grad_Q): Mean Regret = {mean_reg_grad:.6f} +/- {np.std(regrets_grad_q):.6f}")
    print(f"  2. Proposed Continuous Score (S_cont)         : Mean Regret = {mean_reg_scont:.6f} +/- {np.std(regrets_s_cont):.6f}")
    print(f"  3. Shuffled-Curvature Control                 : Mean Regret = {mean_reg_shuf:.6f} +/- {np.std(regrets_shuffled_h):.6f}")
    print(f"  4. Inverse Curvature Only (|H|^-1)            : Mean Regret = {mean_reg_invh:.6f} +/- {np.std(regrets_inv_h_only):.6f}")

    print(f"\n  Paired Gain S_cont vs sigma_grad_Q: {np.mean(paired_diff_scont_vs_grad):+.6f} (t = {t_grad:.3f}, p = {p_grad:.4f})")
    print(f"  Paired Gain S_cont vs Shuffled-H   : {np.mean(paired_diff_scont_vs_shuf):+.6f} (t = {t_shuf:.3f}, p = {p_shuf:.4f})")

    # Check Pre-registered Gate 4B.0
    curvature_advantage_established = bool(np.mean(paired_diff_scont_vs_grad) > 0 and p_grad < 0.05 and np.mean(paired_diff_scont_vs_shuf) > 0 and p_shuf < 0.05)

    if curvature_advantage_established:
        canonical_score = "S_cont (Curvature-Normalized Action-Gradient Disagreement)"
        status = "ESTABLISHED"
    else:
        canonical_score = "sigma_grad_Q (Action-Gradient Epistemic Disagreement)"
        status = "NOT_ESTABLISHED / FROZEN_TO_ACTION_GRADIENT"

    print(f"\n================================================================================")
    print(f"   GATE 4B.0 VERDICT: [{status}]")
    print(f"   CANONICAL INTERVENTION SCORE FROZEN: [{canonical_score}]")
    print(f"================================================================================")

    out_json = {
        "gate_4b0_status": status,
        "canonical_score": canonical_score,
        "mean_regret_grad_q": mean_reg_grad,
        "mean_regret_s_cont": mean_reg_scont,
        "mean_regret_shuffled_h": mean_reg_shuf,
        "mean_regret_inv_h_only": mean_reg_invh,
        "p_val_vs_grad_q": float(p_grad),
        "p_val_vs_shuffled_h": float(p_shuf),
    }
    with open(os.path.join(output_dir, "stage4b0_curvature_closure_report.json"), "w") as f:
        json.dump(out_json, f, indent=2)

    return out_json


if __name__ == "__main__":
    run_stage4b0_curvature_closure()
