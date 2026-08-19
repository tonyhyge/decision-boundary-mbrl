"""
Diagnostic audit script for Gate P1 & P2 across multiple lambda values and both Host A and Host B.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.gridworld_mdp import make_stochastic_choice_gridworld
from src.planning.dp import value_iteration, expected_discounted_return
from src.models.tabular_learned_model import collect_gridworld_experience
from src.models.portability_models import (
    WeightedCategoricalWorldModel,
    ProbabilisticEnsembleWorldModel,
)
from src.correction.portability_weighting import compute_sample_scores
from experiments.empirical_reweighting.run_reweighting_experiment import evaluate_policy_quality


def run_lambda_sweep_audit(num_seeds: int = 30, base_seed: int = 700):
    lambdas = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
    conditions = ["uniform", "prediction_error", "estimated_crossing", "shuffled_crossing", "oracle_crossing"]
    
    records = []

    print(f"Running full diagnostic audit across {num_seeds} seeds and lambda in {lambdas}...")
    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx
        rng = np.random.default_rng(seed)

        true_grid = make_stochastic_choice_gridworld(height=5, width=5, seed=seed)
        V_star, Q_star, pi_star = value_iteration(true_grid)

        train_data = collect_gridworld_experience(true_grid, num_trajectories=80, max_steps=40, seed=seed)
        val_data = collect_gridworld_experience(true_grid, num_trajectories=40, max_steps=40, seed=seed + 5000)

        # Baseline initial model
        init_model = WeightedCategoricalWorldModel(25, 4)
        init_model.fit(train_data, epochs=80, seed=seed)
        init_mdp = init_model.create_learned_mdp(true_grid)
        init_p_mat = init_model.get_transition_matrix()

        for lam in lambdas:
            for host_name in ["Host_A_Deterministic", "Host_B_Ensemble"]:
                for cond in conditions:
                    if cond == "uniform" and lam != 0.0:
                        continue  # Uniform does not depend on lambda

                    actual_cond = "uniform" if lam == 0.0 else cond
                    scores, weights = compute_sample_scores(
                        dataset=train_data,
                        condition=actual_cond,
                        true_mdp=true_grid,
                        initial_mdp=init_mdp,
                        initial_p_matrix=init_p_mat,
                        rng=rng,
                        lambda_val=lam,
                    )

                    t0 = time.perf_counter()
                    if host_name == "Host_A_Deterministic":
                        model = WeightedCategoricalWorldModel(25, 4)
                        model.fit(train_data, sample_weights=weights, epochs=80, seed=seed)
                        param_count = sum(p.numel() for p in model.net.parameters())
                    else:
                        model = ProbabilisticEnsembleWorldModel(25, 4, ensemble_size=5)
                        model.fit(train_data, sample_weights=weights, epochs=80, base_seed=seed)
                        param_count = sum(sum(p.numel() for p in m.parameters()) for m in model.members)
                    
                    wall_clock = time.perf_counter() - t0
                    learned_mdp = model.create_learned_mdp(true_grid)
                    metrics = evaluate_policy_quality(true_grid, learned_mdp, Q_star, pi_star)

                    records.append({
                        "seed": seed,
                        "host": host_name,
                        "condition": cond,
                        "lambda": lam,
                        "j_learned": metrics["j_learned"],
                        "j_optimal": metrics["j_optimal"],
                        "regret": metrics["regret"],
                        "action_agreement": metrics["action_agreement"],
                        "reversal_rate": metrics["reversal_rate"],
                        "transition_mse": metrics["transition_mse"],
                        "wall_clock_sec": wall_clock,
                        "param_count": param_count,
                    })

    df = pd.DataFrame(records)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/portability_full_audit.csv", index=False)

    print("\n--- AUDIT SUMMARY BY HOST, CONDITION, AND LAMBDA ---")
    summary = df.groupby(["host", "condition", "lambda"]).agg({
        "j_learned": ["mean", "std"],
        "regret": ["mean", "std"],
        "action_agreement": ["mean", "std"],
        "reversal_rate": ["mean", "std"],
        "transition_mse": ["mean", "std"],
        "wall_clock_sec": ["mean"],
    })
    print(summary)
    return df


if __name__ == "__main__":
    run_lambda_sweep_audit()
