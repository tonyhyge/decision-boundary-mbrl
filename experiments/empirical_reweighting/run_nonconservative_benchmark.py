"""
Non-Conservative Operationalization Benchmark:
Tests whether avoiding corridor weight starvation (by holding non-crossing transition weights
fixed at 1.0 while increasing crossing transition weights) rescues boundary-aware model learning.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.tabular_mdp import TabularMDP
from src.envs.gridworld_mdp import ChoiceGridWorldMDP, make_stochastic_choice_gridworld
from src.planning.dp import value_iteration, expected_discounted_return
from src.models.tabular_learned_model import (
    collect_gridworld_experience,
    TrajectoryDataset,
)
from src.models.portability_models import (
    WeightedCategoricalWorldModel,
    ProbabilisticEnsembleWorldModel,
)
from src.correction.portability_weighting import compute_sample_scores


def run_nonconservative_benchmark(
    num_seeds: int = 30,
    base_seed: int = 42,
    output_dir: str = "results",
):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(base_seed)

    records = []

    print(f"Running Non-Conservative Operationalization Benchmark across {num_seeds} seeds...")

    for seed_idx in range(num_seeds):
        seed = base_seed + seed_idx
        t_rng = np.random.default_rng(seed)

        # Build stochastic GridWorld
        p_safe = float(t_rng.uniform(0.88, 0.94))
        p_risky = float(t_rng.uniform(0.65, 0.75))
        p_def = float(t_rng.uniform(0.80, 0.88))
        h_pen = float(t_rng.uniform(-2.5, -1.5))
        s_cost = float(t_rng.uniform(-0.08, -0.03))

        true_grid = make_stochastic_choice_gridworld(
            height=5,
            width=5,
            p_succ_safe=p_safe,
            p_succ_risky=p_risky,
            p_succ_default=p_def,
            hazard_penalty=h_pen,
            step_cost=s_cost,
            gamma=0.95,
            seed=seed,
        )

        V_star, Q_star, pi_star = value_iteration(true_grid)
        j_optimal = expected_discounted_return(true_grid, pi_star)

        # Collect 80 trajectories (intermediate data regime)
        train_data = collect_gridworld_experience(true_grid, num_trajectories=80, max_steps=40, seed=seed)

        # Initial reference model
        init_model = WeightedCategoricalWorldModel(25, 4)
        init_model.fit(train_data, epochs=80, seed=seed)
        init_mdp = init_model.create_learned_mdp(true_grid)
        init_p_mat = init_model.get_transition_matrix()

        # Extract oracle and estimated crossing scores
        scores_est, w_norm_est = compute_sample_scores(train_data, "estimated_crossing", true_grid, init_mdp, init_p_mat, lambda_val=1.0)
        scores_orc, w_norm_orc = compute_sample_scores(train_data, "oracle_crossing", true_grid, init_mdp, init_p_mat, lambda_val=1.0)
        scores_shuff = t_rng.permutation(scores_est)
        w_norm_shuff = t_rng.permutation(w_norm_est)

        # Non-conservative weights: w = 1.0 + lambda * s (corridors remain at weight 1.0!)
        w_noncons_est = 1.0 + 1.0 * scores_est
        w_noncons_orc = 1.0 + 1.0 * scores_orc
        w_noncons_shuff = 1.0 + 1.0 * scores_shuff

        conditions = [
            ("uniform", np.ones(len(train_data), dtype=np.float64), True),
            ("normalized_estimated_crossing", w_norm_est, True),
            ("nonconservative_estimated_crossing", w_noncons_est, False),
            ("normalized_oracle_crossing", w_norm_orc, True),
            ("nonconservative_oracle_crossing", w_noncons_orc, False),
            ("normalized_shuffled_control", w_norm_shuff, True),
            ("nonconservative_shuffled_control", w_noncons_shuff, False),
        ]

        # Train both Host A (Deterministic Categorical) and Host B (Ensemble)
        for host_name, HostClass in [("Host_A_Deterministic", WeightedCategoricalWorldModel), ("Host_B_Ensemble", ProbabilisticEnsembleWorldModel)]:
            for cond_name, weights, norm_flag in conditions:
                model = HostClass(true_grid.num_states, true_grid.num_actions, gamma=0.95)
                # Fit model with weights
                if host_name == "Host_A_Deterministic":
                    model.fit(train_data, sample_weights=weights, epochs=80, lr=0.01, seed=seed, normalize_weights=norm_flag)
                else:
                    model.fit(train_data, sample_weights=weights, epochs=80, lr=0.01, base_seed=seed, normalize_weights=norm_flag)

                learned_mdp = model.create_learned_mdp(true_grid)
                _, _, pi_hat = value_iteration(learned_mdp)
                j_learned = expected_discounted_return(true_grid, pi_hat)
                regret = max(0.0, j_optimal - j_learned)

                active_states = [s for s in range(true_grid.num_states) if s != true_grid.goal_state]
                agreement = float(np.mean([int(pi_hat[s] == pi_star[s]) for s in active_states]))

                records.append({
                    "seed": seed_idx,
                    "host": host_name,
                    "condition": cond_name,
                    "j_learned": j_learned,
                    "regret": regret,
                    "action_agreement": agreement,
                })

    df_results = pd.DataFrame(records)
    df_results.to_csv(os.path.join(output_dir, "portability_nonconservative_30seeds.csv"), index=False)

    # Statistical summary
    summary = {}
    for host in ["Host_A_Deterministic", "Host_B_Ensemble"]:
        df_h = df_results[df_results["host"] == host]
        piv = df_h.pivot(index="seed", columns="condition", values="j_learned")

        summary[host] = {}
        for cond_name, _, _ in conditions:
            summary[host][cond_name] = {
                "mean_return": float(piv[cond_name].mean()),
                "std_return": float(piv[cond_name].std()),
            }

        # Compare non-conservative estimated vs uniform
        diff_noncons_est = piv["nonconservative_estimated_crossing"] - piv["uniform"]
        diff_norm_est = piv["normalized_estimated_crossing"] - piv["uniform"]
        diff_noncons_orc = piv["nonconservative_oracle_crossing"] - piv["uniform"]

        summary[host]["contrasts"] = {
            "noncons_est_minus_uniform": {
                "mean": float(diff_noncons_est.mean()),
                "p_val_wilcoxon": float(stats.wilcoxon(diff_noncons_est, zero_method="wilcox")[1]) if np.any(diff_noncons_est != 0) else 1.0,
            },
            "norm_est_minus_uniform": {
                "mean": float(diff_norm_est.mean()),
                "p_val_wilcoxon": float(stats.wilcoxon(diff_norm_est, zero_method="wilcox")[1]) if np.any(diff_norm_est != 0) else 1.0,
            },
            "noncons_orc_minus_uniform": {
                "mean": float(diff_noncons_orc.mean()),
                "p_val_wilcoxon": float(stats.wilcoxon(diff_noncons_orc, zero_method="wilcox")[1]) if np.any(diff_noncons_orc != 0) else 1.0,
            },
        }

    with open(os.path.join(output_dir, "portability_nonconservative_report.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== NON-CONSERVATIVE OPERATIONALIZATION RESULTS ===")
    for host in summary:
        print(f"\n--- {host} ---")
        for cond, metrics in summary[host].items():
            if cond != "contrasts":
                print(f"  {cond:38s}: Mean Return = {metrics['mean_return']:.4f} (std = {metrics['std_return']:.4f})")
        print("  Contrasts:")
        for k, v in summary[host]["contrasts"].items():
            print(f"    {k}: mean diff = {v['mean']:+.4f}, p = {v['p_val_wilcoxon']:.4f}")

    return summary


if __name__ == "__main__":
    run_nonconservative_benchmark(num_seeds=30, base_seed=42)
