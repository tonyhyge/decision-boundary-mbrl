"""
Budgeted Model Correction: Evaluates Recovery@K across prioritized ranking strategies.
"""
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from src.corruptions.injector import CorruptedMDP
from src.baselines.rankers import BaseRanker, get_all_rankers
from src.planning.dp import value_iteration, expected_discounted_return


class BudgetEvaluator:
    """
    Evaluates policy return recovery under a fixed error correction budget K.
    """

    def __init__(self, corrupted_mdp: CorruptedMDP, df_errors: pd.DataFrame):
        self.corrupted_mdp = corrupted_mdp
        self.df_errors = df_errors
        self.j_true = corrupted_mdp.j_true_star
        self.j_corrupt = corrupted_mdp.j_corrupt
        self.j_delta = self.j_true - self.j_corrupt

    def evaluate_ranker(
        self, ranker: BaseRanker, rng: np.random.Generator
    ) -> Dict[int, float]:
        """
        Evaluate Recovery@K for budget levels K = 1, 2, ..., N.
        
        Returns:
            recovery_curve (Dict[int, float]): Map from budget K to Recovery@K in [0, 1].
        """
        n_errors = len(self.corrupted_mdp.errors)
        ranked_indices = ranker.rank(self.df_errors, rng)
        results: Dict[int, float] = {0: 0.0}

        if abs(self.j_delta) < 1e-12:
            # Corrupted policy already achieves optimal return
            return {k: 1.0 for k in range(n_errors + 1)}

        for k in range(1, n_errors + 1):
            top_k_indices = ranked_indices[:k]
            mdp_k_restored = self.corrupted_mdp.restore_subset(top_k_indices)
            _, _, pi_k = value_iteration(mdp_k_restored)
            j_k = expected_discounted_return(self.corrupted_mdp.true_mdp, pi_k)

            recovery = (j_k - self.j_corrupt) / self.j_delta
            # Bound recovery between 0 and 1
            recovery = float(np.clip(recovery, 0.0, 1.0))
            results[k] = recovery

        return results

    def evaluate_dynamic_greedy_oracle(self) -> Dict[int, float]:
        """
        Evaluate sequential Dynamic Greedy Oracle:
        At each step k, greedily choose the remaining error that maximizes marginal return.
        """
        n_errors = len(self.corrupted_mdp.errors)
        results: Dict[int, float] = {0: 0.0}

        if abs(self.j_delta) < 1e-12:
            return {k: 1.0 for k in range(n_errors + 1)}

        s_curr = set()
        for k in range(1, n_errors + 1):
            best_e = None
            best_j = -1e9
            for e_cand in range(n_errors):
                if e_cand not in s_curr:
                    test_set = list(s_curr | {e_cand})
                    mdp_test = self.corrupted_mdp.restore_subset(test_set)
                    _, _, pi_test = value_iteration(mdp_test)
                    j_test = expected_discounted_return(self.corrupted_mdp.true_mdp, pi_test)
                    if j_test > best_j:
                        best_j = j_test
                        best_e = e_cand
            s_curr.add(best_e)
            recovery = float(np.clip((best_j - self.j_corrupt) / self.j_delta, 0.0, 1.0))
            results[k] = recovery

        return results

    def evaluate_combinatorial_optimal_oracle(self) -> Dict[int, float]:
        """
        Evaluate true Combinatorial Optimal Oracle:
        For each budget k, exhaustively find max_{|S|=k} J(M_restored(S)).
        """
        from itertools import combinations
        n_errors = len(self.corrupted_mdp.errors)
        results: Dict[int, float] = {0: 0.0}

        if abs(self.j_delta) < 1e-12:
            return {k: 1.0 for k in range(n_errors + 1)}

        for k in range(1, n_errors + 1):
            best_j = -1e9
            for subset in combinations(range(n_errors), k):
                mdp_test = self.corrupted_mdp.restore_subset(list(subset))
                _, _, pi_test = value_iteration(mdp_test)
                j_test = expected_discounted_return(self.corrupted_mdp.true_mdp, pi_test)
                if j_test > best_j:
                    best_j = j_test
            recovery = float(np.clip((best_j - self.j_corrupt) / self.j_delta, 0.0, 1.0))
            results[k] = recovery

        return results


def run_budget_experiment(
    corrupted_mdp: CorruptedMDP,
    df_errors: pd.DataFrame,
    num_trials: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run budget recovery evaluation across all standard rankers over multiple random seeds.
    """
    rng = np.random.default_rng(seed)
    rankers = get_all_rankers()
    evaluator = BudgetEvaluator(corrupted_mdp, df_errors)
    n_errors = len(corrupted_mdp.errors)

    records: List[Dict] = []

    for ranker in rankers:
        recoveries_by_k = np.zeros((num_trials, n_errors + 1), dtype=np.float64)
        for trial in range(num_trials):
            trial_rng = np.random.default_rng(rng.integers(0, 1000000))
            curve = evaluator.evaluate_ranker(ranker, trial_rng)
            for k in range(n_errors + 1):
                recoveries_by_k[trial, k] = curve[k]

        mean_curve = recoveries_by_k.mean(axis=0)
        std_curve = recoveries_by_k.std(axis=0)

        for k in range(n_errors + 1):
            records.append(
                {
                    "ranker": ranker.name,
                    "budget_k": k,
                    "budget_fraction": k / max(1, n_errors),
                    "recovery_mean": float(mean_curve[k]),
                    "recovery_std": float(std_curve[k]),
                }
            )

    return pd.DataFrame(records)
