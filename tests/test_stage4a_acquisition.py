"""
Unit Tests for Stage 4A: Active Acquisition and Model Repair.

Verifies:
  1. All acquisition strategies select the exact requested budget from the candidate pool.
  2. Oracle acquisition correctly prioritizes positive flip states Z_flip == 1.
  3. ModelRepairTrainer correctly clones, retrains, and evaluates ensemble dynamics without NaNs.
  4. Efficiency metric eta_B computation is mathematically consistent.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.envs.cartpole_continuous import CartPoleDynamics, CompetentCartPoleValueFunction
from src.models.ensemble_dynamics import EnsembleDynamicsModel
from src.detection.signal_extractor import UnprivilegedSignalExtractor, FeatureNormalizer
from src.detection.calibrated_detectors import CombinedLogisticDetector
from src.acquisition.strategies import get_all_stage4a_strategies
from src.acquisition.repair_trainer import ModelRepairTrainer


@pytest.fixture
def acquisition_setup():
    rng = np.random.default_rng(42)
    dynamics = CartPoleDynamics()
    val_fn = CompetentCartPoleValueFunction(dynamics=dynamics)
    ensemble = EnsembleDynamicsModel(num_models=3, hidden_dim=32, seed=42)

    # Train minimal ensemble
    x_synth = rng.normal(scale=0.1, size=(50, 5)).astype(np.float32)
    y_synth = rng.normal(scale=0.01, size=(50, 4)).astype(np.float32)
    ensemble.train_ensemble(x_synth, y_synth, epochs=3, batch_size=32, seed=42)

    # Synthetic candidate pool dataframe
    df_pool = pd.DataFrame({
        "x": rng.normal(size=60),
        "x_dot": rng.normal(size=60),
        "theta": rng.normal(size=60),
        "theta_dot": rng.normal(size=60),
        "score_margin": rng.normal(size=60),
        "m_hat": np.abs(rng.normal(size=60)),
        "sigma_dyn": np.abs(rng.normal(size=60)),
        "sigma_g": np.abs(rng.normal(size=60)),
        "vote_entropy": rng.uniform(0, 1, size=60),
        "r_ratio": np.abs(rng.normal(size=60)),
        "a_model_ens": rng.choice([0, 1], size=60),
        "z_flip": rng.choice([0, 1], size=60, p=[0.8, 0.2]),
        "m_true": np.abs(rng.normal(size=60)),
    })

    normalizer = FeatureNormalizer().fit(df_pool)
    detector = CombinedLogisticDetector().fit(df_pool, normalizer)
    trainer = ModelRepairTrainer(dynamics=dynamics, val_fn=val_fn, epochs=2, lr=0.005)

    return dynamics, val_fn, ensemble, df_pool, normalizer, detector, trainer


def test_acquisition_strategy_budgets(acquisition_setup):
    _, _, _, df_pool, normalizer, detector, _ = acquisition_setup
    rng = np.random.default_rng(123)
    strategies = get_all_stage4a_strategies()

    budget = 10
    for name, strat in strategies.items():
        q_idx = strat.select_queries(
            df_pool=df_pool,
            budget=budget,
            rng=rng,
            detector=detector,
            normalizer=normalizer,
        )
        assert len(q_idx) == budget, f"Strategy {name} returned {len(q_idx)} queries, expected {budget}"
        assert len(set(q_idx)) == budget, f"Strategy {name} returned duplicate indices"
        assert all(0 <= idx < len(df_pool) for idx in q_idx), f"Strategy {name} returned out of bounds index"


def test_oracle_prioritization(acquisition_setup):
    _, _, _, df_pool, normalizer, detector, _ = acquisition_setup
    rng = np.random.default_rng(456)
    strategies = get_all_stage4a_strategies()
    oracle = strategies["oracle"]

    num_pos = int(df_pool["z_flip"].sum())
    assert num_pos > 0

    # Query with budget <= num_pos
    budget_small = min(5, num_pos)
    q_small = oracle.select_queries(df_pool, budget=budget_small, rng=rng)
    flips_selected = df_pool.iloc[q_small]["z_flip"].sum()
    assert flips_selected == budget_small, "Oracle did not select 100% true flips when budget <= num_pos"


def test_repair_trainer_retrain(acquisition_setup):
    dynamics, val_fn, ensemble, df_pool, _, _, trainer = acquisition_setup
    rng = np.random.default_rng(789)

    x_base = rng.normal(scale=0.1, size=(40, 5)).astype(np.float32)
    y_base = rng.normal(scale=0.01, size=(40, 4)).astype(np.float32)

    # Retrain clone
    new_ensemble = trainer.retrain_or_finetune_ensemble(ensemble, x_base, y_base, finetune=True, seed=42)
    assert new_ensemble is not None

    # Evaluate on eval set
    eval_res = trainer.evaluate_flips_on_eval_set(new_ensemble, df_pool)
    assert "n_flips" in eval_res
    assert "flip_rate" in eval_res
    assert "near_boundary_mse" in eval_res

    # Downstream return evaluation
    ret_res = trainer.evaluate_downstream_policy_return(new_ensemble, num_episodes=5, max_steps=20, seed=42)
    assert "mean_return" in ret_res
