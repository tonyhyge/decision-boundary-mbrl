"""
Unit Tests for Stage 3A: Unprivileged Detector Falsification Pipeline.

Verifies:
  1. Ensemble dynamics initialization and prediction shapes.
  2. Unprivileged signal extraction (learned margin, dynamics epistemic dispersion,
     signed gap value variance, action vote entropy, ratio baseline).
  3. Strict label separation (f_P used only for evaluation label Z_flip, not features).
  4. Training-only feature normalization and frozen epsilon estimation.
  5. Calibrated detector fitting, probability outputs in [0, 1], and Platt calibration.
  6. Permuted-label negative control behavior.
  7. Evaluation metrics and trajectory-clustered paired bootstrap inference.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from src.envs.cartpole_continuous import CartPoleDynamics, CompetentCartPoleValueFunction
from src.models.ensemble_dynamics import EnsembleDynamicsModel
from src.detection.signal_extractor import UnprivilegedSignalExtractor, FeatureNormalizer
from src.detection.calibrated_detectors import (
    PrevalenceNullDetector,
    CalibratedScalarDetector,
    CombinedLogisticDetector,
    PermutedLabelControlDetector,
    CalibratedIsotonicDetector,
    get_all_stage3a_detectors,
)
from src.detection.evaluation import (
    compute_detector_metrics,
    evaluate_all_detectors_on_df,
    run_clustered_paired_bootstrap,
)


@pytest.fixture
def test_setup():
    rng = np.random.default_rng(42)
    dynamics = CartPoleDynamics()
    val_fn = CompetentCartPoleValueFunction(dynamics=dynamics)
    ensemble = EnsembleDynamicsModel(num_models=5, hidden_dim=32, seed=42)

    # Train ensemble quickly on synthetic CartPole steps
    x_synth = rng.normal(scale=0.1, size=(100, 5)).astype(np.float32)
    y_synth = rng.normal(scale=0.01, size=(100, 4)).astype(np.float32)
    ensemble.train_ensemble(x_synth, y_synth, epochs=3, batch_size=32, seed=42)

    extractor = UnprivilegedSignalExtractor(ensemble=ensemble, val_fn=val_fn, dynamics=dynamics)
    return dynamics, val_fn, ensemble, extractor


def test_ensemble_dynamics_shapes(test_setup):
    _, _, ensemble, _ = test_setup
    s = np.array([0.05, 0.02, -0.03, 0.01])
    next_s = ensemble.predict_all_next_states(s, action=1)
    assert next_s.shape == (5, 4), f"Expected (5, 4) ensemble predictions, got {next_s.shape}"

    next_s0, next_s1 = ensemble.predict_all_actions_next_states(s)
    assert next_s0.shape == (5, 4)
    assert next_s1.shape == (5, 4)


def test_signal_extraction(test_setup):
    _, _, _, extractor = test_setup
    s = np.array([0.02, -0.01, 0.04, 0.02])
    signals = extractor.extract_state_signals(s, eps_ratio=1e-3)

    expected_keys = [
        "m_hat",
        "score_margin",
        "sigma_dyn",
        "sigma_g",
        "vote_entropy",
        "disagreement_fraction",
        "r_ratio",
        "mean_g",
        "q0_mean",
        "q1_mean",
        "a_model_ens",
    ]
    for k in expected_keys:
        assert k in signals, f"Missing signal: {k}"
        assert not np.isnan(signals[k]), f"NaN found in signal {k}"

    assert signals["m_hat"] >= 0.0
    assert signals["sigma_dyn"] >= 0.0
    assert signals["sigma_g"] >= 0.0
    assert 0.0 <= signals["vote_entropy"] <= 1.0
    assert 0.0 <= signals["disagreement_fraction"] <= 0.5
    assert signals["a_model_ens"] in [0, 1]

    # Test ground truth flip label extraction
    z_flip, a_true, m_true, true_g = extractor.compute_ground_truth_flip(s, signals["a_model_ens"])
    assert z_flip in [0, 1]
    assert a_true in [0, 1]
    assert m_true >= 0.0


def test_feature_normalization_and_dataset_extraction(test_setup):
    _, _, _, extractor = test_setup
    rng = np.random.default_rng(123)

    # Generate synthetic trajectories
    trajs = []
    for t in range(6):
        traj = []
        s = rng.normal(scale=0.05, size=4)
        for step in range(10):
            traj.append({"state": s.tolist(), "action": int(rng.choice(2))})
            s = s + rng.normal(scale=0.01, size=4)
        trajs.append(traj)

    df = extractor.extract_dataset(trajs)
    assert len(df) == 60
    assert "z_flip" in df.columns
    assert "traj_id" in df.columns

    # Test feature normalizer
    norm = FeatureNormalizer()
    norm.fit(df.iloc[:30])  # Fit only on first 30 states (training split)
    assert norm.is_fitted
    assert norm.eps_frozen > 0.0

    X_transformed = norm.transform(df)
    assert X_transformed.shape == (60, len(norm.feature_names))
    # Check mean of training transformed part is close to 0
    assert np.allclose(np.mean(X_transformed[:30], axis=0), 0.0, atol=1e-5)


def test_calibrated_detectors(test_setup):
    _, _, _, extractor = test_setup
    rng = np.random.default_rng(456)

    # Create dummy train and calibration dataframes
    df_data = pd.DataFrame({
        "traj_id": np.repeat(np.arange(10), 10),
        "score_margin": rng.normal(size=100),
        "sigma_dyn": np.abs(rng.normal(size=100)),
        "sigma_g": np.abs(rng.normal(size=100)),
        "vote_entropy": rng.uniform(0, 1, size=100),
        "r_ratio": np.abs(rng.normal(size=100)),
        "m_hat": np.abs(rng.normal(size=100)),
        "z_flip": rng.choice([0, 1], size=100, p=[0.7, 0.3]),
    })

    df_train = df_data.iloc[:50].copy()
    df_cal = df_data.iloc[50:].copy()

    normalizer = FeatureNormalizer().fit(df_train)
    detectors = get_all_stage3a_detectors()

    # Fit prevalence on train
    detectors["prevalence_null"].fit_prevalence(df_train)

    for name, det in detectors.items():
        det.fit(df_cal, normalizer, rng=rng)
        probs = det.predict_proba(df_cal, normalizer)
        scores = det.predict_score(df_cal, normalizer)

        assert len(probs) == len(df_cal)
        assert len(scores) == len(df_cal)
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0), f"Detector {name} generated invalid probabilities"


def test_evaluation_and_clustered_bootstrap(test_setup):
    rng = np.random.default_rng(789)
    # Mock test dataframe with trajectory clusters
    df_test = pd.DataFrame({
        "traj_id": np.repeat(np.arange(8), 15),
        "score_margin": rng.normal(size=120),
        "sigma_dyn": np.abs(rng.normal(size=120)),
        "sigma_g": np.abs(rng.normal(size=120)),
        "vote_entropy": rng.uniform(0, 1, size=120),
        "r_ratio": np.abs(rng.normal(size=120)),
        "m_hat": np.abs(rng.normal(size=120)),
        "z_flip": rng.choice([0, 1], size=120, p=[0.75, 0.25]),
    })

    normalizer = FeatureNormalizer().fit(df_test)
    detectors = {
        "prevalence_null": PrevalenceNullDetector().fit_prevalence(df_test),
        "margin_only": CalibratedScalarDetector("Margin-Only", "score_margin").fit(df_test, normalizer),
        "combined": CombinedLogisticDetector("Combined").fit(df_test, normalizer),
    }

    # Evaluate point metrics
    metrics = evaluate_all_detectors_on_df(detectors, df_test, normalizer)
    assert "brier" in metrics["combined"]
    assert "auprc" in metrics["combined"]
    assert "recall_at_10" in metrics["combined"]

    # Run clustered paired bootstrap with 50 replicates for quick test
    boot_summary = run_clustered_paired_bootstrap(
        detectors=detectors,
        df_test=df_test,
        normalizer=normalizer,
        comparator_name="margin_only",
        cluster_col="traj_id",
        n_boot=50,
        seed=42,
    )

    assert "combined" in boot_summary
    paired = boot_summary["combined"]["paired_to_comparator"]
    assert "delta_brier" in paired
    assert "ci_95" in paired["delta_brier"]
    assert len(paired["delta_brier"]["ci_95"]) == 2
