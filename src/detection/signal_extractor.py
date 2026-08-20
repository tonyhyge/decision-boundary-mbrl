"""
Signal Extraction and Feature Processing for Stage 3A Risk Detectors.

Extracts decision-relevant model-internal signals from ensemble dynamics and reference value function:
  1. Learned Margin: m_hat(s) = |bar{g}(s)|
  2. Dynamics Epistemic Disagreement: sigma_dyn(s)
  3. Value Epistemic Disagreement on Signed Action Gap: sigma_g(s) = sqrt(Var_k[g_k(s)])
  4. Action Disagreement: Vote entropy H_vote(s) and disagreement fraction kappa_dis(s)
  5. Ratio Baseline: r_ratio(s) = sigma_g(s) / (m_hat(s) + eps_frozen)

Extracts Ground-Truth Reference Flip Labels (for evaluation/calibration only):
  Z_flip(s) = 1[a_Mhat_ref(s) != a_P_ref(s)]
"""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

from src.envs.cartpole_continuous import CartPoleDynamics, CompetentCartPoleValueFunction
from src.models.ensemble_dynamics import EnsembleDynamicsModel


class UnprivilegedSignalExtractor:
    """
    Extracts 5 internal feature signal families and ground-truth evaluation labels.
    """

    def __init__(
        self,
        ensemble: EnsembleDynamicsModel,
        val_fn: CompetentCartPoleValueFunction,
        dynamics: Optional[CartPoleDynamics] = None,
        gamma: float = 0.98,
    ):
        self.ensemble = ensemble
        self.val_fn = val_fn
        self.dynamics = dynamics or CartPoleDynamics()
        self.gamma = gamma

    def extract_state_signals(
        self,
        state: np.ndarray,
        eps_ratio: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Extract all model-internal signals for a given state s.
        NO access to true dynamics f_P is used for feature signals.
        """
        # 1. Ensemble next-state predictions for both actions
        # next_s0, next_s1: shape (K, 4)
        next_s0, next_s1 = self.ensemble.predict_all_actions_next_states(state)
        num_models = len(next_s0)

        # 2. Reference Q-values for each model k
        # Q_hat_k(s, a) = r(s, a) + gamma * V_ref(f_hat_k(s, a))
        q0_k = np.zeros(num_models, dtype=np.float64)
        q1_k = np.zeros(num_models, dtype=np.float64)

        for k in range(num_models):
            # Check terminal condition on predicted state
            s0_k = next_s0[k]
            s1_k = next_s1[k]
            done0_k = bool(s0_k[0] < -2.4 or s0_k[0] > 2.4 or abs(s0_k[2]) > 12 * 2 * np.pi / 360)
            done1_k = bool(s1_k[0] < -2.4 or s1_k[0] > 2.4 or abs(s1_k[2]) > 12 * 2 * np.pi / 360)

            r0 = 1.0 if not done0_k else 0.0
            r1 = 1.0 if not done1_k else 0.0

            q0_k[k] = r0 + (self.gamma * self.val_fn.evaluate_v(s0_k) if not done0_k else 0.0)
            q1_k[k] = r1 + (self.gamma * self.val_fn.evaluate_v(s1_k) if not done1_k else 0.0)

        # 3. Signed action gaps for each model: g_k(s) = Q_hat_k(s, 1) - Q_hat_k(s, 0)
        g_k = q1_k - q0_k
        mean_g = float(np.mean(g_k))
        var_g = float(np.var(g_k, ddof=1)) if num_models > 1 else 0.0
        sigma_g = float(np.sqrt(max(0.0, var_g)))

        # Ensemble consensus Q values & action
        q0_mean = float(np.mean(q0_k))
        q1_mean = float(np.mean(q1_k))
        a_model_ens = 1 if mean_g >= 0.0 else 0

        # Signal 1: Learned Margin
        m_hat = float(abs(mean_g))
        score_margin = -m_hat  # Higher score = smaller margin = higher flip risk

        # Signal 2: Dynamics Epistemic Disagreement
        # Dispersion of f_hat_k(s, a) around ensemble mean for actions 0 and 1
        mean_s0 = np.mean(next_s0, axis=0)
        mean_s1 = np.mean(next_s1, axis=0)
        var_dyn_0 = float(np.sum(np.var(next_s0, axis=0, ddof=1))) if num_models > 1 else 0.0
        var_dyn_1 = float(np.sum(np.var(next_s1, axis=0, ddof=1))) if num_models > 1 else 0.0
        sigma_dyn = float(np.sqrt(0.5 * (var_dyn_0 + var_dyn_1)))

        # Signal 3: Value Disagreement on Signed Action Gap (already computed sigma_g)
        # Note: sigma_g directly measures dispersion in preference direction

        # Signal 4: Action Disagreement
        # Fraction of models voting for action 1 vs 0
        votes = (g_k >= 0.0).astype(np.float64)
        p1 = float(np.mean(votes))
        p0 = 1.0 - p1
        if p0 <= 0.0 or p1 <= 0.0:
            vote_entropy = 0.0
        else:
            vote_entropy = float(- (p0 * np.log2(p0) + p1 * np.log2(p1)))
        disagreement_fraction = float(min(p0, p1))

        # Signal 5: Ratio Baseline
        r_ratio = float(sigma_g / (m_hat + eps_ratio))

        return {
            "m_hat": m_hat,
            "score_margin": score_margin,
            "sigma_dyn": sigma_dyn,
            "sigma_g": sigma_g,
            "vote_entropy": vote_entropy,
            "disagreement_fraction": disagreement_fraction,
            "r_ratio": r_ratio,
            "mean_g": mean_g,
            "q0_mean": q0_mean,
            "q1_mean": q1_mean,
            "a_model_ens": a_model_ens,
        }

    def compute_ground_truth_flip(
        self, state: np.ndarray, a_model_ens: int
    ) -> Tuple[int, int, float, float]:
        """
        Compute true-dynamics reference action and ground-truth flip label:
          a_P_ref(s) = argmax_a [r(s, a) + gamma * V_ref(f_P(s, a))]
          Z_flip(s) = 1[a_model_ens != a_P_ref(s)]
        Used STRICTLY for label creation and validation/test evaluation.
        """
        q0_true = self.val_fn.evaluate_q(state, 0)
        q1_true = self.val_fn.evaluate_q(state, 1)
        a_true_ref = 1 if q1_true >= q0_true else 0
        z_flip = int(a_model_ens != a_true_ref)
        m_true = float(abs(q1_true - q0_true))
        return z_flip, a_true_ref, m_true, float(q1_true - q0_true)

    def extract_dataset(
        self,
        trajectories: List[List[Dict]],
        eps_ratio: float = 1e-4,
    ) -> pd.DataFrame:
        """
        Extract full feature set and ground-truth labels across a list of trajectories.
        Each trajectory is a list of step dicts containing 'state', 'action', etc.
        """
        records = []
        for traj_id, traj in enumerate(trajectories):
            for step_idx, step in enumerate(traj):
                s = np.array(step["state"], dtype=np.float64)

                # Extract model-internal signals
                signals = self.extract_state_signals(s, eps_ratio=eps_ratio)

                # Extract ground truth label
                z_flip, a_true_ref, m_true, true_g = self.compute_ground_truth_flip(
                    s, signals["a_model_ens"]
                )

                rec = {
                    "traj_id": traj_id,
                    "step_idx": step_idx,
                    "x": float(s[0]),
                    "x_dot": float(s[1]),
                    "theta": float(s[2]),
                    "theta_dot": float(s[3]),
                    # Signals
                    "m_hat": signals["m_hat"],
                    "score_margin": signals["score_margin"],
                    "sigma_dyn": signals["sigma_dyn"],
                    "sigma_g": signals["sigma_g"],
                    "vote_entropy": signals["vote_entropy"],
                    "disagreement_fraction": signals["disagreement_fraction"],
                    "r_ratio": signals["r_ratio"],
                    "mean_g": signals["mean_g"],
                    "a_model_ens": signals["a_model_ens"],
                    # Ground Truth
                    "a_true_ref": a_true_ref,
                    "m_true": m_true,
                    "true_g": true_g,
                    "z_flip": z_flip,
                }
                records.append(rec)

        return pd.DataFrame(records)


class FeatureNormalizer:
    """
    Stores and applies feature normalization statistics fit STRICTLY on the training split.
    Also computes and freezes eps_frozen for the ratio signal.
    """

    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names or [
            "score_margin",
            "sigma_dyn",
            "sigma_g",
            "vote_entropy",
            "r_ratio",
        ]
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.eps_frozen: float = 1e-3
        self.is_fitted: bool = False

    def fit(self, df_train: pd.DataFrame) -> "FeatureNormalizer":
        """Fit normalization parameters strictly on the training set."""
        # 1. Compute and freeze eps as 25th percentile of m_hat on train (or default if constant)
        if "m_hat" in df_train.columns:
            m_vals = df_train["m_hat"].to_numpy()
            q25 = float(np.percentile(m_vals, 25))
            self.eps_frozen = max(1e-4, q25)
        else:
            self.eps_frozen = 1e-3

        # 2. Compute means and standard deviations for all feature signals
        for feat in self.feature_names:
            vals = df_train[feat].to_numpy(dtype=np.float64)
            self.means[feat] = float(np.mean(vals))
            s = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0
            self.stds[feat] = s if s > 1e-8 else 1.0

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Standardize features according to training statistics.
        Returns: np.ndarray of shape (len(df), len(feature_names)).
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureNormalizer must be fitted on training split first.")

        matrix = []
        for feat in self.feature_names:
            vals = df[feat].to_numpy(dtype=np.float64)
            norm_vals = (vals - self.means[feat]) / self.stds[feat]
            matrix.append(norm_vals)

        return np.column_stack(matrix)

    def transform_single_feature(self, df: pd.DataFrame, feat_name: str) -> np.ndarray:
        """Standardize a single scalar feature."""
        if not self.is_fitted:
            raise RuntimeError("FeatureNormalizer must be fitted on training split first.")
        vals = df[feat_name].to_numpy(dtype=np.float64)
        mean = self.means.get(feat_name, float(np.mean(vals)))
        std = self.stds.get(feat_name, float(np.std(vals, ddof=1)) or 1.0)
        return (vals - mean) / std
