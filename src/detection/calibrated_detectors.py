"""
Calibrated Risk Detectors for Stage 3A Unprivileged Failure Prediction.

Implements:
  - PrevalenceNullDetector: Constant baseline predicting training flip prevalence.
  - CalibratedScalarDetector: 1D Platt scaling (logistic calibration) for scalar signals.
  - CombinedLogisticDetector: Multivariate logistic regression combining all 5 signals.
  - PermutedLabelControlDetector: Mandatory negative control with shuffled training/cal labels.
  - CalibratedIsotonicDetector: Non-parametric 1D isotonic calibration for sensitivity checks.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

from src.detection.signal_extractor import FeatureNormalizer


class BaseDetector(ABC):
    """Abstract base class for calibrated risk detectors."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(
        self,
        df_cal: pd.DataFrame,
        normalizer: FeatureNormalizer,
        rng: Optional[np.random.Generator] = None,
    ) -> "BaseDetector":
        """Fit calibration model strictly on the calibration split."""
        pass

    @abstractmethod
    def predict_proba(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        """Predict calibrated probability of decision flip rho_hat(s) in [0, 1]."""
        pass

    @abstractmethod
    def predict_score(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        """Predict continuous ranking score (higher score = higher flip risk)."""
        pass


class PrevalenceNullDetector(BaseDetector):
    """
    Null baseline: Predicts constant probability equal to training set flip prevalence.
    rho_hat(s) = p_train
    """

    def __init__(self):
        super().__init__("Prevalence Null")
        self.prevalence: float = 0.5

    def fit_prevalence(self, df_train: pd.DataFrame) -> "PrevalenceNullDetector":
        """Fit prevalence strictly from training split."""
        flips = df_train["z_flip"].to_numpy()
        self.prevalence = float(np.mean(flips))
        return self

    def fit(
        self,
        df_cal: pd.DataFrame,
        normalizer: FeatureNormalizer,
        rng: Optional[np.random.Generator] = None,
    ) -> "PrevalenceNullDetector":
        # Prevalence is fixed from training split, fit is a no-op
        return self

    def predict_proba(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        return np.full(len(df), self.prevalence, dtype=np.float64)

    def predict_score(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        return np.full(len(df), self.prevalence, dtype=np.float64)


class CalibratedScalarDetector(BaseDetector):
    """
    1D Platt Scaling (logistic calibration) on a single standardized signal.
    P(Z_flip = 1 | x) = sigma(w * x + b)
    """

    def __init__(self, name: str, feature_col: str):
        super().__init__(name)
        self.feature_col = feature_col
        self.model = LogisticRegression(solver="lbfgs", C=1e4)  # Minimal regularization for clean Platt scaling
        self.is_fitted = False

    def fit(
        self,
        df_cal: pd.DataFrame,
        normalizer: FeatureNormalizer,
        rng: Optional[np.random.Generator] = None,
    ) -> "CalibratedScalarDetector":
        x_norm = normalizer.transform_single_feature(df_cal, self.feature_col).reshape(-1, 1)
        y = df_cal["z_flip"].to_numpy(dtype=np.int32)

        # Check if single class present
        if len(np.unique(y)) < 2:
            self.model.classes_ = np.array([0, 1])
            self.model.coef_ = np.zeros((1, 1))
            self.model.intercept_ = np.array([0.0])
        else:
            self.model.fit(x_norm, y)

        self.is_fitted = True
        return self

    def predict_proba(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Detector {self.name} is not fitted.")
        x_norm = normalizer.transform_single_feature(df, self.feature_col).reshape(-1, 1)
        return self.model.predict_proba(x_norm)[:, 1]

    def predict_score(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        # Return standardized feature value directly for ranking
        return normalizer.transform_single_feature(df, self.feature_col)


class CombinedLogisticDetector(BaseDetector):
    """
    Multivariate Logistic Regression over all 5 standardized signal families:
      x = [score_margin, sigma_dyn, sigma_g, vote_entropy, r_ratio]
      P(Z_flip = 1 | x) = sigma(w^T x + b)
    Fit strictly on the calibration split.
    """

    def __init__(self, name: str = "Combined Detector"):
        super().__init__(name)
        self.model = LogisticRegression(solver="lbfgs", C=1.0)
        self.is_fitted = False
        self.feature_names: List[str] = []

    def fit(
        self,
        df_cal: pd.DataFrame,
        normalizer: FeatureNormalizer,
        rng: Optional[np.random.Generator] = None,
    ) -> "CombinedLogisticDetector":
        self.feature_names = normalizer.feature_names
        X_norm = normalizer.transform(df_cal)
        y = df_cal["z_flip"].to_numpy(dtype=np.int32)

        if len(np.unique(y)) < 2:
            self.model.classes_ = np.array([0, 1])
            self.model.coef_ = np.zeros((1, X_norm.shape[1]))
            self.model.intercept_ = np.array([0.0])
        else:
            self.model.fit(X_norm, y)

        self.is_fitted = True
        return self

    def predict_proba(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("CombinedLogisticDetector is not fitted.")
        X_norm = normalizer.transform(df)
        return self.model.predict_proba(X_norm)[:, 1]

    def predict_score(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("CombinedLogisticDetector is not fitted.")
        X_norm = normalizer.transform(df)
        return self.model.decision_function(X_norm)


class PermutedLabelControlDetector(BaseDetector):
    """
    Mandatory Negative Control:
    Fits the exact same combined detector pipeline on the calibration split,
    but with randomly shuffled labels Z_flip to break any true relationship.
    Verifies that the detector does not manufacture apparent signal from leakage or overfitting.
    """

    def __init__(self, name: str = "Permuted-Label Control"):
        super().__init__(name)
        self.model = LogisticRegression(solver="lbfgs", C=1.0)
        self.is_fitted = False

    def fit(
        self,
        df_cal: pd.DataFrame,
        normalizer: FeatureNormalizer,
        rng: Optional[np.random.Generator] = None,
    ) -> "PermutedLabelControlDetector":
        local_rng = rng or np.random.default_rng(42)
        X_norm = normalizer.transform(df_cal)
        y_true = df_cal["z_flip"].to_numpy(dtype=np.int32)
        # Random permutation to destroy signal
        y_perm = local_rng.permutation(y_true)

        if len(np.unique(y_perm)) < 2:
            self.model.classes_ = np.array([0, 1])
            self.model.coef_ = np.zeros((1, X_norm.shape[1]))
            self.model.intercept_ = np.array([0.0])
        else:
            self.model.fit(X_norm, y_perm)

        self.is_fitted = True
        return self

    def predict_proba(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PermutedLabelControlDetector is not fitted.")
        X_norm = normalizer.transform(df)
        return self.model.predict_proba(X_norm)[:, 1]

    def predict_score(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PermutedLabelControlDetector is not fitted.")
        X_norm = normalizer.transform(df)
        return self.model.decision_function(X_norm)


class CalibratedIsotonicDetector(BaseDetector):
    """
    Non-parametric 1D Isotonic Calibration for sensitivity analysis.
    """

    def __init__(self, name: str, feature_col: str):
        super().__init__(name)
        self.feature_col = feature_col
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.is_fitted = False

    def fit(
        self,
        df_cal: pd.DataFrame,
        normalizer: FeatureNormalizer,
        rng: Optional[np.random.Generator] = None,
    ) -> "CalibratedIsotonicDetector":
        x_norm = normalizer.transform_single_feature(df_cal, self.feature_col)
        y = df_cal["z_flip"].to_numpy(dtype=np.float64)
        self.model.fit(x_norm, y)
        self.is_fitted = True
        return self

    def predict_proba(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"Detector {self.name} is not fitted.")
        x_norm = normalizer.transform_single_feature(df, self.feature_col)
        return np.clip(self.model.predict(x_norm), 0.0, 1.0)

    def predict_score(
        self, df: pd.DataFrame, normalizer: FeatureNormalizer
    ) -> np.ndarray:
        return normalizer.transform_single_feature(df, self.feature_col)


def get_all_stage3a_detectors() -> Dict[str, BaseDetector]:
    """Instantiate the complete suite of Stage 3A detectors."""
    return {
        "prevalence_null": PrevalenceNullDetector(),
        "margin_only": CalibratedScalarDetector("Margin-Only (-m_hat)", "score_margin"),
        "dynamics_only": CalibratedScalarDetector("Dynamics-Only (sigma_dyn)", "sigma_dyn"),
        "gap_variance_only": CalibratedScalarDetector("Value-Gap-Disagreement (sigma_g)", "sigma_g"),
        "action_disagreement_only": CalibratedScalarDetector("Action-Disagreement (H_vote)", "vote_entropy"),
        "ratio_only": CalibratedScalarDetector("Ratio-Only (sigma_g / (m_hat + eps))", "r_ratio"),
        "combined": CombinedLogisticDetector("Combined Linear Detector"),
        "permuted_control": PermutedLabelControlDetector("Permuted-Label Negative Control"),
    }
