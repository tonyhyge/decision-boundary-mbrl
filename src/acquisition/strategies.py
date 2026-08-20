"""
Acquisition Strategies for Stage 4A Targeted World-Model Active Data Selection.

Provides:
  - RandomAcquisition: Uniform random baseline
  - DynamicsAcquisition: Epistemic dynamics disagreement ranker (sigma_dyn)
  - MarginAcquisition: Learned action margin proximity ranker (-m_hat)
  - ActionDisagreementAcquisition: Model voting entropy ranker (H_vote)
  - RatioAcquisition: Normalized uncertainty-to-margin ratio (r_ratio)
  - CombinedAcquisition: Unprivileged multivariate logistic risk ranker
  - OracleAcquisition: Ground-truth decision-flip privileged upper bound (Z_flip)
  - ShuffledScoreControl: Score distribution matched shuffled control
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.detection.signal_extractor import FeatureNormalizer
from src.detection.calibrated_detectors import BaseDetector


class BaseAcquisitionStrategy(ABC):
    """Abstract base class for targeted data acquisition strategies."""

    def __init__(self, name: str, is_privileged: bool = False):
        self.name = name
        self.is_privileged = is_privileged

    @abstractmethod
    def select_queries(
        self,
        df_pool: pd.DataFrame,
        budget: int,
        rng: np.random.Generator,
        detector: Optional[BaseDetector] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> List[int]:
        """
        Select budget indices from df_pool for true dynamics querying.
        Returns: list of integer row indices in df_pool.
        """
        pass


class RandomAcquisition(BaseAcquisitionStrategy):
    """Uniform random data acquisition baseline."""

    def __init__(self):
        super().__init__("Random Acquisition", is_privileged=False)

    def select_queries(
        self,
        df_pool: pd.DataFrame,
        budget: int,
        rng: np.random.Generator,
        detector: Optional[BaseDetector] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> List[int]:
        n_pool = len(df_pool)
        budget = min(budget, n_pool)
        return list(rng.choice(n_pool, size=budget, replace=False))


class DynamicsAcquisition(BaseAcquisitionStrategy):
    """Prioritize states with highest dynamics epistemic disagreement sigma_dyn."""

    def __init__(self):
        super().__init__("Dynamics Uncertainty (sigma_dyn)", is_privileged=False)

    def select_queries(
        self,
        df_pool: pd.DataFrame,
        budget: int,
        rng: np.random.Generator,
        detector: Optional[BaseDetector] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> List[int]:
        scores = df_pool["sigma_dyn"].to_numpy(dtype=np.float64)
        # Random tie-break
        tie_breaker = rng.uniform(0, 1e-9, size=len(scores))
        order = np.argsort(-(scores + tie_breaker))
        return list(order[:budget])


class MarginAcquisition(BaseAcquisitionStrategy):
    """Prioritize states closest to the learned decision boundary (smallest m_hat)."""

    def __init__(self):
        super().__init__("Margin Proximity (-m_hat)", is_privileged=False)

    def select_queries(
        self,
        df_pool: pd.DataFrame,
        budget: int,
        rng: np.random.Generator,
        detector: Optional[BaseDetector] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> List[int]:
        scores = -df_pool["m_hat"].to_numpy(dtype=np.float64)
        tie_breaker = rng.uniform(0, 1e-9, size=len(scores))
        order = np.argsort(-(scores + tie_breaker))
        return list(order[:budget])


class ActionDisagreementAcquisition(BaseAcquisitionStrategy):
    """Prioritize states with highest model voting entropy H_vote."""

    def __init__(self):
        super().__init__("Action Disagreement (H_vote)", is_privileged=False)

    def select_queries(
        self,
        df_pool: pd.DataFrame,
        budget: int,
        rng: np.random.Generator,
        detector: Optional[BaseDetector] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> List[int]:
        scores = df_pool["vote_entropy"].to_numpy(dtype=np.float64)
        tie_breaker = rng.uniform(0, 1e-9, size=len(scores))
        order = np.argsort(-(scores + tie_breaker))
        return list(order[:budget])


class RatioAcquisition(BaseAcquisitionStrategy):
    """Prioritize states with highest ratio r_ratio = sigma_g / (m_hat + eps)."""

    def __init__(self):
        super().__init__("Ratio Baseline (r_ratio)", is_privileged=False)

    def select_queries(
        self,
        df_pool: pd.DataFrame,
        budget: int,
        rng: np.random.Generator,
        detector: Optional[BaseDetector] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> List[int]:
        scores = df_pool["r_ratio"].to_numpy(dtype=np.float64)
        tie_breaker = rng.uniform(0, 1e-9, size=len(scores))
        order = np.argsort(-(scores + tie_breaker))
        return list(order[:budget])


class CombinedAcquisition(BaseAcquisitionStrategy):
    """Prioritize states ranked highest by the unprivileged Combined Linear Detector."""

    def __init__(self):
        super().__init__("Combined Unprivileged Ranker", is_privileged=False)

    def select_queries(
        self,
        df_pool: pd.DataFrame,
        budget: int,
        rng: np.random.Generator,
        detector: Optional[BaseDetector] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> List[int]:
        if detector is None or normalizer is None:
            raise ValueError("CombinedAcquisition requires a fitted detector and normalizer.")
        scores = detector.predict_score(df_pool, normalizer)
        tie_breaker = rng.uniform(0, 1e-9, size=len(scores))
        order = np.argsort(-(scores + tie_breaker))
        return list(order[:budget])


class OracleAcquisition(BaseAcquisitionStrategy):
    """Privileged Oracle: Targets states with true decision flips Z_flip == 1."""

    def __init__(self):
        super().__init__("Oracle Target (Privileged Z_flip)", is_privileged=True)

    def select_queries(
        self,
        df_pool: pd.DataFrame,
        budget: int,
        rng: np.random.Generator,
        detector: Optional[BaseDetector] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> List[int]:
        flips = df_pool["z_flip"].to_numpy(dtype=np.int32)
        pos_idx = np.where(flips == 1)[0]
        neg_idx = np.where(flips == 0)[0]

        # Shuffle positives and negatives
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)

        # Fill with true positives first, then negatives if budget exceeds positive count
        if len(pos_idx) >= budget:
            selected = pos_idx[:budget]
        else:
            needed_neg = budget - len(pos_idx)
            selected = np.concatenate([pos_idx, neg_idx[:needed_neg]])

        return list(selected)


class ShuffledScoreControl(BaseAcquisitionStrategy):
    """Negative Control: Preserves identical budget allocation size but shuffles pool ranking."""

    def __init__(self):
        super().__init__("Shuffled Score Negative Control", is_privileged=False)

    def select_queries(
        self,
        df_pool: pd.DataFrame,
        budget: int,
        rng: np.random.Generator,
        detector: Optional[BaseDetector] = None,
        normalizer: Optional[FeatureNormalizer] = None,
    ) -> List[int]:
        n_pool = len(df_pool)
        perm = rng.permutation(n_pool)
        return list(perm[:budget])


def get_all_stage4a_strategies() -> Dict[str, BaseAcquisitionStrategy]:
    """Return dictionary of all Stage 4A acquisition strategies."""
    return {
        "random": RandomAcquisition(),
        "dynamics": DynamicsAcquisition(),
        "margin": MarginAcquisition(),
        "action_disagreement": ActionDisagreementAcquisition(),
        "ratio": RatioAcquisition(),
        "combined": CombinedAcquisition(),
        "oracle": OracleAcquisition(),
        "shuffled_control": ShuffledScoreControl(),
    }
