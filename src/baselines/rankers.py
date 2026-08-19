"""
Error Ranking Strategies for Budgeted World-Model Correction.
"""
from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np
import pandas as pd


class BaseRanker(ABC):
    """Abstract base class for error prioritization rankers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        """
        Rank error indices (0 to N-1) in descending order of priority.
        """
        pass


class RandomRanker(BaseRanker):
    """Uniform random prioritization."""

    def __init__(self):
        super().__init__("Random")

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        indices = list(range(len(df_errors)))
        rng.shuffle(indices)
        return indices


class PredictionErrorRanker(BaseRanker):
    """Prioritize by maximum prediction error (Total Variation / L1)."""

    def __init__(self):
        super().__init__("Prediction Error (L1)")

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        scores = df_errors["error_l1"].to_numpy()
        return list(np.argsort(-scores))


class OccupancyErrorRanker(BaseRanker):
    """Prioritize by visitation-weighted prediction error (d * E)."""

    def __init__(self):
        super().__init__("Occupancy x Error")

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        scores = df_errors["occupancy"].to_numpy() * df_errors["error_l1"].to_numpy()
        return list(np.argsort(-scores))


class ActionGapRanker(BaseRanker):
    """Prioritize states with the smallest true action gap 1 / (m + eps)."""

    def __init__(self):
        super().__init__("Inverse Action Gap")

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        gaps = df_errors["true_action_gap"].to_numpy()
        scores = 1.0 / (gaps + 1e-4)
        return list(np.argsort(-scores))


class ValueSensitivityAbsRanker(BaseRanker):
    """Prioritize by unsigned first-order value sensitivity |G| (VAML/VaGraM-motivated comparator)."""

    def __init__(self):
        super().__init__("Value Sensitivity (Unsigned |G|)")

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        scores = df_errors["value_sensitivity_abs"].to_numpy()
        return list(np.argsort(-scores))


class ValueSensitivitySignedRanker(BaseRanker):
    r"""Prioritize by signed first-order degradation G^\pm."""

    def __init__(self):
        super().__init__("Value Sensitivity (Signed G±)")

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        # Negative delta P^T V means degradation, so higher priority when delta P^T V is most negative
        scores = -df_errors["value_sensitivity_signed"].to_numpy()
        return list(np.argsort(-scores))


class AdvantageSensitivityRanker(BaseRanker):
    r"""Prioritize by signed advantage perturbation A^\pm."""

    def __init__(self):
        super().__init__("Advantage Sensitivity (A±)")

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        scores = -df_errors["advantage_sensitivity_signed"].to_numpy()
        return list(np.argsort(-scores))


class BoundaryPressureRanker(BaseRanker):
    """Proposed: Prioritize by normalized signed boundary pressure B_i."""

    def __init__(self):
        super().__init__("Boundary Pressure (B_i)")

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        scores = df_errors["boundary_pressure"].to_numpy()
        return list(np.argsort(-scores))


class OccupancyBoundaryPressureRanker(BaseRanker):
    """Proposed: Prioritize by occupancy-weighted signed boundary pressure d * B_i."""

    def __init__(self):
        super().__init__("Occupancy x Boundary Pressure (d·B_i)")

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        scores = df_errors["occupancy"].to_numpy() * df_errors["boundary_pressure"].to_numpy()
        return list(np.argsort(-scores))


class OracleRanker(BaseRanker):
    """Oracle upper bound: Rank by exact counterfactual correction value C_i."""

    def __init__(self):
        super().__init__("Oracle (C_i)")

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        scores = df_errors["correction_value"].to_numpy()
        return list(np.argsort(-scores))


class EstimatedRanker(BaseRanker):
    """Generic wrapper for rankers using features estimated from learned world model."""

    def __init__(self, name: str, feature_col: str, ascending: bool = False):
        super().__init__(name)
        self.feature_col = feature_col
        self.ascending = ascending

    def rank(self, df_errors: pd.DataFrame, rng: np.random.Generator) -> List[int]:
        scores = df_errors[self.feature_col].to_numpy()
        order = np.argsort(scores) if self.ascending else np.argsort(-scores)
        return list(order)


def get_all_rankers() -> List[BaseRanker]:
    """Return an instantiated list of all evaluation rankers."""
    return [
        RandomRanker(),
        PredictionErrorRanker(),
        OccupancyErrorRanker(),
        ActionGapRanker(),
        ValueSensitivityAbsRanker(),
        ValueSensitivitySignedRanker(),
        AdvantageSensitivityRanker(),
        BoundaryPressureRanker(),
        OccupancyBoundaryPressureRanker(),
        OracleRanker(),
    ]
