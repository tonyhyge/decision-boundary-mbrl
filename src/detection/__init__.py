"""
Unprivileged Risk Detection, Calibration, and Evaluation Package.
"""
from src.detection.signal_extractor import UnprivilegedSignalExtractor, FeatureNormalizer
from src.detection.calibrated_detectors import (
    BaseDetector,
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

__all__ = [
    "UnprivilegedSignalExtractor",
    "FeatureNormalizer",
    "BaseDetector",
    "PrevalenceNullDetector",
    "CalibratedScalarDetector",
    "CombinedLogisticDetector",
    "PermutedLabelControlDetector",
    "CalibratedIsotonicDetector",
    "get_all_stage3a_detectors",
    "compute_detector_metrics",
    "evaluate_all_detectors_on_df",
    "run_clustered_paired_bootstrap",
]
