"""
Diagnostic and analytical metrics for Action Margins, Boundary Pressure, and Correction Value.
"""
from src.metrics.diagnostics import (
    compute_action_margins,
    compute_margin_deformation,
    compute_boundary_pressure,
    compute_value_sensitivity,
    compute_advantage_perturbation,
    evaluate_incremental_r2,
    evaluate_matched_pair_effect,
)

__all__ = [
    "compute_action_margins",
    "compute_margin_deformation",
    "compute_boundary_pressure",
    "compute_value_sensitivity",
    "compute_advantage_perturbation",
    "evaluate_incremental_r2",
    "evaluate_matched_pair_effect",
]
