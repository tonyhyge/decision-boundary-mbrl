"""
Stage 4A Active Data Acquisition and World-Model Repair Package.
"""
from src.acquisition.strategies import (
    BaseAcquisitionStrategy,
    RandomAcquisition,
    DynamicsAcquisition,
    MarginAcquisition,
    ActionDisagreementAcquisition,
    RatioAcquisition,
    CombinedAcquisition,
    OracleAcquisition,
    ShuffledScoreControl,
    get_all_stage4a_strategies,
)
from src.acquisition.repair_trainer import ModelRepairTrainer

__all__ = [
    "BaseAcquisitionStrategy",
    "RandomAcquisition",
    "DynamicsAcquisition",
    "MarginAcquisition",
    "ActionDisagreementAcquisition",
    "RatioAcquisition",
    "CombinedAcquisition",
    "OracleAcquisition",
    "ShuffledScoreControl",
    "get_all_stage4a_strategies",
    "ModelRepairTrainer",
]
