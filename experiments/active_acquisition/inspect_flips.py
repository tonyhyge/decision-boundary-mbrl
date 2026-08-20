"""Inspect flip states numerical values."""
import os, sys, numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.envs.cartpole_continuous import CartPoleDynamics, CompetentCartPoleValueFunction
from src.models.ensemble_dynamics import EnsembleDynamicsModel
from src.detection.signal_extractor import UnprivilegedSignalExtractor

dynamics = CartPoleDynamics()
val_fn = CompetentCartPoleValueFunction(dynamics=dynamics)
x_base = np.load("scratch/x_base.npy") if os.path.exists("scratch/x_base.npy") else None
