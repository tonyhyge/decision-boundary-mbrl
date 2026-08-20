"""
Diagnostic Script: Investigating Oracle Model Repair Mechanics in Continuous State CartPole.

Examines:
  1. Training loss on acquired pool transitions Q_B.
  2. Flip elimination on the acquired candidate pool states D_pool (In-sample repair).
  3. Generalization of repair to held-out test trajectories D_eval (Out-of-sample repair).
  4. Effect of training epochs, learning rate, fine-tuning vs scratch training, and oversampling/weighting.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.cartpole_continuous import CartPoleDynamics, CompetentCartPoleValueFunction
from src.models.ensemble_dynamics import EnsembleDynamicsModel
from src.detection.signal_extractor import UnprivilegedSignalExtractor
from src.acquisition.repair_trainer import ModelRepairTrainer
from experiments.unprivileged_detection.run_stage3a_detector_falsification import (
    collect_dynamics_training_data,
    collect_held_out_detector_trajectories,
)

def run_diagnostic():
    dynamics = CartPoleDynamics()
    val_fn = CompetentCartPoleValueFunction(dynamics=dynamics)
    
    # Base dataset
    x_base, y_base = collect_dynamics_training_data(dynamics, val_fn, num_episodes=25, max_steps=60, epsilon=0.40, seed=42)
    base_ensemble = EnsembleDynamicsModel(num_models=7, hidden_dim=64, seed=42)
    base_ensemble.train_ensemble(x_base, y_base, epochs=60, batch_size=64, seed=42)
    
    # Candidate pool & eval
    pool_trajs = collect_held_out_detector_trajectories(dynamics, val_fn, num_trajectories=30, max_steps=70, exploration_epsilon=0.35, seed=142)
    eval_trajs = collect_held_out_detector_trajectories(dynamics, val_fn, num_trajectories=25, max_steps=60, exploration_epsilon=0.25, seed=242)
    
    extractor = UnprivilegedSignalExtractor(ensemble=base_ensemble, val_fn=val_fn, dynamics=dynamics)
    df_pool = extractor.extract_dataset(pool_trajs)
    df_eval = extractor.extract_dataset(eval_trajs)
    
    trainer = ModelRepairTrainer(dynamics=dynamics, val_fn=val_fn)
    base_eval = trainer.evaluate_flips_on_eval_set(base_ensemble, df_eval)
    base_pool = trainer.evaluate_flips_on_eval_set(base_ensemble, df_pool)
    
    print(f"Base Flips on Pool: {base_pool['n_flips']} / {len(df_pool)} ({base_pool['flip_rate']*100:.2f}%)")
    print(f"Base Flips on Eval: {base_eval['n_flips']} / {len(df_eval)} ({base_eval['flip_rate']*100:.2f}%)")
    
    # Oracle: select all true flips in pool
    flip_indices = np.where(df_pool["z_flip"].to_numpy() == 1)[0]
    print(f"Number of true flips in pool: {len(flip_indices)}")
    
    # Query transitions for flips
    x_q, y_q = [], []
    for idx in flip_indices:
        row = df_pool.iloc[idx]
        s = np.array([row["x"], row["x_dot"], row["theta"], row["theta_dot"]], dtype=np.float32)
        for a in [0, 1]:  # Query both actions at the flip state!
            next_s, _, _ = dynamics.step(s, a)
            sa = np.concatenate([s, [1.0 if a == 1 else -1.0]])
            x_q.append(sa)
            y_q.append(next_s - s)
            
    x_q = np.array(x_q, dtype=np.float32)
    y_q = np.array(y_q, dtype=np.float32)
    
    # Test 1: Standard Retraining from Scratch (epochs=80)
    x_aug = np.concatenate([x_base, x_q], axis=0)
    y_aug = np.concatenate([y_base, y_q], axis=0)
    ens_scratch = EnsembleDynamicsModel(num_models=7, hidden_dim=64, seed=42)
    ens_scratch.train_ensemble(x_aug, y_aug, epochs=80, batch_size=64, seed=42)
    
    res_pool_scratch = trainer.evaluate_flips_on_eval_set(ens_scratch, df_pool)
    res_eval_scratch = trainer.evaluate_flips_on_eval_set(ens_scratch, df_eval)
    print(f"\n[Test 1: Scratch Retraining with 1x queries]")
    print(f"  Pool Flips: {res_pool_scratch['n_flips']} (Before: {base_pool['n_flips']}, Fixed: {base_pool['n_flips'] - res_pool_scratch['n_flips']})")
    print(f"  Eval Flips: {res_eval_scratch['n_flips']} (Before: {base_eval['n_flips']})")
    
    # Test 2: Fine-Tuning or High-Weight Retraining on Queries (Oversampling queries 5x)
    x_aug_weighted = np.concatenate([x_base] + [x_q]*5, axis=0)
    y_aug_weighted = np.concatenate([y_base] + [y_q]*5, axis=0)
    ens_weighted = EnsembleDynamicsModel(num_models=7, hidden_dim=64, seed=42)
    ens_weighted.train_ensemble(x_aug_weighted, y_aug_weighted, epochs=100, batch_size=64, seed=42)
    
    res_pool_weighted = trainer.evaluate_flips_on_eval_set(ens_weighted, df_pool)
    res_eval_weighted = trainer.evaluate_flips_on_eval_set(ens_weighted, df_eval)
    print(f"\n[Test 2: Retraining with 5x Oversampled Queries]")
    print(f"  Pool Flips: {res_pool_weighted['n_flips']} (Before: {base_pool['n_flips']}, Fixed: {base_pool['n_flips'] - res_pool_weighted['n_flips']})")
    print(f"  Eval Flips: {res_eval_weighted['n_flips']} (Before: {base_eval['n_flips']})")

if __name__ == "__main__":
    run_diagnostic()
