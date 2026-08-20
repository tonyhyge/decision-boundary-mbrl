"""
World-Model Repair and Retraining Harness for Stage 4A Active Acquisition.

Provides:
  - ModelRepairTrainer: Retrains ensemble dynamics on D_0 U Q_B with matched gradient budgets.
  - Evaluation of decision flip reduction, efficiency eta_B, boundary-local error, and policy return.
"""
from typing import Dict, List, Tuple, Optional
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from src.envs.cartpole_continuous import CartPoleDynamics, CompetentCartPoleValueFunction
from src.models.ensemble_dynamics import EnsembleDynamicsModel
from src.detection.signal_extractor import UnprivilegedSignalExtractor


class ModelRepairTrainer:
    """
    Standardized retraining harness for world-model repair.
    Guarantees strictly identical gradient update budgets, architectures, and hyperparameters.
    """

    def __init__(
        self,
        dynamics: CartPoleDynamics,
        val_fn: CompetentCartPoleValueFunction,
        epochs: int = 50,
        lr: float = 0.005,
        batch_size: int = 64,
        weight_decay: float = 1e-4,
    ):
        self.dynamics = dynamics
        self.val_fn = val_fn
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

    def retrain_or_finetune_ensemble(
        self,
        base_ensemble: EnsembleDynamicsModel,
        x_aug: np.ndarray,
        y_aug: np.ndarray,
        finetune: bool = True,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        seed: int = 42,
    ) -> EnsembleDynamicsModel:
        """
        Update the world-model ensemble on augmented data D_0 U Q_B.
        By default uses continual warm-start fine-tuning (standard in MBPO / online MBRL),
        or fresh scratch retraining if finetune=False.
        """
        n_epochs = epochs or (30 if finetune else self.epochs)
        learning_rate = lr or (0.001 if finetune else self.lr)
        loss_fn = nn.MSELoss()
        n_aug = len(x_aug)
        rng = np.random.default_rng(seed)

        if finetune:
            ensemble_updated = copy.deepcopy(base_ensemble)
            for k, model in enumerate(ensemble_updated.models):
                torch.manual_seed(seed + k * 100)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=self.weight_decay)
                model.train()
                for epoch in range(n_epochs):
                    perm = rng.permutation(n_aug)
                    for b in range(0, n_aug, self.batch_size):
                        idx_b = perm[b : b + self.batch_size]
                        bx = torch.tensor(x_aug[idx_b], dtype=torch.float32)
                        by = torch.tensor(y_aug[idx_b], dtype=torch.float32)
                        optimizer.zero_grad()
                        pred = model(bx)
                        loss = loss_fn(pred, by)
                        loss.backward()
                        optimizer.step()
            return ensemble_updated
        else:
            new_ensemble = EnsembleDynamicsModel(
                num_models=base_ensemble.num_models,
                hidden_dim=base_ensemble.hidden_dim,
                seed=seed,
            )
            new_ensemble.train_ensemble(
                x_data=x_aug,
                y_data=y_aug,
                epochs=self.epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                weight_decay=self.weight_decay,
                bootstrap=True,
                seed=seed,
            )
            return new_ensemble

    def evaluate_flips_on_eval_set(
        self,
        ensemble: EnsembleDynamicsModel,
        df_eval: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Evaluate true reference-action flips and boundary-local model error on df_eval.
        """
        extractor = UnprivilegedSignalExtractor(
            ensemble=ensemble, val_fn=self.val_fn, dynamics=self.dynamics
        )
        flips = []
        sq_errors = []
        near_boundary_errors = []
        far_boundary_errors = []

        tau_boundary = float(df_eval["m_true"].quantile(0.25)) if "m_true" in df_eval.columns else 2.0

        for _, row in df_eval.iterrows():
            s = np.array([row["x"], row["x_dot"], row["theta"], row["theta_dot"]], dtype=np.float64)

            # Model ensemble internal prediction
            signals = extractor.extract_state_signals(s)
            a_model = signals["a_model_ens"]

            # Ground truth reference action
            z_flip, a_true, m_true, _ = extractor.compute_ground_truth_flip(s, a_model)
            flips.append(z_flip)

            # Dynamics prediction accuracy on ground truth step
            # Take true step under action 0 and action 1
            for a in [0, 1]:
                true_ns, _, _ = self.dynamics.step(s, a)
                pred_all_ns = ensemble.predict_all_next_states(s, a)
                mean_pred_ns = np.mean(pred_all_ns, axis=0)
                err = float(np.sum((mean_pred_ns - true_ns) ** 2))
                sq_errors.append(err)
                if m_true <= tau_boundary:
                    near_boundary_errors.append(err)
                else:
                    far_boundary_errors.append(err)

        n_eval = len(flips)
        n_pos = int(np.sum(flips))
        flip_rate = float(n_pos / max(1, n_eval))

        return {
            "n_eval": n_eval,
            "n_flips": n_pos,
            "flip_rate": flip_rate,
            "mean_mse": float(np.mean(sq_errors)),
            "near_boundary_mse": float(np.mean(near_boundary_errors)) if near_boundary_errors else 0.0,
            "far_boundary_mse": float(np.mean(far_boundary_errors)) if far_boundary_errors else 0.0,
        }

    def evaluate_downstream_policy_return(
        self,
        ensemble: EnsembleDynamicsModel,
        num_episodes: int = 50,
        max_steps: int = 200,
        seed: int = 42,
    ) -> Dict[str, float]:
        """
        Evaluate downstream return of model-greedy policy in true CartPole environment P.
        """
        rng = np.random.default_rng(seed)
        extractor = UnprivilegedSignalExtractor(
            ensemble=ensemble, val_fn=self.val_fn, dynamics=self.dynamics
        )
        returns = []

        for _ in range(num_episodes):
            s = rng.uniform(low=[-0.1, -0.1, -0.05, -0.05], high=[0.1, 0.1, 0.05, 0.05])
            ep_ret = 0.0
            for _ in range(max_steps):
                signals = extractor.extract_state_signals(s)
                a = signals["a_model_ens"]
                next_s, r, done = self.dynamics.step(s, a)
                ep_ret += r
                s = next_s
                if done:
                    break
            returns.append(ep_ret)

        returns = np.array(returns)
        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "success_rate_200": float(np.mean(returns >= 200.0)),
        }
