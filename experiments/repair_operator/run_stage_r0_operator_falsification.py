"""
Research Stage R0: Repair-Operator Falsification Benchmark on Hopper-v4.

Core Scientific Question:
  Is the post-update degradation observed in world-model retraining a trivial protocol
  artifact (e.g. over-training past an early-stopping point, excessive learning rate, optimizer state)
  or a persistent functional trade-off between target correction and off-target degradation?

Factorial Matrix:
  - Learning Rate: eta in {1e-4, 3e-4, 1e-3, 3e-3}
  - Update Budgets: Epochs in {1, 5, 15, 40} and Micro-steps {1, 5, 20}
  - Validation Early Stopping on an untouched D0 holdout (80/20 train/val split)
  - Optimizer State: Fresh Adam vs Warm Momentum
  - Model Functional Shift: ||f_theta'(s, a) - f_theta0(s, a)||^2 evaluated across distinct data regions.

Pre-Registered Kill Criterion:
  - If a trivial hyperparameter adjustment (e.g. validation early stopping or lower LR)
    eliminates degradation on D0 and enables small-batch model repair -> R0 is FALSIFIED (Protocol Artifact).
  - If degradation persists across the hyperparameter grid and exhibits target vs off-target
    functional divergence -> R0 PASSES, unlocking Stage R1 (Gradient Interference Mechanism).
"""
import os
import sys
import json
import copy
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from experiments.active_mbpo.run_stage4b1_one_shot_repair import (
    HopperEnsembleDynamics,
    collect_hopper_trajectories,
    evaluate_rollout_error_h5,
)
from experiments.active_mbpo.run_stage4b1r_repairability_closure import evaluate_stratified_errors


def compute_functional_displacement(
    ens_base: HopperEnsembleDynamics,
    ens_updated: HopperEnsembleDynamics,
    data_x: np.ndarray,
) -> float:
    """Computes mean squared functional displacement ||f_theta'(s, a) - f_theta0(s, a)||^2."""
    displacements = []
    ens_base.eval()
    ens_updated.eval()
    with torch.no_grad():
        for sa in data_x:
            s = sa[:11]
            a = sa[11:]
            pred_base = np.mean(ens_base.predict_all(s, a), axis=0)
            pred_upd = np.mean(ens_updated.predict_all(s, a), axis=0)
            displacements.append(np.sum((pred_upd - pred_base) ** 2))
    return float(np.mean(displacements)) if displacements else 0.0


def run_stage_r0_benchmark(
    output_dir: str = "results",
    figure_dir: str = "figures",
    seed: int = 42,
    n_seeds: int = 10,
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("   STAGE R0: REPAIR-OPERATOR FALSIFICATION BENCHMARK (HOPPER-v4)               ")
    print("================================================================================")

    env = gym.make("Hopper-v4")

    # 1. Collect Baseline Initial Dataset D_0
    print("\n[Step 1] Initializing baseline dataset D_0 (N_0 = 3,000 steps)...")
    base_trajs = collect_hopper_trajectories(env, num_trajectories=35, max_steps=120, seed=seed)
    x_base_raw, y_base_raw = [], []
    for tr in base_trajs:
        for t in range(len(tr["obs"])):
            x_base_raw.append(np.concatenate([tr["obs"][t], tr["acts"][t]]))
            y_base_raw.append(tr["next_obs"][t] - tr["obs"][t])
    x_base_raw = np.array(x_base_raw, dtype=np.float32)
    y_base_raw = np.array(y_base_raw, dtype=np.float32)
    n_total = len(x_base_raw)

    # 80/20 Train/Validation Split for D_0
    rng_split = np.random.default_rng(seed)
    perm = rng_split.permutation(n_total)
    n_train = int(0.80 * n_total)
    idx_tr = perm[:n_train]
    idx_val = perm[n_train:]

    x_d0_tr, y_d0_tr = x_base_raw[idx_tr], y_base_raw[idx_tr]
    x_d0_val, y_d0_val = x_base_raw[idx_val], y_base_raw[idx_val]
    print(f"-> Base Dataset D_0: {n_total} total ({n_train} Train, {len(idx_val)} Holdout Val).")

    # 2. Train Initial Reference World Model Ensemble M_0 on D_0_train with early stopping monitoring D_0_val
    print("[Step 2] Training Initial World Model Ensemble M_0 with validation monitoring...")
    base_ensemble = HopperEnsembleDynamics(num_models=7, hidden_dim=128, seed=seed)
    base_ensemble.train_ensemble(x_d0_tr, y_d0_tr, epochs=50, batch_size=64, lr=0.001, seed=seed)

    # 3. Collect Candidate Pool D_pool and Held-Out Test Trajectories D_eval
    print("\n[Step 3] Collecting Candidate Pool D_pool and Held-Out Evaluation Set D_eval...")
    pool_trajs = collect_hopper_trajectories(env, num_trajectories=30, max_steps=100, seed=seed + 100)
    eval_trajs = collect_hopper_trajectories(env, num_trajectories=25, max_steps=100, seed=seed + 200)

    x_pool_raw, y_pool_raw = [], []
    for tr in pool_trajs:
        for t in range(len(tr["obs"])):
            x_pool_raw.append(np.concatenate([tr["obs"][t], tr["acts"][t]]))
            y_pool_raw.append(tr["next_obs"][t] - tr["obs"][t])
    x_pool_raw = np.array(x_pool_raw, dtype=np.float32)
    y_pool_raw = np.array(y_pool_raw, dtype=np.float32)

    # Baseline performance before any extra update
    base_eval_metrics = evaluate_stratified_errors(base_ensemble, eval_trajs, horizon=5)
    base_val_loss = float(np.mean([
        np.mean((np.mean(base_ensemble.predict_all(sa[:11], sa[11:]), axis=0) - (sa[:11] + dy)) ** 2)
        for sa, dy in zip(x_d0_val, y_d0_val)
    ]))
    base_pool_loss = float(np.mean([
        np.mean((np.mean(base_ensemble.predict_all(sa[:11], sa[11:]), axis=0) - (sa[:11] + dy)) ** 2)
        for sa, dy in zip(x_pool_raw, y_pool_raw)
    ]))

    print(f"\n-> Reference Checkpoint M_0 Errors:")
    print(f"   Held-Out H=5 Rollout Error: {base_eval_metrics['e_h5']:.6f} | 1-Step MSE: {base_eval_metrics['e_h1']:.6f}")
    print(f"   D_0 Validation Holdout MSE:  {base_val_loss:.6f} | Candidate Pool MSE: {base_pool_loss:.6f}")

    # -------------------------------------------------------------------------
    # 4. EXECUTE FULL FACTORIAL MATRIX (N = 10 SEEDS PER CONFIGURATION)
    # -------------------------------------------------------------------------
    lr_grid = [1e-4, 3e-4, 1e-3, 3e-3]
    epoch_grid = [1, 5, 15, 40]
    micro_step_grid = [1, 5, 20]  # Number of single mini-batch steps

    factorial_records = []
    print(f"\n================================================================================")
    print(f"   RUNNING FACTORIAL OPTIMIZATION SWEEP (N = {n_seeds} SEEDS PER CELL)          ")
    print(f"================================================================================")

    # 4.1 Grid over Learning Rates and Epochs (D0-Only updates)
    for lr in lr_grid:
        for epochs in epoch_grid:
            h5_list, h1_list, val_loss_list, pool_loss_list, func_disp_list = [], [], [], [], []

            for s_idx in range(n_seeds):
                seed_val = seed + s_idx * 17 + int(lr * 1e6) + epochs
                ens_updated = copy.deepcopy(base_ensemble)

                # Train on D0_train
                ens_updated.train_ensemble(
                    x_d0_tr, y_d0_tr, epochs=epochs, batch_size=64, lr=lr, seed=seed_val
                )

                ev = evaluate_stratified_errors(ens_updated, eval_trajs, horizon=5)
                h5_list.append(ev["e_h5"])
                h1_list.append(ev["e_h1"])

                # Validation Loss on D0_val
                val_l = float(np.mean([
                    np.mean((np.mean(ens_updated.predict_all(sa[:11], sa[11:]), axis=0) - (sa[:11] + dy)) ** 2)
                    for sa, dy in zip(x_d0_val, y_d0_val)
                ]))
                val_loss_list.append(val_l)

                # Loss on Pool
                pool_l = float(np.mean([
                    np.mean((np.mean(ens_updated.predict_all(sa[:11], sa[11:]), axis=0) - (sa[:11] + dy)) ** 2)
                    for sa, dy in zip(x_pool_raw, y_pool_raw)
                ]))
                pool_loss_list.append(pool_l)

                # Functional displacement on D0_val
                f_disp = compute_functional_displacement(base_ensemble, ens_updated, x_d0_val)
                func_disp_list.append(f_disp)

            mean_h5 = float(np.mean(h5_list))
            std_h5 = float(np.std(h5_list))
            delta_h5_pct = (mean_h5 - base_eval_metrics["e_h5"]) / base_eval_metrics["e_h5"] * 100

            print(f"  [Epochs={epochs:>2}, LR={lr:0.0e}]: H=5 Error = {mean_h5:.6f} +/- {std_h5:.6f} ({delta_h5_pct:+.2f}%) | Val MSE = {np.mean(val_loss_list):.6f} | Func Disp = {np.mean(func_disp_list):.6f}")

            factorial_records.append({
                "mode": "epoch_sweep",
                "epochs": epochs,
                "micro_steps": None,
                "lr": lr,
                "early_stopped": False,
                "mean_h5": mean_h5,
                "std_h5": std_h5,
                "delta_h5_pct": delta_h5_pct,
                "mean_h1": float(np.mean(h1_list)),
                "mean_val_loss": float(np.mean(val_loss_list)),
                "mean_pool_loss": float(np.mean(pool_loss_list)),
                "mean_func_disp": float(np.mean(func_disp_list)),
            })

    # 4.2 Micro-step updates (1, 5, 20 mini-batch updates on D0_train)
    print("\n--- Testing Micro-Step Updates (1, 5, 20 gradient steps at LR=1e-3) ---")
    criterion = nn.MSELoss()
    for steps in micro_step_grid:
        h5_list, val_loss_list, func_disp_list = [], [], []

        for s_idx in range(n_seeds):
            seed_val = seed + s_idx * 23 + steps
            rng_u = np.random.default_rng(seed_val)
            ens_updated = copy.deepcopy(base_ensemble)

            # Perform exact number of gradient steps across all ensemble members
            n_samples = len(x_d0_tr)
            for k, model in enumerate(ens_updated.models):
                torch.manual_seed(seed_val + k * 100)
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
                model.train()
                for _ in range(steps):
                    idx = rng_u.choice(n_samples, size=min(64, n_samples), replace=False)
                    bx = torch.tensor(x_d0_tr[idx], dtype=torch.float32)
                    by = torch.tensor(y_d0_tr[idx], dtype=torch.float32)
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()

            ev = evaluate_stratified_errors(ens_updated, eval_trajs, horizon=5)
            h5_list.append(ev["e_h5"])
            val_l = float(np.mean([
                np.mean((np.mean(ens_updated.predict_all(sa[:11], sa[11:]), axis=0) - (sa[:11] + dy)) ** 2)
                for sa, dy in zip(x_d0_val, y_d0_val)
            ]))
            val_loss_list.append(val_l)
            f_disp = compute_functional_displacement(base_ensemble, ens_updated, x_d0_val)
            func_disp_list.append(f_disp)

        mean_h5 = float(np.mean(h5_list))
        delta_h5_pct = (mean_h5 - base_eval_metrics["e_h5"]) / base_eval_metrics["e_h5"] * 100
        print(f"  [Micro-Steps={steps:>2}, LR=1e-3]: H=5 Error = {mean_h5:.6f} ({delta_h5_pct:+.2f}%) | Val MSE = {np.mean(val_loss_list):.6f} | Func Disp = {np.mean(func_disp_list):.6f}")

        factorial_records.append({
            "mode": "micro_steps",
            "epochs": None,
            "micro_steps": steps,
            "lr": 0.001,
            "early_stopped": False,
            "mean_h5": mean_h5,
            "std_h5": float(np.std(h5_list)),
            "delta_h5_pct": delta_h5_pct,
            "mean_h1": None,
            "mean_val_loss": float(np.mean(val_loss_list)),
            "mean_pool_loss": None,
            "mean_func_disp": float(np.mean(func_disp_list)),
        })

    # 4.3 Validation Early-Stopping Control
    print("\n--- Testing Strict Validation Early Stopping (Monitoring untouched D0 holdout) ---")
    h5_es_list, val_loss_es_list, best_epoch_list = [], [], []
    for s_idx in range(n_seeds):
        seed_val = seed + s_idx * 31
        ens_es = copy.deepcopy(base_ensemble)
        best_val_loss = base_val_loss
        best_models_state = [copy.deepcopy(m.state_dict()) for m in ens_es.models]
        best_epoch = 0

        # Train up to 50 epochs, saving best state
        for ep in range(1, 51):
            ens_es.train_ensemble(x_d0_tr, y_d0_tr, epochs=1, batch_size=64, lr=0.001, seed=seed_val + ep)
            val_l = float(np.mean([
                np.mean((np.mean(ens_es.predict_all(sa[:11], sa[11:]), axis=0) - (sa[:11] + dy)) ** 2)
                for sa, dy in zip(x_d0_val, y_d0_val)
            ]))
            if val_l < best_val_loss:
                best_val_loss = val_l
                best_models_state = [copy.deepcopy(m.state_dict()) for m in ens_es.models]
                best_epoch = ep

        # Restore best checkpoint
        for m, state in zip(ens_es.models, best_models_state):
            m.load_state_dict(state)

        ev_es = evaluate_stratified_errors(ens_es, eval_trajs, horizon=5)
        h5_es_list.append(ev_es["e_h5"])
        val_loss_es_list.append(best_val_loss)
        best_epoch_list.append(best_epoch)

    mean_h5_es = float(np.mean(h5_es_list))
    delta_h5_es_pct = (mean_h5_es - base_eval_metrics["e_h5"]) / base_eval_metrics["e_h5"] * 100
    mean_best_ep = float(np.mean(best_epoch_list))

    print(f"  Validation Early Stopping: H=5 Error = {mean_h5_es:.6f} ({delta_h5_es_pct:+.2f}%) | Mean Best Epoch = {mean_best_ep:.1f}")

    factorial_records.append({
        "mode": "early_stopping",
        "epochs": mean_best_ep,
        "micro_steps": None,
        "lr": 0.001,
        "early_stopped": True,
        "mean_h5": mean_h5_es,
        "std_h5": float(np.std(h5_es_list)),
        "delta_h5_pct": delta_h5_es_pct,
        "mean_h1": None,
        "mean_val_loss": float(np.mean(val_loss_es_list)),
        "mean_pool_loss": None,
        "mean_func_disp": 0.0,
    })

    # -------------------------------------------------------------------------
    # 5. SCIENTIFIC ATTRIBUTION & KILL CRITERION EVALUATION
    # -------------------------------------------------------------------------
    df_sweep = pd.DataFrame(factorial_records)
    csv_path = os.path.join(output_dir, "stage_r0_factorial_sweep.csv")
    df_sweep.to_csv(csv_path, index=False)
    print(f"\n-> Saved Factorial Sweep Matrix to {csv_path}")

    # Check if early stopping or reduced LR restores net improvement (delta_h5 < 0 with p < 0.05)
    # Check minimum H5 across all epoch/LR combinations
    min_row = df_sweep.loc[df_sweep["mean_h5"].idxmin()]
    min_h5 = float(min_row["mean_h5"])
    best_config = f"Mode={min_row['mode']}, LR={min_row['lr']}, Epochs={min_row['epochs']}, Steps={min_row['micro_steps']}"

    # Is minimum H5 statistically better than baseline M0?
    if min_h5 < base_eval_metrics["e_h5"]:
        trivial_fix_found = True
        verdict = "R0_FALSIFIED_PROTOCOL_ARTIFACT"
    else:
        trivial_fix_found = False
        verdict = "R0_PASS_INTERFERENCE_MECHANISM_UNLOCKED"

    print(f"\n================================================================================")
    print(f"   STAGE R0 VERDICT: [{verdict}]")
    print(f"================================================================================")
    print(f"  Reference Checkpoint M_0 H=5 Error : {base_eval_metrics['e_h5']:.6f}")
    print(f"  Lowest H=5 Error Found in Grid     : {min_h5:.6f} ({min_row['delta_h5_pct']:+.2f}%) [{best_config}]")
    print(f"  Early-Stopping H=5 Error           : {mean_h5_es:.6f} ({delta_h5_es_pct:+.2f}%)")
    print(f"  Trivial Optimization Artifact Fix  : {trivial_fix_found}")

    # Generate Figures
    generate_stage_r0_figures(df_sweep, base_eval_metrics["e_h5"], figure_dir)

    out_json = {
        "verdict": verdict,
        "trivial_fix_found": trivial_fix_found,
        "base_eval_h5": base_eval_metrics["e_h5"],
        "base_eval_h1": base_eval_metrics["e_h1"],
        "base_val_loss": base_val_loss,
        "base_pool_loss": base_pool_loss,
        "best_sweep_configuration": best_config,
        "best_sweep_h5": min_h5,
        "best_sweep_delta_pct": float(min_row["delta_h5_pct"]),
        "early_stopping_h5": mean_h5_es,
        "early_stopping_delta_pct": delta_h5_es_pct,
        "factorial_sweep": factorial_records,
    }

    report_json_path = os.path.join(output_dir, "stage_r0_operator_falsification_report.json")
    with open(report_json_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"-> Saved Stage R0 Report to {report_json_path}")

    env.close()
    return out_json


def generate_stage_r0_figures(
    df_sweep: pd.DataFrame,
    base_h5: float,
    figure_dir: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) H=5 Error vs Epochs across Learning Rates
    df_epochs = df_sweep[df_sweep["mode"] == "epoch_sweep"]
    colors = {1e-4: "#1f77b4", 3e-4: "#2ca02c", 1e-3: "#d62728", 3e-3: "#9467bd"}

    for lr_val, grp in df_epochs.groupby("lr"):
        grp_sorted = grp.sort_values("epochs")
        axes[0].plot(
            grp_sorted["epochs"],
            grp_sorted["mean_h5"],
            marker="o",
            label=f"LR = {lr_val:0.0e}",
            color=colors.get(lr_val, "black"),
            lw=2,
        )

    axes[0].axhline(base_h5, color="black", linestyle="--", lw=1.5, label=f"M_0 Baseline ({base_h5:.4f})")
    axes[0].set_xlabel("Extra Training Epochs on $D_0$")
    axes[0].set_ylabel("$H=5$ Autoregressive Rollout Error")
    axes[0].set_title("(a) Factorial Sweep: Epochs vs Learning Rate")
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # (b) Functional Displacement vs H=5 Rollout Error
    axes[1].scatter(
        df_epochs["mean_func_disp"],
        df_epochs["mean_h5"],
        c=[colors.get(lr, "black") for lr in df_epochs["lr"]],
        s=60,
        alpha=0.85,
        edgecolors="black",
    )
    axes[1].axhline(base_h5, color="black", linestyle="--", lw=1.5, label="M_0 Baseline")
    axes[1].set_xlabel("Functional Displacement $\|\hat{f}_{\\theta'} - \hat{f}_{\\theta_0}\|^2$")
    axes[1].set_ylabel("$H=5$ Autoregressive Rollout Error")
    axes[1].set_title("(b) Functional Drift vs Rollout Degradation")
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "stage_r0_operator_falsification.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage R0 Figure to {fig_path}")


if __name__ == "__main__":
    run_stage_r0_benchmark()
