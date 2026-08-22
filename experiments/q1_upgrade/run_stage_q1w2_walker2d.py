"""
Stage Q1-W2: Full-Budget Walker2d-v4 External Validity & Environment Transfer Benchmark.
Evaluates MLE vs. VaGraM under canonical-scale MBPO training on Walker2d-v4 up to 125k environment steps.

Preregistered Protocol:
- Environment: Walker2d-v4 (MuJoCo, obs_dim=17, act_dim=6)
- Methods: MLE (Gaussian NLL) vs VaGraM (Value-Gradient Weighted Coordinate MSE)
- Architecture: 7 dynamics MLPs (elite 5, 4x200 hidden ReLU), SAC Double-Q (2x256)
- Model retrain frequency: every 250 environment steps
- SAC policy updates: 20 updates per environment step
- Rollout schedule: [20, 150, 1, 5] (horizon grows from 1 to 5)
- Evaluation: 10 deterministic episodes every 5,000 environment steps
- Primary Estimand: Mean terminal return over final 3 checkpoints (115k, 120k, 125k)
- Paired Confirmatory Seeds: [142, 143, 144, 145, 146, 147, 148, 149, 150, 151] (N=10)
- Smoke Test Seed: 101
"""
import os
import sys
import time
import json
import psutil
import argparse
from typing import Dict, Any, List, Optional
import numpy as np
import torch
import omegaconf

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

_VAGRAM_PATH = os.path.abspath(os.path.join(_ROOT, "src", "baselines", "external", "vagram"))
if _VAGRAM_PATH not in sys.path:
    sys.path.insert(0, _VAGRAM_PATH)

import mbrl.models
import mbrl.models.vaml_mlp as vaml_mlp
import mbrl.models.gaussian_mlp as gaussian_mlp
import mbrl.planning.sac_wrapper as sac_wrapper
import mbrl.third_party.pytorch_sac as pytorch_sac
import mbrl.util
import mbrl.util.math
from src.baselines.adapters.vagram_mbpo import GymnasiumToGymAdapter, create_termination_fn, NullLogger


def get_rollout_horizon(epoch: int, start_epoch: int = 20, end_epoch: int = 150, min_h: int = 1, max_h: int = 5) -> int:
    """Calculates rollout horizon according to official MBPO/VaGraM schedule [20, 150, 1, 5]."""
    if epoch <= start_epoch:
        return min_h
    if epoch >= end_epoch:
        return max_h
    frac = (epoch - start_epoch) / (end_epoch - start_epoch)
    return int(min_h + np.floor(frac * (max_h - min_h)))


def run_full_budget_mbpo(
    env_id: str = "Walker2d-v4",
    algorithm_type: str = "mle",  # 'mle' or 'vagram'
    seed: int = 142,
    total_steps: int = 125000,
    epoch_length: int = 250,
    model_batch_size: int = 256,
    ensemble_size: int = 7,
    num_elites: int = 5,
    eval_freq: int = 5000,
    eval_episodes: int = 10,
    device: str = "cuda:3",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes a complete full-budget run of MLE or VaGraM on Walker2d-v4.
    """
    t_start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    env = GymnasiumToGymAdapter(env_id, seed=seed)
    eval_env = GymnasiumToGymAdapter(env_id, seed=seed + 1000)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    term_fn = create_termination_fn(env_id)

    num_epochs = total_steps // epoch_length

    # Canonical Dynamics Model architecture: 4 hidden layers x 200 units, ReLU activations
    if algorithm_type.lower() == "mle":
        model = gaussian_mlp.GaussianMLP(
            in_size=obs_dim + act_dim,
            out_size=obs_dim + 1,  # Predict (delta_s, reward)
            device=torch.device(device),
            num_layers=4,
            ensemble_size=ensemble_size,
            hid_size=200,
            deterministic=False,
            propagation_method="expectation",
            learn_logvar_bounds=False,
        )
    elif algorithm_type.lower() == "vagram":
        model = vaml_mlp.VAMLGaussianMLP(
            in_size=obs_dim + act_dim,
            out_size=obs_dim + 1,
            device=torch.device(device),
            num_layers=4,
            ensemble_size=ensemble_size,
            hid_size=200,
            deterministic=False,
            propagation_method="expectation",
            learn_logvar_bounds=False,
            loss_fn="vagram",
            target_is_delta=True,
            quantile=0.95,
        )
    else:
        raise ValueError(f"Unknown algorithm_type: {algorithm_type}")

    # Canonical Dynamics Optimizer: Adam lr=1e-3, weight_decay=1e-5
    dynamics_model = mbrl.models.ModelEnv(
        env,
        model,
        term_fn,
        generator=torch.Generator(device=torch.device(device)).manual_seed(seed),
    )
    model_trainer = mbrl.models.ModelTrainer(
        model,
        optim_lr=1e-3,
        weight_decay=1e-5,
        logger=NullLogger(),
    )

    # Replay Buffers
    env_buffer = mbrl.util.ReplayBuffer(
        capacity=total_steps + 10000,
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
        obs_type=np.float32,
        action_type=np.float32,
        max_trajectory_length=1000,
        rng=np.random.default_rng(seed),
    )

    model_buffer_capacity = 200000
    model_buffer = mbrl.util.ReplayBuffer(
        capacity=model_buffer_capacity,
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
        obs_type=np.float32,
        action_type=np.float32,
        max_trajectory_length=1000,
        rng=np.random.default_rng(seed + 1),
    )

    # Canonical SAC Agent: Double-Q Critic (2x256), Actor (2x256), lr=3e-4, actor_lr=3e-5
    sac_cfg = omegaconf.OmegaConf.create({
        "_target_": "mbrl.third_party.pytorch_sac.sac.SACAgent",
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "action_range": [float(env.action_space.low.min()), float(env.action_space.high.max())],
        "device": device,
        "critic_cfg": {
            "_target_": "mbrl.third_party.pytorch_sac.sac.DoubleQCritic",
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "hidden_dim": 256,
            "hidden_depth": 2,
        },
        "actor_cfg": {
            "_target_": "mbrl.third_party.pytorch_sac.sac.DiagGaussianActor",
            "obs_dim": obs_dim,
            "action_dim": act_dim,
            "hidden_dim": 256,
            "hidden_depth": 2,
            "log_std_bounds": [-10, 2],
        },
        "discount": 0.99,
        "init_temperature": 0.1,
        "alpha_lr": 3e-4,
        "alpha_betas": [0.9, 0.999],
        "actor_lr": 3e-5,
        "actor_betas": [0.9, 0.999],
        "actor_update_frequency": 1,
        "critic_lr": 3e-4,
        "critic_betas": [0.9, 0.999],
        "critic_tau": 0.005,
        "critic_target_update_frequency": 2,
        "batch_size": 256,
        "learnable_temperature": True,
    })
    agent = sac_wrapper.SACAgent(sac_cfg)

    # Warmup exploration: 5,000 steps with random actions
    warmup_steps = 5000
    obs = env.reset(seed=seed)
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        next_obs, reward, done, info = env.step(action)
        env_buffer.add(obs, action, next_obs, reward, done)
        obs = next_obs
        if done:
            obs = env.reset()

    evaluation_history: List[Dict[str, Any]] = []
    total_env_steps = warmup_steps
    sac_updates_count = 0
    dyn_updates_count = 0
    imagined_rollouts_count = 0

    # Initial Evaluation at step 5k
    eval_rets = []
    for ep in range(eval_episodes):
        e_obs = eval_env.reset(seed=seed + 1000 + ep)
        e_ret = 0.0
        e_done = False
        while not e_done:
            with torch.no_grad():
                e_act = agent.act(e_obs, sample=False, batched=False)
            e_obs, e_r, e_done, _ = eval_env.step(e_act)
            e_ret += e_r
        eval_rets.append(e_ret)

    eval_mean = float(np.mean(eval_rets))
    eval_std = float(np.std(eval_rets))
    evaluation_history.append({
        "step": total_env_steps,
        "epoch": 0,
        "eval_mean_return": eval_mean,
        "eval_std_return": eval_std,
        "rollout_horizon": 1,
        "wall_clock_seconds": time.time() - t_start,
    })

    if verbose:
        print(f"[{algorithm_type.upper()}] Initial Eval @ Step {total_env_steps}: Return = {eval_mean:.2f} ± {eval_std:.2f}")

    # Main Training Loop
    obs = env.reset()
    start_epoch = warmup_steps // epoch_length

    for epoch in range(start_epoch, num_epochs):
        # 1. Retrain Dynamics Model at start of epoch
        if algorithm_type.lower() == "vagram":
            model.update_value_function(agent.sac_agent.critic, agent.sac_agent.actor)

        # Train model using standard MBRL trainer
        dataset_train, dataset_val = env_buffer.get_train_val_split(0.2)
        model_train_info = model_trainer.train(
            dataset_train,
            dataset_val=dataset_val,
            num_epochs=None,
            patience=50,
            batch_size=model_batch_size,
            valid_ratio=0.2,
        )
        dyn_updates_count += model_train_info.get("best_epoch", 1) * max(1, len(dataset_train) // model_batch_size)

        # Determine Rollout Horizon for current epoch
        rollout_horizon = get_rollout_horizon(epoch, start_epoch=20, end_epoch=150, min_h=1, max_h=5)

        # 2. Collect synthetic rollouts into model buffer
        model_buffer.empty()
        rollout_batch_size = 100000
        init_obs_batch = env_buffer.sample(rollout_batch_size).obs

        with torch.no_grad():
            cur_obs = torch.from_numpy(init_obs_batch).float().to(device)
            for h in range(rollout_horizon):
                cur_act = agent.act(cur_obs.cpu().numpy(), sample=True, batched=True)
                cur_act_tensor = torch.from_numpy(cur_act).float().to(device)
                next_obs_tensor, rew_tensor, done_tensor, _ = dynamics_model.step(
                    cur_obs, cur_act_tensor, sample=True
                )
                model_buffer.add_batch(
                    cur_obs.cpu().numpy(),
                    cur_act,
                    next_obs_tensor.cpu().numpy(),
                    rew_tensor.cpu().numpy(),
                    done_tensor.cpu().numpy(),
                )
                cur_obs = next_obs_tensor
                imagined_rollouts_count += rollout_batch_size

        # 3. Environment Step & SAC Policy Updates (20 SAC updates per step)
        for _ in range(epoch_length):
            action = agent.act(obs, sample=True, batched=False)
            next_obs, reward, done, info = env.step(action)
            env_buffer.add(obs, action, next_obs, reward, done)
            obs = next_obs
            total_env_steps += 1

            if done:
                obs = env.reset()

            # 20 SAC Updates per environment step
            for _ in range(20):
                batch_real = env_buffer.sample(int(256 * 0.05))
                batch_model = model_buffer.sample(int(256 * 0.95))
                
                b_obs = np.concatenate([batch_real.obs, batch_model.obs], axis=0)
                b_act = np.concatenate([batch_real.act, batch_model.act], axis=0)
                b_next_obs = np.concatenate([batch_real.next_obs, batch_model.next_obs], axis=0)
                b_rewards = np.concatenate([batch_real.rewards, batch_model.rewards], axis=0)
                b_dones = np.concatenate([batch_real.dones, batch_model.dones], axis=0)

                agent.update(
                    b_obs,
                    b_act,
                    b_rewards,
                    b_next_obs,
                    b_dones,
                    NullLogger(),
                    sac_updates_count,
                )
                sac_updates_count += 1

        # 4. Periodic Evaluation Checkpoints
        if total_env_steps % eval_freq == 0:
            eval_rets = []
            for ep in range(eval_episodes):
                e_obs = eval_env.reset(seed=seed + 1000 + ep + total_env_steps)
                e_ret = 0.0
                e_done = False
                while not e_done:
                    with torch.no_grad():
                        e_act = agent.act(e_obs, sample=False, batched=False)
                    e_obs, e_r, e_done, _ = eval_env.step(e_act)
                    e_ret += e_r
                eval_rets.append(e_ret)

            eval_mean = float(np.mean(eval_rets))
            eval_std = float(np.std(eval_rets))
            eval_record = {
                "step": total_env_steps,
                "epoch": epoch + 1,
                "eval_mean_return": eval_mean,
                "eval_std_return": eval_std,
                "rollout_horizon": rollout_horizon,
                "wall_clock_seconds": time.time() - t_start,
            }
            evaluation_history.append(eval_record)

            if verbose:
                print(
                    f"[{algorithm_type.upper()}] Step {total_env_steps}/{total_steps} | "
                    f"Horizon: {rollout_horizon} | Return: {eval_mean:.2f} ± {eval_std:.2f} | "
                    f"Wall: {time.time() - t_start:.1f}s"
                )

    wall_clock_total = time.time() - t_start
    env.close()
    eval_env.close()

    # Compute terminal metrics: mean over final 3 checkpoints (115k, 120k, 125k)
    final_3_returns = [h["eval_mean_return"] for h in evaluation_history[-3:]]
    terminal_return_mean = float(np.mean(final_3_returns))
    terminal_return_std = float(np.std(final_3_returns))

    peak_ram_mb = float(psutil.Process().memory_info().rss / (1024 * 1024))

    result = {
        "benchmark": "Stage Q1-W2: Walker2d-v4 Full-Budget MBPO Replication",
        "environment": env_id,
        "algorithm": algorithm_type.lower(),
        "seed": seed,
        "total_steps": total_steps,
        "evaluation_history": evaluation_history,
        "terminal_return_mean": terminal_return_mean,
        "terminal_return_std": terminal_return_std,
        "final_return_at_125k": evaluation_history[-1]["eval_mean_return"],
        "final_return_std_at_125k": evaluation_history[-1]["eval_std_return"],
        "wall_clock_seconds": wall_clock_total,
        "peak_ram_mb": peak_ram_mb,
        "total_dynamics_updates": dyn_updates_count,
        "total_sac_updates": sac_updates_count,
        "total_imagined_rollouts": imagined_rollouts_count,
        "device": device,
    }

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Stage Q1-W2 Full-Budget Walker2d-v4 MBPO Benchmark")
    parser.add_argument("--algorithms", nargs="+", default=["mle", "vagram"], choices=["mle", "vagram"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[142, 143, 144, 145, 146, 147, 148, 149, 150, 151])
    parser.add_argument("--total-steps", type=int, default=125000, help="Total environment interaction steps")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency in environment steps")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of eval episodes per checkpoint")
    parser.add_argument("--device", type=str, default="cuda:3", help="Device (e.g. cuda:3 or cpu)")
    parser.add_argument("--out-dir", type=str, default=os.path.join(_ROOT, "results", "stage_q1w2_cuda"), help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE Q1-W2: FULL-BUDGET WALKER2D-V4 REPLICATION BENCHMARK (125k STEPS)")
    print("=" * 80)
    print(f"Algorithms:       {args.algorithms}")
    print(f"Seeds:            {args.seeds} (N={len(args.seeds)})")
    print(f"Total Steps/Seed: {args.total_steps}")
    print(f"Eval Frequency:   Every {args.eval_freq} steps ({args.eval_episodes} deterministic episodes)")
    print(f"Output Directory: {args.out_dir}")
    print("=" * 80)

    for seed in args.seeds:
        for algo in args.algorithms:
            out_file = os.path.join(args.out_dir, f"q1w2_walker2d_{algo}_seed{seed}.json")
            if os.path.exists(out_file):
                print(f"\n[CACHE HIT] Found existing result: {out_file}")
                continue

            print(f"\n---> Running Q1-W2 {algo.upper()} on Walker2d-v4 | Seed {seed} (Budget: {args.total_steps} steps) <---")
            t0 = time.time()
            res = run_full_budget_mbpo(
                env_id="Walker2d-v4",
                algorithm_type=algo,
                seed=seed,
                total_steps=args.total_steps,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                device=args.device,
                verbose=True,
            )
            elapsed = time.time() - t0
            print(f"[COMPLETED] Q1-W2 {algo.upper()} Seed {seed} in {elapsed:.1f}s | Terminal Return: {res['terminal_return_mean']:.2f}")

            with open(out_file, "w") as f:
                json.dump(res, f, indent=2)

    print("\nStage Q1-W2 execution completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
