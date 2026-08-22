"""
Stage Q1-FB: Full-Budget Hopper Representativeness Confirmation Benchmark.
Evaluates MLE vs. VaGraM under canonical-scale MBPO training on Hopper-v4 up to 125k environment steps.

Preregistered Protocol:
- Environment: Hopper-v4 (MuJoCo)
- Methods: MLE (Gaussian NLL) vs VaGraM (Value-Gradient Weighted Coordinate MSE)
- Architecture: 7 dynamics MLPs (elite 5, 4x200 hidden ReLU), SAC Double-Q (2x256)
- Model retrain frequency: every 250 environment steps
- SAC policy updates: 20 updates per environment step
- Rollout schedule: [20, 150, 1, 5] (horizon grows from 1 to 5)
- Evaluation: 10 deterministic episodes every 5,000 environment steps
- Primary Estimand: Mean terminal return over final 3 checkpoints (115k, 120k, 125k)
- Secondary Estimands: Learning-curve AUC, H-step rollout MSE, optimization diagnostics
- Paired Independent Confirmatory Seeds: [142, 143, 144, 145, 146, 147, 148, 149, 150, 151]
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
    env_id: str = "Hopper-v4",
    algorithm_type: str = "mle",  # 'mle' or 'vagram'
    seed: int = 142,
    total_steps: int = 125000,
    freq_train_model: int = 250,
    sac_updates_per_step: int = 20,
    model_batch_size: int = 1024,
    eval_freq: int = 5000,
    eval_episodes: int = 10,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes full-budget MBPO training for MLE or VaGraM with exact compute accounting.
    """
    assert algorithm_type in ["mle", "vagram"], f"Unknown algorithm_type: {algorithm_type}"

    # 1. Deterministic Seeding
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and "cuda" in device:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    process = psutil.Process(os.getpid())
    start_time = time.time()
    peak_memory_mb = 0.0

    # 2. Environments
    train_env = GymnasiumToGymAdapter(env_id, seed=seed)
    eval_env = GymnasiumToGymAdapter(env_id, seed=seed + 50000)
    term_fn = create_termination_fn(env_id)

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]

    # 3. Model construction (7 MLPs, elite 5, 4x200 hidden units)
    if algorithm_type == "vagram":
        dynamics_model = vaml_mlp.VAMLMLP(
            in_size=obs_dim + act_dim,
            out_size=obs_dim + 1,
            device=device,
            num_layers=4,
            ensemble_size=7,
            hid_size=200,
            use_silu=False,
            propagation_method="fixed_model",
            bound_clipping=True,
            bound_clipping_quantile=0.95,
            use_vaml=False,
            use_scaling=True,
            add_mse=False,
        )
    else:  # 'mle'
        dynamics_model = gaussian_mlp.GaussianMLP(
            in_size=obs_dim + act_dim,
            out_size=obs_dim + 1,
            device=device,
            num_layers=4,
            ensemble_size=7,
            hid_size=200,
            use_silu=False,
            propagation_method="fixed_model",
            deterministic=False,
        )
    dynamics_model.set_elite(list(range(5)))

    # 4. SAC Agent setup (identical across MLE and VaGraM)
    actor_cfg = omegaconf.DictConfig({
        "_target_": "mbrl.third_party.pytorch_sac.agent.actor.DiagGaussianActor",
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "hidden_dim": 256,
        "hidden_depth": 2,
        "log_std_bounds": [-10, 2],
    })
    critic_cfg = omegaconf.DictConfig({
        "_target_": "mbrl.third_party.pytorch_sac.agent.critic.DoubleQCritic",
        "obs_dim": obs_dim,
        "action_dim": act_dim,
        "hidden_dim": 256,
        "hidden_depth": 2,
    })

    agent = pytorch_sac.agent.sac.SACAgent(
        obs_dim=obs_dim,
        action_dim=act_dim,
        action_range=[-1.0, 1.0],
        device=device,
        critic_cfg=critic_cfg,
        actor_cfg=actor_cfg,
        discount=0.99,
        init_temperature=0.2,
        alpha_lr=3e-4,
        alpha_betas=[0.9, 0.999],
        actor_lr=3e-5,
        actor_betas=[0.9, 0.999],
        actor_update_frequency=1,
        critic_lr=3e-4,
        critic_betas=[0.9, 0.999],
        critic_tau=0.005,
        critic_target_update_frequency=4,
        batch_size=256,
        learnable_temperature=True,
        target_entropy=-1.0,
    )
    sac_agent = sac_wrapper.SACAgent(agent)

    if algorithm_type == "vagram":
        dynamics_model.set_gradient_buffer(
            (obs_dim,), (act_dim,),
            omegaconf.DictConfig({"overrides": {"num_steps": max(200000, total_steps + 10000)}})
        )
        dynamics_model.set_agent(agent)

    # 5. Buffers
    env_buffer = mbrl.util.ReplayBuffer(capacity=max(200000, total_steps + 10000), obs_shape=(obs_dim,), action_shape=(act_dim,))
    sac_buffer = pytorch_sac.ReplayBuffer(
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
        capacity=100000,
        device=torch.device(device),
    )

    one_d_model = mbrl.models.OneDTransitionRewardModel(
        dynamics_model,
        target_is_delta=True,
        normalize=True,
        learned_rewards=True,
    )

    model_env = mbrl.models.ModelEnv(
        train_env,
        one_d_model,
        termination_fn=term_fn,
        reward_fn=None,
        generator=torch.Generator(device=device),
    )

    null_logger = NullLogger()
    total_env_interactions = 0
    total_dyn_updates = 0
    total_sac_updates = 0
    total_imagined_transitions = 0

    evaluation_history = []
    dynamics_loss_history = []

    # Warmup
    warmup_steps = 1000
    obs = train_env.reset()
    for _ in range(warmup_steps):
        action = train_env.action_space.sample()
        next_obs, reward, done, info = train_env.step(action)
        env_buffer.add(obs, action, next_obs, reward, done)
        total_env_interactions += 1
        obs = next_obs if not done else train_env.reset()

    step = warmup_steps

    while step < total_steps:
        epoch = step // 1000
        rollout_horizon = get_rollout_horizon(epoch, start_epoch=20, end_epoch=150, min_h=1, max_h=5)

        # Periodic Model Retraining (every 250 steps)
        if step % freq_train_model == 0 or step == warmup_steps:
            dataset = env_buffer.get_all()
            if len(dataset) > 0:
                in_obs, in_act, out_obs, out_r, _ = dataset.astuple()
                model_in = np.concatenate([in_obs, in_act], axis=-1)
                model_target = np.concatenate([out_obs - in_obs, out_r[:, None]], axis=-1)

                t_in = torch.tensor(model_in, dtype=torch.float32, device=device)
                t_tgt = torch.tensor(model_target, dtype=torch.float32, device=device)

                optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=1e-3, weight_decay=5e-5)
                n_batches = max(1, len(t_in) // model_batch_size)

                epoch_loss = 0.0
                for _ in range(5):  # 5 mini-epochs
                    idx = torch.randperm(len(t_in))
                    for b in range(n_batches):
                        b_idx = idx[b * model_batch_size : (b + 1) * model_batch_size]
                        optimizer.zero_grad()
                        if algorithm_type == "vagram":
                            loss = dynamics_model._vaml_loss(
                                t_in[b_idx], t_tgt[b_idx], b_idx,
                                next_obs=torch.tensor(out_obs[b_idx.cpu().numpy()], dtype=torch.float32, device=device),
                            ).mean()
                        else:
                            mean, logvar = dynamics_model.forward(t_in[b_idx], use_propagation=False)
                            t_tgt_exp = t_tgt[b_idx].unsqueeze(0).repeat(dynamics_model.num_members, 1, 1)
                            loss = mbrl.util.math.gaussian_nll(mean, logvar, t_tgt_exp, reduce=True)

                        loss.backward()
                        optimizer.step()
                        total_dyn_updates += 1
                        epoch_loss += loss.item()

                dynamics_loss_history.append({"step": step, "loss": epoch_loss / (5 * n_batches)})

            # Model rollouts: 400 transitions per batch * rollout_horizon
            raw_batch_size = min(len(env_buffer), 400)
            rollout_batch_size = (raw_batch_size // 5) * 5
            if rollout_batch_size >= 5:
                sample_batch = env_buffer.sample(rollout_batch_size)
                init_obs, *_ = sample_batch.astuple()
                curr_obs = model_env.reset(initial_obs_batch=init_obs, return_as_np=True)

                for _ in range(rollout_horizon):
                    act = sac_agent.act(curr_obs, sample=True, batched=True)
                    pred_next, pred_rew, pred_d, _ = model_env.step(act, sample=True)
                    sac_buffer.add_batch(curr_obs, act, pred_rew, pred_next, pred_d, pred_d)
                    total_imagined_transitions += len(curr_obs)
                    curr_obs = pred_next

        # Environment interaction
        action = sac_agent.act(obs, sample=True)
        next_obs, reward, done, info = train_env.step(action)
        env_buffer.add(obs, action, next_obs, reward, done)
        total_env_interactions += 1
        step += 1

        # SAC updates: 20 updates per step
        if len(sac_buffer) >= 256:
            for _ in range(sac_updates_per_step):
                agent.update(sac_buffer, null_logger, step)
                total_sac_updates += 1

        obs = next_obs if not done else train_env.reset()

        # Periodic Evaluation (every 5,000 steps with 10 deterministic episodes)
        if step % eval_freq == 0 or step == total_steps:
            eval_returns = []
            for ep in range(eval_episodes):
                ep_seed = int(1000000 + step * 100 + ep)
                e_obs = eval_env.reset(seed=ep_seed)
                e_done = False
                e_ret = 0.0
                while not e_done:
                    e_act = sac_agent.act(e_obs, sample=False)
                    e_next, e_r, e_done, _ = eval_env.step(e_act)
                    e_ret += e_r
                    e_obs = e_next
                eval_returns.append(float(e_ret))

            mean_ret = float(np.mean(eval_returns))
            std_ret = float(np.std(eval_returns))
            curr_mem = process.memory_info().rss / (1024 * 1024)
            peak_memory_mb = max(peak_memory_mb, curr_mem)

            rec = {
                "step": step,
                "epoch": step // 1000,
                "rollout_horizon": rollout_horizon,
                "eval_mean_return": mean_ret,
                "eval_std_return": std_ret,
                "eval_returns": eval_returns,
                "env_interactions": total_env_interactions,
                "dyn_updates": total_dyn_updates,
                "sac_updates": total_sac_updates,
                "imagined_transitions": total_imagined_transitions,
                "wall_clock_sec": time.time() - start_time,
                "memory_mb": curr_mem,
            }
            evaluation_history.append(rec)
            if verbose:
                print(f"[{algorithm_type.upper()}] Step {step:6d}/{total_steps} | Return: {mean_ret:7.2f} ± {std_ret:6.2f} | DynUpd: {total_dyn_updates:5d} | SACUpd: {total_sac_updates:7d} | Wall: {rec['wall_clock_sec']:.1f}s")

    train_env.close()
    eval_env.close()

    total_wall_clock = time.time() - start_time

    # Get git metadata for strict audit
    import subprocess
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT).decode("ascii").strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=_ROOT).decode("ascii").strip())
    except Exception:
        git_commit = "unknown"
        git_dirty = False

    # Compute terminal return over final 3 checkpoints
    if len(evaluation_history) >= 3:
        terminal_returns = [h["eval_mean_return"] for h in evaluation_history[-3:]]
        terminal_mean = float(np.mean(terminal_returns))
    elif evaluation_history:
        terminal_mean = evaluation_history[-1]["eval_mean_return"]
    else:
        terminal_mean = 0.0

    return {
        "stage": "Q1-FB",
        "algorithm": algorithm_type,
        "env_id": env_id,
        "seed": seed,
        "total_steps": total_steps,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "terminal_return_mean": terminal_mean,
        "final_return_mean": evaluation_history[-1]["eval_mean_return"] if evaluation_history else 0.0,
        "final_return_std": evaluation_history[-1]["eval_std_return"] if evaluation_history else 0.0,
        "evaluation_history": evaluation_history,
        "dynamics_loss_history": dynamics_loss_history,
        "total_env_interactions": total_env_interactions,
        "total_dyn_updates": total_dyn_updates,
        "total_sac_updates": total_sac_updates,
        "total_imagined_transitions": total_imagined_transitions,
        "total_wall_clock_sec": total_wall_clock,
        "peak_memory_mb": peak_memory_mb,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage Q1-FB Full-Budget Hopper Confirmation.")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["mle", "vagram"], help="Algorithms (mle, vagram)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[142, 143, 144, 145, 146, 147, 148, 149, 150, 151], help="Seeds")
    parser.add_argument("--total-steps", type=int, default=125000, help="Total environment interaction steps")
    parser.add_argument("--eval-freq", type=int, default=5000, help="Evaluation frequency")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of eval episodes per checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--out-dir", type=str, default=os.path.join(_ROOT, "results", "stage_q1fb"), help="Output directory")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("STAGE Q1-FB: FULL-BUDGET HOPPER REPRESENTATIVENESS CONFIRMATION (125k STEPS)")
    print("=" * 80)
    print(f"Algorithms:       {args.algorithms}")
    print(f"Seeds:            {args.seeds} (N={len(args.seeds)})")
    print(f"Total Steps/Seed: {args.total_steps}")
    print(f"Eval Frequency:   Every {args.eval_freq} steps ({args.eval_episodes} deterministic episodes)")
    print(f"Output Directory: {args.out_dir}")
    print("=" * 80)

    # Interleave execution by seed pair: (seed: MLE, seed: VaGraM)
    for seed in args.seeds:
        for algo in args.algorithms:
            out_file = os.path.join(args.out_dir, f"q1fb_hopper_{algo}_seed{seed}.json")
            if os.path.exists(out_file):
                print(f"\n[CACHE HIT] Found existing result: {out_file}")
                continue

            print(f"\n---> Running Q1-FB {algo.upper()} on Hopper-v4 | Seed {seed} (Budget: {args.total_steps} steps) <---")
            t0 = time.time()
            res = run_full_budget_mbpo(
                env_id="Hopper-v4",
                algorithm_type=algo,
                seed=seed,
                total_steps=args.total_steps,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                device=args.device,
                verbose=True,
            )
            elapsed = time.time() - t0
            print(f"[COMPLETED] Q1-FB {algo.upper()} Seed {seed} in {elapsed:.1f}s | Terminal Return: {res['terminal_return_mean']:.2f}")

            with open(out_file, "w") as f:
                json.dump(res, f, indent=2)

    print("\nStage Q1-FB execution completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
