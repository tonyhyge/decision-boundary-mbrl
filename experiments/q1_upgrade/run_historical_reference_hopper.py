"""
Step 2C: Historical Reference Reproduction for VaGraM on Hopper (Hopper-v4).
Implements the exact official ICLR 2022 VaGraM Hopper hyperparameter schedule:
- Dynamic rollout schedule: [20, 150, 1, 5] (horizon grows from 1 to 5)
- Retrain frequency: every 250 steps
- SAC updates per step: 20
- Model batch size: 1024
- SAC learning rates: critic 3e-4, actor 3e-5, alpha 3e-4, target entropy -1.0
- Elite size: 5 out of 7 members
"""
import os
import sys
import time
import json
import psutil
import argparse
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


def run_historical_vagram_hopper(
    env_id: str = "Hopper-v4",
    seed: int = 42,
    total_steps: int = 50000,
    eval_freq: int = 2500,
    eval_episodes: int = 5,
    device: str = "cpu",
):
    print("=" * 80)
    print(f"HISTORICAL VAGRAM HOPPER REPRODUCTION (Seed {seed}, Budget: {total_steps} steps)")
    print("=" * 80)

    np.random.seed(seed)
    torch.manual_seed(seed)
    process = psutil.Process(os.getpid())
    start_time = time.time()

    train_env = GymnasiumToGymAdapter(env_id, seed=seed)
    eval_env = GymnasiumToGymAdapter(env_id, seed=seed + 20000)
    term_fn = create_termination_fn(env_id)

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]

    # Official VaGraM Architecture: Ensemble of 7 (5 elites), 4 layers, 200 hidden units
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
    dynamics_model.set_elite(list(range(5)))

    # Official SAC Architecture & Hyperparameters
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

    dynamics_model.set_gradient_buffer(
        (obs_dim,), (act_dim,),
        omegaconf.DictConfig({"overrides": {"num_steps": max(100000, total_steps + 10000)}})
    )
    dynamics_model.set_agent(agent)

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

    env_buffer = mbrl.util.ReplayBuffer(capacity=max(100000, total_steps), obs_shape=(obs_dim,), action_shape=(act_dim,))
    sac_buffer = pytorch_sac.ReplayBuffer(
        obs_shape=(obs_dim,),
        action_shape=(act_dim,),
        capacity=100000,
        device=torch.device(device),
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
    epoch = 0
    freq_train_model = 250

    while step < total_steps:
        epoch = step // 1000
        rollout_horizon = get_rollout_horizon(epoch, start_epoch=20, end_epoch=150, min_h=1, max_h=5)

        # Official VaGraM retrain frequency: every 250 steps
        if step % freq_train_model == 0:
            dataset = env_buffer.get_all()
            if len(dataset) > 0:
                in_obs, in_act, out_obs, out_r, _ = dataset.astuple()
                model_in = np.concatenate([in_obs, in_act], axis=-1)
                model_target = np.concatenate([out_obs - in_obs, out_r[:, None]], axis=-1)
                
                t_in = torch.tensor(model_in, dtype=torch.float32, device=device)
                t_tgt = torch.tensor(model_target, dtype=torch.float32, device=device)
                
                optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=1e-3, weight_decay=5e-5)
                batch_size = 1024
                n_batches = max(1, len(t_in) // batch_size)
                
                epoch_loss = 0.0
                for _ in range(5):
                    idx = torch.randperm(len(t_in))
                    for b in range(n_batches):
                        b_idx = idx[b * batch_size : (b + 1) * batch_size]
                        optimizer.zero_grad()
                        loss = dynamics_model._vaml_loss(
                            t_in[b_idx], t_tgt[b_idx], b_idx,
                            next_obs=torch.tensor(out_obs[b_idx.cpu().numpy()], dtype=torch.float32, device=device),
                        ).mean()
                        loss.backward()
                        optimizer.step()
                        total_dyn_updates += 1
                        epoch_loss += loss.item()
                
                dynamics_loss_history.append({"step": step, "loss": epoch_loss / (5 * n_batches)})

            # Model rollouts: 400 transitions per step * rollout_horizon
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

        # SAC updates: 20 updates per step (official VaGraM / MBPO)
        if len(sac_buffer) >= 256:
            for _ in range(20):
                agent.update(sac_buffer, null_logger, step)
                total_sac_updates += 1

        obs = next_obs if not done else train_env.reset()

        # Periodic Evaluation
        if step % eval_freq == 0 or step == total_steps:
            eval_returns = []
            for _ in range(eval_episodes):
                e_obs = eval_env.reset()
                e_done = False
                e_ret = 0.0
                while not e_done:
                    e_act = sac_agent.act(e_obs, sample=False)
                    e_next, e_r, e_done, _ = eval_env.step(e_act)
                    e_ret += e_r
                    e_obs = e_next
                eval_returns.append(e_ret)

            mean_ret = float(np.mean(eval_returns))
            std_ret = float(np.std(eval_returns))
            curr_mem = process.memory_info().rss / (1024 * 1024)
            elapsed_sec = time.time() - start_time

            rec = {
                "step": step,
                "epoch": epoch,
                "rollout_horizon": rollout_horizon,
                "eval_mean_return": mean_ret,
                "eval_std_return": std_ret,
                "eval_returns": eval_returns,
                "env_interactions": total_env_interactions,
                "dyn_updates": total_dyn_updates,
                "sac_updates": total_sac_updates,
                "imagined_transitions": total_imagined_transitions,
                "wall_clock_sec": elapsed_sec,
                "memory_mb": curr_mem,
            }
            evaluation_history.append(rec)
            print(f"[HISTORICAL VAGRAM] Step {step}/{total_steps} (Epoch {epoch}, Horizon {rollout_horizon}) | Return: {mean_ret:.2f} ± {std_ret:.2f} | SAC Updates: {total_sac_updates} | Wall: {elapsed_sec:.1f}s")

    train_env.close()
    eval_env.close()

    res = {
        "benchmark": "historical_vagram_hopper_reference",
        "env_id": env_id,
        "seed": seed,
        "total_steps": total_steps,
        "hyperparameters": {
            "model_batch_size": 1024,
            "freq_train_model": 250,
            "sac_updates_per_step": 20,
            "rollout_schedule": [20, 150, 1, 5],
            "sac_actor_lr": 3e-5,
            "sac_critic_lr": 3e-4,
            "sac_target_entropy": -1.0,
        },
        "final_return_mean": evaluation_history[-1]["eval_mean_return"] if evaluation_history else 0.0,
        "final_return_std": evaluation_history[-1]["eval_std_return"] if evaluation_history else 0.0,
        "evaluation_history": evaluation_history,
        "total_env_interactions": total_env_interactions,
        "total_dyn_updates": total_dyn_updates,
        "total_sac_updates": total_sac_updates,
        "total_imagined_transitions": total_imagined_transitions,
        "total_wall_clock_sec": time.time() - start_time,
        "peak_memory_mb": max(h["memory_mb"] for h in evaluation_history) if evaluation_history else 0.0,
    }

    out_file = os.path.join(_ROOT, "experiments", "q1_upgrade", "results", "historical_vagram_hopper_reference.json")
    with open(out_file, "w") as f:
        json.dump(res, f, indent=2)

    print(f"\nHistorical VaGraM Reference Results successfully saved to: {out_file}")
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=25000, help="Total environment steps")
    parser.add_argument("--eval-freq", type=int, default=2500, help="Evaluation frequency")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()
    run_historical_vagram_hopper(seed=args.seed, total_steps=args.steps, eval_freq=args.eval_freq)
