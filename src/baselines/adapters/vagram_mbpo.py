"""
Adapter and unified runner for MLE-MBPO and VaGraM-MBPO baselines.
Provides a clean Gymnasium-to-Gym wrapper and comprehensive compute/metric accounting.
DOES NOT MODIFY EXTERNAL REPOSITORY CODE DIRECTLY.
"""
import os
import sys
import time
import psutil
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import gymnasium as gym

# Add external vagram repo to sys.path
_VAGRAM_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "external", "vagram"))
if _VAGRAM_PATH not in sys.path:
    sys.path.insert(0, _VAGRAM_PATH)

import mbrl.models
import mbrl.models.vaml_mlp as vaml_mlp
import mbrl.models.gaussian_mlp as gaussian_mlp
import mbrl.planning.sac_wrapper as sac_wrapper
import mbrl.third_party.pytorch_sac as pytorch_sac
import mbrl.util
import mbrl.util.math


class NullLogger:
    def log(self, *args, **kwargs):
        pass
    def log_histogram(self, *args, **kwargs):
        pass
    def log_param(self, *args, **kwargs):
        pass


class GymnasiumToGymAdapter:
    """
    Adapts a standard Gymnasium environment to the legacy Gym interface
    expected by MBRL-lib / VaGraM without modifying external libraries.
    """
    def __init__(self, env_id: str, seed: Optional[int] = None):
        self.env_id = env_id
        self._seed = seed
        self.env = gym.make(env_id)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self) -> np.ndarray:
        obs, info = self.env.reset(seed=self._seed)
        # Advance seed for future resets if needed
        if self._seed is not None:
            self._seed += 1
        return np.asarray(obs, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info["terminated"] = terminated
        info["truncated"] = truncated
        return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info

    def close(self):
        self.env.close()

    def render(self):
        return self.env.render()


def create_termination_fn(env_id: str):
    """
    Returns termination function for continuous control environments.
    """
    if "InvertedPendulum" in env_id:
        def term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
            assert len(next_obs.shape) == 2
            # Angle is index 1
            sin_th = torch.sin(next_obs[:, 1])
            cos_th = torch.cos(next_obs[:, 1])
            not_done = (
                torch.isfinite(next_obs).all(-1)
                * (next_obs[:, 1] <= 0.2)
                * (next_obs[:, 1] >= -0.2)
            )
            done = ~not_done
            return done.unsqueeze(-1)
        return term_fn
    elif "Hopper" in env_id:
        def term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
            assert len(next_obs.shape) == 2
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (
                torch.isfinite(next_obs).all(-1)
                * (height > 0.7)
                * (torch.abs(angle) < 0.2)
                * (torch.abs(next_obs[:, 2:]) < 100).all(-1)
            )
            done = ~not_done
            return done.unsqueeze(-1)
        return term_fn
    elif "Walker2d" in env_id:
        def term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
            assert len(next_obs.shape) == 2
            height = next_obs[:, 0]
            angle = next_obs[:, 1]
            not_done = (
                torch.isfinite(next_obs).all(-1)
                * (height > 0.8)
                * (height < 2.0)
                * (torch.abs(angle) < 1.0)
            )
            done = ~not_done
            return done.unsqueeze(-1)
        return term_fn
    else:
        # Default (e.g. HalfCheetah): no early termination during rollouts
        def term_fn(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
            return torch.zeros(next_obs.shape[0], 1, dtype=torch.bool, device=next_obs.device)
        return term_fn


def run_mbpo_training(
    env_id: str,
    algorithm_type: str,  # 'mle', 'vagram', 'vaml', 'lambda_vaml'
    seed: int,
    total_steps: int = 10000,
    epoch_length: int = 250,
    model_batch_size: int = 256,
    ensemble_size: int = 7,
    num_elites: int = 5,
    eval_freq: int = 250,
    eval_episodes: int = 5,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes controlled MBPO baseline training for MLE, VaGraM, VAML, or Lambda-VAML with exact compute accounting.
    """
    valid_algos = ["mle", "vagram", "vaml", "lambda_vaml"]
    assert algorithm_type in valid_algos, f"Unknown algorithm_type: {algorithm_type}. Supported: {valid_algos}"
    
    # 1. Deterministic seeding
    np.random.seed(seed)
    torch.manual_seed(seed)
    process = psutil.Process(os.getpid())
    start_time = time.time()
    peak_memory_bytes = 0

    # 2. Environments
    train_env = GymnasiumToGymAdapter(env_id, seed=seed)
    eval_env = GymnasiumToGymAdapter(env_id, seed=seed + 10000)
    term_fn = create_termination_fn(env_id)

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]

    # 3. Model construction
    # Target dimension includes reward (obs_dim + 1)
    if algorithm_type == "vagram":
        dynamics_model = vaml_mlp.VAMLMLP(
            in_size=obs_dim + act_dim,
            out_size=obs_dim + 1,
            device=device,
            num_layers=4,
            ensemble_size=ensemble_size,
            hid_size=200,
            use_silu=False,
            propagation_method="fixed_model",
            bound_clipping=True,
            bound_clipping_quantile=0.95,
            use_vaml=False,
            use_scaling=True,
            add_mse=False,
        )
    elif algorithm_type == "vaml":
        dynamics_model = vaml_mlp.VAMLMLP(
            in_size=obs_dim + act_dim,
            out_size=obs_dim + 1,
            device=device,
            num_layers=4,
            ensemble_size=ensemble_size,
            hid_size=200,
            use_silu=False,
            propagation_method="fixed_model",
            bound_clipping=True,
            bound_clipping_quantile=0.95,
            use_vaml=True,
            use_scaling=False,
            add_mse=False,
        )
    elif algorithm_type == "lambda_vaml":
        dynamics_model = vaml_mlp.VAMLMLP(
            in_size=obs_dim + act_dim,
            out_size=obs_dim + 1,
            device=device,
            num_layers=4,
            ensemble_size=ensemble_size,
            hid_size=200,
            use_silu=False,
            propagation_method="fixed_model",
            bound_clipping=True,
            bound_clipping_quantile=0.95,
            use_vaml=True,
            use_scaling=False,
            add_mse=True,
        )
    else:  # 'mle'
        dynamics_model = gaussian_mlp.GaussianMLP(
            in_size=obs_dim + act_dim,
            out_size=obs_dim + 1,
            device=device,
            num_layers=4,
            ensemble_size=ensemble_size,
            hid_size=200,
            use_silu=False,
            propagation_method="fixed_model",
            deterministic=False,
        )

    # 4. SAC Agent setup
    import omegaconf

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

    actor_lr = 3e-4 if "InvertedPendulum" in env_id else 3e-5
    target_entropy = -0.05 if "InvertedPendulum" in env_id else -act_dim

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
        actor_lr=actor_lr,
        actor_betas=[0.9, 0.999],
        actor_update_frequency=1,
        critic_lr=3e-4,
        critic_betas=[0.9, 0.999],
        critic_tau=0.005,
        critic_target_update_frequency=2 if "InvertedPendulum" in env_id else 4,
        batch_size=256,
        learnable_temperature=True,
        target_entropy=target_entropy,
    )
    sac_agent = sac_wrapper.SACAgent(agent)

    if algorithm_type in ["vagram", "vaml", "lambda_vaml"]:
        dynamics_model.set_gradient_buffer(
            (obs_dim,), (act_dim,),
            omegaconf.DictConfig({"overrides": {"num_steps": max(100000, total_steps + 10000)}})
        )
        dynamics_model.set_agent(agent)

    # 5. Buffers
    env_buffer = mbrl.util.ReplayBuffer(capacity=max(100000, total_steps), obs_shape=(obs_dim,), action_shape=(act_dim,))
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

    # 6. Accounting counters
    null_logger = NullLogger()
    total_env_interactions = 0
    total_dyn_updates = 0
    total_sac_updates = 0
    total_imagined_transitions = 0

    evaluation_history = []
    dynamics_loss_history = []

    # Initial random warm-up steps
    warmup_steps = 1000 if total_steps >= 10000 else min(250, total_steps // 2)
    obs = train_env.reset()
    for _ in range(warmup_steps):
        action = train_env.action_space.sample()
        next_obs, reward, done, info = train_env.step(action)
        env_buffer.add(obs, action, next_obs, reward, done)
        total_env_interactions += 1
        obs = next_obs if not done else train_env.reset()

    dynamics_model.set_elite(list(range(num_elites)))

    # Main training loop
    step = warmup_steps
    epoch = 0
    rollout_horizon = 1

    while step < total_steps:
        # Periodic model retraining
        if step % epoch_length == 0 or step == warmup_steps:
            # Train dynamics model
            dataset = env_buffer.get_all()
            if len(dataset) > 0:
                # Prepare training tensors
                in_obs, in_act, out_obs, out_r, _ = dataset.astuple()
                model_in = np.concatenate([in_obs, in_act], axis=-1)
                model_target = np.concatenate([out_obs - in_obs, out_r[:, None]], axis=-1)
                
                # Fit dynamics model
                t_in = torch.tensor(model_in, dtype=torch.float32, device=device)
                t_tgt = torch.tensor(model_target, dtype=torch.float32, device=device)
                
                optimizer = torch.optim.Adam(dynamics_model.parameters(), lr=1e-3, weight_decay=5e-5)
                n_batches = max(1, len(t_in) // model_batch_size)
                
                epoch_loss = 0.0
                for _ in range(5):  # mini-epochs
                    idx = torch.randperm(len(t_in))
                    for b in range(n_batches):
                        b_idx = idx[b * model_batch_size : (b + 1) * model_batch_size]
                        optimizer.zero_grad()
                        if algorithm_type in ["vagram", "vaml", "lambda_vaml"]:
                            loss = dynamics_model._vaml_loss(
                                t_in[b_idx], t_tgt[b_idx], b_idx,
                                next_obs=torch.tensor(out_obs[b_idx.cpu().numpy()], dtype=torch.float32, device=device),
                            ).mean()
                            if getattr(dynamics_model, "add_mse", False):
                                loss = loss + dynamics_model._mse_loss(t_in[b_idx], t_tgt[b_idx]).mean()
                        else:
                            mean, logvar = dynamics_model.forward(t_in[b_idx], use_propagation=False)
                            t_tgt_exp = t_tgt[b_idx].unsqueeze(0).repeat(dynamics_model.num_members, 1, 1)
                            loss = mbrl.util.math.gaussian_nll(mean, logvar, t_tgt_exp, reduce=True)
                        
                        loss.backward()
                        optimizer.step()
                        total_dyn_updates += 1
                        epoch_loss += loss.item()
                
                dynamics_loss_history.append({"step": step, "loss": epoch_loss / (5 * n_batches)})

            # Perform model rollouts into SAC buffer
            raw_batch_size = min(len(env_buffer), 400)
            rollout_batch_size = (raw_batch_size // num_elites) * num_elites
            if rollout_batch_size >= num_elites:
                sample_batch = env_buffer.sample(rollout_batch_size)
                init_obs, *_ = sample_batch.astuple()
                curr_obs = model_env.reset(initial_obs_batch=init_obs, return_as_np=True)
                
                for _ in range(rollout_horizon):
                    act = sac_agent.act(curr_obs, sample=True, batched=True)
                    pred_next, pred_rew, pred_d, _ = model_env.step(act, sample=True)
                    sac_buffer.add_batch(
                        curr_obs, act, pred_rew, pred_next, pred_d, pred_d
                    )
                    total_imagined_transitions += len(curr_obs)
                    curr_obs = pred_next

        # Environment step
        action = sac_agent.act(obs, sample=True)
        next_obs, reward, done, info = train_env.step(action)
        env_buffer.add(obs, action, next_obs, reward, done)
        total_env_interactions += 1
        step += 1

        # SAC updates
        if len(sac_buffer) >= 256:
            for _ in range(20 if "InvertedPendulum" in env_id else 5):
                agent.update(sac_buffer, null_logger, step)
                total_sac_updates += 1

        obs = next_obs if not done else train_env.reset()

        # Periodic evaluation
        if step % eval_freq == 0 or step == total_steps:
            epoch += 1
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
            peak_memory_bytes = max(peak_memory_bytes, curr_mem)

            rec = {
                "step": step,
                "epoch": epoch,
                "algorithm": algorithm_type,
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
                print(f"[{algorithm_type.upper()}] Step {step}/{total_steps} | Return: {mean_ret:.2f} ± {std_ret:.2f} | DynUpdates: {total_dyn_updates} | SACUpdates: {total_sac_updates} | Wall: {rec['wall_clock_sec']:.1f}s")

    train_env.close()
    eval_env.close()

    total_wall_clock = time.time() - start_time

    return {
        "algorithm": algorithm_type,
        "env_id": env_id,
        "seed": seed,
        "final_return_mean": evaluation_history[-1]["eval_mean_return"] if evaluation_history else 0.0,
        "final_return_std": evaluation_history[-1]["eval_std_return"] if evaluation_history else 0.0,
        "evaluation_history": evaluation_history,
        "dynamics_loss_history": dynamics_loss_history,
        "total_env_interactions": total_env_interactions,
        "total_dyn_updates": total_dyn_updates,
        "total_sac_updates": total_sac_updates,
        "total_imagined_transitions": total_imagined_transitions,
        "total_wall_clock_sec": total_wall_clock,
        "peak_memory_mb": peak_memory_bytes,
    }
