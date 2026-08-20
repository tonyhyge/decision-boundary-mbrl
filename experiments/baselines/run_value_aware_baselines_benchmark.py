"""
Benchmark runner for Value-Aware Baselines on MuJoCo (Hopper-v4, Walker2d-v4, HalfCheetah-v4).
Evaluates 4 matched objective formulations:
1. Standard MLE (Gaussian NLL)
2. VaGraM (Value-Gradient Weighted Coordinate MSE)
3. VAML (Taylor Value-Aware Model Learning)
4. Lambda-VAML / Calibrated VAML (VAML + MSE regularizer)

Logs closed-loop evaluation return, dynamics loss, and full resource accounting.
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.baselines.adapters.vagram_mbpo import run_mbpo_training


def parse_args():
    parser = argparse.ArgumentParser(description="Run Value-Aware Baselines Benchmark Suite.")
    parser.add_argument("--envs", type=str, nargs="+", default=["Hopper-v4"], 
                        help="Environments to evaluate (e.g. Hopper-v4, Walker2d-v4, HalfCheetah-v4)")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["mle", "vagram", "vaml", "lambda_vaml"],
                        help="Algorithms to benchmark (mle, vagram, vaml, lambda_vaml)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="Random seeds to evaluate")
    parser.add_argument("--total-steps", type=int, default=10000,
                        help="Total environment interaction steps per seed")
    parser.add_argument("--epoch-length", type=int, default=1000,
                        help="Steps per model retraining epoch")
    parser.add_argument("--eval-freq", type=int, default=1000,
                        help="Evaluation frequency in environment steps")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Compute device (cpu or cuda)")
    parser.add_argument("--out-dir", type=str, default=os.path.join(_ROOT, "results", "baselines"),
                        help="Output directory for benchmark results")
    return parser.parse_args()


def bootstrap_ci(data, num_bootstraps=5000, alpha=0.05):
    """Computes non-parametric bootstrap mean and 95% confidence interval."""
    data = np.array(data, dtype=np.float64)
    if len(data) == 0:
        return 0.0, (0.0, 0.0)
    if len(data) == 1:
        return float(data[0]), (float(data[0]), float(data[0]))
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(num_bootstraps)]
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(data)), (float(low), float(high))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("VALUE-AWARE BASELINES COMPARISON BENCHMARK (FEEDBACK #6)")
    print("=" * 80)
    print(f"Environments:     {args.envs}")
    print(f"Algorithms:       {args.algorithms}")
    print(f"Seeds:            {args.seeds} (N={len(args.seeds)})")
    print(f"Budget per run:   {args.total_steps} steps (Epoch length: {args.epoch_length})")
    print(f"Output Directory: {args.out_dir}")
    print("=" * 80)

    master_results = {}

    for env_id in args.envs:
        master_results[env_id] = {}
        for algo in args.algorithms:
            master_results[env_id][algo] = []
            for seed in args.seeds:
                seed_fname = f"{env_id.lower().replace('-', '_')}_{algo}_seed{seed}.json"
                seed_fpath = os.path.join(args.out_dir, seed_fname)

                # Check if result already exists in q1_upgrade/results or out_dir
                legacy_fpath = os.path.join(_ROOT, "experiments", "q1_upgrade", "results", seed_fname)
                if os.path.exists(seed_fpath):
                    print(f"\n[CACHE HIT] Loading existing result for {algo.upper()} on {env_id} (Seed {seed})")
                    with open(seed_fpath, "r") as f:
                        res = json.load(f)
                    master_results[env_id][algo].append(res)
                    continue
                elif os.path.exists(legacy_fpath):
                    print(f"\n[LEGACY CACHE HIT] Loading existing result for {algo.upper()} on {env_id} (Seed {seed})")
                    with open(legacy_fpath, "r") as f:
                        res = json.load(f)
                    with open(seed_fpath, "w") as f:
                        json.dump(res, f, indent=2)
                    master_results[env_id][algo].append(res)
                    continue

                print(f"\n---> Running {algo.upper()} on {env_id} | Seed {seed} (Budget: {args.total_steps} steps) <---")
                t0 = time.time()
                res = run_mbpo_training(
                    env_id=env_id,
                    algorithm_type=algo,
                    seed=seed,
                    total_steps=args.total_steps,
                    epoch_length=args.epoch_length,
                    eval_freq=args.eval_freq,
                    eval_episodes=args.eval_episodes,
                    device=args.device,
                    verbose=True,
                )
                elapsed = time.time() - t0
                print(f"[COMPLETED] {algo.upper()} on {env_id} Seed {seed} in {elapsed:.1f}s | Final Return: {res['final_return_mean']:.2f} ± {res['final_return_std']:.2f}")

                with open(seed_fpath, "w") as f:
                    json.dump(res, f, indent=2)

                master_results[env_id][algo].append(res)

    # Compile and generate summary tables
    summary_path = os.path.join(args.out_dir, "value_aware_baselines_summary.json")
    with open(summary_path, "w") as f:
        json.dump(master_results, f, indent=2)

    print("\n" + "=" * 80)
    print("STATISTICAL AGGREGATION & SUMMARY TABLE")
    print("=" * 80)

    summary_rows = []
    for env_id in args.envs:
        for algo in args.algorithms:
            runs = master_results[env_id][algo]
            if not runs:
                continue
            final_returns = [r["final_return_mean"] for r in runs]
            mean_ret, (ci_low, ci_high) = bootstrap_ci(final_returns)
            std_ret = float(np.std(final_returns, ddof=1)) if len(final_returns) > 1 else 0.0
            mean_wall = float(np.mean([r["total_wall_clock_sec"] for r in runs]))
            mean_dyn_upd = int(np.mean([r["total_dyn_updates"] for r in runs]))
            mean_sac_upd = int(np.mean([r["total_sac_updates"] for r in runs]))
            mean_peak_ram = float(np.mean([r["peak_memory_mb"] for r in runs]))

            summary_rows.append({
                "Environment": env_id,
                "Algorithm": algo.upper(),
                "Seeds (N)": len(runs),
                "Mean Return": f"{mean_ret:.2f}",
                "Sample SD": f"{std_ret:.2f}",
                "95% Bootstrap CI": f"[{ci_low:.2f}, {ci_high:.2f}]",
                "Mean Dyn Updates": mean_dyn_upd,
                "Mean SAC Updates": mean_sac_upd,
                "Mean Wall (s)": f"{mean_wall:.1f}",
                "Mean RAM (MB)": f"{mean_peak_ram:.1f}",
            })

    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_markdown(index=False))

    csv_path = os.path.join(args.out_dir, "value_aware_baselines_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to: {csv_path}")
    print(f"Summary JSON saved to:  {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
