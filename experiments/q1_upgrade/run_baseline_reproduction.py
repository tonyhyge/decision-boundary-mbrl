"""
Step 2B: Baseline Reproduction Gate on Hopper (Hopper-v4 / Hopper-v5).
Evaluates MLE-MBPO vs. VaGraM-MBPO across multiple seeds under matched hyperparameters.
Logs all resource accounting metrics (env steps, dyn updates, sac updates, imagined transitions, wall-clock, memory).
"""
import os
import sys
import json
import argparse
import time
import numpy as np

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.baselines.adapters.vagram_mbpo import run_mbpo_training


def parse_args():
    parser = argparse.ArgumentParser(description="Run Step 2B Baseline Reproduction on Hopper.")
    parser.add_argument("--env-id", type=str, default="Hopper-v4", help="Environment ID (Hopper-v4 or Hopper-v5)")
    parser.add_argument("--total-steps", type=int, default=25000, help="Total environment interaction steps per seed")
    parser.add_argument("--epoch-length", type=int, default=1000, help="Steps per training epoch")
    parser.add_argument("--eval-freq", type=int, default=1000, help="Evaluation frequency in environment steps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds to evaluate")
    parser.add_argument("--algorithms", type=str, nargs="+", default=["mle", "vagram"], help="Algorithms to run (mle, vagram)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 75)
    print("STEP 2B: BASELINE REPRODUCTION BENCHMARK (MLE vs VaGraM on Hopper)")
    print("=" * 75)
    print(f"Environment:       {args.env_id}")
    print(f"Total Steps/Seed:  {args.total_steps}")
    print(f"Seeds:             {args.seeds}")
    print(f"Algorithms:        {args.algorithms}")
    print("=" * 75)

    out_dir = os.path.join(_ROOT, "experiments", "q1_upgrade", "results")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    for algo in args.algorithms:
        all_results[algo] = []
        for seed in args.seeds:
            print(f"\n---> Running {algo.upper()} on {args.env_id} | Seed {seed} (Budget: {args.total_steps} steps) <---")
            t0 = time.time()
            res = run_mbpo_training(
                env_id=args.env_id,
                algorithm_type=algo,
                seed=seed,
                total_steps=args.total_steps,
                epoch_length=args.epoch_length,
                eval_freq=args.eval_freq,
                eval_episodes=args.eval_episodes,
                verbose=True,
            )
            elapsed = time.time() - t0
            print(f"[COMPLETED] {algo.upper()} Seed {seed} in {elapsed:.1f}s | Final Return: {res['final_return_mean']:.2f} ± {res['final_return_std']:.2f}")
            all_results[algo].append(res)

            # Save intermediate per-seed result
            seed_file = os.path.join(out_dir, f"hopper_{algo}_seed{seed}.json")
            with open(seed_file, "w") as f:
                json.dump(res, f, indent=2)

    # Save summary results
    summary_file = os.path.join(out_dir, "step2b_hopper_reproduction_summary.json")
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nAll reproduction results saved to: {summary_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
