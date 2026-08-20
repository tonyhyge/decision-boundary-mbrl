"""
Evaluation and statistical reporting script for Value-Aware Baselines Benchmark (Feedback #6).
Aggregates all 4 algorithms (MLE, VaGraM, VAML, Lambda-VAML) across N=5 seeds on Hopper-v4,
computes non-parametric bootstrap CIs, generates comparison figures, and produces markdown reporting artifacts.
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


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


def load_baseline_data(env_id="hopper_v4"):
    """Loads all per-seed JSON files for all 4 algorithms."""
    results_dir = os.path.join(_ROOT, "results", "baselines")
    legacy_dir = os.path.join(_ROOT, "experiments", "q1_upgrade", "results")
    
    algorithms = ["mle", "vagram", "vaml", "lambda_vaml"]
    data = {algo: {} for algo in algorithms}
    
    for algo in algorithms:
        for seed in [42, 43, 44, 45, 46]:
            fname = f"{env_id}_{algo}_seed{seed}.json"
            fpath = os.path.join(results_dir, fname)
            if not os.path.exists(fpath):
                fpath = os.path.join(legacy_dir, f"hopper_{algo}_seed{seed}.json")
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    data[algo][seed] = json.load(f)
    return data


def plot_learning_curves(data, env_name="Hopper-v4", out_path=None):
    """Generates comparison plot of learning curves across algorithms."""
    plt.figure(figsize=(9, 5.5))
    
    color_map = {
        "mle": "#1f77b4",         # Blue
        "vagram": "#ff7f0e",      # Orange
        "vaml": "#2ca02c",        # Green
        "lambda_vaml": "#d62728", # Red
    }
    label_map = {
        "mle": "Standard MLE (Gaussian NLL)",
        "vagram": "VaGraM (Value-Gradient MSE)",
        "vaml": "Taylor-VAML (Inner Product)",
        "lambda_vaml": r"$\lambda$-VAML (VAML + MSE)",
    }
    
    for algo, runs in data.items():
        if not runs:
            continue
        # Extract steps and evaluation returns
        steps_list = []
        returns_per_seed = []
        
        for seed, res in runs.items():
            hist = res.get("evaluation_history", [])
            steps = [h["step"] for h in hist]
            returns = [h["eval_mean_return"] for h in hist]
            if not steps_list:
                steps_list = steps
            returns_per_seed.append(returns)
            
        if not returns_per_seed:
            continue
            
        returns_arr = np.array(returns_per_seed) # shape (N, T)
        mean_curve = np.mean(returns_arr, axis=0)
        std_curve = np.std(returns_arr, axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean_curve)
        
        plt.plot(steps_list, mean_curve, label=f"{label_map[algo]} (N={len(runs)})", 
                 color=color_map[algo], linewidth=2.2)
        plt.fill_between(steps_list, mean_curve - std_curve, mean_curve + std_curve, 
                         color=color_map[algo], alpha=0.18)
        
    plt.title(f"Value-Aware World-Model Baselines Comparison on {env_name} (Matched Compute)", fontsize=13, fontweight="bold")
    plt.xlabel("Environment Steps", fontsize=11)
    plt.ylabel("Evaluation Return (Mean ± 1 SD)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"Saved figure to: {out_path}")
    plt.close()


def main():
    print("=" * 80)
    print("STATISTICAL AUDIT: VALUE-AWARE BASELINES COMPARISON (FEEDBACK #6)")
    print("=" * 80)
    
    env_id = "hopper_v4"
    data = load_baseline_data(env_id)
    
    # 1. Per-seed table
    per_seed_rows = []
    for algo, runs in data.items():
        for seed, r in runs.items():
            per_seed_rows.append({
                "Algorithm": algo.upper(),
                "Seed": seed,
                "Final Return": f"{r['final_return_mean']:.2f} ± {r['final_return_std']:.2f}",
                "Dyn Updates": r["total_dyn_updates"],
                "SAC Updates": r["total_sac_updates"],
                "Imagined Steps": r["total_imagined_transitions"],
                "Wall Clock (s)": f"{r['total_wall_clock_sec']:.1f}",
                "Peak RAM (MB)": f"{r['peak_memory_mb']:.1f}",
            })
            
    if per_seed_rows:
        df_per_seed = pd.DataFrame(per_seed_rows)
        print("\n### Individual Per-Seed Breakdown:")
        print(df_per_seed.to_markdown(index=False))
        
    # 2. Aggregated summary table
    summary_rows = []
    for algo, runs in data.items():
        if not runs:
            continue
        final_returns = [r["final_return_mean"] for r in runs.values()]
        mean_ret, (ci_low, ci_high) = bootstrap_ci(final_returns)
        std_ret = float(np.std(final_returns, ddof=1)) if len(final_returns) > 1 else 0.0
        mean_wall = float(np.mean([r["total_wall_clock_sec"] for r in runs.values()]))
        mean_dyn = int(np.mean([r["total_dyn_updates"] for r in runs.values()]))
        mean_sac = int(np.mean([r["total_sac_updates"] for r in runs.values()]))
        mean_ram = float(np.mean([r["peak_memory_mb"] for r in runs.values()]))
        
        summary_rows.append({
            "Algorithm": algo.upper(),
            "Seeds (N)": len(runs),
            "Mean Return": f"{mean_ret:.2f}",
            "Sample SD": f"{std_ret:.2f}",
            "95% Bootstrap CI": f"[{ci_low:.2f}, {ci_high:.2f}]",
            "Mean Dyn Updates": mean_dyn,
            "Mean SAC Updates": mean_sac,
            "Mean Wall Clock (s)": f"{mean_wall:.1f}",
            "Mean RAM (MB)": f"{mean_ram:.1f}",
        })
        
    df_summary = pd.DataFrame(summary_rows)
    print("\n### Aggregated Statistical Summary:")
    print(df_summary.to_markdown(index=False))
    
    # 3. Paired Seed Delta Analysis vs MLE
    paired_rows = []
    if "mle" in data and len(data["mle"]) == 5:
        mle_returns = {s: data["mle"][s]["final_return_mean"] for s in [42, 43, 44, 45, 46]}
        for algo in ["vagram", "vaml", "lambda_vaml"]:
            if algo in data and len(data[algo]) == 5:
                deltas = [data[algo][s]["final_return_mean"] - mle_returns[s] for s in [42, 43, 44, 45, 46]]
                wins = sum(1 for d in deltas if d > 0)
                paired_rows.append({
                    "Baseline": algo.upper(),
                    "Seed 42 Δ": f"{deltas[0]:+.2f}",
                    "Seed 43 Δ": f"{deltas[1]:+.2f}",
                    "Seed 44 Δ": f"{deltas[2]:+.2f}",
                    "Seed 45 Δ": f"{deltas[3]:+.2f}",
                    "Seed 46 Δ": f"{deltas[4]:+.2f}",
                    "Wins vs MLE": f"{wins} / 5",
                    "Median Δ": f"{np.median(deltas):+.2f}",
                })
    if paired_rows:
        df_paired = pd.DataFrame(paired_rows)
        print("\n### Paired-Seed Delta Analysis (Baseline − MLE):")
        print(df_paired.to_markdown(index=False))

    # 4. Generate visualization
    fig_path = os.path.join(_ROOT, "figures", "value_aware_baselines_comparison.pdf")
    plot_learning_curves(data, env_name="Hopper-v4", out_path=fig_path)


if __name__ == "__main__":
    sys.exit(main())
