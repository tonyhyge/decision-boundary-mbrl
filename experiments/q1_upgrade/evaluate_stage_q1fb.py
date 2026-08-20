"""
Evaluation and statistical reporting script for Stage Q1-FB: Full-Budget Hopper Confirmation.
Computes primary paired terminal return deltas (Δ_i) and secondary horizon-interaction estimands (H_i).
Runs blind from raw JSON results files.
"""
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def paired_sign_flip_test(deltas, num_permutations=10000):
    """Exact/Monte Carlo paired sign-flip permutation test against H0: median/mean delta = 0."""
    deltas = np.array(deltas, dtype=np.float64)
    n = len(deltas)
    if n == 0:
        return 1.0
    obs_stat = np.mean(deltas)
    if n <= 12:
        # Exact enumeration of all 2^n sign flips
        all_signs = np.array(np.meshgrid(*[[1, -1] for _ in range(n)])).T.reshape(-1, n)
        perm_stats = np.mean(all_signs * deltas, axis=1)
    else:
        # Monte Carlo sign flips
        random_signs = np.random.choice([1, -1], size=(num_permutations, n))
        perm_stats = np.mean(random_signs * deltas, axis=1)
    p_val = float(np.mean(np.abs(perm_stats) >= np.abs(obs_stat)))
    return p_val


def load_q1fb_results(results_dir):
    """Loads all per-seed JSON files for MLE and VaGraM."""
    data = {"mle": {}, "vagram": {}}
    for algo in ["mle", "vagram"]:
        pattern = os.path.join(results_dir, f"q1fb_hopper_{algo}_seed*.json")
        for fpath in sorted(glob.glob(pattern)):
            with open(fpath, "r") as f:
                d = json.load(f)
                seed = d["seed"]
                data[algo][seed] = d
    return data


def main():
    results_dir = os.path.join(_ROOT, "results", "stage_q1fb")
    data = load_q1fb_results(results_dir)

    print("=" * 80)
    print("STAGE Q1-FB: FULL-BUDGET HOPPER CONFIRMATION EVALUATION (125k STEPS)")
    print("=" * 80)

    mle_seeds = set(data["mle"].keys())
    vagram_seeds = set(data["vagram"].keys())
    paired_seeds = sorted(list(mle_seeds.intersection(vagram_seeds)))

    print(f"Completed MLE seeds:    {sorted(list(mle_seeds))}")
    print(f"Completed VaGraM seeds: {sorted(list(vagram_seeds))}")
    print(f"Common Paired seeds:    {paired_seeds} (N={len(paired_seeds)})")
    print("=" * 80)

    # 1. Individual Per-Seed Table
    per_seed_rows = []
    for algo in ["mle", "vagram"]:
        for seed, r in data[algo].items():
            per_seed_rows.append({
                "Algorithm": algo.upper(),
                "Seed": seed,
                "Terminal Return (Final 3)": f"{r['terminal_return_mean']:.2f}",
                "Final Return (Step 125k)": f"{r['final_return_mean']:.2f} ± {r['final_return_std']:.2f}",
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

    # 2. Primary Confirmatory Paired Analysis: Δ_i^terminal
    paired_rows = []
    deltas = []
    horizon_interactions = []
    h_rows = []

    for seed in paired_seeds:
        mle_ret = data["mle"][seed]["terminal_return_mean"]
        vag_ret = data["vagram"][seed]["terminal_return_mean"]
        delta = vag_ret - mle_ret
        deltas.append(delta)

        # Extract 10k step and terminal step returns for horizon interaction
        mle_hist = {h["step"]: h["eval_mean_return"] for h in data["mle"][seed].get("evaluation_history", [])}
        vag_hist = {h["step"]: h["eval_mean_return"] for h in data["vagram"][seed].get("evaluation_history", [])}

        mle_10k = mle_hist.get(10000, None)
        vag_10k = vag_hist.get(10000, None)
        mle_final = data["mle"][seed]["final_return_mean"]
        vag_final = data["vagram"][seed]["final_return_mean"]

        if mle_10k is not None and vag_10k is not None:
            delta_10k = vag_10k - mle_10k
            delta_final = vag_final - mle_final
            H_i = delta_final - delta_10k
            horizon_interactions.append(H_i)
            h_rows.append({
                "Seed": seed,
                "Δ(10k)": f"{delta_10k:+.2f}",
                "Δ(Terminal)": f"{delta_final:+.2f}",
                "H_i = Δ(Terminal) - Δ(10k)": f"{H_i:+.2f}",
                "Horizon Growth?": "YES" if H_i > 0 else "NO",
            })

        paired_rows.append({
            "Seed": seed,
            "MLE Terminal Return": f"{mle_ret:.2f}",
            "VaGraM Terminal Return": f"{vag_ret:.2f}",
            "Paired Δ (VaGraM - MLE)": f"{delta:+.2f}",
            "Win?": "VaGraM" if delta > 0 else "MLE",
        })

    if paired_rows:
        df_paired = pd.DataFrame(paired_rows)
        print("\n### Primary Confirmatory Paired Analysis (N={}):".format(len(paired_seeds)))
        print(df_paired.to_markdown(index=False))

        mean_delta, (ci_low, ci_high) = bootstrap_ci(deltas)
        median_delta = float(np.median(deltas))
        p_val_delta = paired_sign_flip_test(deltas)
        wins = sum(1 for d in deltas if d > 0)
        win_rate = (wins / len(deltas)) * 100.0

        print(f"\n--- Primary Terminal Return Estimand (N={len(paired_seeds)}) ---")
        print(f"Mean Paired Δ:              {mean_delta:+.2f}")
        print(f"95% Paired Bootstrap CI:    [{ci_low:+.2f}, {ci_high:+.2f}]")
        print(f"Median Paired Δ:            {median_delta:+.2f}")
        print(f"Paired Permutation p-value: {p_val_delta:.4f}")
        print(f"VaGraM Wins vs MLE:         {wins} / {len(deltas)} ({win_rate:.1f}%)")

    # 3. Secondary Mechanistic Horizon Interaction Analysis: H_i
    if h_rows:
        df_h = pd.DataFrame(h_rows)
        print("\n### Secondary Horizon-Interaction Analysis (H_i = Δ_final - Δ_10k):")
        print(df_h.to_markdown(index=False))

        mean_h, (h_ci_low, h_ci_high) = bootstrap_ci(horizon_interactions)
        median_h = float(np.median(horizon_interactions))
        p_val_h = paired_sign_flip_test(horizon_interactions)
        h_wins = sum(1 for h in horizon_interactions if h > 0)

        print(f"\n--- Secondary Horizon Interaction Summary (N={len(horizon_interactions)}) ---")
        print(f"Mean Horizon Interaction H: {mean_h:+.2f}")
        print(f"95% Bootstrap CI of H:      [{h_ci_low:+.2f}, {h_ci_high:+.2f}]")
        print(f"Median Horizon Interaction: {median_h:+.2f}")
        print(f"Permutation p-value:        {p_val_h:.4f}")
        print(f"Horizon Positive Fraction:  {h_wins} / {len(horizon_interactions)} ({(h_wins/len(horizon_interactions))*100:.1f}%)")

    # 4. Learning Curves Plot
    fig_path = os.path.join(_ROOT, "figures", "stage_q1fb_full_budget_curves.pdf")
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.figure(figsize=(9, 5.5))

    color_map = {"mle": "#1f77b4", "vagram": "#ff7f0e"}
    label_map = {"mle": "Standard MLE (Gaussian NLL)", "vagram": "VaGraM (Value-Gradient MSE)"}

    for algo in ["mle", "vagram"]:
        if not data[algo]:
            continue
        curves = []
        steps = None
        for seed, r in data[algo].items():
            hist = r.get("evaluation_history", [])
            s_list = [h["step"] for h in hist]
            r_list = [h["eval_mean_return"] for h in hist]
            if steps is None:
                steps = s_list
            curves.append(r_list)
        if curves:
            arr = np.array(curves)
            mean_c = np.mean(arr, axis=0)
            std_c = np.std(arr, axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean_c)
            plt.plot(steps, mean_c, label=f"{label_map[algo]} (N={len(curves)})", color=color_map[algo], linewidth=2.2)
            plt.fill_between(steps, mean_c - std_c, mean_c + std_c, color=color_map[algo], alpha=0.18)

    plt.title("Stage Q1-FB: Full-Budget MBPO on Hopper-v4 (10 Eval Episodes / Checkpoint)", fontsize=12, fontweight="bold")
    plt.xlabel("Environment Interaction Steps", fontsize=11)
    plt.ylabel("Evaluation Return (Mean ± 1 SD)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"\nSaved learning curve figure to: {fig_path}")

    # 5. Generate Markdown Artifact
    report_path = os.path.join(_ROOT, "docs", "STAGE_Q1FB_FULL_BUDGET_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Scientific Report: Stage Q1-FB Full-Budget Hopper Representativeness Confirmation\n\n")
        f.write("**Date:** 2026-08-21  \n")
        f.write("**Target Milestone:** Q1 Journal Upgrade — Stage Q1-FB (Full-Budget Hopper Confirmation)  \n")
        f.write("**Core Scientific Question:** Does the conclusion that VaGraM lacks a robust advantage over MLE hold when MBPO is trained in a canonical locomotion regime up to 125,000 environment steps?  \n")
        f.write("**Accounting Controls:** Matched 7-model ensemble (elite 5, 4x200 ReLU), 20 SAC updates/step, retrain every 250 steps, 10 deterministic eval episodes per checkpoint.  \n\n")
        if paired_rows:
            f.write("## 1. Primary Confirmatory Paired Analysis (Terminal Return)\n\n")
            f.write(pd.DataFrame(paired_rows).to_markdown(index=False))
            f.write(f"\n\n- **Mean Paired $\\Delta$:** ${mean_delta:+.2f}$ (95% BCI: $[{ci_low:+.2f}, {ci_high:+.2f}]$)\n")
            f.write(f"- **Median Paired $\\Delta$:** ${median_delta:+.2f}$\n")
            f.write(f"- **Paired Permutation $p$-value:** ${p_val_delta:.4f}$\n")
            f.write(f"- **Win Rate:** {wins} / {len(deltas)} ({win_rate:.1f}%)\n\n")
        if h_rows:
            f.write("## 2. Secondary Mechanistic Horizon-Interaction Analysis\n\n")
            f.write(pd.DataFrame(h_rows).to_markdown(index=False))
            f.write(f"\n\n- **Mean Horizon Interaction $\\bar{{H}}$:** ${mean_h:+.2f}$ (95% BCI: $[{h_ci_low:+.2f}, {h_ci_high:+.2f}]$)\n")
            f.write(f"- **Median Horizon Interaction:** ${median_h:+.2f}$\n")
            f.write(f"- **Permutation $p$-value:** ${p_val_h:.4f}$\n")
            f.write(f"- **Horizon Positive Fraction:** {h_wins} / {len(horizon_interactions)} ({(h_wins/len(horizon_interactions))*100:.1f}%)\n\n")
        if per_seed_rows:
            f.write("## 3. Individual Per-Seed Breakdown\n\n")
            f.write(pd.DataFrame(per_seed_rows).to_markdown(index=False))
            f.write("\n\n")
        f.write("## 4. Epistemic Demarcation & Interpretation Rules\n\n")
        f.write("1. **Primary Estimand Priority:** Primary conclusion is determined strictly by the terminal return paired difference $\\Delta_i^{\\text{terminal}}$. Horizon interaction $H_i$ serves as a secondary mechanistic diagnostic.\n")
        f.write("2. **No Claim Changes from Pilot:** Pilot Seed 101 represents a strong positive signal but does not close the confirmatory stage. Final conclusions require the full $N=10$ independent paired block ($[142, \\dots, 151]$).\n")

    print(f"Saved report artifact to: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
