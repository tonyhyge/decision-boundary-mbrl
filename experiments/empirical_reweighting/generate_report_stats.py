"""
Detailed statistical synthesis and report generation from portability audit dataset.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

def generate_report_data(csv_path: str = "results/portability_full_audit.csv", output_dir: str = "results", figure_dir: str = "figures"):
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    # Let's inspect lambda=1.0 and best-performing lambda
    print("=== OVERALL METRICS PER HOST AND CONDITION AT LAMBDA = 1.0 ===")
    df_l1 = df[(df["lambda"] == 1.0) | (df["condition"] == "uniform")].copy()
    
    # Map uniform into lambda=1.0 for pivot
    df_uniform = df[df["condition"] == "uniform"].copy()
    df_uniform["lambda"] = 1.0

    df_combined = pd.concat([df[df["lambda"] == 1.0], df_uniform[df_uniform["condition"] == "uniform"]]).drop_duplicates(subset=["seed", "host", "condition"])

    report_stats = {}

    for host in ["Host_A_Deterministic", "Host_B_Ensemble"]:
        df_h = df_combined[df_combined["host"] == host]
        piv_j = df_h.pivot(index="seed", columns="condition", values="j_learned")
        piv_regret = df_h.pivot(index="seed", columns="condition", values="regret")
        piv_agree = df_h.pivot(index="seed", columns="condition", values="action_agreement")
        piv_rev = df_h.pivot(index="seed", columns="condition", values="reversal_rate")
        piv_mse = df_h.pivot(index="seed", columns="condition", values="transition_mse")
        piv_time = df_h.pivot(index="seed", columns="condition", values="wall_clock_sec")

        n_seeds = len(piv_j)

        # Contrasts
        # 1. Delta J = Estimated Crossing - Uniform
        d_j = piv_j["estimated_crossing"] - piv_j["uniform"]
        mean_dj = float(d_j.mean())
        std_dj = float(d_j.std(ddof=1))
        d_eff_dj = float(mean_dj / (std_dj + 1e-12))
        t_dj, p_dj = stats.ttest_1samp(d_j, 0.0)
        w_dj, wp_dj = stats.wilcoxon(d_j, zero_method="wilcox")
        boot_dj = [float(np.mean(rng.choice(d_j, size=n_seeds, replace=True))) for _ in range(2000)]
        bci_dj = [float(np.percentile(boot_dj, 2.5)), float(np.percentile(boot_dj, 97.5))]

        # 2. Delta J_semantic = Estimated Crossing - Shuffled Crossing
        d_sem = piv_j["estimated_crossing"] - piv_j["shuffled_crossing"]
        mean_dsem = float(d_sem.mean())
        std_dsem = float(d_sem.std(ddof=1))
        d_eff_dsem = float(mean_dsem / (std_dsem + 1e-12))
        t_dsem, p_dsem = stats.ttest_1samp(d_sem, 0.0)
        w_dsem, wp_dsem = stats.wilcoxon(d_sem, zero_method="wilcox")
        boot_dsem = [float(np.mean(rng.choice(d_sem, size=n_seeds, replace=True))) for _ in range(2000)]
        bci_dsem = [float(np.percentile(boot_dsem, 2.5)), float(np.percentile(boot_dsem, 97.5))]

        # 3. Oracle - Uniform
        d_orc = piv_j["oracle_crossing"] - piv_j["uniform"]
        mean_dorc = float(d_orc.mean())
        std_dorc = float(d_orc.std(ddof=1))
        d_eff_dorc = float(mean_dorc / (std_dorc + 1e-12))
        t_dorc, p_dorc = stats.ttest_1samp(d_orc, 0.0)
        w_dorc, wp_dorc = stats.wilcoxon(d_orc, zero_method="wilcox")
        boot_dorc = [float(np.mean(rng.choice(d_orc, size=n_seeds, replace=True))) for _ in range(2000)]
        bci_dorc = [float(np.percentile(boot_dorc, 2.5)), float(np.percentile(boot_dorc, 97.5))]

        # 4. Prediction Error - Uniform
        d_pe = piv_j["prediction_error"] - piv_j["uniform"]
        mean_dpe = float(d_pe.mean())
        std_dpe = float(d_pe.std(ddof=1))
        d_eff_dpe = float(mean_dpe / (std_dpe + 1e-12))
        t_dpe, p_dpe = stats.ttest_1samp(d_pe, 0.0)
        w_dpe, wp_dpe = stats.wilcoxon(d_pe, zero_method="wilcox")
        boot_dpe = [float(np.mean(rng.choice(d_pe, size=n_seeds, replace=True))) for _ in range(2000)]
        bci_dpe = [float(np.percentile(boot_dpe, 2.5)), float(np.percentile(boot_dpe, 97.5))]

        cond_summary = {}
        for c in ["uniform", "prediction_error", "estimated_crossing", "shuffled_crossing", "oracle_crossing"]:
            cond_summary[c] = {
                "return_mean": float(piv_j[c].mean()),
                "return_std": float(piv_j[c].std(ddof=1)),
                "regret_mean": float(piv_regret[c].mean()),
                "regret_std": float(piv_regret[c].std(ddof=1)),
                "agreement_mean": float(piv_agree[c].mean()),
                "agreement_std": float(piv_agree[c].std(ddof=1)),
                "reversal_mean": float(piv_rev[c].mean()),
                "reversal_std": float(piv_rev[c].std(ddof=1)),
                "mse_mean": float(piv_mse[c].mean()),
                "mse_std": float(piv_mse[c].std(ddof=1)),
                "time_sec_mean": float(piv_time[c].mean()),
            }

        report_stats[host] = {
            "n_seeds": n_seeds,
            "conditions": cond_summary,
            "delta_j": {
                "mean": mean_dj,
                "std": std_dj,
                "cohen_d": d_eff_dj,
                "t_stat": float(t_dj),
                "p_val": float(p_dj),
                "wilcoxon_stat": float(w_dj),
                "wilcoxon_p": float(wp_dj),
                "bci_95": bci_dj,
            },
            "delta_semantic": {
                "mean": mean_dsem,
                "std": std_dsem,
                "cohen_d": d_eff_dsem,
                "t_stat": float(t_dsem),
                "p_val": float(p_dsem),
                "wilcoxon_stat": float(w_dsem),
                "wilcoxon_p": float(wp_dsem),
                "bci_95": bci_dsem,
            },
            "oracle_vs_uniform": {
                "mean": mean_dorc,
                "std": std_dorc,
                "cohen_d": d_eff_dorc,
                "t_stat": float(t_dorc),
                "p_val": float(p_dorc),
                "wilcoxon_stat": float(w_dorc),
                "wilcoxon_p": float(wp_dorc),
                "bci_95": bci_dorc,
            },
            "prediction_error_vs_uniform": {
                "mean": mean_dpe,
                "std": std_dpe,
                "cohen_d": d_eff_dpe,
                "t_stat": float(t_dpe),
                "p_val": float(p_dpe),
                "wilcoxon_stat": float(w_dpe),
                "wilcoxon_p": float(wp_dpe),
                "bci_95": bci_dpe,
            },
        }

    # Generate Figures
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cond_order = ["uniform", "prediction_error", "shuffled_crossing", "estimated_crossing", "oracle_crossing"]
    cond_labels = ["Uniform\nBaseline", "Prediction\nError", "Shuffled\nControl", "Estimated\nCrossing", "Oracle\nCrossing"]
    colors = ["#7f7f7f", "#ff7f0e", "#d62728", "#2ca02c", "#1f77b4"]

    for idx, (host, ax, title) in enumerate([
        ("Host_A_Deterministic", axes[0], "(a) Host A: Deterministic Dynamics"),
        ("Host_B_Ensemble", axes[1], "(b) Host B: Probabilistic Ensemble Dynamics"),
    ]):
        means = [report_stats[host]["conditions"][c]["return_mean"] for c in cond_order]
        stds = [report_stats[host]["conditions"][c]["return_std"] for c in cond_order]
        sems = [s / np.sqrt(30) for s in stds]

        bars = ax.bar(range(len(cond_order)), means, yerr=sems, capsize=4, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
        ax.set_xticks(range(len(cond_order)))
        ax.set_xticklabels(cond_labels, fontsize=10)
        ax.set_ylabel("True Environment Return $J(\\pi^*_{\\hat{P}}; P)$")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, axis="y")

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, m * 0.94, f"{m:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold", color="white")

    plt.tight_layout()
    fig_path = os.path.join(figure_dir, "portability_final_comparison.png")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # Sweep figure across lambdas
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lambdas = [0.0, 0.2, 0.5, 1.0, 2.0, 5.0]
    line_colors = {
        "prediction_error": "#ff7f0e",
        "estimated_crossing": "#2ca02c",
        "shuffled_crossing": "#d62728",
        "oracle_crossing": "#1f77b4",
    }
    line_labels = {
        "prediction_error": "Prediction Error",
        "estimated_crossing": "Estimated Crossing",
        "shuffled_crossing": "Shuffled Control",
        "oracle_crossing": "Oracle Crossing",
    }

    for idx, (host, ax, title) in enumerate([
        ("Host_A_Deterministic", axes[0], "(a) Host A $\\lambda$-Sensitivity"),
        ("Host_B_Ensemble", axes[1], "(b) Host B $\\lambda$-Sensitivity"),
    ]):
        df_h = df[df["host"] == host]
        unif_mean = float(df_h[df_h["condition"] == "uniform"]["j_learned"].mean())
        ax.axhline(unif_mean, color="#7f7f7f", linestyle="--", linewidth=1.5, label=f"Uniform Baseline ({unif_mean:.2f})")

        for cond in ["prediction_error", "estimated_crossing", "shuffled_crossing", "oracle_crossing"]:
            df_c = df_h[df_h["condition"] == cond]
            means = [float(df_c[df_c["lambda"] == lam]["j_learned"].mean()) for lam in lambdas]
            sems = [float(df_c[df_c["lambda"] == lam]["j_learned"].std(ddof=1) / np.sqrt(30)) for lam in lambdas]
            ax.errorbar(lambdas, means, yerr=sems, marker="o", capsize=3, label=line_labels[cond], color=line_colors[cond], linewidth=1.8)

        ax.set_xlabel("Reweighting Hyperparameter $\\lambda$")
        ax.set_ylabel("True Environment Return $J(\\pi^*_{\\hat{P}}; P)$")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="lower left", fontsize=9)

    plt.tight_layout()
    fig_sweep_path = os.path.join(figure_dir, "portability_lambda_sensitivity.png")
    fig.savefig(fig_sweep_path, dpi=300)
    plt.close(fig)

    with open(os.path.join(output_dir, "portability_final_report_stats.json"), "w") as f:
        json.dump(report_stats, f, indent=2)

    print("-> Generated final report statistics and figures successfully.")
    return report_stats

if __name__ == "__main__":
    generate_report_data()
