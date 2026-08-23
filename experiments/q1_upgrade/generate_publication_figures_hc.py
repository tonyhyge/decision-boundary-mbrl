"""
Publication-Quality Figure Generation Suite for Stage Q1-HC (HalfCheetah-v4) and 3-Environment Cross-Domain Synthesis.
Outputs vector PDFs and high-res PNGs to `figures/`.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESULTS_HOPPER = os.path.join(_ROOT, "results", "stage_q1fb_cuda")
_RESULTS_WALKER = os.path.join(_ROOT, "results", "stage_q1w2_cuda")
_RESULTS_CHEETAH = os.path.join(_ROOT, "results", "stage_q1hc_cuda")
_FIGURES_DIR = os.path.join(_ROOT, "figures")

CONFIRMATORY_SEEDS = [142, 143, 144, 145, 146, 147, 148, 149, 150, 151]


def setup_matplotlib_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.0,
        "figure.titlesize": 13,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.3,
    })


def load_cohort_data(results_dir, prefix="q1hc_halfcheetah", seeds=CONFIRMATORY_SEEDS):
    mle_runs = {}
    vagram_runs = {}
    for s in seeds:
        mf = os.path.join(results_dir, f"{prefix}_mle_seed{s}.json")
        vf = os.path.join(results_dir, f"{prefix}_vagram_seed{s}.json")
        if os.path.exists(mf):
            with open(mf, "r") as f: mle_runs[s] = json.load(f)
        if os.path.exists(vf):
            with open(vf, "r") as f: vagram_runs[s] = json.load(f)
    return mle_runs, vagram_runs


def generate_all_figures():
    setup_matplotlib_style()
    os.makedirs(_FIGURES_DIR, exist_ok=True)
    
    mle_runs, vagram_runs = load_cohort_data(_RESULTS_CHEETAH, prefix="q1hc_halfcheetah")
    common_seeds = sorted(list(set(mle_runs.keys()) & set(vagram_runs.keys())))
    
    if not common_seeds:
        print("No completed HalfCheetah paired seeds found.")
        return

    steps = [h["step"] for h in mle_runs[common_seeds[0]]["evaluation_history"]]
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Full-Budget Learning Curves J(t) on HalfCheetah-v4
    # -------------------------------------------------------------------------
    mle_curves = np.array([[h["eval_mean_return"] for h in mle_runs[s]["evaluation_history"]] for s in common_seeds])
    vag_curves = np.array([[h["eval_mean_return"] for h in vagram_runs[s]["evaluation_history"]] for s in common_seeds])

    mle_mean = np.mean(mle_curves, axis=0)
    mle_std = np.std(mle_curves, axis=0, ddof=1) if len(common_seeds) > 1 else np.zeros_like(mle_mean)
    vag_mean = np.mean(vag_curves, axis=0)
    vag_std = np.std(vag_curves, axis=0, ddof=1) if len(common_seeds) > 1 else np.zeros_like(vag_mean)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, s in enumerate(common_seeds):
        ax.plot(steps, mle_curves[i], color="#3498db", alpha=0.25, linestyle="--", linewidth=1.0)
        ax.plot(steps, vag_curves[i], color="#e74c3c", alpha=0.25, linestyle="-", linewidth=1.0)

    ax.plot(steps, mle_mean, color="#2980b9", label=f"Standard MLE (Mean, N={len(common_seeds)})", linewidth=2.5)
    ax.fill_between(steps, mle_mean - mle_std, mle_mean + mle_std, color="#2980b9", alpha=0.15)

    ax.plot(steps, vag_mean, color="#c0392b", label=f"VaGraM Value-Aware (Mean, N={len(common_seeds)})", linewidth=2.5)
    ax.fill_between(steps, vag_mean - vag_std, vag_mean + vag_std, color="#c0392b", alpha=0.15)

    ax.set_title("HalfCheetah-v4 MBPO Confirmatory Replication (125k Steps, N=10)")
    ax.set_xlabel("Environment Interaction Steps")
    ax.set_ylabel("Deterministic Evaluation Return")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    
    fig.savefig(os.path.join(_FIGURES_DIR, "stage_q1hc_fig1_learning_curves.pdf"), dpi=300)
    fig.savefig(os.path.join(_FIGURES_DIR, "stage_q1hc_fig1_learning_curves.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 2: 3-Environment Forest Plot (Hopper vs Walker2d vs HalfCheetah)
    # -------------------------------------------------------------------------
    hopper_f = os.path.join(_RESULTS_HOPPER, "stage_q1fb_confirmatory_closure.json")
    walker_f = os.path.join(_RESULTS_WALKER, "stage_q1w2_confirmatory_closure.json")
    cheetah_f = os.path.join(_RESULTS_CHEETAH, "stage_q1hc_confirmatory_closure.json")
    
    if os.path.exists(hopper_f) and os.path.exists(walker_f) and os.path.exists(cheetah_f):
        with open(hopper_f) as f: h_data = json.load(f)
        with open(walker_f) as f: w_data = json.load(f)
        with open(cheetah_f) as f: c_data = json.load(f)
        
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
        seeds = [r["seed"] for r in h_data["per_seed_breakdown"]]
        y_pos = np.arange(len(seeds))
        
        # Panel (a): Hopper
        axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        axes[0].barh(y_pos, [r["paired_delta_terminal"] for r in h_data["per_seed_breakdown"]], color="#27ae60", alpha=0.75, height=0.6, edgecolor="#1e8449")
        h_mean = h_data["primary_estimand"]["mean_paired_delta"]
        axes[0].axvline(h_mean, color="#c0392b", linestyle="-", linewidth=1.8, label=rf"Mean $\bar{{\Delta}} = {h_mean:+.1f}$")
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels([f"Seed {s}" for s in seeds])
        axes[0].set_xlabel(r"Terminal $\Delta_i$ (VaGraM - MLE)")
        axes[0].set_title(r"(a) Hopper-v4 ($10/10$ Wins)", fontweight="bold")
        axes[0].legend(loc="lower right", frameon=True)
        axes[0].grid(True, linestyle=":", alpha=0.5)

        # Panel (b): Walker2d
        axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        axes[1].barh(y_pos, [r["paired_delta_terminal"] for r in w_data["per_seed_breakdown"]], color="#2980b9", alpha=0.75, height=0.6, edgecolor="#1f618d")
        w_mean = w_data["primary_estimand"]["mean_paired_delta"]
        axes[1].axvline(w_mean, color="#c0392b", linestyle="-", linewidth=1.8, label=rf"Mean $\bar{{\Delta}} = {w_mean:+.1f}$")
        axes[1].set_xlabel(r"Terminal $\Delta_i$ (VaGraM - MLE)")
        axes[1].set_title(r"(b) Walker2d-v4 ($10/10$ Wins)", fontweight="bold")
        axes[1].legend(loc="lower right", frameon=True)
        axes[1].grid(True, linestyle=":", alpha=0.5)

        # Panel (c): HalfCheetah
        axes[2].axvline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        axes[2].barh(y_pos, [r["paired_delta_terminal"] for r in c_data["per_seed_breakdown"]], color="#8e44ad", alpha=0.75, height=0.6, edgecolor="#5b2c6f")
        c_mean = c_data["primary_estimand"]["mean_paired_delta"]
        axes[2].axvline(c_mean, color="#c0392b", linestyle="-", linewidth=1.8, label=rf"Mean $\bar{{\Delta}} = {c_mean:+.1f}$")
        axes[2].set_xlabel(r"Terminal $\Delta_i$ (VaGraM - MLE)")
        p_c = c_data["primary_estimand"]["exact_permutation_p_value"]
        axes[2].set_title(rf"(c) HalfCheetah-v4 ($p={p_c:.4f}$)", fontweight="bold")
        axes[2].legend(loc="lower right", frameon=True)
        axes[2].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        fig.savefig(os.path.join(_FIGURES_DIR, "stage_q1_3env_cross_environment_synthesis.pdf"), dpi=300)
        fig.savefig(os.path.join(_FIGURES_DIR, "stage_q1_3env_cross_environment_synthesis.png"), dpi=300)
        plt.close(fig)
        print("Generated 3-environment publication figure successfully!")


if __name__ == "__main__":
    generate_all_figures()
