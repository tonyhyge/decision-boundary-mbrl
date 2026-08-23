"""
Publication-Quality Figure Generation Suite for Stage Q1-W2 (Walker2d-v4) and Cross-Environment Transfer.
Outputs vector PDFs and high-res PNGs to `figures/`.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_RESULTS_HOPPER = os.path.join(_ROOT, "results", "stage_q1fb_cuda")
_RESULTS_WALKER = os.path.join(_ROOT, "results", "stage_q1w2_cuda")
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
        "legend.fontsize": 9.5,
        "figure.titlesize": 13,
        "lines.linewidth": 2.0,
        "grid.alpha": 0.3,
    })


def load_cohort_data(results_dir, prefix="q1w2_walker2d", seeds=CONFIRMATORY_SEEDS):
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
    
    mle_runs, vagram_runs = load_cohort_data(_RESULTS_WALKER, prefix="q1w2_walker2d")
    common_seeds = sorted(list(set(mle_runs.keys()) & set(vagram_runs.keys())))
    
    if not common_seeds:
        print("No completed Walker2d paired seeds found.")
        return

    steps = [h["step"] for h in mle_runs[common_seeds[0]]["evaluation_history"]]
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Full-Budget Learning Curves J(t) on Walker2d-v4
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

    ax.set_title("Walker2d-v4 MBPO Confirmatory Replication (125k Steps, N=10)")
    ax.set_xlabel("Environment Interaction Steps")
    ax.set_ylabel("Deterministic Evaluation Return")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    
    fig.savefig(os.path.join(_FIGURES_DIR, "stage_q1w2_fig1_learning_curves.pdf"), dpi=300)
    fig.savefig(os.path.join(_FIGURES_DIR, "stage_q1w2_fig1_learning_curves.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 2: Paired Deltas Forest Plot (Walker2d-v4)
    # -------------------------------------------------------------------------
    walker_closure_file = os.path.join(_RESULTS_WALKER, "stage_q1w2_confirmatory_closure.json")
    with open(walker_closure_file) as f: w_data = json.load(f)
    
    w_seeds = [r["seed"] for r in w_data["per_seed_breakdown"]]
    w_deltas = [r["paired_delta_terminal"] for r in w_data["per_seed_breakdown"]]
    y_pos = np.arange(len(w_seeds))

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    bars = ax.barh(y_pos, w_deltas, color="#2980b9", alpha=0.75, height=0.55, edgecolor="#1f618d")
    mean_val = w_data["primary_estimand"]["mean_paired_delta"]
    ax.axvline(mean_val, color="#c0392b", linestyle="-", linewidth=2.0,
               label=rf"Cohort Mean $\bar{{\Delta}} = {mean_val:+.2f}$ (p=0.00195)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Seed {s}" for s in w_seeds])
    ax.set_xlabel(r"Paired Terminal Delta $\Delta_i = J_{\mathrm{VaGraM}} - J_{\mathrm{MLE}}$")
    ax.set_title(r"Walker2d-v4 Paired Confirmatory Deltas ($N=10$, 10/10 Wins)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    
    fig.savefig(os.path.join(_FIGURES_DIR, "stage_q1w2_fig2_paired_deltas.pdf"), dpi=300)
    fig.savefig(os.path.join(_FIGURES_DIR, "stage_q1w2_fig2_paired_deltas.png"), dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 3: Cross-Environment Comparison (Hopper vs Walker2d Forest Plot)
    # -------------------------------------------------------------------------
    hopper_closure_file = os.path.join(_RESULTS_HOPPER, "stage_q1fb_confirmatory_closure.json")
    if os.path.exists(hopper_closure_file):
        with open(hopper_closure_file) as f: h_data = json.load(f)
        
        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
        h_seeds = [r["seed"] for r in h_data["per_seed_breakdown"]]
        h_deltas = [r["paired_delta_terminal"] for r in h_data["per_seed_breakdown"]]
        
        # Panel (a): Hopper
        axes[0].axvline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        axes[0].barh(y_pos, h_deltas, color="#27ae60", alpha=0.75, height=0.6, edgecolor="#1e8449")
        h_mean = h_data["primary_estimand"]["mean_paired_delta"]
        axes[0].axvline(h_mean, color="#c0392b", linestyle="-", linewidth=1.8,
                        label=rf"Mean $\bar{{\Delta}} = {h_mean:+.1f}$")
        axes[0].set_yticks(y_pos)
        axes[0].set_yticklabels([f"Seed {s}" for s in h_seeds])
        axes[0].set_xlabel(r"Paired Terminal Delta $\Delta_i$")
        axes[0].set_title(r"(a) Hopper-v4 ($10/10$ Wins, $p=0.00195$)", fontweight="bold")
        axes[0].legend(loc="lower right", frameon=True)
        axes[0].grid(True, linestyle=":", alpha=0.5)

        # Panel (b): Walker2d
        axes[1].axvline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        axes[1].barh(y_pos, w_deltas, color="#2980b9", alpha=0.75, height=0.6, edgecolor="#1f618d")
        axes[1].axvline(mean_val, color="#c0392b", linestyle="-", linewidth=1.8,
                        label=rf"Mean $\bar{{\Delta}} = {mean_val:+.1f}$")
        axes[1].set_xlabel(r"Paired Terminal Delta $\Delta_i$")
        axes[1].set_title(r"(b) Walker2d-v4 ($10/10$ Wins, $p=0.00195$)", fontweight="bold")
        axes[1].legend(loc="lower right", frameon=True)
        axes[1].grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        fig.savefig(os.path.join(_FIGURES_DIR, "stage_q1_cross_environment_transfer.pdf"), dpi=300)
        fig.savefig(os.path.join(_FIGURES_DIR, "stage_q1_cross_environment_transfer.png"), dpi=300)
        plt.close(fig)
        print("Generated all publication figures successfully!")


if __name__ == "__main__":
    generate_all_figures()
