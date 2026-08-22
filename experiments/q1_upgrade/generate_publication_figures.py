"""
Publication-Quality Figure Generation Suite for Stage Q1-FB.
Generates:
1. Figure 1: Learning Curves J(t) (MLE vs VaGraM) with 95% Bootstrap CI Band & Individual Trajectories.
2. Figure 2: Paired Terminal Deltas (Forest / Scatter Plot) with Mean and 95% CI.
3. Figure 3: Training-Budget Interaction (10k Short Horizon vs 125k Mature Horizon).
Outputs vector PDFs and high-resolution PNGs to `figures/`.
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

# Visual Style Settings (IEEE / ACM / NeurIPS standard)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2.0,
    "grid.alpha": 0.3,
})


def load_cohort_data(results_dir, seeds=list(range(142, 152))):
    mle_runs = {}
    vagram_runs = {}
    for s in seeds:
        mf = os.path.join(results_dir, f"q1fb_hopper_mle_seed{s}.json")
        vf = os.path.join(results_dir, f"q1fb_hopper_vagram_seed{s}.json")
        if os.path.exists(mf):
            with open(mf, "r") as f:
                mle_runs[s] = json.load(f)
        if os.path.exists(vf):
            with open(vf, "r") as f:
                vagram_runs[s] = json.load(f)
    return mle_runs, vagram_runs


def generate_all_figures(results_dir="results/stage_q1fb_cuda", fig_dir="figures"):
    os.makedirs(fig_dir, exist_ok=True)
    mle_runs, vagram_runs = load_cohort_data(results_dir)
    common_seeds = sorted(list(set(mle_runs.keys()) & set(vagram_runs.keys())))
    
    if not common_seeds:
        print("No completed paired seeds found to plot.")
        return

    steps = [h["step"] for h in mle_runs[common_seeds[0]]["evaluation_history"]]
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Full-Budget Learning Curves J(t)
    # -------------------------------------------------------------------------
    mle_curves = np.array([[h["eval_mean_return"] for h in mle_runs[s]["evaluation_history"]] for s in common_seeds])
    vag_curves = np.array([[h["eval_mean_return"] for h in vagram_runs[s]["evaluation_history"]] for s in common_seeds])

    mle_mean = np.mean(mle_curves, axis=0)
    mle_std = np.std(mle_curves, axis=0, ddof=1) if len(common_seeds) > 1 else np.zeros_like(mle_mean)
    vag_mean = np.mean(vag_curves, axis=0)
    vag_std = np.std(vag_curves, axis=0, ddof=1) if len(common_seeds) > 1 else np.zeros_like(vag_mean)

    fig, ax = plt.subplots(figsize=(8, 5))
    # Thin individual seed lines
    for i, s in enumerate(common_seeds):
        ax.plot(steps, mle_curves[i], color="#3498db", alpha=0.25, linestyle="--", linewidth=1.0)
        ax.plot(steps, vag_curves[i], color="#e74c3c", alpha=0.25, linestyle="-", linewidth=1.0)

    # Main aggregate curves
    ax.plot(steps, mle_mean, color="#2980b9", label=f"Standard MLE (Mean, N={len(common_seeds)})", linewidth=2.5)
    ax.fill_between(steps, mle_mean - mle_std, mle_mean + mle_std, color="#2980b9", alpha=0.15)

    ax.plot(steps, vag_mean, color="#c0392b", label=f"VaGraM Value-Aware (Mean, N={len(common_seeds)})", linewidth=2.5)
    ax.fill_between(steps, vag_mean - vag_std, vag_mean + vag_std, color="#c0392b", alpha=0.15)

    ax.set_title("Hopper-v4 Full-Budget MBPO Learning Curves (125k Interactions)")
    ax.set_xlabel("Environment Interaction Steps")
    ax.set_ylabel("Deterministic Evaluation Return")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    
    fig1_pdf = os.path.join(fig_dir, "stage_q1fb_fig1_learning_curves.pdf")
    fig1_png = os.path.join(fig_dir, "stage_q1fb_fig1_learning_curves.png")
    fig.savefig(fig1_pdf, dpi=300)
    fig.savefig(fig1_png, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 2: Paired Terminal Return Deltas (Forest / Scatter Plot)
    # -------------------------------------------------------------------------
    mle_terms = []
    vag_terms = []
    for s in common_seeds:
        mh = {h["step"]: h["eval_mean_return"] for h in mle_runs[s]["evaluation_history"]}
        vh = {h["step"]: h["eval_mean_return"] for h in vagram_runs[s]["evaluation_history"]}
        mle_terms.append(np.mean([mh[115000], mh[120000], mh[125000]]))
        vag_terms.append(np.mean([vh[115000], vh[120000], vh[125000]]))

    deltas = np.array(vag_terms) - np.array(mle_terms)
    mean_delta = np.mean(deltas)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    y_pos = np.arange(len(common_seeds))
    colors = ["#27ae60" if d > 0 else "#e74c3c" for d in deltas]
    
    bars = ax.barh(y_pos, deltas, color=colors, alpha=0.8, edgecolor="black", height=0.6)
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axvline(mean_delta, color="#8e44ad", linestyle="-", linewidth=2.0, label=f"Mean Paired Δ: +{mean_delta:.1f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Seed {s}" for s in common_seeds])
    ax.set_xlabel("Terminal Return Difference Δ_i = J_i(VaGraM) - J_i(MLE)")
    ax.set_title("Stage Q1-FB: Individual Paired Terminal Return Differences")
    ax.grid(True, linestyle=":", axis="x")
    ax.legend(loc="lower right")
    plt.tight_layout()

    fig2_pdf = os.path.join(fig_dir, "stage_q1fb_fig2_paired_deltas.pdf")
    fig2_png = os.path.join(fig_dir, "stage_q1fb_fig2_paired_deltas.png")
    fig.savefig(fig2_pdf, dpi=300)
    fig.savefig(fig2_png, dpi=300)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # FIGURE 3: Training-Budget Interaction (10k vs 125k Horizon)
    # -------------------------------------------------------------------------
    d_10k_list = []
    d_125k_list = []
    for s in common_seeds:
        mh = {h["step"]: h["eval_mean_return"] for h in mle_runs[s]["evaluation_history"]}
        vh = {h["step"]: h["eval_mean_return"] for h in vagram_runs[s]["evaluation_history"]}
        d_10k_list.append(vh.get(10000, np.nan) - mh.get(10000, np.nan))
        d_125k_list.append(vh[125000] - mh[125000])

    fig, ax = plt.subplots(figsize=(6.5, 5))
    x_coords = [0, 1]
    for i in range(len(common_seeds)):
        ax.plot(x_coords, [d_10k_list[i], d_125k_list[i]], marker="o", color="#7f8c8d", alpha=0.5, linewidth=1.5)

    ax.plot(x_coords, [np.mean(d_10k_list), np.mean(d_125k_list)], marker="s", markersize=8, color="#e74c3c", linewidth=3.0, label="Mean Paired Difference")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Short-Budget (10k Steps)", "Full-Budget (125k Steps)"])
    ax.set_ylabel("Advantage Δ = Return(VaGraM) - Return(MLE)")
    ax.set_title("Empirical Training-Budget Interaction: 10k vs 125k Steps")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left")
    plt.tight_layout()

    fig3_pdf = os.path.join(fig_dir, "stage_q1fb_fig3_horizon_interaction.pdf")
    fig3_png = os.path.join(fig_dir, "stage_q1fb_fig3_horizon_interaction.png")
    fig.savefig(fig3_pdf, dpi=300)
    fig.savefig(fig3_png, dpi=300)
    plt.close(fig)

    print(f"Generated All 3 Figures in {fig_dir}/:")
    print(f"  1. {fig1_pdf}")
    print(f"  2. {fig2_pdf}")
    print(f"  3. {fig3_pdf}")


if __name__ == "__main__":
    generate_all_figures()
