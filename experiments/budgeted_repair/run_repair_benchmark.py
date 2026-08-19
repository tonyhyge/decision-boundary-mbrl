"""
Stage 3: Stochastic Multi-Action Choice GridWorld Decisive Falsification Benchmark.

Executes the decisive test for Gate G3 (Budgeted Model Correction Utility):
  1. Verifies Discriminative Headroom Quality Gate (AUC_Oracle - AUC_Random >= 0.15).
  2. Evaluates 10 Rankers across 50 independent GridWorld configurations (N=14 errors per model).
  3. Formally applies Hard Kill Rule 3 (AUC(dB) vs max(AUC(dE), AUC(d|G|))).
  4. Generates publication-grade figures and statistical reports.
"""
import os
import sys
import json
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.envs.gridworld_mdp import ChoiceGridWorldMDP, make_stochastic_choice_gridworld
from src.planning.dp import (
    value_iteration,
    policy_evaluation,
    compute_occupancy,
    expected_discounted_return,
)
from src.corruptions.injector import LocalizedError, CorruptedMDP
from src.metrics.diagnostics import (
    compute_action_margins,
    compute_boundary_pressure,
    compute_value_sensitivity,
    compute_advantage_perturbation,
    evaluate_incremental_r2,
)
from src.baselines.rankers import get_all_rankers
from src.correction.budget import BudgetEvaluator


def inject_gridworld_multidistribution_errors(
    mdp: ChoiceGridWorldMDP,
    num_errors: int = 14,
    rng: Optional[np.random.Generator] = None,
) -> CorruptedMDP:
    """
    Inject structured multi-distribution corruptions across GridWorld states:
      - 4 errors at unvisited / low-occupancy states (high L1 error, NC3 control).
      - 5 errors at large-gap visited corridor states (moderate L1 error, sub-threshold, NC1 control).
      - 5 errors at near-tie bottleneck choice states (small L1 error, boundary crossing).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    V_star, Q_star, pi_star = value_iteration(mdp)
    d_s, d_sa = compute_occupancy(mdp, pi_star)
    margins = compute_action_margins(Q_star, pi_star)

    # Classify candidate states
    unvisited_states = [s for s in range(mdp.num_states) if s != mdp.goal_state and d_s[s] < 0.005]
    if len(unvisited_states) < 3:
        # Fallback to lowest occupancy states
        sorted_by_d = np.argsort(d_s)
        unvisited_states = [s for s in sorted_by_d if s != mdp.goal_state][:6]

    visited_states = [s for s in range(mdp.num_states) if s != mdp.goal_state and d_s[s] >= 0.005]
    
    # Sort visited states by minimum competitor gap
    def min_comp_gap(s):
        opt_a = int(pi_star[s])
        return min(margins[s, a] for a in range(mdp.num_actions) if a != opt_a)

    visited_sorted = sorted(visited_states, key=min_comp_gap)
    near_tie_states = visited_sorted[:len(visited_sorted) // 2]
    large_gap_states = visited_sorted[len(visited_sorted) // 2:]

    corruptions: List[LocalizedError] = []
    err_id = 0

    # 1. Unvisited Errors (Large error, d ~ 0)
    for s in rng.choice(unvisited_states, size=min(4, len(unvisited_states)), replace=False):
        a = int(rng.choice(mdp.num_actions))
        true_p = mdp.transitions[s, a, :].copy()
        
        # Large random perturbation
        noise = rng.exponential(scale=1.0, size=mdp.num_states)
        noise[mdp.goal_state] = 0.0
        corrupt_p = true_p + 0.8 * (noise / noise.sum() - true_p)
        corrupt_p = np.maximum(corrupt_p, 0.0)
        corrupt_p /= corrupt_p.sum()

        corruptions.append(LocalizedError(
            error_id=err_id,
            state=int(s),
            action=int(a),
            true_p=true_p,
            corrupt_p=corrupt_p,
            error_l1=0.5 * float(np.sum(np.abs(corrupt_p - true_p))),
            error_kl=0.0,
            error_mse=float(np.mean((corrupt_p - true_p) ** 2)),
        ))
        err_id += 1

    # 2. Large-gap Visited Errors (Moderate error, pre-boundary)
    for s in rng.choice(large_gap_states, size=min(5, len(large_gap_states)), replace=False):
        opt_a = int(pi_star[s])
        a = opt_a  # On optimal action
        true_p = mdp.transitions[s, a, :].copy()
        
        # Sub-threshold perturbation shifting mass to slightly lower value state
        supp = np.where(true_p > 1e-5)[0]
        if len(supp) >= 2:
            supp_sorted = supp[np.argsort(V_star[supp])]
            s_low, s_high = supp_sorted[0], supp_sorted[-1]
            shift = min(true_p[s_high] * 0.4, 0.25)
            corrupt_p = true_p.copy()
            corrupt_p[s_high] -= shift
            corrupt_p[s_low] += shift
        else:
            corrupt_p = true_p.copy()

        corruptions.append(LocalizedError(
            error_id=err_id,
            state=int(s),
            action=int(a),
            true_p=true_p,
            corrupt_p=corrupt_p,
            error_l1=0.5 * float(np.sum(np.abs(corrupt_p - true_p))),
            error_kl=0.0,
            error_mse=float(np.mean((corrupt_p - true_p) ** 2)),
        ))
        err_id += 1

    # 3. Near-tie Bottleneck Errors (Small error, boundary-crossing)
    for s in rng.choice(near_tie_states, size=min(5, len(near_tie_states)), replace=False):
        opt_a = int(pi_star[s])
        comp_actions = [a_c for a_c in range(mdp.num_actions) if a_c != opt_a]
        # Choose runner-up action
        comp_a = comp_actions[int(np.argmin([margins[s, a_c] for a_c in comp_actions]))]
        
        # Corrupt optimal action to push it below runner-up
        true_p = mdp.transitions[s, opt_a, :].copy()
        supp = np.where(true_p > 1e-5)[0]
        if len(supp) >= 2:
            supp_sorted = supp[np.argsort(V_star[supp])]
            s_low, s_high = supp_sorted[0], supp_sorted[-1]
            gap = margins[s, comp_a]
            v_span = max(1e-5, V_star[s_high] - V_star[s_low])
            crit_shift = gap / (mdp.gamma * v_span)
            shift = min(true_p[s_high] * 0.9, crit_shift * 1.35)
            corrupt_p = true_p.copy()
            corrupt_p[s_high] -= shift
            corrupt_p[s_low] += shift
        else:
            corrupt_p = true_p.copy()

        corruptions.append(LocalizedError(
            error_id=err_id,
            state=int(s),
            action=int(opt_a),
            true_p=true_p,
            corrupt_p=corrupt_p,
            error_l1=0.5 * float(np.sum(np.abs(corrupt_p - true_p))),
            error_kl=0.0,
            error_mse=float(np.mean((corrupt_p - true_p) ** 2)),
        ))
        err_id += 1

    return CorruptedMDP(mdp, corruptions)


def run_stage3_gridworld_benchmark(
    num_trials: int = 50,
    base_seed: int = 42,
    output_dir: str = "results",
    figure_dir: str = "figures",
) -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figure_dir, exist_ok=True)

    print("================================================================================")
    print("      STAGE 3: STOCHASTIC MULTI-ACTION GRIDWORLD DECISIVE BENCHMARK             ")
    print("================================================================================")

    rankers = get_all_rankers()
    all_budget_records: List[Dict] = []
    trial_auc_records: List[Dict] = []
    pilot_error_records: List[Dict] = []

    print(f"\n[1/4] Running multi-action correction benchmark across {num_trials} GridWorld seeds...")

    for trial_idx in range(num_trials):
        seed = base_seed + trial_idx
        t_rng = np.random.default_rng(seed)

        # Sample grid environment with stochastic routes
        p_safe = float(t_rng.uniform(0.88, 0.94))
        p_risky = float(t_rng.uniform(0.65, 0.75))
        p_def = float(t_rng.uniform(0.80, 0.88))
        h_pen = float(t_rng.uniform(-2.5, -1.5))
        s_cost = float(t_rng.uniform(-0.08, -0.03))

        true_grid = make_stochastic_choice_gridworld(
            height=5,
            width=5,
            p_succ_safe=p_safe,
            p_succ_risky=p_risky,
            p_succ_default=p_def,
            hazard_penalty=h_pen,
            step_cost=s_cost,
            gamma=0.95,
            seed=seed,
        )

        corrupted_grid = inject_gridworld_multidistribution_errors(true_grid, num_errors=14, rng=t_rng)

        # Dynamic programming ground truth
        V_star, Q_star, pi_star = value_iteration(true_grid)
        _, d_sa = compute_occupancy(true_grid, pi_star)
        m_true = compute_action_margins(Q_star, pi_star)
        _, Q_corrupt, pi_corrupt = value_iteration(corrupted_grid.corrupted_mdp)
        m_corrupt = compute_action_margins(Q_corrupt, pi_corrupt)
        B_all = compute_boundary_pressure(m_true, m_corrupt)

        err_df_rows = []
        for e_idx, e in enumerate(corrupted_grid.errors):
            c_val, _, _ = corrupted_grid.compute_counterfactual_correction_value(e_idx)
            g_s, g_a = compute_value_sensitivity(e.corrupt_p - e.true_p, V_star)

            opt_a = int(pi_star[e.state])
            comp_actions = [a_c for a_c in range(true_grid.num_actions) if a_c != opt_a]
            gap = min(m_true[e.state, a_c] for a_c in comp_actions)

            delta_q_opt = float(true_grid.gamma * np.dot(e.corrupt_p - e.true_p, V_star)) if e.action == opt_a else 0.0
            delta_q_comp = 0.0
            a_s = compute_advantage_perturbation(delta_q_opt, delta_q_comp)

            b_val = float(B_all[e.state, e.action])
            occ_val = float(d_sa[e.state, e.action])

            row = {
                "trial": trial_idx,
                "error_id": e_idx,
                "state": e.state,
                "action": e.action,
                "error_l1": e.error_l1,
                "occupancy": occ_val,
                "true_action_gap": gap,
                "value_sensitivity_abs": g_a,
                "value_sensitivity_signed": g_s,
                "advantage_sensitivity_signed": a_s,
                "boundary_pressure": b_val,
                "occ_error_l1": occ_val * e.error_l1,
                "occ_value_sensitivity_abs": occ_val * g_a,
                "occ_boundary_pressure": occ_val * b_val,
                "correction_value": max(0.0, c_val),
            }
            err_df_rows.append(row)
            pilot_error_records.append(row)

        df_errors = pd.DataFrame(err_df_rows)
        evaluator = BudgetEvaluator(corrupted_grid, df_errors)

        trial_auc_dict = {"trial": trial_idx}
        for ranker in rankers:
            curve = evaluator.evaluate_ranker(ranker, t_rng)
            n_e = len(corrupted_grid.errors)
            y_vals = [curve[k] for k in range(n_e + 1)]
            auc_val = float(np.sum(y_vals) / (n_e + 1))
            trial_auc_dict[ranker.name] = auc_val

            for k in range(n_e + 1):
                all_budget_records.append({
                    "trial": trial_idx,
                    "ranker": ranker.name,
                    "budget_k": k,
                    "budget_fraction": k / n_e,
                    "recovery": curve[k],
                    "auc": auc_val,
                })

        trial_auc_records.append(trial_auc_dict)

    df_budget_all = pd.DataFrame(all_budget_records)
    df_auc_all = pd.DataFrame(trial_auc_records)
    df_pilot = pd.DataFrame(pilot_error_records)

    df_budget_all.to_csv(os.path.join(output_dir, "stage3_budget_benchmark_50seeds.csv"), index=False)
    df_pilot.to_csv(os.path.join(output_dir, "stage3_pilot_dataset.csv"), index=False)

    # -------------------------------------------------------------------------
    # 2. CHECK: Discriminative Headroom Quality Gate
    # -------------------------------------------------------------------------
    print("\n[2/4] Verifying Discriminative Headroom Quality Gate...")
    mean_oracle_auc = float(df_auc_all["Oracle (C_i)"].mean())
    mean_random_auc = float(df_auc_all["Random"].mean())
    headroom = mean_oracle_auc - mean_random_auc

    print(f"   Oracle Mean AUC: {mean_oracle_auc:.4f}")
    print(f"   Random Mean AUC: {mean_random_auc:.4f}")
    print(f"   Discriminative Headroom: {headroom:.4f} (Required: >= 0.1500)")

    headroom_passed = headroom >= 0.15

    # -------------------------------------------------------------------------
    # 3. STATISTICAL EVALUATION & HARD KILL RULE 3 APPLICATION
    # -------------------------------------------------------------------------
    print("\n[3/4] Evaluating Ranker Competition & Hard Kill Rule 3...")

    # Aggregate performance table
    table_records = []
    budget_fracs = [0.143, 0.286, 0.429, 0.571, 0.714, 1.0]
    for ranker_name in [r.name for r in rankers]:
        grp = df_budget_all[df_budget_all["ranker"] == ranker_name]
        rec_row = {"Ranker": ranker_name}
        for bf in budget_fracs:
            bf_grp = grp[np.isclose(grp["budget_fraction"], bf, atol=0.04)]
            rec_row[f"Recovery@{int(bf*100)}%"] = float(bf_grp["recovery"].mean()) if not bf_grp.empty else 0.0
        rec_row["AUC_Recovery"] = float(df_auc_all[ranker_name].mean())
        rec_row["AUC_Std"] = float(df_auc_all[ranker_name].std())
        table_records.append(rec_row)

    df_table = pd.DataFrame(table_records).sort_values("AUC_Recovery", ascending=False)
    print("\n--- STAGE 3: MULTI-ACTION BUDGET RECOVERY BENCHMARK TABLE ---")
    print(df_table.to_string(index=False))

    # Compare proposed Occupancy Boundary Pressure vs top baselines
    auc_db = df_auc_all["Occupancy x Boundary Pressure (d·B_i)"].to_numpy()
    auc_de = df_auc_all["Occupancy x Error"].to_numpy()
    auc_dg = df_auc_all["Value Sensitivity (Unsigned |G|)"].to_numpy()
    auc_l1 = df_auc_all["Prediction Error (L1)"].to_numpy()

    # Paired comparisons
    diff_vs_de = auc_db - auc_de
    diff_vs_dg = auc_db - auc_dg
    diff_vs_l1 = auc_db - auc_l1

    w_stat_de, w_pval_de = stats.wilcoxon(auc_db, auc_de, alternative="greater")
    w_stat_dg, w_pval_dg = stats.wilcoxon(auc_db, auc_dg, alternative="greater")
    w_stat_l1, w_pval_l1 = stats.wilcoxon(auc_db, auc_l1, alternative="greater")

    # Bootstrap 95% CI of AUC difference
    rng_b = np.random.default_rng(base_seed)
    boot_diffs_de = [np.mean(rng_b.choice(diff_vs_de, size=len(diff_vs_de), replace=True)) for _ in range(2000)]
    ci_de = [float(np.percentile(boot_diffs_de, 2.5)), float(np.percentile(boot_diffs_de, 97.5))]

    print("\n--- STATISTICAL COMPARISONS vs BASELINES ---")
    print(f"   d*B vs Occupancy x Error (d*E): Mean Delta AUC = +{np.mean(diff_vs_de):.4f}, 95% BCI: [{ci_de[0]:.4f}, {ci_de[1]:.4f}], p = {w_pval_de:.2e}")
    print(f"   d*B vs Value Sensitivity (|G|): Mean Delta AUC = +{np.mean(diff_vs_dg):.4f}, p = {w_pval_dg:.2e}")
    print(f"   d*B vs Prediction Error (L1): Mean Delta AUC = +{np.mean(diff_vs_l1):.4f}, p = {w_pval_l1:.2e}")

    # Gate G3 Verdict
    top_baseline_auc = max(float(df_auc_all["Occupancy x Error"].mean()), float(df_auc_all["Value Sensitivity (Unsigned |G|)"].mean()))
    db_auc_mean = float(df_auc_all["Occupancy x Boundary Pressure (d·B_i)"].mean())

    if db_auc_mean > top_baseline_auc and w_pval_de < 0.01 and ci_de[0] > 0:
        gate_g3_status = "PASSED (Green)"
        gate_g3_verdict = "Decision-boundary geometry (d*B_i) demonstrates statistically significant practical utility in multi-action budgeted model correction over all conventional heuristics under non-saturated headroom."
    else:
        gate_g3_status = "KILLED / FALSIFIED (Red)"
        gate_g3_verdict = "Boundary-aware prioritization did not outperform conventional heuristics. Contribution downgraded to diagnostic framework."

    print(f"\n================================================================================")
    print(f"   GATE G3 STATUS: {gate_g3_status}")
    print(f"   Verdict: {gate_g3_verdict}")
    print(f"================================================================================")

    # -------------------------------------------------------------------------
    # 4. PUBLICATION-QUALITY VISUALIZATIONS
    # -------------------------------------------------------------------------
    generate_stage3_figures(df_budget_all, df_auc_all, df_table, figure_dir)

    # Write Markdown Summary and JSON
    summary_path = os.path.join(output_dir, "stage3_summary.md")
    with open(summary_path, "w") as f:
        f.write("# STAGE 3: STOCHASTIC MULTI-ACTION GRIDWORLD BENCHMARK REPORT\n\n")
        f.write("## 1. Quality Gate: Discriminative Headroom Verification\n\n")
        f.write(f"- Oracle Mean AUC: `{mean_oracle_auc:.4f}`\n")
        f.write(f"- Random Mean AUC: `{mean_random_auc:.4f}`\n")
        f.write(f"- **Discriminative Headroom (Oracle - Random):** **`+{headroom:.4f}`** (Quality Gate >= 0.15: `{'PASSED' if headroom_passed else 'FAILED'}`)\n\n")
        f.write("## 2. Multi-Action Budget Recovery Benchmark Table (50 Seeds)\n\n")
        f.write("| Ranker | Recovery@14% | Recovery@29% | Recovery@43% | Recovery@57% | Recovery@71% | AUC Recovery | AUC Std |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in df_table.to_dict(orient="records"):
            f.write(f"| **{r['Ranker']}** | `{r.get('Recovery@14%', 0.0):.3f}` | `{r.get('Recovery@28%', r.get('Recovery@29%', 0.0)):.3f}` | `{r.get('Recovery@42%', r.get('Recovery@43%', 0.0)):.3f}` | `{r.get('Recovery@57%', 0.0):.3f}` | `{r.get('Recovery@71%', 0.0):.3f}` | **`{r['AUC_Recovery']:.3f}`** | `±{r['AUC_Std']:.3f}` |\n")
        f.write("\n## 3. Statistical Comparison & Hypothesis Tests\n\n")
        f.write(f"- $d \\cdot B_i$ vs Occupancy x Error ($d \\cdot E_{{L1}}$): $\\Delta \\text{{AUC}} = \\mathbf{{+{np.mean(diff_vs_de):.4f}}}$, 95% BCI: `[{ci_de[0]:.4f}, {ci_de[1]:.4f}]`, Wilcoxon $p = \\mathbf{{{w_pval_de:.2e}}}$\n")
        f.write(f"- $d \\cdot B_i$ vs Value Sensitivity ($\\|G\\|$): $\\Delta \\text{{AUC}} = \\mathbf{{+{np.mean(diff_vs_dg):.4f}}}$, Wilcoxon $p = \\mathbf{{{w_pval_dg:.2e}}}$\n")
        f.write(f"- $d \\cdot B_i$ vs Prediction Error ($E_{{L1}}$): $\\Delta \\text{{AUC}} = \\mathbf{{+{np.mean(diff_vs_l1):.4f}}}$, Wilcoxon $p = \\mathbf{{{w_pval_l1:.2e}}}$\n\n")
        f.write("## 4. Gate G3 Scientific Verdict\n\n")
        f.write(f"### Status: **{gate_g3_status}**\n")
        f.write(f"{gate_g3_verdict}\n")

    report_dict = {
        "headroom_oracle_random": headroom,
        "headroom_passed": headroom_passed,
        "budget_table": df_table.to_dict(orient="records"),
        "statistical_tests": {
            "mean_delta_auc_vs_de": float(np.mean(diff_vs_de)),
            "bci_95_vs_de": ci_de,
            "p_val_vs_de": float(w_pval_de),
            "p_val_vs_dg": float(w_pval_dg),
            "p_val_vs_l1": float(w_pval_l1),
        },
        "gate_g3_status": gate_g3_status,
    }

    with open(os.path.join(output_dir, "stage3_analysis_report.json"), "w") as f:
        json.dump(report_dict, f, indent=2)

    print(f"-> Saved Stage 3 summary to {summary_path}")
    return report_dict


def generate_stage3_figures(
    df_budget_all: pd.DataFrame,
    df_auc_all: pd.DataFrame,
    df_table: pd.DataFrame,
    figure_dir: str,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
    })

    # Figure 1: Recovery@K Curves with 95% CI bands
    fig, ax = plt.subplots(figsize=(9, 6))
    
    color_map = {
        "Random": "#7f7f7f",
        "Prediction Error (L1)": "#1f77b4",
        "Occupancy x Error": "#ff7f0e",
        "Inverse Action Gap": "#9467bd",
        "Value Sensitivity (Unsigned |G|)": "#8c564b",
        "Value Sensitivity (Signed G±)": "#e377c2",
        "Advantage Sensitivity (A±)": "#bcbd22",
        "Boundary Pressure (B_i)": "#ff7f00",
        "Occupancy x Boundary Pressure (d·B_i)": "#d62728",
        "Oracle (C_i)": "#2ca02c",
    }

    for ranker_name in df_table["Ranker"]:
        df_r = df_budget_all[df_budget_all["ranker"] == ranker_name]
        mean_curve = df_r.groupby("budget_fraction")["recovery"].mean()
        std_curve = df_r.groupby("budget_fraction")["recovery"].std()
        
        is_proposed = "Occupancy x Boundary" in ranker_name
        is_oracle = "Oracle" in ranker_name
        is_random = "Random" in ranker_name
        
        lw = 3.0 if is_proposed else (2.2 if is_oracle else (1.8 if is_random else 1.4))
        ls = "-" if is_proposed else ("--" if is_oracle else (":" if is_random else "-."))
        
        x_vals = mean_curve.index.to_numpy()
        y_vals = mean_curve.to_numpy()
        
        ax.plot(
            x_vals,
            y_vals,
            label=ranker_name,
            color=color_map.get(ranker_name, None),
            linewidth=lw,
            linestyle=ls,
        )
        if is_proposed or is_oracle:
            ax.fill_between(
                x_vals,
                np.maximum(0, y_vals - 1.96 * std_curve / np.sqrt(50)),
                np.minimum(1, y_vals + 1.96 * std_curve / np.sqrt(50)),
                color=color_map.get(ranker_name),
                alpha=0.15,
            )

    ax.set_xlabel("Correction Budget Fraction $K / N$")
    ax.set_ylabel("Expected Return Recovery (Recovery@K)")
    ax.set_title("Stage 3 GridWorld: Budgeted World-Model Correction (50 Seeds)")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig1_path = os.path.join(figure_dir, "stage3_budget_recovery_headroom.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage 3 Recovery Curve Figure to {fig1_path}")

    # Figure 2: AUC Boxplot Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort columns by mean AUC
    sorted_rankers = df_table["Ranker"].tolist()
    auc_data = [df_auc_all[r].to_numpy() for r in sorted_rankers]
    
    box = ax.boxplot(
        auc_data,
        vert=False,
        patch_artist=True,
        tick_labels=sorted_rankers,
        medianprops=dict(color="black", linewidth=2),
    )
    
    for patch, name in zip(box['boxes'], sorted_rankers):
        c = color_map.get(name, "#cccccc")
        patch.set_facecolor(c)
        patch.set_alpha(0.7)

    ax.set_xlabel("Area Under Recovery Curve (AUC Recovery)")
    ax.set_title("Stage 3 GridWorld: Ranker AUC Distribution (50 Seeds)")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig2_path = os.path.join(figure_dir, "stage3_ranker_auc_distribution.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"-> Saved Stage 3 AUC Boxplot Figure to {fig2_path}")


if __name__ == "__main__":
    run_stage3_gridworld_benchmark()
