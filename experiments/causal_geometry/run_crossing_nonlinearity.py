"""
Stage 0 / C2 Within-Crossing Sub-Sample Nonlinearity & Monotonicity Analysis
Analyzes linear, polynomial, rank (Spearman, Kendall), and monotonic (Isotonic) associations
within the crossing-only subset (N = 174) from stage0_matched_pairs.csv.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

def run_within_crossing_audit(
    matched_csv: str = "results/stage0_matched_pairs.csv",
    output_dir: str = "results",
):
    df = pd.read_csv(matched_csv)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    df_cross = df[df["z_cross_comp"] == 1].copy()
    n_samples = len(df_cross)
    b = df_cross["boundary_pressure_comp"].to_numpy()
    c = df_cross["correction_value_comp"].to_numpy()

    # 1. Linear Pearson
    r_p, p_p = stats.pearsonr(b, c)

    # 2. Spearman & Kendall
    rho_s, p_s = stats.spearmanr(b, c)
    tau_k, p_k = stats.kendalltau(b, c)

    # 3. Regressions
    lin = LinearRegression().fit(b.reshape(-1, 1), c)
    r2_lin = float(r2_score(c, lin.predict(b.reshape(-1, 1))))

    poly2 = PolynomialFeatures(degree=2)
    X_poly2 = poly2.fit_transform(b.reshape(-1, 1))
    lin2 = LinearRegression().fit(X_poly2, c)
    r2_poly2 = float(r2_score(c, lin2.predict(X_poly2)))

    poly3 = PolynomialFeatures(degree=3)
    X_poly3 = poly3.fit_transform(b.reshape(-1, 1))
    lin3 = LinearRegression().fit(X_poly3, c)
    r2_poly3 = float(r2_score(c, lin3.predict(X_poly3)))

    iso = IsotonicRegression(out_of_bounds="clip").fit(b, c)
    r2_iso = float(r2_score(c, iso.predict(b)))

    # Cluster Bootstrap for Pearson & Spearman
    mdps = df_cross["mdp_id"].unique()
    boot_r = []
    boot_rho = []
    for _ in range(2000):
        sampled = rng.choice(mdps, size=len(mdps), replace=True)
        sample_df = pd.concat([df_cross[df_cross["mdp_id"] == m] for m in sampled])
        br, _ = stats.pearsonr(sample_df["boundary_pressure_comp"], sample_df["correction_value_comp"])
        brho, _ = stats.spearmanr(sample_df["boundary_pressure_comp"], sample_df["correction_value_comp"])
        boot_r.append(br)
        boot_rho.append(brho)

    ci_r = [float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))]
    ci_rho = [float(np.percentile(boot_rho, 2.5)), float(np.percentile(boot_rho, 97.5))]

    report = {
        "n_crossing_samples": n_samples,
        "num_mdp_clusters": len(mdps),
        "linear_pearson": {
            "r": float(r_p),
            "p_val": float(p_p),
            "bci_95": ci_r,
            "r2": r2_lin,
        },
        "rank_correlations": {
            "spearman_rho": float(rho_s),
            "spearman_p_val": float(p_s),
            "spearman_bci_95": ci_rho,
            "kendall_tau": float(tau_k),
            "kendall_p_val": float(p_k),
        },
        "nonlinear_fits": {
            "r2_quadratic_poly2": r2_poly2,
            "r2_cubic_poly3": r2_poly3,
            "r2_isotonic_monotonic": r2_iso,
        }
    }

    with open(os.path.join(output_dir, "stage0_crossing_nonlinearity_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("=== WITHIN-CROSSING AUDIT COMPLETE ===")
    print(f"N = {n_samples}, Pearson r = {r_p:.4f} (p = {p_p:.4f}, 95% BCI: [{ci_r[0]:.4f}, {ci_r[1]:.4f}])")
    print(f"Spearman rho = {rho_s:.4f} (p = {p_s:.4f}, 95% BCI: [{ci_rho[0]:.4f}, {ci_rho[1]:.4f}])")
    print(f"R^2 Lin = {r2_lin:.4f}, R^2 Poly2 = {r2_poly2:.4f}, R^2 Poly3 = {r2_poly3:.4f}, R^2 Isotonic = {r2_iso:.4f}")
    return report

if __name__ == "__main__":
    run_within_crossing_audit()
