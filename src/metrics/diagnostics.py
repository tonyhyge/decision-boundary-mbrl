r"""
Mathematical diagnostics: Action-Margin Deformation, Boundary Pressure, Value Sensitivity, and Statistical Evaluation.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats as stats
import pandas as pd


def compute_action_margins(
    Q: np.ndarray, optimal_actions: np.ndarray
) -> np.ndarray:
    """
    Compute action margins for every state and action.
    For state s and action a:
        margin(s, a) = Q(s, a*(s)) - Q(s, a)
    
    Returns:
        margins (np.ndarray): Shape (|S|, |A|). Note that margin(s, a*(s)) == 0.0.
    """
    num_states, num_actions = Q.shape
    margins = np.zeros_like(Q)
    for s in range(num_states):
        opt_a = int(optimal_actions[s])
        opt_q = Q[s, opt_a]
        margins[s, :] = opt_q - Q[s, :]
    return margins


def compute_margin_deformation(
    m_true: np.ndarray, m_corrupt: np.ndarray
) -> np.ndarray:
    r"""
    Compute signed margin deformation \Delta m(s, a) = m_{\hat{P}}(s, a) - m_P(s, a).
    """
    return m_corrupt - m_true


def compute_boundary_pressure(
    m_true: np.ndarray, m_corrupt: np.ndarray, eps: float = 1e-4
) -> np.ndarray:
    r"""
    Compute normalized signed boundary pressure:
        B(s, a) = - \Delta m(s, a) / (gap(s) + eps)
    where gap(s) is the true action gap at state s against the runner-up action.
    
    Interpretation:
        B < 0: Error widens/stabilizes the optimal action ranking.
        0 < B < 1: Error compresses the margin toward the decision boundary.
        B >= 1: Error crosses the decision boundary, flipping the optimal action.
    """
    delta_m = compute_margin_deformation(m_true, m_corrupt)
    num_states, num_actions = m_true.shape
    b_pressure = np.zeros_like(m_true)
    for s in range(num_states):
        # Action gap against competitor
        non_zero_margins = [m_true[s, a] for a in range(num_actions) if m_true[s, a] > 1e-8]
        gap_s = min(non_zero_margins) if non_zero_margins else float(np.max(m_true[s, :]))
        denom = max(gap_s, 1e-4) + eps
        b_pressure[s, :] = -delta_m[s, :] / denom
    return b_pressure


def compute_value_sensitivity(
    delta_p: np.ndarray, V_star: np.ndarray
) -> Tuple[float, float]:
    r"""
    Compute signed and unsigned first-order value sensitivity.
        G^\pm = \delta P^T V^*
        |G|   = |\delta P^T V^*|
    """
    signed_g = float(np.dot(delta_p, V_star))
    abs_g = float(np.abs(signed_g))
    return signed_g, abs_g


def compute_advantage_perturbation(
    delta_q_opt: float, delta_q_comp: float
) -> float:
    r"""
    Signed advantage perturbation A^\pm = \Delta Q(a^*) - \Delta Q(a_comp).
    """
    return float(delta_q_opt - delta_q_comp)


def evaluate_incremental_r2(
    df: pd.DataFrame,
    target_col: str = "correction_value",
    control_cols: Optional[List[str]] = None,
    proposed_col: str = "boundary_pressure",
) -> Dict[str, Union[float, int]]:
    """
    Fit hierarchical linear regression models and compute incremental R^2 (Delta R^2)
    and partial F-test statistic when adding the proposed decision-margin diagnostic.
    """
    if control_cols is None:
        control_cols = [
            "error_l1",
            "occupancy",
            "true_action_gap",
            "value_sensitivity_abs",
            "value_sensitivity_signed",
            "advantage_sensitivity_signed",
        ]

    # Filter out missing or constant columns
    valid_controls = [c for c in control_cols if c in df.columns and df[c].std() > 1e-12]
    y = df[target_col].to_numpy(dtype=np.float64)
    n = len(y)
    
    # Spearman correlation
    try:
        spearman_corr, spearman_p = stats.spearmanr(df[proposed_col], y)
        if np.isnan(spearman_corr):
            spearman_corr, spearman_p = 0.0, 1.0
    except Exception:
        spearman_corr, spearman_p = 0.0, 1.0

    if n <= len(valid_controls) + 2:
        return {
            "n_samples": n,
            "r2_base": 0.0,
            "r2_full": 0.0,
            "delta_r2": 0.0,
            "f_stat": 0.0,
            "p_val": 1.0,
            "spearman_rho": float(spearman_corr),
            "spearman_p": float(spearman_p),
        }

    # Standardize features for numerical stability
    X_raw_base = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in valid_controls])
    # Remove collinear / zero-variance columns
    stds_base = np.std(X_raw_base, axis=0)
    valid_mask_base = stds_base > 1e-10
    if not np.any(valid_mask_base):
        X_base = np.ones((n, 1), dtype=np.float64)
    else:
        X_norm_base = (X_raw_base[:, valid_mask_base] - np.mean(X_raw_base[:, valid_mask_base], axis=0)) / stds_base[valid_mask_base]
        X_base = np.column_stack([np.ones(n, dtype=np.float64), X_norm_base])

    # Proposed feature standardized
    x_prop = df[proposed_col].to_numpy(dtype=np.float64)
    std_prop = float(np.std(x_prop))
    if std_prop > 1e-10:
        x_prop_norm = (x_prop - np.mean(x_prop)) / std_prop
        X_full = np.column_stack([X_base, x_prop_norm])
    else:
        X_full = X_base.copy()

    # SVD orthogonal projection for perfectly stable least-squares under any collinearity
    u_b, s_b, _ = np.linalg.svd(X_base, full_matrices=False)
    k_b = int(np.sum(s_b > 1e-8 * s_b[0])) if len(s_b) > 0 and s_b[0] > 1e-12 else 0
    if k_b > 0:
        u_bk = np.ascontiguousarray(u_b[:, :k_b], dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        y_pred_base = np.dot(u_bk, np.dot(u_bk.T, y_c))
        rank_base = k_b
    else:
        y_pred_base = np.zeros_like(y)
        rank_base = 0

    u_f, s_f, _ = np.linalg.svd(X_full, full_matrices=False)
    k_f = int(np.sum(s_f > 1e-8 * s_f[0])) if len(s_f) > 0 and s_f[0] > 1e-12 else 0
    if k_f > 0:
        u_fk = np.ascontiguousarray(u_f[:, :k_f], dtype=np.float64)
        y_c = np.ascontiguousarray(y, dtype=np.float64)
        y_pred_full = np.dot(u_fk, np.dot(u_fk.T, y_c))
        rank_full = k_f
    else:
        y_pred_full = np.zeros_like(y)
        rank_full = 0

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot < 1e-12:
        return {
            "n_samples": n,
            "r2_base": 0.0,
            "r2_full": 0.0,
            "delta_r2": 0.0,
            "f_stat": 0.0,
            "p_val": 1.0,
            "spearman_rho": float(spearman_corr),
            "spearman_p": float(spearman_p),
        }

    ss_res_base = float(np.sum((y - y_pred_base) ** 2))
    ss_res_full = float(np.sum((y - y_pred_full) ** 2))

    r2_base = max(0.0, 1.0 - ss_res_base / ss_tot)
    r2_full = max(0.0, 1.0 - ss_res_full / ss_tot)
    delta_r2 = max(0.0, r2_full - r2_base)

    # Partial F-test
    df_num = max(1, rank_full - rank_base)
    df_denom = max(1, n - rank_full)

    if df_denom > 0 and ss_res_full > 1e-12 and ss_res_base >= ss_res_full:
        f_stat = float(((ss_res_base - ss_res_full) / df_num) / (ss_res_full / df_denom))
        p_val = float(1.0 - stats.f.cdf(f_stat, df_num, df_denom))
    else:
        f_stat = 0.0
        p_val = 1.0

    return {
        "n_samples": n,
        "r2_base": float(r2_base),
        "r2_full": float(r2_full),
        "delta_r2": float(delta_r2),
        "f_stat": float(f_stat),
        "p_val": float(p_val),
        "spearman_rho": float(spearman_corr),
        "spearman_p": float(spearman_p),
    }


def evaluate_matched_pair_effect(
    df_matched: pd.DataFrame,
    val_col_compressive: str = "correction_value_comp",
    val_col_expansive: str = "correction_value_exp",
) -> Dict[str, float]:
    """
    Perform paired t-test and Wilcoxon signed-rank test on matched error pairs.
    """
    c_comp = df_matched[val_col_compressive].to_numpy(dtype=np.float64)
    c_exp = df_matched[val_col_expansive].to_numpy(dtype=np.float64)
    diff = c_comp - c_exp

    n = len(diff)
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    cohen_d = float(mean_diff / (std_diff + 1e-12))

    t_stat, t_pval = stats.ttest_rel(c_comp, c_exp)
    
    # Wilcoxon signed-rank test (handles zero differences)
    try:
        w_stat, w_pval = stats.wilcoxon(c_comp, c_exp, zero_method="wilcox")
    except Exception:
        w_stat, w_pval = 0.0, 1.0

    return {
        "n_pairs": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohen_d": cohen_d,
        "t_stat": float(t_stat),
        "t_pval": float(t_pval),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pval": float(w_pval),
    }


def evaluate_cluster_hierarchical_bootstrap(
    df_matched: pd.DataFrame,
    cluster_col: str = "mdp_id",
    val_col_compressive: str = "correction_value_comp",
    val_col_expansive: str = "correction_value_exp",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Union[float, int, List[float]]]:
    """
    Perform hierarchical block bootstrap over MDP configurations to eliminate pseudoreplication.
    
    Procedure:
      1. Group matched pairs by MDP configuration (cluster).
      2. For b = 1, ..., n_boot:
         a. Sample unique clusters with replacement.
         b. Within each sampled cluster, sample pairs with replacement.
         c. Compute mean paired difference for bootstrap replicate.
      3. Compute 95% bootstrap percentile confidence interval.
    """
    rng = np.random.default_rng(seed)
    clusters = np.unique(df_matched[cluster_col])
    n_clusters = len(clusters)
    
    cluster_groups = {
        c: df_matched[df_matched[cluster_col] == c]
        for c in clusters
    }

    boot_means = np.zeros(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        sampled_diffs = []
        for c in sampled_clusters:
            grp = cluster_groups[c]
            n_rows = len(grp)
            if n_rows > 0:
                idx = rng.choice(n_rows, size=n_rows, replace=True)
                c_comp = grp[val_col_compressive].iloc[idx].to_numpy(dtype=np.float64)
                c_exp = grp[val_col_expansive].iloc[idx].to_numpy(dtype=np.float64)
                sampled_diffs.extend(c_comp - c_exp)

        boot_means[b] = np.mean(sampled_diffs) if sampled_diffs else 0.0

    original_diff = (df_matched[val_col_compressive] - df_matched[val_col_expansive]).to_numpy(dtype=np.float64)
    obs_mean = float(np.mean(original_diff))
    boot_se = float(np.std(boot_means, ddof=1))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    # Cluster-level empirical p-value: fraction of bootstrap replicates <= 0 (two-sided)
    frac_le_zero = float(np.mean(boot_means <= 0.0))
    p_boot = min(1.0, 2.0 * min(frac_le_zero, 1.0 - frac_le_zero))

    return {
        "n_clusters": n_clusters,
        "n_pairs_total": len(df_matched),
        "observed_mean_diff": obs_mean,
        "bootstrap_se": boot_se,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "bootstrap_p_val": p_boot,
        "boot_means_sample": boot_means[:100].tolist(),
    }


def evaluate_cluster_robust_ols(
    df: pd.DataFrame,
    target_col: str = "correction_value",
    feature_cols: Optional[List[str]] = None,
    cluster_col: str = "mdp_id",
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Fit Ordinary Least Squares with Huber-White Cluster-Robust Standard Errors (CRSE).
    """
    if feature_cols is None:
        feature_cols = [
            "error_l1",
            "occupancy",
            "true_action_gap",
            "value_sensitivity_abs",
            "value_sensitivity_signed",
            "advantage_sensitivity_signed",
            "occupancy_boundary_pressure",
        ]

    valid_features = [c for c in feature_cols if c in df.columns and df[c].std() > 1e-12]
    y = np.ascontiguousarray(df[target_col].to_numpy(dtype=np.float64))
    n = len(y)
    clusters = df[cluster_col].to_numpy()
    unique_clusters = np.unique(clusters)
    g_clusters = len(unique_clusters)

    # Standardize features
    X_raw = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in valid_features])
    stds = np.std(X_raw, axis=0)
    X_norm = (X_raw - np.mean(X_raw, axis=0)) / stds
    X = np.column_stack([np.ones(n, dtype=np.float64), X_norm])
    k_params = X.shape[1]

    # OLS coefficient fit via SVD pseudo-inverse
    XtX_inv = np.linalg.pinv(np.dot(X.T, X))
    beta = np.dot(XtX_inv, np.dot(X.T, y))
    residuals = y - np.dot(X, beta)

    # Cluster-robust sandwich covariance:
    meat = np.zeros((k_params, k_params), dtype=np.float64)
    for c in unique_clusters:
        mask = (clusters == c)
        X_g = np.ascontiguousarray(X[mask], dtype=np.float64)
        e_g = np.ascontiguousarray(residuals[mask], dtype=np.float64)
        score_g = np.dot(X_g.T, e_g)
        meat += np.outer(score_g, score_g)

    df_correction = (g_clusters / max(1, g_clusters - 1)) * ((n - 1) / max(1, n - k_params))
    vcov_cluster = np.dot(XtX_inv, np.dot(meat, XtX_inv)) * df_correction
    se_cluster = np.sqrt(np.maximum(0.0, np.diag(vcov_cluster)))

    coef_names = ["Intercept"] + valid_features
    results_dict: Dict[str, Dict[str, float]] = {}

    for i, name in enumerate(coef_names):
        b = float(beta[i])
        se = float(se_cluster[i])
        t_stat = float(b / (se + 1e-12))
        p_val = float(2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=max(1, g_clusters - 1))))
        results_dict[name] = {
            "coef": b,
            "cluster_robust_se": se,
            "t_stat": t_stat,
            "p_val": p_val,
        }

    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2 = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    return {
        "n_samples": n,
        "n_clusters": g_clusters,
        "r2": r2,
        "coefficients": results_dict,
    }
