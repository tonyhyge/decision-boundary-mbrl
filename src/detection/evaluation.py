"""
Evaluation Metrics and Clustered Bootstrap Inference for Stage 3A Risk Detectors.

Endpoints:
  - Primary: Brier Score and Expected Calibration Error (ECE).
  - Ranking: AUPRC (primary ranking), AUROC (secondary ranking).
  - Operational / Selective Risk-Coverage: Recall@10%, Recall@20%, Precision@10%, Precision@20%.
  - Inference: Trajectory-level clustered block bootstrap (2000 replicates) with 95% BCIs.
"""
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)
import scipy.stats as stats

from src.detection.calibrated_detectors import BaseDetector
from src.detection.signal_extractor import FeatureNormalizer


def compute_detector_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute all primary, ranking, and operational endpoints for a single detector.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = len(y_true)
    num_pos = int(np.sum(y_true))

    # Base rate
    base_rate = float(num_pos / max(1, n))

    # 1. Primary Endpoint: Brier Score
    brier = float(np.mean((y_prob - y_true) ** 2))

    # 2. Expected Calibration Error (ECE)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = (y_prob >= bin_edges[b]) & (y_prob < bin_edges[b + 1] if b < n_bins - 1 else y_prob <= bin_edges[b + 1])
        if np.any(in_bin):
            bin_acc = float(np.mean(y_true[in_bin]))
            bin_conf = float(np.mean(y_prob[in_bin]))
            bin_weight = float(np.sum(in_bin) / n)
            ece += bin_weight * abs(bin_acc - bin_conf)

    # 3. Ranking Endpoints: AUPRC and AUROC
    if num_pos == 0 or num_pos == n:
        auprc = base_rate
        auroc = 0.5
    else:
        try:
            auprc = float(average_precision_score(y_true, y_score))
        except Exception:
            auprc = base_rate

        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auroc = 0.5

    # 4. Operational Endpoints: Recall@K and Precision@K
    def get_at_k(k_pct: float) -> Tuple[float, float]:
        k_count = max(1, int(np.ceil(k_pct * n)))
        # Sort by descending score
        order = np.argsort(-y_score)
        top_k_labels = y_true[order[:k_count]]
        pos_in_top_k = int(np.sum(top_k_labels))

        rec = float(pos_in_top_k / max(1, num_pos))
        prec = float(pos_in_top_k / k_count)
        return rec, prec

    rec_10, prec_10 = get_at_k(0.10)
    rec_20, prec_20 = get_at_k(0.20)
    rec_30, prec_30 = get_at_k(0.30)

    return {
        "n_samples": n,
        "n_positives": num_pos,
        "prevalence": base_rate,
        "brier": brier,
        "ece": float(ece),
        "auprc": auprc,
        "auroc": auroc,
        "recall_at_10": rec_10,
        "precision_at_10": prec_10,
        "recall_at_20": rec_20,
        "precision_at_20": prec_20,
        "recall_at_30": rec_30,
        "precision_at_30": prec_30,
    }


def evaluate_all_detectors_on_df(
    detectors: Dict[str, BaseDetector],
    df_test: pd.DataFrame,
    normalizer: FeatureNormalizer,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate all detectors on a test dataframe and return point metrics.
    """
    y_true = df_test["z_flip"].to_numpy(dtype=np.int32)
    results = {}
    for name, detector in detectors.items():
        y_prob = detector.predict_proba(df_test, normalizer)
        y_score = detector.predict_score(df_test, normalizer)
        results[name] = compute_detector_metrics(y_true, y_prob, y_score)
    return results


def run_clustered_paired_bootstrap(
    detectors: Dict[str, BaseDetector],
    df_test: pd.DataFrame,
    normalizer: FeatureNormalizer,
    comparator_name: str,
    cluster_col: str = "traj_id",
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Perform trajectory-level clustered block bootstrap to evaluate paired differences
    relative to the pre-registered comparator baseline:
      Delta_Brier = Brier_detector - Brier_comparator
      Delta_AUPRC = AUPRC_detector - AUPRC_comparator
      Delta_AUROC = AUROC_detector - AUROC_comparator
      Delta_Recall@10 = Recall@10_detector - Recall@10_comparator
      Delta_Recall@20 = Recall@20_detector - Recall@20_comparator
    """
    rng = np.random.default_rng(seed)
    clusters = np.unique(df_test[cluster_col])
    n_clusters = len(clusters)

    # Pre-group dataframe by cluster for fast sampling
    cluster_dfs = {c: df_test[df_test[cluster_col] == c] for c in clusters}

    # Precompute predictions across all detectors on entire test set
    preds = {}
    for name, det in detectors.items():
        prob = det.predict_proba(df_test, normalizer)
        score = det.predict_score(df_test, normalizer)
        preds[name] = {"prob": prob, "score": score}

    y_true_all = df_test["z_flip"].to_numpy(dtype=np.int32)

    # Point metrics on observed sample
    obs_metrics = evaluate_all_detectors_on_df(detectors, df_test, normalizer)

    # Bootstrap storage
    boot_records = {name: [] for name in detectors.keys()}
    boot_paired_diffs = {name: [] for name in detectors.keys()}

    for b in range(n_boot):
        # Sample clusters with replacement
        sampled_clusters = rng.choice(clusters, size=n_clusters, replace=True)
        # Gather indices of selected clusters
        sampled_indices = []
        for c in sampled_clusters:
            idx = df_test.index[df_test[cluster_col] == c].tolist()
            sampled_indices.extend(idx)

        sampled_indices = np.array(sampled_indices, dtype=np.int64)
        b_y_true = y_true_all[sampled_indices]

        # Evaluate each detector on this identical resampled cluster set
        b_res = {}
        for name in detectors.keys():
            b_prob = preds[name]["prob"][sampled_indices]
            b_score = preds[name]["score"][sampled_indices]
            m = compute_detector_metrics(b_y_true, b_prob, b_score)
            b_res[name] = m
            boot_records[name].append(m)

        # Paired differences relative to comparator
        comp_m = b_res[comparator_name]
        for name in detectors.keys():
            curr_m = b_res[name]
            paired_diff = {
                "delta_brier": curr_m["brier"] - comp_m["brier"],
                "delta_auprc": curr_m["auprc"] - comp_m["auprc"],
                "delta_auroc": curr_m["auroc"] - comp_m["auroc"],
                "delta_recall_at_10": curr_m["recall_at_10"] - comp_m["recall_at_10"],
                "delta_recall_at_20": curr_m["recall_at_20"] - comp_m["recall_at_20"],
                "delta_ece": curr_m["ece"] - comp_m["ece"],
            }
            boot_paired_diffs[name].append(paired_diff)

    # Summarize bootstrap distributions
    summary = {}
    for name in detectors.keys():
        obs_m = obs_metrics[name]
        comp_obs = obs_metrics[comparator_name]

        # Observed paired differences
        obs_d_brier = obs_m["brier"] - comp_obs["brier"]
        obs_d_auprc = obs_m["auprc"] - comp_obs["auprc"]
        obs_d_auroc = obs_m["auroc"] - comp_obs["auroc"]
        obs_d_rec10 = obs_m["recall_at_10"] - comp_obs["recall_at_10"]
        obs_d_rec20 = obs_m["recall_at_20"] - comp_obs["recall_at_20"]

        # Bootstrap distributions of paired differences
        d_brier_arr = np.array([r["delta_brier"] for r in boot_paired_diffs[name]])
        d_auprc_arr = np.array([r["delta_auprc"] for r in boot_paired_diffs[name]])
        d_auroc_arr = np.array([r["delta_auroc"] for r in boot_paired_diffs[name]])
        d_rec10_arr = np.array([r["delta_recall_at_10"] for r in boot_paired_diffs[name]])
        d_rec20_arr = np.array([r["delta_recall_at_20"] for r in boot_paired_diffs[name]])

        # 95% BCIs for individual metrics
        brier_arr = np.array([r["brier"] for r in boot_records[name]])
        auprc_arr = np.array([r["auprc"] for r in boot_records[name]])
        auroc_arr = np.array([r["auroc"] for r in boot_records[name]])

        def get_bci(arr: np.ndarray) -> List[float]:
            return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

        def get_two_sided_p(arr: np.ndarray, null_val: float = 0.0) -> float:
            frac_le = float(np.mean(arr <= null_val))
            return float(min(1.0, 2.0 * min(frac_le, 1.0 - frac_le)))

        summary[name] = {
            "name": name,
            "point_metrics": obs_m,
            "brier_95_bci": get_bci(brier_arr),
            "auprc_95_bci": get_bci(auprc_arr),
            "auroc_95_bci": get_bci(auroc_arr),
            "paired_to_comparator": {
                "comparator_name": comparator_name,
                "delta_brier": {
                    "observed": float(obs_d_brier),
                    "mean_boot": float(np.mean(d_brier_arr)),
                    "ci_95": get_bci(d_brier_arr),
                    "p_val": get_two_sided_p(d_brier_arr, 0.0),
                },
                "delta_auprc": {
                    "observed": float(obs_d_auprc),
                    "mean_boot": float(np.mean(d_auprc_arr)),
                    "ci_95": get_bci(d_auprc_arr),
                    "p_val": get_two_sided_p(d_auprc_arr, 0.0),
                },
                "delta_auroc": {
                    "observed": float(obs_d_auroc),
                    "mean_boot": float(np.mean(d_auroc_arr)),
                    "ci_95": get_bci(d_auroc_arr),
                    "p_val": get_two_sided_p(d_auroc_arr, 0.0),
                },
                "delta_recall_at_10": {
                    "observed": float(obs_d_rec10),
                    "mean_boot": float(np.mean(d_rec10_arr)),
                    "ci_95": get_bci(d_rec10_arr),
                    "p_val": get_two_sided_p(d_rec10_arr, 0.0),
                },
                "delta_recall_at_20": {
                    "observed": float(obs_d_rec20),
                    "mean_boot": float(np.mean(d_rec20_arr)),
                    "ci_95": get_bci(d_rec20_arr),
                    "p_val": get_two_sided_p(d_rec20_arr, 0.0),
                },
            },
        }

    return summary
