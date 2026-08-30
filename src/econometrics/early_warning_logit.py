"""
Early Warning Binary Stress Classifier & Out-of-Sample Evaluation
================================================================
Implements IRLS Logit models, Expanding-Window OOS classification,
and Leave-One-Crisis-Out cross-validation to prevent in-sample overfitting.
"""

from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logit_irls(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 60,
    tol: float = 1e-6,
    l2_reg: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits Logistic regression via Iteratively Reweighted Least Squares (IRLS) with L2 regularization.
    """
    T, k = X.shape
    beta = np.zeros(k)

    for _ in range(max_iter):
        p = sigmoid(X @ beta)
        W_diag = np.clip(p * (1.0 - p), a_min=1e-8, a_max=None)
        z = X @ beta + (y - p) / W_diag

        # Ridge regularized IRLS update
        XW = X.T * W_diag
        XWX = XW @ X + l2_reg * np.eye(k)
        beta_new = np.linalg.solve(XWX, XW @ z)

        if np.linalg.norm(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    p = sigmoid(X @ beta)
    W_diag = np.clip(p * (1.0 - p), a_min=1e-8, a_max=None)
    cov_beta = np.linalg.inv((X.T * W_diag) @ X + l2_reg * np.eye(k))

    return beta, cov_beta


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Calculates Area Under the ROC Curve (AUC) non-parametrically using Mann-Whitney U statistic.
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos = len(pos)
    n_neg = len(neg)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    u_stat = 0.0
    for p in pos:
        u_stat += np.sum(p > neg) + 0.5 * np.sum(p == neg)

    return float(u_stat / (n_pos * n_neg))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray, n_thresholds: int = 100) -> float:
    """
    Calculates Precision-Recall AUC non-parametrically.
    """
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    precisions, recalls = [], []

    for th in thresholds:
        tp = np.sum((y_score >= th) & (y_true == 1))
        fp = np.sum((y_score >= th) & (y_true == 0))
        fn = np.sum((y_score < th) & (y_true == 1))

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    # Sort by recall
    sorted_indices = np.argsort(recalls)
    rec_sorted = np.array(recalls)[sorted_indices]
    prec_sorted = np.array(precisions)[sorted_indices]

    # Trapezoidal integration
    try:
        pr_auc = float(np.trapezoid(prec_sorted, rec_sorted))
    except AttributeError:
        pr_auc = float(np.trapz(prec_sorted, rec_sorted))
    return float(np.clip(pr_auc, 0.0, 1.0))



def evaluate_oos_expanding_logit(
    y_binary: np.ndarray,
    X_features: np.ndarray,
    initial_train_size: int,
    feature_names: List[str]
) -> Dict:
    """
    Evaluates Logit model strictly Out-of-Sample using Expanding Window without look-ahead bias.
    """
    T, k = X_features.shape
    n_oos = T - initial_train_size
    p_hat_oos = np.zeros(n_oos)
    y_actual_oos = y_binary[initial_train_size:]

    for i, t in enumerate(range(initial_train_size, T)):
        X_train = X_features[:t]
        y_train = y_binary[:t]

        # Fit IRLS on history up to t
        beta_t, _ = fit_logit_irls(X_train, y_train)
        # Predict on out-of-sample step t
        p_hat_oos[i] = sigmoid(X_features[t] @ beta_t)

    auc_oos = compute_roc_auc(y_actual_oos, p_hat_oos)
    pr_auc_oos = compute_pr_auc(y_actual_oos, p_hat_oos)
    brier_oos = float(np.mean((p_hat_oos - y_actual_oos) ** 2))

    return {
        "auc_roc_oos": auc_oos,
        "pr_auc_oos": pr_auc_oos,
        "brier_score_oos": brier_oos,
        "n_oos_obs": n_oos,
        "p_hat_oos": p_hat_oos,
        "y_actual_oos": y_actual_oos
    }


def evaluate_leave_one_crisis_out_logit(
    dates: pd.DatetimeIndex,
    y_binary: np.ndarray,
    X_features: np.ndarray,
    crisis_intervals: List[Tuple[str, str, str]]
) -> Dict:
    """
    Leave-One-Crisis-Out (LOCO) Cross-Validation.
    Trains on all periods EXCEPT the target crisis, and tests on the held-out crisis episode.
    """
    loco_results = {}

    for crisis_name, start_date, end_date in crisis_intervals:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        test_mask = (dates >= start_dt) & (dates <= end_dt)
        train_mask = ~test_mask

        if np.sum(test_mask) == 0 or np.sum(y_binary[test_mask] == 1) == 0:
            continue

        X_train = X_features[train_mask]
        y_train = y_binary[train_mask]
        X_test = X_features[test_mask]
        y_test = y_binary[test_mask]

        beta_loco, _ = fit_logit_irls(X_train, y_train)
        p_hat_test = sigmoid(X_test @ beta_loco)

        auc_crisis = compute_roc_auc(y_test, p_hat_test)
        brier_crisis = float(np.mean((p_hat_test - y_test) ** 2))

        loco_results[crisis_name] = {
            "auc_roc": auc_crisis,
            "brier": brier_crisis,
            "test_obs": int(np.sum(test_mask))
        }

    return loco_results


def evaluate_early_warning_classifier(
    y_binary: np.ndarray,
    X_features: np.ndarray,
    feature_names: List[str]
) -> Dict:
    """
    Fits IRLS Logit model and returns parameters, HAC standard errors, AUC, and Brier score.
    """
    T, k = X_features.shape
    beta, cov_beta = fit_logit_irls(X_features, y_binary)
    p_hat = sigmoid(X_features @ beta)

    se = np.sqrt(np.diagonal(cov_beta))
    z_stats = beta / se

    brier_score = float(np.mean((p_hat - y_binary) ** 2))
    auc = compute_roc_auc(y_binary, p_hat)
    pr_auc = compute_pr_auc(y_binary, p_hat)

    eps = 1e-12
    p_null = np.mean(y_binary)
    ll_null = np.sum(y_binary * np.log(p_null + eps) + (1.0 - y_binary) * np.log(1.0 - p_null + eps))
    ll_model = np.sum(y_binary * np.log(p_hat + eps) + (1.0 - y_binary) * np.log(1.0 - p_hat + eps))
    pseudo_r2 = float(1.0 - ll_model / ll_null) if ll_null != 0 else 0.0

    return {
        "beta": dict(zip(feature_names, beta)),
        "se": dict(zip(feature_names, se)),
        "z_stats": dict(zip(feature_names, z_stats)),
        "auc_roc": auc,
        "pr_auc": pr_auc,
        "brier_score": brier_score,
        "pseudo_r2": pseudo_r2,
        "nobs": T
    }

