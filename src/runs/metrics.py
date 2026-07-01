from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
    root_mean_squared_error,
)

from src.helpers import _encode_labels, class_ratios


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray
) -> dict[str, float]:

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    n = len(y_true)
    prior = float(np.mean(y_train))

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))
    mae_prior = float(mean_absolute_error(y_true, np.full(n, prior)))
    rmse_prior = float(root_mean_squared_error(y_true, np.full(n, prior)))

    return {
        "mean_absolute_error": mae,
        "mean_prior_absolute_error": mae_prior,
        "number_of_instances": float(n),
        "root_mean_squared_error": rmse,
        "root_mean_prior_squared_error": rmse_prior,
        "relative_absolute_error": (mae / mae_prior) if mae_prior > 0 else 0.0,
        "root_relative_squared_error": (rmse / rmse_prior) if rmse_prior > 0 else 0.0,
    }


def kb_relative_information(
    y_true: np.ndarray, conf: np.ndarray, prior: np.ndarray
) -> float:
    eps = 1e-12
    prior = np.asarray(prior, dtype=float)
    if not (prior > 0).any():
        return 0.0
    h_prior = float(entropy(prior, base=2))
    if h_prior == 0.0:
        return 0.0

    y_true = np.asarray(y_true, dtype=int)
    conf = np.asarray(conf, dtype=float)

    p_true_pred = conf[np.arange(len(y_true)), y_true]
    p_true_pred = np.clip(p_true_pred, eps, 1.0)
    p_true_prior = prior[y_true]
    p_true_prior = np.clip(p_true_prior, eps, 1.0)

    info_pred = -np.log2(p_true_pred)
    info_prior = -np.log2(p_true_prior)
    kb = float((info_prior - info_pred).mean())
    return kb / h_prior


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    class_names: list[str],
    y_train_labels: Iterable,
    cost_matrix: Optional[np.ndarray] = None,
) -> dict:
    y_true = _encode_labels(y_true, class_names)
    y_pred = _encode_labels(y_pred, class_names)
    conf = np.asarray(conf, dtype=float)
    y_train_arr = _encode_labels(y_train_labels, class_names)
    n = len(y_true)
    num_classes = len(class_names)
    classes = list(range(num_classes))

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # Probability-vector MAE/RMSE (Weka's definition for classification).
    onehot = np.eye(num_classes)[y_true]
    mae = float(np.abs(conf - onehot).sum(axis=1).mean())
    rmse = float(np.sqrt(np.square(conf - onehot).sum(axis=1).mean()))

    prior = class_ratios(y_train_arr, num_classes)
    mae_prior = float(np.abs(onehot - prior).sum(axis=1).mean())
    rmse_prior = float(np.sqrt(np.square(onehot - prior).sum(axis=1).mean()))
    h_prior = float(entropy(prior, base=2)) if prior.sum() > 0 else 0.0

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, average=None, zero_division=0
    )
    instances_per_class = cm.sum(axis=1)
    total_inst = int(instances_per_class.sum())
    weights = (
        instances_per_class / total_inst if total_inst > 0 else np.zeros(num_classes)
    )

    try:
        auroc_per_class = roc_auc_score(
            np.eye(num_classes)[y_true],
            conf,
            average=None,
            multi_class="ovr",
            labels=classes,
        )
        auroc_per_class = [float(x) for x in auroc_per_class]
        auroc_global = (
            float(np.average(auroc_per_class, weights=weights))
            if weights.sum() > 0
            else None
        )
    except (ValueError, IndexError):
        auroc_per_class = [None] * num_classes
        auroc_global = None

    result: dict = {
        "mean_absolute_error": mae,
        "mean_prior_absolute_error": mae_prior,
        "root_mean_squared_error": rmse,
        "root_mean_prior_squared_error": rmse_prior,
        "relative_absolute_error": (mae / mae_prior) if mae_prior > 0 else 0.0,
        "root_relative_squared_error": (rmse / rmse_prior) if rmse_prior > 0 else 0.0,
        "prior_entropy": h_prior,
        "kb_relative_information_score": kb_relative_information(y_true, conf, prior),
        "predictive_accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "precision": (
            float(np.average(p, weights=weights)) if weights.sum() > 0 else 0.0
        ),
        "weighted_recall": (
            float(np.average(r, weights=weights)) if weights.sum() > 0 else 0.0
        ),
        "unweighted_recall": float(r.mean()),
        "f_measure": (
            float(np.average(f, weights=weights)) if weights.sum() > 0 else 0.0
        ),
        "number_of_instances": float(n),
        "confusion_matrix": cm.tolist(),
        "_per_class": {
            "precision": p.tolist(),
            "recall": r.tolist(),
            "f_measure": f.tolist(),
            "auroc": auroc_per_class,
            "instances_per_class": instances_per_class.tolist(),
        },
    }
    if auroc_global is not None:
        result["area_under_roc_curve"] = auroc_global

    if cost_matrix is not None:
        cm_cost = np.asarray(cost_matrix, dtype=float)
        total_cost = float((cm * cm_cost).sum())
        result["total_cost"] = total_cost
        result["average_cost"] = total_cost / n if n > 0 else 0.0

    return result
