"""Python port of OpenML's run evaluation engine.

Mirrors the metric-computation core of the Java EvaluationEngine
(``EvaluateRun.java``, ``EvaluateBatchPredictions.java``,
``EvaluateStreamPredictions.java``, ``EvaluateSurvivalAnalysisPredictions.java``,
``OpenMLEvaluation.java``, ``Output.java``, ``InstancesHelper.java``,
``FoldsPredictionCounter.java``).

Out of scope (will be added in a later phase): the upload pipeline
(``RunEvaluation`` POST, ``RunTrace`` upload, retry logic), trace parsing,
run-description parsing, and the consistency check against user-defined
measures. Everything here assumes the caller has already loaded the dataset,
splits, and predictions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional

import arff
import numpy as np
import pandas as pd
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

from src.folds import EstimationProcedureType

# ============================================================================
# Constants
# ============================================================================

EVALUATION_ENGINE_ID = 1
SUPPORTED_TASK_TYPES_EVALUATION = {1, 2, 3, 4, 5, 6, 7, 8}


class TaskType(Enum):
    """Mirrors the Java ``org.openml.webapplication.evaluate.TaskType`` enum."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    LEARNINGCURVE = "learning_curve"
    TESTTHENTRAIN = "test_then_train"


# OpenML task_type_id -> evaluator kind. ``None`` means "separate evaluator"
# (currently only task 7, survival analysis, which is handled by
# ``evaluate_survival`` rather than ``evaluate_batch``).
TASK_TYPE_ID_TO_TASK_TYPE: dict[int, Optional[TaskType]] = {
    1: TaskType.CLASSIFICATION,
    2: TaskType.REGRESSION,
    3: TaskType.LEARNINGCURVE,
    4: TaskType.TESTTHENTRAIN,
    5: TaskType.CLASSIFICATION,
    6: TaskType.CLASSIFICATION,
    7: None,
    8: TaskType.CLASSIFICATION,
}


# ============================================================================
# Result types
# ============================================================================


@dataclass
class EvaluationScore:
    """One computed metric. Mirrors ``org.openml.apiconnector.xml.EvaluationScore``."""

    function: str
    value: Optional[float] = None
    stdev: Optional[float] = None
    array: Optional[list] = None
    repeat: Optional[int] = None
    fold: Optional[int] = None
    sample: Optional[int] = None
    sample_size: Optional[int] = None


@dataclass
class RunEvaluation:
    """Aggregated result of evaluating one run. Mirrors ``RunEvaluation``."""

    run_id: Optional[int] = None
    evaluation_engine_id: int = EVALUATION_ENGINE_ID
    scores: list[EvaluationScore] = field(default_factory=list)
    error: Optional[str] = None
    warning: Optional[str] = None

    def add_scores(self, scores: Iterable[EvaluationScore]) -> None:
        self.scores.extend(scores)


# ============================================================================
# InstancesHelper ports
# ============================================================================


def get_row_index(name: str, columns: Iterable[str]) -> int:
    """Return the 0-based index of ``name`` in ``columns``, or -1 if absent.

    Mirrors ``InstancesHelper.getRowIndex(String, Instances)``.
    """
    cols = list(columns)
    return cols.index(name) if name in cols else -1


def get_row_index_multi(names: Iterable[str], columns: Iterable[str]) -> int:
    """Return the index of the first name in ``names`` present in ``columns``.

    Raises ``ValueError`` if none of the names are found. Mirrors
    ``InstancesHelper.getRowIndex(String[], Instances)``.
    """
    cols = list(columns)
    for name in names:
        if name in cols:
            return cols.index(name)
    raise ValueError(
        f"ARFF file contains none of the specified attributes: {list(names)}"
    )


def to_prob_dist(d: Iterable[float]) -> np.ndarray:
    """Normalize a vector to a probability distribution.

    Replicates ``InstancesHelper.toProbDist`` exactly:
      * If any element is +/-inf, the first such element becomes 1.0 and the
        rest become 0.
      * If all (non-nan) elements sum to 0, the first element becomes 1.0.
      * Otherwise, divide each non-nan element by the total. NaNs become 0.
    """
    arr = np.asarray(d, dtype=float)
    result = np.zeros_like(arr)

    inf_mask = np.isinf(arr)
    if inf_mask.any():
        result[np.argmax(inf_mask)] = 1.0
        return result

    nan_mask = np.isnan(arr)
    total = float(np.sum(arr[~nan_mask]))

    if total == 0.0:
        result[0] = 1.0
        return result

    for i in range(len(arr)):
        if nan_mask[i]:
            result[i] = 0.0
        elif total > 0.0:
            result[i] = arr[i] / total
        else:
            result[i] = arr[i]
    return result


def prediction_to_confidences(
    confidence_values: Iterable[float],
    prediction_value: object,
    class_names: list[str],
) -> np.ndarray:
    """Build a confidence vector from a prediction row.

    Mirrors ``InstancesHelper.predictionToConfidences``. Raises ``ValueError``
    on missing values. If every confidence is 0, falls back to placing all
    mass on the predicted class.

    ``prediction_value`` may be either a class label (string) or a 0-based
    integer class index — both are accepted, matching how Weka's
    ``Instance.value()`` returns either form depending on attribute type.
    """
    conf = np.asarray(confidence_values, dtype=float)
    if np.isnan(conf).any():
        raise ValueError(
            "Prediction file contains missing values for a confidence attribute."
        )
    if not (conf > 0).any():
        label_to_idx = {c: i for i, c in enumerate(class_names)}
        if isinstance(prediction_value, str):
            idx = label_to_idx[prediction_value]
        else:
            idx = int(prediction_value)
        conf = conf.copy()
        conf[idx] = 1.0
    return conf


def class_counts(y: Iterable, num_classes: int) -> np.ndarray:
    """Bin integer-coded labels into a length-``num_classes`` count vector."""
    counts = np.zeros(num_classes, dtype=int)
    for c in y:
        counts[int(c)] += 1
    return counts


def class_ratios(y: Iterable, num_classes: int) -> np.ndarray:
    """Class frequency ratios. Mirrors ``InstancesHelper.classRatios``."""
    counts = class_counts(y, num_classes)
    total = counts.sum()
    if total == 0:
        return np.zeros(num_classes, dtype=float)
    return counts / total


def _encode_labels(values, class_names):
    """Map an iterable of string/integer labels to 0-based integer codes."""
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    return np.array([label_to_idx[v] if isinstance(v, str) else int(v) for v in values])


# ============================================================================
# FoldsPredictionCounter
# ============================================================================


class FoldsPredictionCounter:
    """Validates that a predictions file covers exactly the expected rows.

    Full port of ``FoldsPredictionCounter.java``. The constructor reads the
    splits table and records, per ``(repeat, fold, sample)``, the row ids
    marked as ``TEST`` (the rows we expect predictions for) and the count of
    ``TRAIN`` rows (used downstream to populate ``sample_size`` on user-defined
    measures). As predictions arrive via ``add_prediction``, they land in an
    "actual" structure; ``check`` confirms the two match.
    """

    def __init__(
        self,
        splits: pd.DataFrame,
        type_name: str = "TEST",
        shadow_type: str = "TRAIN",
    ) -> None:
        cols = list(splits.columns)

        self._col_type = get_row_index("type", cols)
        self._col_rowid = get_row_index_multi(["rowid", "row_id"], cols)
        self._col_repeat = get_row_index_multi(["repeat", "repeat_nr"], cols)
        self._col_fold = get_row_index_multi(["fold", "fold_nr"], cols)
        try:
            self._col_sample = get_row_index_multi(["sample", "sample_nr"], cols)
        except ValueError:
            self._col_sample = -1

        # Dimensions: max + 1, matching Weka's ``attributeStats(...).numericStats.max + 1``
        self._num_repeats = (
            int(splits.iloc[:, self._col_repeat].max()) + 1
            if "repeat" in cols or "repeat_nr" in cols
            else 1
        )
        self._num_folds = (
            int(splits.iloc[:, self._col_fold].max()) + 1
            if "fold" in cols or "fold_nr" in cols
            else 1
        )
        self._num_samples = (
            int(splits.iloc[:, self._col_sample].max()) + 1
            if self._col_sample >= 0
            else 1
        )

        self.expected = [
            [[[] for _ in range(self._num_samples)] for _ in range(self._num_folds)]
            for _ in range(self._num_repeats)
        ]
        self.actual = [
            [[[] for _ in range(self._num_samples)] for _ in range(self._num_folds)]
            for _ in range(self._num_repeats)
        ]
        self.shadow_type_size = np.zeros(
            (self._num_repeats, self._num_folds, self._num_samples), dtype=int
        )
        self.expected_total = 0
        self.error_message = ""

        for _, row in splits.iterrows():
            row_type = row.iloc[self._col_type]
            repeat = int(row.iloc[self._col_repeat])
            fold = int(row.iloc[self._col_fold])
            sample = int(row.iloc[self._col_sample]) if self._col_sample >= 0 else 0
            rowid = int(row.iloc[self._col_rowid])

            if row_type == type_name:
                self.expected[repeat][fold][sample].append(rowid)
                self.expected_total += 1
            elif row_type == shadow_type:
                self.shadow_type_size[repeat][fold][sample] += 1

        for i in range(self._num_repeats):
            for j in range(self._num_folds):
                for k in range(self._num_samples):
                    self.expected[i][j][k].sort()

    def add_prediction(self, repeat: int, fold: int, sample: int, rowid: int) -> None:
        if repeat >= len(self.actual):
            raise RuntimeError(f"Repeat #{repeat} not defined by task.")
        if fold >= len(self.actual[repeat]):
            raise RuntimeError(f"Fold #{fold} not defined by task.")
        self.actual[repeat][fold][sample].append(rowid)

    def check(self) -> bool:
        for i in range(self._num_repeats):
            for j in range(self._num_folds):
                for k in range(self._num_samples):
                    self.actual[i][j][k].sort()
                    if self.actual[i][j][k] != self.expected[i][j][k]:
                        self.error_message = (
                            f"Repeat {i} fold {j} sample {k} expected predictions "
                            f"with row id's {self.expected[i][j][k]}, but got "
                            f"predictions with row id's {self.actual[i][j][k]}"
                        )
                        return False
        return True

    def get_expected_rowids(self, i: int, j: int, k: int) -> list[int]:
        return list(self.expected[i][j][k])

    def get_shadow_type_size(self, i: int, j: int, k: int) -> int:
        return int(self.shadow_type_size[i][j][k])

    def get_error_message(self) -> str:
        return self.error_message

    @property
    def repeats(self) -> int:
        return self._num_repeats

    @property
    def folds(self) -> int:
        return self._num_folds

    @property
    def samples(self) -> int:
        return self._num_samples

    def get_expected_total(self) -> int:
        return self.expected_total


# ============================================================================
# Metric functions (Output.java + OpenMLEvaluation.java)
# ============================================================================


def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray
) -> dict[str, float]:
    """Port of ``Output.evaluatorToMap`` regression branch.

    ``y_train`` is the full training-set target vector; the "prior" model
    predicts ``mean(y_train)`` for every instance, and the relative metrics
    normalize by that baseline. Weka returns ``relative_absolute_error`` and
    ``root_relative_squared_error`` as percentages; we mirror the Java code's
    ``/ 100`` so the values land in [0, ∞) as fractions.
    """
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
    """Kononenko-Bratko relative information score, divided by 100.

    Mirrors ``Evaluation.KBRelativeInformation() / 100``. The exact Weka
    formula clips probabilities and applies sign conventions that aren't
    fully documented; this implementation uses the textbook form and may
    differ slightly from Weka on edge cases. Validate against a known run
    before relying on bit-parity.
    """
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
    """Port of ``Output.evaluatorToMap`` classification branch.

    ``y_pred`` may be string labels or integer codes; ``conf`` is an
    ``(N, num_classes)`` array of per-instance confidence probabilities.
    ``y_train_labels`` is the full training-set target vector (used to
    compute the class prior — Weka's prior is the training distribution,
    not the test distribution).
    """
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


# ============================================================================
# Evaluator pipelines
# ============================================================================


def _resolve_class_names(dataset_df: pd.DataFrame, target_feature: str) -> list[str]:
    col = dataset_df[target_feature]
    if isinstance(col.dtype, pd.CategoricalDtype):
        return list(col.cat.categories)
    if pd.api.types.is_numeric_dtype(col):
        return []
    return sorted(col.dropna().unique().tolist())


def evaluate_batch(
    dataset_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    target_feature: str,
    task_type: TaskType,
    cost_matrix: Optional[np.ndarray] = None,
    estimation_procedure_type: Optional[EstimationProcedureType] = None,
) -> tuple[list[EvaluationScore], list[EvaluationScore], FoldsPredictionCounter]:
    """Port of ``EvaluateBatchPredictions``.

    Returns ``(per_cell_scores, global_scores, prediction_counter)``.
    Per-cell scores are suppressed for ``LEAVEONEOUT`` and
    ``TESTONTRAININGDATA`` procedures (matching Output.java:245). Global
    scores carry a standard deviation across folds (computed on the last
    sample of each fold for learning curves — matching the
    ``measureGlobalScore`` rule at EvaluateBatchPredictions.java:189).
    """
    if target_feature not in dataset_df.columns:
        raise ValueError(f"Class attribute ({target_feature}) not found")

    class_names = (
        [] if task_type is TaskType.REGRESSION else _resolve_class_names(dataset_df, target_feature)
    )
    num_classes = len(class_names)

    pc = FoldsPredictionCounter(splits_df)

    pred_cols = list(predictions_df.columns)
    col_rowid = get_row_index("row_id", pred_cols)
    if col_rowid < 0:
        raise ValueError("Predictions are missing the 'row_id' column")
    col_repeat = get_row_index_multi(["repeat", "repeat_nr"], pred_cols)
    col_fold = get_row_index_multi(["fold", "fold_nr"], pred_cols)
    col_sample = (
        get_row_index_multi(["sample", "sample_nr"], pred_cols)
        if task_type is TaskType.LEARNINGCURVE
        else -1
    )
    col_prediction = get_row_index("prediction", pred_cols)
    if col_prediction < 0:
        raise ValueError("Predictions are missing the 'prediction' column")

    confidence_cols: dict[str, int] = {}
    if task_type is not TaskType.REGRESSION:
        for cls in class_names:
            col_name = f"confidence.{cls}"
            if col_name not in pred_cols:
                raise ValueError(
                    f"Attribute {col_name} not found among predictions."
                )
            confidence_cols[cls] = get_row_index(col_name, pred_cols)

    target_values = dataset_df[target_feature].to_numpy()
    n_dataset = len(dataset_df)
    last_sample = pc.samples - 1

    if task_type is TaskType.REGRESSION:
        y_train = target_values.astype(float)
    else:
        y_train = _encode_labels(target_values, class_names)

    label_to_idx = {c: i for i, c in enumerate(class_names)}

    cells: dict[tuple[int, int, int], dict] = {}
    global_rows: list[dict] = []

    for _, pred_row in predictions_df.iterrows():
        repeat = int(pred_row.iloc[col_repeat])
        fold = int(pred_row.iloc[col_fold])
        sample = int(pred_row.iloc[col_sample]) if col_sample >= 0 else 0
        rowid = int(pred_row.iloc[col_rowid])

        pc.add_prediction(repeat, fold, sample, rowid)
        if rowid >= n_dataset:
            raise RuntimeError(
                f"Making a prediction for row_id {rowid} (0-based) while "
                f"dataset has only {n_dataset} instances."
            )

        cell_key = (repeat, fold, sample)
        cell = cells.setdefault(
            cell_key, {"y_true": [], "y_pred": [], "conf": []}
        )

        y_true_i = y_train[rowid]
        cell["y_true"].append(y_true_i)

        measure_global = not (
            task_type is TaskType.LEARNINGCURVE and sample != last_sample
        )

        if task_type is TaskType.REGRESSION:
            y_pred_i = float(pred_row.iloc[col_prediction])
            cell["y_pred"].append(y_pred_i)
            if measure_global:
                global_rows.append({"y_true": y_true_i, "y_pred": y_pred_i})
        else:
            pred_value = pred_row.iloc[col_prediction]
            conf_vec = np.array(
                [float(pred_row.iloc[confidence_cols[c]]) for c in class_names]
            )
            conf_vec = prediction_to_confidences(conf_vec, pred_value, class_names)
            y_pred_code = (
                label_to_idx[pred_value] if isinstance(pred_value, str) else int(pred_value)
            )
            cell["y_pred"].append(y_pred_code)
            cell["conf"].append(conf_vec)
            if measure_global:
                global_rows.append(
                    {"y_true": y_true_i, "y_pred": y_pred_code, "conf": conf_vec}
                )

    if not pc.check():
        raise RuntimeError(f"Prediction count does not match: {pc.get_error_message()}")

    suppress_per_fold = estimation_procedure_type in (
        EstimationProcedureType.LEAVEONEOUT,
        EstimationProcedureType.TESTONTRAININGDATA,
    )

    per_cell_scores: list[EvaluationScore] = []
    fold_values_by_metric: dict[str, list[float]] = {}

    for cell_key in sorted(cells):
        rep, fold, sample = cell_key
        data = cells[cell_key]
        y_t = np.asarray(data["y_true"])

        if task_type is TaskType.REGRESSION:
            y_p = np.asarray(data["y_pred"], dtype=float)
            metrics = regression_metrics(y_t, y_p, y_train.astype(float))
        else:
            y_p = np.asarray(data["y_pred"], dtype=int)
            conf_arr = np.asarray(data["conf"]) if data["conf"] else np.zeros((0, num_classes))
            metrics = classification_metrics(
                y_t, y_p, conf_arr, class_names, y_train, cost_matrix
            )

        if not suppress_per_fold:
            sample_size = pc.get_shadow_type_size(rep, fold, sample)
            for k, v in metrics.items():
                if k == "_per_class":
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    per_cell_scores.append(
                        EvaluationScore(
                            function=k,
                            value=float(v),
                            repeat=rep,
                            fold=fold,
                            sample=sample,
                            sample_size=sample_size if task_type is TaskType.LEARNINGCURVE else None,
                        )
                    )

        if task_type is not TaskType.LEARNINGCURVE or sample == last_sample:
            for k, v in metrics.items():
                if k == "_per_class":
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    fold_values_by_metric.setdefault(k, []).append(float(v))

    if task_type is TaskType.REGRESSION:
        g_y_true = np.asarray([r["y_true"] for r in global_rows], dtype=float)
        g_y_pred = np.asarray([r["y_pred"] for r in global_rows], dtype=float)
        global_metrics = regression_metrics(g_y_true, g_y_pred, y_train.astype(float))
    else:
        g_y_true = np.asarray([r["y_true"] for r in global_rows], dtype=int)
        g_y_pred = np.asarray([r["y_pred"] for r in global_rows], dtype=int)
        g_conf = (
            np.asarray([r["conf"] for r in global_rows])
            if global_rows
            else np.zeros((0, num_classes))
        )
        global_metrics = classification_metrics(
            g_y_true, g_y_pred, g_conf, class_names, y_train, cost_matrix
        )

    global_scores: list[EvaluationScore] = []
    per_class = global_metrics.get("_per_class", {})
    for k, v in global_metrics.items():
        if k == "_per_class":
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            fold_vals = fold_values_by_metric.get(k, [])
            stdev = float(np.std(fold_vals, ddof=0)) if len(fold_vals) > 0 else None
            array = per_class.get(k)
            global_scores.append(
                EvaluationScore(
                    function=k,
                    value=float(v),
                    stdev=stdev,
                    array=array,
                )
            )

    return per_cell_scores, global_scores, pc


def evaluate_stream(
    dataset_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    target_feature: str,
) -> list[EvaluationScore]:
    """Port of ``EvaluateStreamPredictions``.

    Streams dataset and predictions in lockstep; enforces that
    ``row_id == instance_index`` (i.e., predictions appear in dataset order).
    Returns a flat list of global ``EvaluationScore``s — there are no folds or
    samples in the stream setting.
    """
    class_names = _resolve_class_names(dataset_df, target_feature)
    if not class_names:
        raise ValueError(
            f"Class attribute ({target_feature}) not found or has no values"
        )

    pred_cols = list(predictions_df.columns)
    col_rowid = get_row_index("row_id", pred_cols)
    col_prediction = get_row_index("prediction", pred_cols)
    confidence_cols: dict[str, int] = {}
    for cls in class_names:
        col_name = f"confidence.{cls}"
        if col_name not in pred_cols:
            raise ValueError(f"Attribute {col_name} not found among predictions.")
        confidence_cols[cls] = get_row_index(col_name, pred_cols)

    target_values = dataset_df[target_feature].to_numpy()
    label_to_idx = {c: i for i, c in enumerate(class_names)}

    if len(predictions_df) != len(dataset_df):
        raise ValueError(
            "Predictions need to be done in the same order as the dataset. "
            f"Dataset has {len(dataset_df)} instances, predictions has {len(predictions_df)}."
        )

    y_true: list[int] = []
    y_pred: list[int] = []
    confs: list[np.ndarray] = []

    for i, pred_row in predictions_df.iterrows():
        rowid = int(pred_row.iloc[col_rowid])
        if rowid != i:
            raise ValueError(
                "Predictions need to be done in the same order as the dataset. "
                f"Could not find prediction for instance #{i}. "
                f"Found prediction for instance #{rowid} instead."
            )
        pred_value = pred_row.iloc[col_prediction]
        raw_conf = np.array(
            [float(pred_row.iloc[confidence_cols[c]]) for c in class_names]
        )
        conf = to_prob_dist(raw_conf)
        y_true.append(label_to_idx[target_values[rowid]])
        y_pred.append(
            label_to_idx[pred_value] if isinstance(pred_value, str) else int(pred_value)
        )
        confs.append(conf)

    metrics = classification_metrics(
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(confs),
        class_names,
        target_values,
    )

    return [
        EvaluationScore(function=k, value=float(v))
        for k, v in metrics.items()
        if k != "_per_class" and isinstance(v, (int, float)) and not isinstance(v, bool)
    ]


def evaluate_survival(
    dataset_df: pd.DataFrame,
    splits_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    target_feature: Optional[str] = None,
) -> tuple[list[EvaluationScore], FoldsPredictionCounter]:
    """Port of ``EvaluateSurvivalAnalysisPredictions``.

    The Java source only validates prediction counts and bounds —
    ``evaluationScores`` is left empty (see
    EvaluateSurvivalAnalysisPredictions.java:89-91). This function faithfully
    reproduces that behavior: it returns an empty score list and the
    prediction counter (whose ``check()`` will have already been called and
    can be inspected for errors). Wire in ``sksurv`` here when extending.
    """
    pc = FoldsPredictionCounter(splits_df)

    pred_cols = list(predictions_df.columns)
    col_rowid = get_row_index_multi(["row_id", "rowid"], pred_cols)
    col_repeat = get_row_index_multi(["repeat", "repeat_nr"], pred_cols)
    col_fold = get_row_index_multi(["fold", "fold_nr"], pred_cols)
    n_dataset = len(dataset_df)

    for _, pred_row in predictions_df.iterrows():
        repeat = int(pred_row.iloc[col_repeat])
        fold = int(pred_row.iloc[col_fold])
        rowid = int(pred_row.iloc[col_rowid])
        pc.add_prediction(repeat, fold, 0, rowid)
        if rowid >= n_dataset:
            raise RuntimeError(
                f"Making a prediction for row_id {rowid} (0-based) while "
                f"dataset has only {n_dataset} instances."
            )

    if not pc.check():
        raise RuntimeError(f"Prediction count does not match: {pc.get_error_message()}")

    return [], pc


# ============================================================================
# Dispatcher (mirrors EvaluateRun.evaluate, minus the upload phase)
# ============================================================================


def evaluate_run(
    task_type_id: int,
    dataset_df: pd.DataFrame,
    splits_df: Optional[pd.DataFrame],
    predictions_df: pd.DataFrame,
    target_feature: str,
    cost_matrix: Optional[np.ndarray] = None,
    estimation_procedure_type: Optional[EstimationProcedureType] = None,
    run_id: Optional[int] = None,
) -> RunEvaluation:
    """Route to the right evaluator by OpenML ``task_type_id``.

    Mirrors ``EvaluateRun.evaluate`` lines 152-177, minus the API and upload
    machinery. The caller supplies already-loaded data. If predictions or
    splits are missing, returns a ``RunEvaluation`` with ``error`` set rather
    than raising — same short-circuit the Java code uses at lines 120-134.
    """
    result = RunEvaluation(run_id=run_id)

    if task_type_id not in SUPPORTED_TASK_TYPES_EVALUATION:
        result.error = f"Task type not supported: {task_type_id}"
        return result

    if predictions_df is None or len(predictions_df) == 0:
        result.error = "Required output files not present (e.g., arff predictions)."
        return result

    try:
        if task_type_id == 4:
            result.scores = evaluate_stream(dataset_df, predictions_df, target_feature)
        elif task_type_id == 7:
            if splits_df is None:
                result.error = "Splits required for survival analysis tasks."
                return result
            scores, _ = evaluate_survival(
                dataset_df, splits_df, predictions_df, target_feature
            )
            result.scores = scores
        else:
            task_type = TASK_TYPE_ID_TO_TASK_TYPE[task_type_id]
            if splits_df is None:
                result.error = "Splits required for batch evaluation tasks."
                return result
            _, global_scores, _ = evaluate_batch(
                dataset_df,
                splits_df,
                predictions_df,
                target_feature,
                task_type,
                cost_matrix=cost_matrix,
                estimation_procedure_type=estimation_procedure_type,
            )
            result.scores = global_scores
    except Exception as exc:
        result.error = str(exc)

    return result


# ============================================================================
# ARFF I/O helpers (used by the notebook examples)
# ============================================================================


def load_arff_to_df(path: str) -> pd.DataFrame:
    """Load any ARFF file into a DataFrame, preserving column order.

    Nominal columns become ``pd.Categorical`` with the declared categories,
    matching what ``src.folds.load_dataset`` does for dataset ARFFs.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        payload = arff.load(f)
    attributes = payload["attributes"]
    columns = [name for name, _ in attributes]
    df = pd.DataFrame(payload["data"], columns=columns)
    for name, type_spec in attributes:
        if isinstance(type_spec, list):
            df[name] = pd.Categorical(df[name], categories=type_spec)
    return df
