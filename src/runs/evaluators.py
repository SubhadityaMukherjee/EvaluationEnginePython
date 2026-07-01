from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from models import EstimationProcedureType
from src.helpers import (
    _encode_labels,
    get_row_index,
    get_row_index_multi,
    prediction_to_confidences,
    to_prob_dist,
)
from src.models import EvaluationScore, RunEvaluation
from src.runs.metrics import classification_metrics, regression_metrics
from src.runs.prediction_counter import FoldsPredictionCounter

# ============================================================================
# Constants
# ============================================================================

EVALUATION_ENGINE_ID = 1
SUPPORTED_TASK_TYPES_EVALUATION = {1, 2, 3, 4, 5, 6, 7, 8}


class TaskType(Enum):
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
