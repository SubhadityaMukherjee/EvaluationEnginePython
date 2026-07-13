"""Python port of ``org.openml.webapplication.EvaluateRun``.

Single-run path is fully wired: fetch run/task/dataset/predictions/splits via
the raw OpenML REST API (openml.runs.get_run has a parser bug for multi-dataset
runs), dispatch to the right evaluator in ``src.runs``, and return a
``RunEvaluation`` with both per-cell and global scores.

Out of scope (marked with TODOs):
  * the polling loop that asks the server for the next batch of unevaluated
    runs (``evaluationRequest``) — only useful once uploads land.
  * trace parsing (``traceToXML``) — only useful once ``runTraceUpload`` lands.
  * run-description parsing and the consistency check against user-defined
    measures (EvaluateRun.java:180-217).
  * all upload paths (``runEvaluate``, ``runTraceUpload``).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.helpers import (download_to_temp_file, get_run_xml, get_task_xml,
                         load_arff_to_df, openml_file_url, run_output_file_ids,
                         task_cost_matrix, task_estimation_procedure,
                         task_source_data)
from src.models import EstimationProcedureType, RunEvaluation
from src.runs import (EVALUATION_ENGINE_ID, SUPPORTED_TASK_TYPES_EVALUATION,
                      TASK_TYPE_ID_TO_TASK_TYPE, TaskType, evaluate_batch,
                      evaluate_stream, evaluate_survival)

_MAX_LENGTH_WARNING = 1024

# Task XML estimation_procedure/oml:type → EstimationProcedureType.
# Source: org.openml.apiconnector.xml.EstimationProcedureType.
_PROCEDURE_TYPE_MAP: dict[str, EstimationProcedureType] = {
    "crossvalidation": EstimationProcedureType.CROSSVALIDATION,
    "holdout": EstimationProcedureType.HOLDOUT,
    "holdout_ordered": EstimationProcedureType.HOLDOUT_ORDERED,
    "leaveoneout": EstimationProcedureType.LEAVEONEOUT,
    "testontrainingdata": EstimationProcedureType.TESTONTRAININGDATA,
    "testonthetrainingdata": EstimationProcedureType.TESTONTRAININGDATA,
    "learningcurve": EstimationProcedureType.LEARNINGCURVE_CV,
}


def _estimation_procedure_type(task_xml: dict) -> Optional[EstimationProcedureType]:
    ep = task_estimation_procedure(task_xml)
    if not ep:
        return None
    type_str = (ep.get("oml:type") or "").lower()
    return _PROCEDURE_TYPE_MAP.get(type_str)


def _cost_matrix_from_task(task_xml: dict) -> Optional[np.ndarray]:
    """Parse the ``cost_matrix`` input into a numpy matrix, if present.

    Java delegates to ``TaskInformation.getCostMatrix(task)`` and
    ``InstancesHelper.doubleToCostMatrix``. We mirror the shape only —
    a 2D float matrix. The OpenML XML serialises it as a JSON-style
    array of arrays; we accept either a list-of-lists or a JSON string.
    """
    import json

    cm = task_cost_matrix(task_xml)
    if not cm:
        return None
    raw = cm.get("#text") or cm
    if isinstance(raw, str):
        raw = json.loads(raw)
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2:
        return None
    return arr


class EvaluateRun:
    """Port of ``EvaluateRun``. Construct with a ``run_id`` to evaluate one
    run immediately; construct without one to get an instance ready for
    ``poll``. The ``RunEvaluation`` from the last call is kept on
    ``self.last_result`` for inspection (Java uploads it as a side effect —
    TODO here)."""

    def __init__(
        self,
        run_id: Optional[int] = None,
        evaluation_mode: str = "normal",
        task_type_ids: Optional[set[int]] = None,
        task_ids: Optional[str] = None,
        tag: Optional[str] = None,
        uploader_id: Optional[int] = None,
    ) -> None:
        self.evaluation_mode = evaluation_mode
        self.task_type_ids = (
            set(task_type_ids) if task_type_ids else SUPPORTED_TASK_TYPES_EVALUATION
        )
        self.task_ids = task_ids
        self.tag = tag
        self.uploader_id = uploader_id
        self.last_result: Optional[RunEvaluation] = None

        if run_id is not None:
            self.last_result = self.evaluate(run_id)

    # ----------------------------------------------------------------------
    # Single-run path
    # ----------------------------------------------------------------------

    def evaluate(self, run_id: int) -> RunEvaluation:
        """Port of ``EvaluateRun.evaluate``. Returns the assembled
        ``RunEvaluation`` (Java returns void — the side effect there is the
        upload, which is TODO here)."""
        result = RunEvaluation(run_id=run_id, evaluation_engine_id=EVALUATION_ENGINE_ID)

        try:
            run_xml = get_run_xml(run_id)
            task_id = int(run_xml["oml:task_id"])
            task_xml = get_task_xml(task_id)
            task_type_id = int(task_xml["oml:task_type_id"])

            if task_type_id not in self.task_type_ids:
                raise ValueError(
                    f"Task type not supported: {task_xml.get('oml:task_type')}"
                )

            file_ids = run_output_file_ids(run_xml)
            source_data = task_source_data(task_xml)
            dataset_id = int(
                source_data.get("oml:labeled_data_set_id")
                or source_data["oml:data_set_id"]
            )
            target_feature = source_data["oml:target_feature"]

            # Java short-circuits if description / predictions are missing.
            if "description" not in file_ids:
                result.error = "Run description file not present."
                # TODO: apiconnector.runEvaluate(result)
                return result
            if not any(
                k in file_ids for k in ("predictions", "subgroups", "predictions_0")
            ):
                result.error = (
                    "Required output files not present (e.g., arff predictions)."
                )
                # TODO: apiconnector.runEvaluate(result)
                return result

            # TODO: trace parsing. If "trace" in file_ids:
            #   trace = self._trace_to_xml(file_ids["trace"], task_id, run_id)

            # TODO: download description XML, parse to Run description, run the
            # consistency check against user-defined measures
            # (EvaluateRun.java:180-217).

            dataset_df, splits_df, predictions_df = _load_run_inputs(
                task_xml=task_xml,
                dataset_id=dataset_id,
                file_ids=file_ids,
                run_id=run_id,
            )

            scores = self._compute_scores(
                task_type_id=task_type_id,
                task_xml=task_xml,
                dataset_df=dataset_df,
                splits_df=splits_df,
                predictions_df=predictions_df,
                target_feature=target_feature,
            )
            result.scores = scores

        except Exception as exc:  # noqa: BLE001 — mirrors Java's catch-all
            result.error = str(exc)[:_MAX_LENGTH_WARNING]

        # TODO: apiconnector.runEvaluate(result) + runTraceUpload(trace).
        return result

    # ----------------------------------------------------------------------
    # Score assembly
    # ----------------------------------------------------------------------

    def _compute_scores(
        self,
        *,
        task_type_id: int,
        task_xml: dict,
        dataset_df: pd.DataFrame,
        splits_df: Optional[pd.DataFrame],
        predictions_df: pd.DataFrame,
        target_feature: str,
    ) -> list:
        """Dispatch to the right evaluator and return the concatenated
        per-cell + global score list (matches ``EvaluateBatchPredictions
        .getEvaluationScores``)."""
        ep_type = _estimation_procedure_type(task_xml)
        cost_matrix = _cost_matrix_from_task(task_xml)

        if task_type_id == 4:  # Supervised Data Stream Classification
            return evaluate_stream(dataset_df, predictions_df, target_feature)

        if task_type_id == 7:  # Survival Analysis — count validation only
            if splits_df is None:
                raise ValueError("Splits required for survival analysis tasks.")
            scores, _ = evaluate_survival(
                dataset_df, splits_df, predictions_df, target_feature
            )
            return scores

        task_type = TASK_TYPE_ID_TO_TASK_TYPE[task_type_id]
        if splits_df is None:
            raise ValueError("Splits required for batch evaluation tasks.")

        per_cell, global_scores, _ = evaluate_batch(
            dataset_df,
            splits_df,
            predictions_df,
            target_feature,
            task_type,
            cost_matrix=cost_matrix,
            estimation_procedure_type=ep_type,
        )
        return list(per_cell) + list(global_scores)

    # ----------------------------------------------------------------------
    # Polling loop (TODO)
    # ----------------------------------------------------------------------

    def poll(self) -> None:
        """Port of EvaluateRun.java:57-94.

        Original behaviour: build a filter map (ttid, task, tag, uploader),
        call ``evaluationRequest`` in a loop with ``numRequests=1000``, and
        evaluate each returned run. Stops when the server returns
        ``NO_UNEVALUATED_RUNS`` (API error 1013).
        """
        # TODO: implement once the upload path lands. Without uploads the loop
        # has no observable side effect, so we refuse to spin rather than burn
        # the server. Filters would be:
        #   {"ttid": self.task_type_ids, "task": self.task_ids,
        #    "tag": self.tag, "uploader": self.uploader_id}
        raise NotImplementedError(
            "EvaluateRun polling loop is not implemented yet — pass a run_id."
        )


# ============================================================================
# Input loading helpers (module-level — no mutable state needed)
# ============================================================================


def _load_run_inputs(
    *,
    task_xml: dict,
    dataset_id: int,
    file_ids: dict[str, str],
    run_id: int,
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], pd.DataFrame]:
    """Download dataset, splits, and predictions for a run.

    Returns ``(dataset_df, splits_df, predictions_df)``. ``splits_df`` is
    ``None`` only for stream tasks (task_type_id 4), which have no splits.
    """
    from src.process_dataset.module import load_dataset

    dataset_df, _ = load_dataset(dataset_id)

    # Splits URL comes from the task's estimation_procedure. Stream tasks (4)
    # and survival (7) — survival still uses splits — handle both.
    splits_df: Optional[pd.DataFrame] = None
    ep = task_estimation_procedure(task_xml)
    splits_url = ep.get("oml:data_splits_url") if ep else None
    if splits_url:
        splits_path = download_to_temp_file(splits_url, suffix=".arff")
        splits_df = load_arff_to_df(splits_path)

    predictions_path = download_to_temp_file(
        openml_file_url(file_ids["predictions"], f"Run_{run_id}_predictions.arff"),
        suffix=".arff",
    )
    predictions_df = load_arff_to_df(predictions_path)

    return dataset_df, splits_df, predictions_df
