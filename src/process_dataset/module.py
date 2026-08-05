"""OpenML EvaluationEngine fold generators.

Each generator produces a *splits table*: one row per
``(instance, repeat, fold, [sample])`` assignment, telling OpenML whether that
original instance belongs to ``TRAIN`` or ``TEST`` for that combination.

The splits table schema mirrors the Java ``ArffMapping``:

    column   meaning
    ------   -------------------------------------------------------
    type     'TRAIN' or 'TEST'
    rowid    original 0-based row index in the dataset
    repeat   repeat index (0-based)
    fold     fold index (0-based)
    sample   subsample index (learning-curve tasks only)

"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.data_loader import DataLoader
from src.helpers import (
    DEFAULT_API_BASE,
    get_data_and_meta_information_from_did,
    get_task_xml,
    task_estimation_procedure,
    task_source_data,
)
from src.models import (
    DataFormat,
    DatasetDownloadInfo,
    EstimationProcedure,
    EstimationProcedureType,
)
from src.process_dataset.splitting import (
    crossvalidation_splits,
    holdout_ordered_splits,
    holdout_splits,
    learning_curve_splits,
    leave_one_out_splits,
    train_on_test_splits,
)


def load_dataset(
    did: int,
    base_url: str = DEFAULT_API_BASE,
    *,
    data_format: DataFormat = "arff",
) -> tuple[pd.DataFrame, Optional[str]]:
    """Download an OpenML dataset by id and return ``(DataFrame, target)``.

    Downloads the dataset (ARFF or Parquet, per ``data_format``) and parses it
    via :class:`~src.data_loader.DataLoader` into a DataFrame, preserving file
    order so that row indices are stable ``rowid`` s. Nominal columns become
    ``pd.Categorical`` with their declared categories.
    """
    info: DatasetDownloadInfo = get_data_and_meta_information_from_did(
        did, dataset_type=data_format, base_url=base_url
    )

    attributes, rows = DataLoader(data_format).load(info)

    columns = [name for name, _ in attributes]
    df = pd.DataFrame(rows, columns=columns)

    for name, type_spec in attributes:
        if isinstance(type_spec, list):
            df[name] = pd.Categorical(df[name], categories=type_spec)

    return df, info.default_target_attribute


# Java's Main.FOLD_GENERATION_SEED (== 0). Default seed for fold generation.
FOLD_GENERATION_SEED = 0


def _splits_for_procedure(
    df: pd.DataFrame,
    procedure: EstimationProcedure,
    *,
    target: Optional[str] = None,
    seed: int = FOLD_GENERATION_SEED,
) -> pd.DataFrame:
    """Route to the matching splitter based on ``procedure.type``."""
    t = procedure.type
    if t is EstimationProcedureType.CROSSVALIDATION:
        return crossvalidation_splits(df, procedure, target=target, seed=seed)
    if t is EstimationProcedureType.LEARNINGCURVE_CV:
        return learning_curve_splits(df, procedure, target=target, seed=seed)
    if t is EstimationProcedureType.HOLDOUT:
        return holdout_splits(df, procedure, seed=seed)
    if t is EstimationProcedureType.HOLDOUT_ORDERED:
        return holdout_ordered_splits(df, procedure)
    if t is EstimationProcedureType.LEAVEONEOUT:
        return leave_one_out_splits(df)
    if t is EstimationProcedureType.TESTONTRAININGDATA:
        return train_on_test_splits(df)
    raise ValueError(f"No fold generator for procedure type {t!r}.")


def generate_folds(
    did: int,
    procedure: EstimationProcedure,
    seed: int = 1,
    base_url: str = DEFAULT_API_BASE,
    *,
    data_format: DataFormat = "arff",
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """Download dataset ``did`` and compute its splits table for an explicit
    ``procedure``. Low-level entry point; the CLI uses
    :func:`generate_folds_for_task`, which mirrors Java's task-driven
    ``GenerateFolds``.
    """
    df, target = load_dataset(did, base_url, data_format=data_format)
    return _splits_for_procedure(df, procedure, target=target, seed=seed), df, target


def _procedure_parameters(ep: dict) -> dict[str, str]:
    """Flatten a task estimation_procedure's ``oml:parameter`` list into a
    ``{name: text}`` dict, dropping empty parameters."""
    raw = ep.get("oml:parameter", [])
    if isinstance(raw, dict):  # single-parameter edge case
        raw = [raw]
    out: dict[str, str] = {}
    for p in raw:
        name = p.get("@name")
        text = p.get("#text")
        if name and text is not None:
            out[name] = text
    return out


def _estimation_procedure_from_task(ep: dict) -> EstimationProcedure:
    """Build an ``EstimationProcedure`` from a task's ``oml:estimation_procedure``
    node — port of how Java consumes ``ac.estimationProcedureGet(epId)``."""
    type_str = ep.get("oml:type") or ""
    ep_type = EstimationProcedureType.from_oml_type(type_str)
    if ep_type is None:
        raise ValueError(
            f"Unsupported estimation procedure type {type_str!r}; expected "
            f"one of {', '.join(sorted(t.value for t in EstimationProcedureType))}."
        )

    params = _procedure_parameters(ep)
    folds = int(params["number_folds"]) if params.get("number_folds") else None
    repeats = int(params["number_repeats"]) if params.get("number_repeats") else None
    percentage = float(params["percentage"]) if params.get("percentage") else None
    return EstimationProcedure(
        type=ep_type, folds=folds, repeats=repeats, percentage=percentage
    )


def generate_folds_for_task(
    task_id: int,
    *,
    base_url: str = DEFAULT_API_BASE,
    seed: int = FOLD_GENERATION_SEED,
    data_format: DataFormat = "arff",
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """Port of ``GenerateFolds.java`` — ``task_id`` is a TASK id (matching
    Java's ``-f generate_folds -id <task_id>``).

    Downloads the task's source dataset, reads the estimation-procedure type
    and ``number_folds`` / ``number_repeats`` / ``percentage`` from the task,
    and generates the splits with ``seed`` (default ``FOLD_GENERATION_SEED``,
    matching Java's ``Main.FOLD_GENERATION_SEED``). Multitask tasks (which Java
    serves from a pre-merged dataset) are not handled here.
    """
    task_xml = get_task_xml(task_id, base_url)
    source_data = task_source_data(task_xml)
    did = int(source_data["oml:data_set_id"])
    target = source_data.get("oml:target_feature")

    ep = task_estimation_procedure(task_xml)
    if ep is None:
        raise ValueError(
            "Task has no estimation_procedure input; cannot generate folds."
        )
    procedure = _estimation_procedure_from_task(ep)

    df, _ = load_dataset(did, base_url, data_format=data_format)
    splits = _splits_for_procedure(df, procedure, target=target, seed=seed)
    return splits, df, target
