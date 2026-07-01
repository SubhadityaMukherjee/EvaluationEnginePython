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

import arff
import pandas as pd

from helpers import get_data_and_meta_information_from_did
from models import DatasetDownloadInfo, EstimationProcedure, EstimationProcedureType
from process_dataset.splitting import crossvalidation_splits, learning_curve_splits, holdout_splits, \
    holdout_ordered_splits, leave_one_out_splits, train_on_test_splits


def load_dataset(did: int) -> tuple[pd.DataFrame, Optional[str]]:
    """Download an OpenML dataset by id and return ``(DataFrame, target)``.

    downloads the ARFF for a given dataset id and returns its temp file path plus the declared
    target attribute. We wrap ``liac-arff`` to turn that into a DataFrame,
    preserving file order so that row indices are stable ``rowid`` s.
    """
    info: DatasetDownloadInfo = get_data_and_meta_information_from_did(did)

    with open(info.file_path, "r", encoding="utf-8", errors="replace") as f:
        payload = arff.load(f)

    attributes = payload["attributes"]
    columns = [name for name, _ in attributes]

    df = pd.DataFrame(payload["data"], columns=columns)

    for name, type_spec in attributes:
        if isinstance(type_spec, list):
            df[name] = pd.Categorical(df[name], categories=type_spec)

    return df, info.default_target_attribute


def generate_folds(
    did: int,
    procedure: EstimationProcedure,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """Download dataset ``did`` and compute its splits table.

     download the dataset, then route to
    the matching generator based on ``procedure.type``.

    """
    df, target = load_dataset(did)

    if procedure.type is EstimationProcedureType.CROSSVALIDATION:
        splits = crossvalidation_splits(df, procedure, target=target, seed=seed)
    elif procedure.type is EstimationProcedureType.LEARNINGCURVE_CV:
        splits = learning_curve_splits(df, procedure, target=target, seed=seed)
    elif procedure.type is EstimationProcedureType.HOLDOUT:
        splits = holdout_splits(df, procedure, seed=seed)
    elif procedure.type is EstimationProcedureType.HOLDOUT_ORDERED:
        splits = holdout_ordered_splits(df, procedure)
    elif procedure.type is EstimationProcedureType.LEAVEONEOUT:
        splits = leave_one_out_splits(df)
    elif procedure.type is EstimationProcedureType.TESTONTRAININGDATA:
        splits = train_on_test_splits(df)
    else:
        raise ValueError(f"Unsupported procedure type: {procedure.type}")

    return splits, df, target
