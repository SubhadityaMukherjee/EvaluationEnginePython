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

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import arff
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    ShuffleSplit,
    StratifiedKFold,
)

from src.helpers import get_data_and_meta_information_from_did
from src.models import DatasetDownloadInfo

# ============================================================================
# Dataset loading
# ============================================================================


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


# ============================================================================
# Estimation-procedure configuration
# ============================================================================


class EstimationProcedureType(str, Enum):
    CROSSVALIDATION = "CROSSVALIDATION"
    HOLDOUT = "HOLDOUT"
    HOLDOUT_ORDERED = "HOLDOUT_ORDERED"
    LEAVEONEOUT = "LEAVEONEOUT"
    TESTONTRAININGDATA = "TESTONTRAININGDATA"
    LEARNINGCURVE_CV = "LEARNINGCURVE_CV"


@dataclass(frozen=True)
class EstimationProcedure:
    """Estimation-procedure configuration read by the Java dispatcher.

    ``folds`` / ``repeats`` / ``percentage`` correspond to the procedure fields
    consumed by ``GenerateFolds.java``; ``percentage`` is a test-set size in the
    range 0..100.
    """

    type: EstimationProcedureType
    folds: Optional[int] = None
    repeats: Optional[int] = None
    percentage: Optional[float] = None


def _is_nominal(df: pd.DataFrame, target: Optional[str]) -> bool:
    """Match Weka's ``classAttribute().isNominal()`` check for stratification."""
    if target is None or target not in df.columns:
        return False
    dtype = df[target].dtype
    return isinstance(dtype, pd.CategoricalDtype) or dtype == object


# ============================================================================
# Split generators
# ============================================================================


def holdout_splits(
    df: pd.DataFrame,
    procedure: EstimationProcedure,
    seed: int = 1,
) -> pd.DataFrame:
    """Generate holdout splits .

    For each repeat, shuffle the data with the seed, then take the first
    ``round(N * percentage / 100)`` instances as ``TEST``, the rest as
    ``TRAIN``. ``sklearn.model_selection.ShuffleSplit`` does exactly this and
    yields ``repeats`` independent shuffles in one call.
    """
    n = len(df)
    test_size = procedure.percentage / 100.0
    ss = ShuffleSplit(
        n_splits=procedure.repeats,
        test_size=test_size,
        random_state=seed,
    )

    rows: list[tuple] = []
    # rowid is the ORIGINAL row index, so we split an index array, not df rows.
    index = np.arange(n)
    for repeat, (train_idx, test_idx) in enumerate(ss.split(index)):
        for i in train_idx:
            rows.append(("TRAIN", int(i), repeat, 0))
        for i in test_idx:
            rows.append(("TEST", int(i), repeat, 0))

    return pd.DataFrame(rows, columns=["type", "rowid", "repeat", "fold"])


def holdout_ordered_splits(
    df: pd.DataFrame,
    procedure: EstimationProcedure,
) -> pd.DataFrame:
    """Generate holdout-ordered splits .

    No shuffling — the first ``(100 - percentage)%`` of instances (in file order)
    become ``TRAIN``, the tail becomes ``TEST``. Exactly one repeat and one fold.
    """
    n = len(df)
    test_size = n * procedure.percentage / 100.0
    threshold = n - test_size  # rows at index <= threshold are TRAIN

    rows = [("TRAIN" if i <= threshold else "TEST", i, 0, 0) for i in range(n)]
    return pd.DataFrame(rows, columns=["type", "rowid", "repeat", "fold"])


def crossvalidation_splits(
    df: pd.DataFrame,
    procedure: EstimationProcedure,
    target: Optional[str] = None,
    seed: int = 1,
) -> pd.DataFrame:
    """Generate cross-validation splits .

    For each repeat:

    1. shuffle the dataset with the seed;
    2. stratify if the target is nominal;
    3. carve into ``folds`` slices and emit every instance as ``TRAIN`` or
       ``TEST`` depending on whether it belongs to fold *f*.

    ``StratifiedKFold(shuffle=True)`` is the scikit-learn equivalent of Weka's
    ``randomize`` + ``stratify`` + ``trainCV``/``testCV``. We instantiate one
    splitter per repeat with a distinct child seed so each repeat gets a fresh
    permutation consecutively).
    """
    n = len(df)
    stratify = _is_nominal(df, target)
    y = df[target].to_numpy() if stratify else None
    index = np.arange(n)

    rows: list[tuple] = []
    for repeat in range(procedure.repeats):
        if stratify:
            splitter = StratifiedKFold(
                n_splits=procedure.folds,
                shuffle=True,
                random_state=seed + repeat,
            )
            splits = splitter.split(index, y)
        else:
            splitter = KFold(
                n_splits=procedure.folds,
                shuffle=True,
                random_state=seed + repeat,
            )
            splits = splitter.split(index)

        for fold, (train_idx, test_idx) in enumerate(splits):
            for i in train_idx:
                rows.append(("TRAIN", int(i), repeat, fold))
            for i in test_idx:
                rows.append(("TEST", int(i), repeat, fold))

    return pd.DataFrame(rows, columns=["type", "rowid", "repeat", "fold"])


def leave_one_out_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Generate leave-one-out splits .

    For each fold *f* in ``0..N-1``, instance *f* is ``TEST`` and all others are
    ``TRAIN``. The output has ``N * N`` rows. ``sklearn.model_selection.LeaveOneOut``
    yields the train/test index pairs directly.
    """
    index = np.arange(len(df))
    rows: list[tuple] = []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneOut().split(index)):
        for i in train_idx:
            rows.append(("TRAIN", int(i), 0, fold))
        for i in test_idx:
            rows.append(("TEST", int(i), 0, fold))
    return pd.DataFrame(rows, columns=["type", "rowid", "repeat", "fold"])


def train_on_test_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Generate test-on-training-data splits .

    Every instance appears twice in fold 0 — once as ``TRAIN`` and once as
    ``TEST``. Trains and tests on the same data; used for in-sample evaluation.
    Output size is ``2 * N``.
    """
    rows: list[tuple] = []
    for i in range(len(df)):
        rows.append(("TEST", i, 0, 0))
        rows.append(("TRAIN", i, 0, 0))
    return pd.DataFrame(rows, columns=["type", "rowid", "repeat", "fold"])


# ============================================================================
# Learning-curve subsampling
# ============================================================================


def sample_size(number: int, train_size: int) -> int:
    """Return ``2 ** (6 + 0.5 * number)`` capped at ``train_size``."""
    return int(min(train_size, round(2 ** (6 + number * 0.5))))


def num_samples(train_size: int) -> int:
    """Number of subsamples for a training set of this size (incl. the full set)."""
    i = 0
    while sample_size(i, train_size) < train_size:
        i += 1
    return i + 1


def learning_curve_splits(
    df: pd.DataFrame,
    procedure: EstimationProcedure,
    target: Optional[str] = None,
    seed: int = 1,
) -> pd.DataFrame:
    """Generate learning-curve splits.


     Same CV skeleton as `crossvalidation_splits`, but inside each fold the *training* side is
    subsampled at geometrically growing sizes ``2 ** (6 + 0.5 * s)`` (capped at
    the full train size), and each subsample is emitted under its own ``sample``
    index. The full test fold is repeated for every sample.

    This is the only generator that emits a ``sample`` column.
    """
    n = len(df)
    stratify = _is_nominal(df, target)
    y = df[target].to_numpy() if stratify else None
    index = np.arange(n)

    rows: list[tuple] = []
    for repeat in range(procedure.repeats):
        if stratify:
            splitter = StratifiedKFold(
                n_splits=procedure.folds,
                shuffle=True,
                random_state=seed + repeat,
            )
            splits = splitter.split(index, y)
        else:
            splitter = KFold(
                n_splits=procedure.folds,
                shuffle=True,
                random_state=seed + repeat,
            )
            splits = splitter.split(index)

        for fold, (train_idx, test_idx) in enumerate(splits):
            train_size = len(train_idx)
            for s in range(num_samples(train_size)):
                k = sample_size(s, train_size)
                # first k (already-shuffled) training rows at this sample size
                for i in train_idx[:k]:
                    rows.append(("TRAIN", int(i), repeat, fold, s))
                for i in test_idx:
                    rows.append(("TEST", int(i), repeat, fold, s))

    return pd.DataFrame(
        rows,
        columns=["type", "rowid", "repeat", "fold", "sample"],
    )


# ============================================================================
# Dispatcher
# ============================================================================


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


# ============================================================================
# ARFF serialization
# ============================================================================


def splits_to_arff(splits: pd.DataFrame, relation: str = "splits") -> str:
    """Serialize a splits DataFrame to an OpenML-format ARFF string.

    Java's ``ArffMapping`` emits a 4- or 5-column ARFF (``type``, ``rowid``,
    ``repeat``, ``fold``, optional ``sample``). We reproduce that schema with
    ``liac-arff`` so the output is byte-compatible with what the OpenML server
    accepts. ``type`` is nominal ``{TRAIN, TEST}``; every other column is
    ``NUMERIC``. Columns are emitted in declaration order regardless of
    DataFrame column order.
    """
    attributes: list[tuple[str, object]] = [("type", ["TRAIN", "TEST"])]
    for col in ("rowid", "repeat", "fold", "sample"):
        if col in splits.columns:
            attributes.append((col, "NUMERIC"))

    ordered = splits[[name for name, _ in attributes]]
    data = ordered.astype(object).to_numpy().tolist()

    return arff.dumps(
        {
            "relation": relation,
            "attributes": attributes,
            "data": data,
        }
    )


def save_splits_arff(
    splits: pd.DataFrame,
    path: str,
    relation: str = "splits",
) -> None:
    """Write a splits DataFrame to ``path`` as OpenML-format ARFF."""
    text = splits_to_arff(splits, relation=relation)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def arff_head(text: str, n: int = 15) -> str:
    """First ``n`` lines of an ARFF string — handy for quick inspection."""
    return "\n".join(text.splitlines()[:n])
