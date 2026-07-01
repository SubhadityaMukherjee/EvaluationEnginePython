from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold, LeaveOneOut, ShuffleSplit

from models import EstimationProcedure


def _is_nominal(df: pd.DataFrame, target: Optional[str]) -> bool:
    """Match Weka's ``classAttribute().isNominal()`` check for stratification."""
    if target is None or target not in df.columns:
        return False
    dtype = df[target].dtype
    return isinstance(dtype, pd.CategoricalDtype) or dtype == object


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
