import math
from collections import Counter

import arff
import numpy as np
import pandas as pd
from pymfe.mfe import MFE

from src.helpers import normalize_target_names
from src.models import (
    DataQuality,
    DatasetDownloadInfo,
    Quality,
    _NUMERIC_TYPES,
)


_DEFAULT_MFE_GROUPS = ("general", "statistical", "info-theory")


def _pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den * 100


def _compute_dataset_qualities(
    attributes,
    rows,
    target_names: set[str],
) -> list[Quality]:
    attr_names = [a[0] for a in attributes]
    attr_types = [a[1] for a in attributes]

    target_idxs = [i for i, n in enumerate(attr_names) if n in target_names]
    target_idx = target_idxs[0] if target_idxs else None

    n_instances = len(rows)
    n_features = len(attributes)

    target_type_spec = (
        attr_types[target_idx] if target_idx is not None else None
    )
    target_is_numeric = isinstance(target_type_spec, str) and (
        target_type_spec.upper() in _NUMERIC_TYPES
    )

    qualities: list[Quality] = []

    def add(name: str, value: float | None) -> None:
        qualities.append(Quality(name=name, value=value))

    # --- Counts / dimensions ---
    add("NumberOfInstances", float(n_instances))
    add("NumberOfFeatures", float(n_features))

    if target_idx is None or target_is_numeric:
        add("NumberOfClasses", None)
    else:
        target_col = (row[target_idx] for row in rows)
        distinct = {v for v in target_col if v is not None}
        add("NumberOfClasses", float(len(distinct)))

    add(
        "Dimensionality",
        n_features / n_instances if n_instances else None,
    )

    # --- Missing values ---
    n_missing = 0
    rows_with_missing = 0
    for row in rows:
        row_missing = sum(1 for v in row if v is None)
        n_missing += row_missing
        if row_missing:
            rows_with_missing += 1

    add(
        "NumberOfInstancesWithMissingValues",
        float(rows_with_missing),
    )
    add("NumberOfMissingValues", float(n_missing))
    add(
        "PercentageOfInstancesWithMissingValues",
        _pct(rows_with_missing, n_instances),
    )
    add(
        "PercentageOfMissingValues",
        _pct(n_missing, n_instances * n_features),
    )

    # --- Feature types ---
    n_numeric = 0
    n_symbolic = 0
    n_binary = 0
    for i, type_spec in enumerate(attr_types):
        if isinstance(type_spec, list):
            n_symbolic += 1
            if len(type_spec) == 2:
                n_binary += 1
        elif isinstance(
            type_spec, str,
        ) and type_spec.upper() in _NUMERIC_TYPES:
            n_numeric += 1
            col = (row[i] for row in rows)
            if len({v for v in col if v is not None}) == 2:
                n_binary += 1

    add("NumberOfNumericFeatures", float(n_numeric))
    add("NumberOfSymbolicFeatures", float(n_symbolic))
    add("NumberOfBinaryFeatures", float(n_binary))
    add(
        "PercentageOfNumericFeatures",
        _pct(n_numeric, n_features),
    )
    add(
        "PercentageOfSymbolicFeatures",
        _pct(n_symbolic, n_features),
    )
    add(
        "PercentageOfBinaryFeatures",
        _pct(n_binary, n_features),
    )

    # --- Class distribution ---
    if target_idx is None or target_is_numeric:
        add("MajorityClassSize", None)
        add("MinorityClassSize", None)
        add("MajorityClassPercentage", None)
        add("MinorityClassPercentage", None)
    else:
        target_col = (row[target_idx] for row in rows)
        counts = Counter(v for v in target_col if v is not None)
        if counts:
            total = sum(counts.values())
            majority = max(counts.values())
            minority = min(counts.values())
            add("MajorityClassSize", float(majority))
            add("MinorityClassSize", float(minority))
            add(
                "MajorityClassPercentage",
                majority / total * 100,
            )
            add(
                "MinorityClassPercentage",
                minority / total * 100,
            )
        else:
            add("MajorityClassSize", None)
            add("MinorityClassSize", None)
            add("MajorityClassPercentage", None)
            add("MinorityClassPercentage", None)

    return qualities


def _build_xy(
    attributes,
    rows,
    target_names: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    attr_names = [a[0] for a in attributes]

    target_idxs = [i for i, n in enumerate(attr_names) if n in target_names]

    if not target_idxs:
        raise ValueError(
            "default_target_attribute not found in ARFF attributes",
        )

    target_idx = target_idxs[0]

    arr = np.array(rows, dtype=object)

    feature_idxs = [
        i for i in range(len(attr_names)) if i != target_idx
    ]

    X_full = pd.DataFrame(
        {attr_names[i]: arr[:, i] for i in feature_idxs}
    )
    y_raw = arr[:, target_idx]

    y_obj = np.asarray(y_raw, dtype=object)
    if y_obj.dtype == object:
        y = pd.Categorical(y_raw).codes.astype(float)
    else:
        y = y_raw.astype(float)

    string_cols = X_full.select_dtypes(
        include=["object", "string"],
    ).columns
    multi_word_cols = [
        c
        for c in string_cols
        if X_full[c]
        .dropna()
        .astype(str)
        .str.contains(r"\s")
        .any()
    ]
    X_clean = X_full.drop(columns=multi_word_cols)

    for col in X_clean.select_dtypes(
        include=["object", "string"],
    ).columns:
        X_clean[col] = pd.Categorical(
            X_full[col],
        ).codes.astype(float)

    return X_clean.to_numpy(dtype=float), y


def load_arff_qualities(
    dataset: DatasetDownloadInfo,
    *,
    did: int | None = None,
    evaluation_engine_id: int | None = None,
    groups: tuple[str, ...] = _DEFAULT_MFE_GROUPS,
    random_state: int = 42,
    timeout: int = 30,
) -> DataQuality:
    try:
        with open(
            dataset.file_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:
            data = arff.load(f)

        attributes = data["attributes"]
        rows = data["data"]

        target_names = normalize_target_names(
            dataset.default_target_attribute,
        )

        qualities = _compute_dataset_qualities(
            attributes,
            rows,
            target_names,
        )

        X, y = _build_xy(attributes, rows, target_names)

        mfe = MFE(
            groups=tuple(groups),
            random_state=random_state,
        )
        mfe.fit(X, y)
        names, values = mfe.extract(
            cat_cols="auto",
            suppress_warnings=True,
            verbose=0,
            timeout=timeout,
        )

        for name, value in zip(names, values):
            if value is None:
                parsed: float | None = None
            else:
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    parsed = None

            if parsed is not None and math.isnan(parsed):
                parsed = None

            qualities.append(Quality(name=str(name), value=parsed))

    except Exception as exc:
        return DataQuality(
            did=did,
            evaluation_engine_id=evaluation_engine_id,
            error=str(exc),
        )

    return DataQuality(
        did=did,
        evaluation_engine_id=evaluation_engine_id,
        qualities=qualities,
    )
