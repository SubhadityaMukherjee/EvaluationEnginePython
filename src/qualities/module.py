from collections import Counter

import numpy as np
import pandas as pd

from src.models import NUMERIC_TYPES, Quality


def _pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return num / den * 100


def compute_dataset_qualities(
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

    target_type_spec = attr_types[target_idx] if target_idx is not None else None
    target_is_numeric = isinstance(target_type_spec, str) and (
        target_type_spec.upper() in NUMERIC_TYPES
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
        elif (
            isinstance(
                type_spec,
                str,
            )
            and type_spec.upper() in NUMERIC_TYPES
        ):
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

    add(
        "AutoCorrelation",
        _autocorrelation(rows, target_idx, target_is_numeric, n_instances),
    )

    return qualities


def _autocorrelation(
    rows,
    target_idx: int | None,
    target_is_numeric: bool,
    n_instances: int,
) -> float | None:
    """Port of ``SimpleMetaFeatures`` AutoCorrelation (Java lines 100-124).

    Nominal target: counts how often consecutive class labels differ, then
    returns ``(N-1-changes)/(N-1)``. Numeric target: same formula but
    ``changes`` is the sum of absolute consecutive differences. Undefined
    (returns None) if any target value is missing — Java bails on the first
    NaN via ``classCur.isNaN()``. N<=1 returns 1.0 (Java parity)."""
    if target_idx is None or n_instances == 0:
        return None

    time_based_changes = 0.0
    class_cur = rows[0][target_idx]
    for i in range(1, n_instances):
        class_prev = class_cur
        class_cur = rows[i][target_idx]
        # Java checks classCur.isNaN() and breaks; None is our missing marker.
        if class_cur is None:
            return None
        if target_is_numeric:
            # Java doesn't guard the first value; if it was missing, math
            # produces NaN upstream. We surface that as undefined.
            if class_prev is None:
                return None
            time_based_changes += abs(class_prev - class_cur)
        else:
            time_based_changes += 0 if class_prev == class_cur else 1

    if n_instances > 1:
        return (n_instances - 1 - time_based_changes) / (n_instances - 1.0)
    return 1.0


def _numeric_column(values) -> bool:
    """True when every present value is a number (object arrays only).

    ``build_xy`` works on an object array, so dtype inspection cannot
    distinguish numeric columns — check the values instead."""
    seen_present = False
    for v in values:
        if v is None:
            continue
        if not isinstance(v, (int, float, np.integer, np.floating)):
            return False
        seen_present = True
    return seen_present


def build_xy(
    attributes,
    rows,
    target_names: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    attr_names = [a[0] for a in attributes]

    target_idxs = [i for i, n in enumerate(attr_names) if n in target_names]

    if not target_idxs:
        raise ValueError(
            f"Target feature not found among dataset attributes "
            f"(default_target_attribute={target_names!r})."
        )

    target_idx = target_idxs[0]

    arr = np.array(rows, dtype=object)

    feature_idxs = [i for i in range(len(attr_names)) if i != target_idx]

    X_full = pd.DataFrame({attr_names[i]: arr[:, i] for i in feature_idxs})
    y_raw = arr[:, target_idx]

    if _numeric_column(y_raw):
        y = np.asarray(
            [np.nan if v is None else float(v) for v in y_raw]
        )
    else:
        y = pd.Categorical(y_raw).codes.astype(float)

    string_cols = X_full.select_dtypes(
        include=["object", "string"],
    ).columns
    multi_word_cols = [
        c
        for c in string_cols
        if X_full[c].dropna().astype(str).str.contains(r"\s").any()
    ]
    X_clean = X_full.drop(columns=multi_word_cols)

    for col in X_clean.select_dtypes(
        include=["object", "string"],
    ).columns:
        values = X_full[col].tolist()
        if _numeric_column(values):
            X_clean[col] = [
                np.nan if v is None else float(v) for v in values
            ]
        else:
            X_clean[col] = pd.Categorical(
                X_full[col],
            ).codes.astype(float)

    return X_clean.to_numpy(dtype=float), y
