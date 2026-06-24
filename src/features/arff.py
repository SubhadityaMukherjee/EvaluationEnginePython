from collections import Counter

import arff
import numpy as np

from src.helpers import normalize_target_names
from src.models import (
    DataFeature,
    DatasetDownloadInfo,
    Feature,
    _NUMERIC_TYPES,
)


def _liac_type(type_spec):
    if isinstance(type_spec, list):
        return "nominal", tuple(type_spec)

    normalized = type_spec.upper()

    if normalized in _NUMERIC_TYPES:
        return "numeric", None

    return {
        "STRING": ("string", None),
        "DATE": ("date", None),
    }.get(normalized, (normalized.lower(), None))


def _fill_numeric_feature(col, feat: Feature) -> None:
    feat.number_of_values = len(col)

    present = np.asarray(
        [v for v in col if v is not None],
        dtype=np.float64,
    )

    feat.number_of_missing_values = len(col) - present.size

    if present.size == 0:
        return

    int_mask = present == np.floor(present)

    feat.number_of_integer_values = int(int_mask.sum())
    feat.number_of_real_values = int((~int_mask).sum())

    feat.maximum_value = float(np.max(present))
    feat.minimum_value = float(np.min(present))
    feat.mean_value = float(np.mean(present))
    feat.standard_deviation = float(np.std(present, ddof=0))

    unique_vals, counts = np.unique(
        present,
        return_counts=True,
    )

    feat.number_of_distinct_values = len(unique_vals)
    feat.number_of_unique_values = int(np.sum(counts == 1))


def _fill_nominal_feature(col, feat: Feature, schema_values) -> None:
    feat.nominal_values = sorted(schema_values)
    feat.number_of_values = len(col)

    counts = Counter(v for v in col if v is not None)

    present = sum(counts.values())

    feat.number_of_missing_values = len(col) - present
    feat.number_of_nominal_values = present
    feat.number_of_distinct_values = len(counts)
    feat.number_of_unique_values = sum(c == 1 for c in counts.values())

    feat.class_distribution = ",".join(f"{k}:{v}" for k, v in sorted(counts.items()))


def load_arff_features(
    dataset: DatasetDownloadInfo,
    *,
    did: int | None = None,
    evaluation_engine_id: int | None = None,
) -> DataFeature:
    try:
        with open(
            dataset.file_path,
            "r",
            encoding="utf-8",
            errors="replace",
        ) as f:
            data = arff.load(f)

    except Exception as exc:
        return DataFeature(
            did=did,
            evaluation_engine_id=evaluation_engine_id,
            error=str(exc),
        )

    attributes = data["attributes"]
    rows = data["data"]

    target_names = normalize_target_names(dataset.default_target_attribute)

    if rows:
        columns = list(zip(*rows))
    else:
        columns = [tuple() for _ in attributes]

    features: list[Feature] = []

    for idx, ((name, type_spec), col) in enumerate(zip(attributes, columns)):
        type_name, type_range = _liac_type(type_spec)

        feat = Feature(
            index=idx,
            name=name,
            data_type=type_name,
            is_target=name in target_names,
        )

        if type_name == "numeric":
            _fill_numeric_feature(col, feat)

        elif type_name == "nominal":
            _fill_nominal_feature(col, feat, type_range)

        else:
            feat.number_of_values = len(col)

        features.append(feat)

    return DataFeature(
        did=did,
        evaluation_engine_id=evaluation_engine_id,
        features=features,
    )
