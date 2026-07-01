import arff

from features.module import _fill_numeric_feature, _fill_nominal_feature
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
