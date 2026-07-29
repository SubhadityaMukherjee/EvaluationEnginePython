"""Dataset → ``DataFeature`` extraction (format-agnostic).

The actual file parsing lives in :class:`src.data_loader.DataLoader`; this
module owns the per-column statistical filling (``Feature`` objects) and is
unaware of whether the source was ARFF or Parquet.
"""

from src.data_loader import DataLoader
from src.features.module import _fill_nominal_feature, _fill_numeric_feature
from src.helpers import normalize_target_names
from src.models import (_NUMERIC_TYPES, DataFeature, DataFormat,
                        DatasetDownloadInfo, Feature)


def _liac_type(type_spec):
    """Normalize an ARFF-style ``type_spec`` into ``(type_name, type_range)``.

    Shared between the ARFF and Parquet backends because both produce the same
    ``liac-arff``-shaped ``type_spec`` values (see ``DataLoader``).
    """
    if isinstance(type_spec, list):
        return "nominal", tuple(type_spec)

    normalized = type_spec.upper()

    if normalized in _NUMERIC_TYPES:
        return "numeric", None

    return {
        "STRING": ("string", None),
        "DATE": ("date", None),
    }.get(normalized, (normalized.lower(), None))


def load_features(
    dataset: DatasetDownloadInfo,
    *,
    data_format: DataFormat = "arff",
    did: int | None = None,
    evaluation_engine_id: int | None = None,
) -> DataFeature:
    """Extract a :class:`DataFeature` from a downloaded dataset.

    Parameters
    ----------
    dataset:
        Already-downloaded dataset (``file_path`` points at an ``.arff`` or
        ``.parquet`` file depending on how it was fetched).
    data_format:
        ``"arff"`` (default) or ``"parquet"`` — selects the
        :class:`~src.data_loader.DataLoader` backend.
    """
    try:
        attributes, rows = DataLoader(data_format).load(dataset)
    except Exception as exc:
        return DataFeature(
            did=did,
            evaluation_engine_id=evaluation_engine_id,
            error=str(exc),
        )

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
