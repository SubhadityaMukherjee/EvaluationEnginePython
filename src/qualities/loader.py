"""Dataset → ``DataQuality`` extraction (format-agnostic).

File parsing is delegated to :class:`src.data_loader.DataLoader`; this module
computes the SimpleMetaFeatures-equivalent counts plus the pymfe meta-features.
"""

import math

from pymfe.mfe import MFE

from src.data_loader import DataLoader
from src.helpers import normalize_target_names
from src.models import DataFormat, DataQuality, DatasetDownloadInfo, Quality
from src.qualities.module import build_xy, compute_dataset_qualities

_DEFAULT_MFE_GROUPS = ("general", "statistical", "info-theory")


def load_qualities(
    dataset: DatasetDownloadInfo,
    *,
    data_format: DataFormat = "arff",
    did: int | None = None,
    evaluation_engine_id: int | None = None,
    groups: tuple[str, ...] = _DEFAULT_MFE_GROUPS,
    random_state: int = 42,
    timeout: int = 30,
) -> DataQuality:
    """Extract a :class:`DataQuality` from a downloaded dataset.

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

        target_names = normalize_target_names(
            dataset.default_target_attribute,
        )

        qualities = compute_dataset_qualities(
            attributes,
            rows,
            target_names,
        )

        X, y = build_xy(attributes, rows, target_names)

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
