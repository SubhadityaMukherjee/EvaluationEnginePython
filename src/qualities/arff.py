import math

import arff
from pymfe.mfe import MFE

from qualities.module import _compute_dataset_qualities, _build_xy
from src.helpers import normalize_target_names
from src.models import (
    DataQuality,
    DatasetDownloadInfo,
)
from models import Quality

_DEFAULT_MFE_GROUPS = ("general", "statistical", "info-theory")


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
