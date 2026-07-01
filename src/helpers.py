import arff
import numpy as np
import pandas as pd
import xmltodict
import requests
from tempfile import NamedTemporaryFile
from typing import Iterable
from typing import Literal
from src.models import DatasetDownloadInfo


def download_and_parse(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()
    return xmltodict.parse(response.content)


def download_to_temp_file(
    url: str,
    suffix: str = "",
    chunk_size: int = 8192,
) -> str:
    """
    Download a URL to a temporary file.

    Returns
    -------
    str
        Path to the downloaded file.
    """
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        with NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as tmp:
            for chunk in response.iter_content(chunk_size=chunk_size):
                tmp.write(chunk)

    return tmp.name


def normalize_target_names(target: str | list[str] | None) -> set[str]:
    if target is None:
        return set()

    if isinstance(target, str):
        return {t.strip() for t in target.split(",") if t.strip()}

    return {t.strip() for t in target if t and t.strip()}


# ============================================================================
# Dataset retrieval
# ============================================================================


def get_data_and_meta_information_from_did(
    did: int,
    dataset_type: Literal["arff", "parquet"] = "arff",
) -> DatasetDownloadInfo:
    dataset_type = dataset_type.lower()

    if dataset_type not in {"arff", "parquet"}:
        raise ValueError("dataset_type must be 'arff' or 'parquet'")

    metadata = download_and_parse(f"https://www.openml.org/api/v1/xml/data/{did}")[
        "oml:data_set_description"
    ]

    url_key = "oml:url" if dataset_type == "arff" else "oml:parquet_url"

    return DatasetDownloadInfo(
        file_path=download_to_temp_file(
            metadata[url_key],
            suffix=f".{dataset_type}",
        ),
        default_target_attribute=metadata.get("oml:default_target_attribute"),
    )


# ============================================================================
# ARFF / prediction helpers (ported from InstancesHelper.java)
# ============================================================================


def get_row_index(name: str, columns: Iterable[str]) -> int:
    """Return the 0-based index of ``name`` in ``columns``, or -1 if absent.

    Mirrors ``InstancesHelper.getRowIndex(String, Instances)``.
    """
    cols = list(columns)
    return cols.index(name) if name in cols else -1


def get_row_index_multi(names: Iterable[str], columns: Iterable[str]) -> int:
    """Return the index of the first name in ``names`` present in ``columns``.

    Raises ``ValueError`` if none of the names are found. Mirrors
    ``InstancesHelper.getRowIndex(String[], Instances)``.
    """
    cols = list(columns)
    for name in names:
        if name in cols:
            return cols.index(name)
    raise ValueError(
        f"ARFF file contains none of the specified attributes: {list(names)}"
    )


def to_prob_dist(d: Iterable[float]) -> np.ndarray:
    """Normalize a vector to a probability distribution.

    Replicates ``InstancesHelper.toProbDist`` exactly:
      * If any element is +/- inf, the first such element becomes 1.0 and the
        rest become 0.
      * If all (non-nan) elements sum to 0, the first element becomes 1.0.
      * Otherwise, divide each non-nan element by the total. NaNs become 0.
    """
    arr = np.asarray(d, dtype=float)
    result = np.zeros_like(arr)

    inf_mask = np.isinf(arr)
    if inf_mask.any():
        result[np.argmax(inf_mask)] = 1.0
        return result

    nan_mask = np.isnan(arr)
    total = float(np.sum(arr[~nan_mask]))

    if total == 0.0:
        result[0] = 1.0
        return result

    for i in range(len(arr)):
        if nan_mask[i]:
            result[i] = 0.0
        elif total > 0.0:
            result[i] = arr[i] / total
        else:
            result[i] = arr[i]
    return result


def prediction_to_confidences(
    confidence_values: Iterable[float],
    prediction_value: object,
    class_names: list[str],
) -> np.ndarray:
    """Build a confidence vector from a prediction row.

    Mirrors ``InstancesHelper.predictionToConfidences``. Raises ``ValueError``
    on missing values. If every confidence is 0, falls back to placing all
    mass on the predicted class.

    ``prediction_value`` may be either a class label (string) or a 0-based
    integer class index — both are accepted, matching how Weka's
    ``Instance.value()`` returns either form depending on attribute type.
    """
    conf = np.asarray(confidence_values, dtype=float)
    if np.isnan(conf).any():
        raise ValueError(
            "Prediction file contains missing values for a confidence attribute."
        )
    if not (conf > 0).any():
        label_to_idx = {c: i for i, c in enumerate(class_names)}
        if isinstance(prediction_value, str):
            idx = label_to_idx[prediction_value]
        else:
            idx = int(prediction_value)
        conf = conf.copy()
        conf[idx] = 1.0
    return conf


def class_counts(y: Iterable, num_classes: int) -> np.ndarray:
    """Bin integer-coded labels into a length-``num_classes`` count vector."""
    counts = np.zeros(num_classes, dtype=int)
    for c in y:
        counts[int(c)] += 1
    return counts


def class_ratios(y: Iterable, num_classes: int) -> np.ndarray:
    """Class frequency ratios. Mirrors ``InstancesHelper.classRatios``."""
    counts = class_counts(y, num_classes)
    total = counts.sum()
    if total == 0:
        return np.zeros(num_classes, dtype=float)
    return counts / total


def _encode_labels(values, class_names):
    """Map an iterable of string/integer labels to 0-based integer codes."""
    label_to_idx = {c: i for i, c in enumerate(class_names)}
    return np.array([label_to_idx[v] if isinstance(v, str) else int(v) for v in values])


def load_arff_to_df(path: str) -> pd.DataFrame:
    """Load any ARFF file into a DataFrame, preserving column order.

    Nominal columns become ``pd.Categorical`` with the declared categories,
    matching what ``src.folds.load_dataset`` does for dataset ARFFs.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        payload = arff.load(f)
    attributes = payload["attributes"]
    columns = [name for name, _ in attributes]
    df = pd.DataFrame(payload["data"], columns=columns)
    for name, type_spec in attributes:
        if isinstance(type_spec, list):
            df[name] = pd.Categorical(df[name], categories=type_spec)
    return df
