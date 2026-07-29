"""Generic OpenML dataset loader.

Reads a downloaded dataset file (ARFF or Parquet) into a single normalized
``(attributes, rows)`` structure that the feature / quality extractors and the
fold generators all consume. Picking the format happens once at the edge (CLI
flag, notebook cell, or explicit caller) — everything downstream is
format-agnostic.

The normalized form deliberately mirrors the ``liac-arff`` schema so the
existing extractors work unchanged:

``attributes``
    list of ``(name, type_spec)`` pairs where ``type_spec`` is

      * ``"NUMERIC"``                       — numeric columns,
      * ``[label, ...]`` (a list)           — nominal columns (declared
                                              categories),
      * ``"STRING"`` / ``"DATE"``           — free-form / temporal columns.

``rows``
    list of row lists; missing values are ``None`` (matching ARFF's ``?``).
"""

from __future__ import annotations

import arff
import pandas as pd

from src.models import DataFormat, DatasetDownloadInfo

# ARFF-style type tag used for every numeric column. It is a member of
# ``src.models._NUMERIC_TYPES`` so the downstream extractors (which check
# membership in that set) treat it as numeric regardless of integer/float.
_NUMERIC_TYPE_TAG = "NUMERIC"

_SUPPORTED_FORMATS = ("arff", "parquet")


class DataLoader:
    """Load a downloaded OpenML dataset into a normalized ``(attributes, rows)`` form.

    Parameters
    ----------
    fmt:
        ``"arff"`` (default) or ``"parquet"``. ARFF support is the incumbent
        format; Parquet is being phased in as the dataset storage backend.

    Examples
    --------
    >>> loader = DataLoader("parquet")
    >>> attributes, rows = loader.load(dataset_info)
    """

    def __init__(self, fmt: DataFormat = "arff") -> None:
        fmt = fmt.lower()
        if fmt not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported data format: {fmt!r} "
                f"(expected one of {_SUPPORTED_FORMATS})."
            )
        self.fmt: DataFormat = fmt

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(
        self,
        dataset: DatasetDownloadInfo,
    ) -> tuple[list[tuple[str, object]], list[list]]:
        """Return ``(attributes, rows)`` for ``dataset.file_path``.

        ``attributes`` and ``rows`` follow the schema documented at the top of
        this module, identical for both backends.
        """
        if self.fmt == "arff":
            return self._load_arff(dataset.file_path)
        return self._load_parquet(dataset.file_path)

    # ------------------------------------------------------------------
    # Backends
    # ------------------------------------------------------------------

    @staticmethod
    def _load_arff(
        path: str,
    ) -> tuple[list[tuple[str, object]], list[list]]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            payload = arff.load(f)
        return payload["attributes"], payload["data"]

    @staticmethod
    def _load_parquet(
        path: str,
    ) -> tuple[list[tuple[str, object]], list[list]]:
        df = pd.read_parquet(path)
        return _df_to_attributes_and_rows(df)


def _df_to_attributes_and_rows(
    df: pd.DataFrame,
) -> tuple[list[tuple[str, object]], list[list]]:
    """Convert a DataFrame to the normalized ``(attributes, rows)`` form.

    Column → ``type_spec`` mapping (matches ARFF semantics):

      * ``CategoricalDtype`` → nominal ``[label, ...]``
      * numeric dtype        → ``"NUMERIC"``
      * datetime dtype       → ``"DATE"``
      * anything else        → ``"STRING"``
    """
    attributes: list[tuple[str, object]] = []
    for name in df.columns:
        col_name = str(name)
        dtype = df[name].dtype

        if isinstance(dtype, pd.CategoricalDtype):
            categories = [str(c) for c in dtype.categories]
            attributes.append((col_name, categories))
        elif pd.api.types.is_numeric_dtype(dtype):
            attributes.append((col_name, _NUMERIC_TYPE_TAG))
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            attributes.append((col_name, "DATE"))
        else:
            attributes.append((col_name, "STRING"))

    # Cast to object so ``None`` survives (otherwise NaN creeps back in for
    # numeric columns), then replace every null with ``None`` to match ARFF's
    # missing-value marker. Categorical values come through as their labels.
    obj = df.astype(object)
    obj = obj.where(pd.notnull(df), None)
    rows = obj.to_numpy().tolist()

    return attributes, rows
