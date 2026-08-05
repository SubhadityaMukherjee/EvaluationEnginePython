"""Python port of ``org.openml.webapplication.features.FantailConnector``.

Drives dataset-level meta-feature (quality) computation and upload. Two
characterizer sets, mirroring ``CharacterizerFactory``:
  * ``simple`` — the base qualities from ``load_qualities``
    (SimpleMetaFeatures-equivalent counts + pymfe groups). Matches Java's
    ``CharacterizerFactory.simple()`` scope, with pymfe groups added on the
    Python side.
  * ``all`` — everything in ``simple`` plus the sklearn landmarker port from
    ``src.qualities.landmarkers``. Java's ``CfsSubsetEval_*`` landmarkers are
    omitted (see landmarkers module docstring).

Deviations from Java (documented):
  * Java's ``FantailConnector.extractFeatures`` removes ``row_id_attribute``
    and ``is_ignore`` columns before characterizing. The Python path computes
    over the full dataset (matching what ``ProcessDataset`` / existing
    ``load_qualities`` already do). Landmarker values on datasets with
    ID-like columns will diverge from Java for this reason.
  * Java polls via ``dataqualitiesUnprocessed`` with the full expected-quality
    list. We do the same, but only the static SimpleMetaFeatures + landmarker
    IDs are used as the filter — pymfe-derived names vary per dataset and
    aren't included.
"""

from __future__ import annotations

from typing import Optional

from src.client import OpenmlClient
from src.data_loader import DataLoader
from src.exceptions import OpenmlApiError
from src.helpers import get_data_and_meta_information_from_did
from src.models import DataFormat, DataQuality, DatasetDownloadInfo, Quality
from src.qualities.landmarkers import compute_all_landmarkers, expected_landmarker_ids
from src.qualities.loader import load_qualities
from src.qualities.module import build_xy
from src.constants import EVALUATION_ENGINE_ID

# The 19 SimpleMetaFeatures IDs (Java's SimpleMetaFeatures.ids). Used as the
# polling filter so the server returns datasets missing any of these.
_SIMPLE_META_FEATURE_IDS: tuple[str, ...] = (
    "NumberOfInstances",
    "NumberOfFeatures",
    "NumberOfClasses",
    "Dimensionality",
    "NumberOfInstancesWithMissingValues",
    "NumberOfMissingValues",
    "PercentageOfInstancesWithMissingValues",
    "PercentageOfMissingValues",
    "NumberOfNumericFeatures",
    "NumberOfSymbolicFeatures",
    "NumberOfBinaryFeatures",
    "PercentageOfNumericFeatures",
    "PercentageOfSymbolicFeatures",
    "PercentageOfBinaryFeatures",
    "MajorityClassSize",
    "MinorityClassSize",
    "MajorityClassPercentage",
    "MinorityClassPercentage",
    "AutoCorrelation",
)


class ExtractFeatures:
    """Port of ``FantailConnector``. Construct with a ``characterizer_set``
    of ``"simple"`` or ``"all"``; call ``process(did)`` for one dataset or
    ``poll()`` to loop the qualities-unprocessed endpoint."""

    def __init__(
        self,
        client: OpenmlClient,
        mode: str = "normal",
        characterizer_set: str = "simple",
        priority_tag: Optional[str] = None,
        dataset_format: DataFormat = "arff",
    ) -> None:
        if characterizer_set not in ("simple", "all"):
            raise ValueError(
                f"characterizer_set must be 'simple' or 'all', got {characterizer_set!r}"
            )
        self.client = client
        self.mode = mode
        self.characterizer_set = characterizer_set
        self.priority_tag = priority_tag
        self.dataset_format: DataFormat = dataset_format
        self.last_result: Optional[DataQuality] = None

    # ----------------------------------------------------------------------
    # Single-dataset path (Java: FantailConnector.computeMetafeatures)
    # ----------------------------------------------------------------------

    def process(self, did: int) -> DataQuality:
        """Compute qualities for one dataset and upload. Returns the
        ``DataQuality`` (Java uploads as a side effect — same here)."""
        info = get_data_and_meta_information_from_did(
            did, dataset_type=self.dataset_format, base_url=self.client.base_url
        )
        data_quality = load_qualities(
            info,
            data_format=self.dataset_format,
            did=did,
            evaluation_engine_id=EVALUATION_ENGINE_ID,
        )

        # Java's "all" set runs landmarkers on top of the base statistical
        # characterizers. We append sklearn-landmarker Quality objects to the
        # same DataQuality before upload.
        if self.characterizer_set == "all" and not data_quality.error:
            self._append_landmarkers(data_quality, info)

        if data_quality.qualities and not data_quality.error:
            try:
                self.client.data_qualities_upload(data_quality)
            except OpenmlApiError:
                # Java's FantailConnector lets upload errors propagate; we
                # match that — the caller (CLI / poll) decides what to do.
                raise

        self.last_result = data_quality
        return data_quality

    def _append_landmarkers(
        self,
        data_quality: DataQuality,
        info: DatasetDownloadInfo,
    ) -> None:
        """Compute all landmarkers and append to ``data_quality.qualities``.

        Silent on failure: if X/y can't be built (no target, unparseable),
        we leave the base qualities intact and skip — Java's GenericLandmarker
        returns null per-landmarker on exception; here the whole landmarker
        pass is all-or-nothing since they share one X/y."""
        from src.helpers import normalize_target_names

        try:
            attributes, rows = DataLoader(self.dataset_format).load(info)
            target_names = normalize_target_names(info.default_target_attribute)
            X, y = build_xy(attributes, rows, target_names)
            landmark_values = compute_all_landmarkers(X, y)
        except Exception:  # noqa: BLE001 — Java parity: landmarkers fail soft
            return

        for name, value in landmark_values.items():
            data_quality.qualities.append(Quality(name=name, value=value))

    # ----------------------------------------------------------------------
    # Polling loop (Java: FantailConnector.start with id == null)
    # ----------------------------------------------------------------------

    def poll(self) -> None:
        """Loop ``data_qualities_unprocessed``, processing each dataset id
        until the server returns "No unprocessed" (Java's
        ``fetchUnprocessedQualities`` catches the same string)."""
        expected = self._expected_quality_ids()
        while True:
            try:
                dids = self.client.data_qualities_unprocessed(
                    EVALUATION_ENGINE_ID,
                    self.mode,
                    expected,
                    priority_tag=self.priority_tag,
                )
            except OpenmlApiError as e:
                if "No unprocessed" in e.message:
                    return
                raise
            if not dids:
                return
            for did in dids:
                self.process(did)

    def _expected_quality_ids(self) -> list[str]:
        ids = list(_SIMPLE_META_FEATURE_IDS)
        if self.characterizer_set == "all":
            ids.extend(expected_landmarker_ids())
        return ids
