"""Python port of ``org.openml.webapplication.ProcessDataset``.

Two entry points:
  * ``ProcessDataset(dataset_id=...)`` — process one dataset (or, without an
    id, poll the server for unprocessed datasets via ``.poll()``).
  * ``ProcessDataset().process_and_print(did)`` — local-only feature
    extraction, prints the feature XML. Mirrors Java's ``process_dataset_print``.

Uploads hit the live OpenML server via ``src.client.OpenmlClient`` (port of
``OpenmlConnector``). The client is constructed lazily on first use, so the
print-only path and tests don't need ``OPENML_API_KEY`` set.
"""

from __future__ import annotations

from typing import Optional

from src.client import OpenmlApiError, OpenmlClient
from src.features import DataFeature, features_to_xml, load_arff_features
from src.helpers import (get_data_and_meta_information_from_did,
                         get_dataset_description_xml)
from src.models import DataQuality
from src.qualities import load_arff_qualities, qualities_to_xml

# OpenML dataset status constants — Java's
# org.openml.apiconnector.settings.Constants.DATA_STATUS_*
DATA_STATUS_PREP = "in_preparation"
DATA_STATUS_ACTIVE = "active"

# Mirrors Settings.EVALUATION_ENGINE_ID (== 1 on production).
EVALUATION_ENGINE_ID = 1

# OpenML API error codes that ProcessDataset treats specially. See
# ProcessDataset.java:74-80 (441) and :88-95 (431).
_CODE_FEATURES_ALREADY_UPLOADED = 441
_CODE_DATASET_ALREADY_PROCESSED = 431


class ProcessDataset:
    """Port of ``ProcessDataset``. Construct with a ``dataset_id`` to process
    one dataset immediately (mirrors Java's 3-arg constructor); construct
    without one to get an instance ready for ``process_and_print`` or
    ``poll``. The ``DataFeature`` / ``DataQuality`` from the last call are
    kept on ``self.last_features`` / ``self.last_qualities`` for inspection."""

    def __init__(
        self,
        dataset_id: Optional[int] = None,
        mode: str = "normal",
        client: Optional[OpenmlClient] = None,
    ) -> None:
        self.mode = mode
        self._client = client
        self.last_features: Optional[DataFeature] = None
        self.last_qualities: Optional[DataQuality] = None

        if dataset_id is not None:
            self.process(dataset_id)

    def _get_client(self) -> OpenmlClient:
        """Lazily construct the REST client so the print-only path doesn't
        require ``OPENML_API_KEY``."""
        if self._client is None:
            self._client = OpenmlClient()
        return self._client

    # ----------------------------------------------------------------------
    # Single-dataset path
    # ----------------------------------------------------------------------

    def process(self, did: int) -> tuple[DataFeature, DataQuality]:
        """Port of ``ProcessDataset.process``. Returns ``(features, qualities)``.

        Mirrors Java's structure: download → extract features → upload features
        → (maybe status update) → compute qualities → upload qualities. Errors
        are captured into ``DataFeature.error`` / ``DataQuality.error`` rather
        than raised, matching the Java ``processDatasetWithError`` fallthrough.
        """
        client = self._get_client()
        base_url = client.base_url
        try:
            dsd = get_dataset_description_xml(did, base_url)
            default_target = dsd.get("oml:default_target_attribute")
            status = dsd.get("oml:status")

            info = get_data_and_meta_information_from_did(did, base_url=base_url)

            features = (
                load_arff_features(
                    info,
                    did=did,
                    evaluation_engine_id=EVALUATION_ENGINE_ID,
                )
                if default_target is not None
                else DataFeature(
                    did=did,
                    evaluation_engine_id=EVALUATION_ENGINE_ID,
                    error="Dataset has no default_target_attribute; cannot extract features.",
                )
            )

            # Java: ProcessDataset.java:74-80 — swallow 441 (already uploaded),
            # propagate every other ApiException.
            try:
                client.data_features_upload(features)
            except OpenmlApiError as ae:
                if ae.code != _CODE_FEATURES_ALREADY_UPLOADED:
                    raise

            # Java: ProcessDataset.java:81-83 — flip PREP → ACTIVE.
            if status == DATA_STATUS_PREP:
                client.data_status_update(did, DATA_STATUS_ACTIVE)

            qualities = load_arff_qualities(
                info,
                did=did,
                evaluation_engine_id=EVALUATION_ENGINE_ID,
            )
            client.data_qualities_upload(qualities)

            self.last_features = features
            self.last_qualities = qualities
            return features, qualities

        except OpenmlApiError as e:
            # Java: ProcessDataset.java:88-95 — code 431 means the dataset was
            # already fully processed; we log and bail without re-uploading.
            if e.code == _CODE_DATASET_ALREADY_PROCESSED:
                self.last_features = DataFeature(
                    did=did,
                    evaluation_engine_id=EVALUATION_ENGINE_ID,
                    error=f"Dataset already processed: {e.message}",
                )
                self.last_qualities = DataQuality(
                    did=did,
                    evaluation_engine_id=EVALUATION_ENGINE_ID,
                    error=f"Dataset already processed: {e.message}",
                )
                return self.last_features, self.last_qualities
            self._record_error(did, str(e))
            return self.last_features, self.last_qualities

        except Exception as exc:  # noqa: BLE001 — matches Java's catch-all
            self._record_error(did, str(exc))
            return self.last_features, self.last_qualities

    def _record_error(self, did: int, error: str) -> None:
        """Port of ``ProcessDataset.processDatasetWithError`` — capture the
        error on both objects for inspection, and upload just the feature
        (Java uploads only the feature, not qualities)."""
        self.last_features = DataFeature(
            did=did,
            evaluation_engine_id=EVALUATION_ENGINE_ID,
            error=error,
        )
        self.last_qualities = DataQuality(
            did=did,
            evaluation_engine_id=EVALUATION_ENGINE_ID,
            error=error,
        )
        try:
            self._get_client().data_features_upload(self.last_features)
        except Exception:
            # Java: if the error upload itself fails, there's nothing more to
            # do — the dataset will remain "unprocessed" and be retried.
            pass

    # ----------------------------------------------------------------------
    # Print-only path (Java: process_dataset_print)
    # ----------------------------------------------------------------------

    def process_and_print(self, did: int) -> None:
        """Port of Java's ``processAndPrint`` — compute features locally and
        print them as XML, without any upload. Java only prints features here
        (no qualities), so we match that."""
        info = get_data_and_meta_information_from_did(
            did, base_url=self._get_client().base_url
        )
        features = load_arff_features(
            info,
            did=did,
            evaluation_engine_id=EVALUATION_ENGINE_ID,
        )
        self.last_features = features
        print(features_to_xml(features))

    # ----------------------------------------------------------------------
    # Polling loop
    # ----------------------------------------------------------------------

    def poll(self) -> None:
        """Port of ``ProcessDataset.java:35-43``.

        Calls ``dataUnprocessed(engine_id, mode)`` in a loop, processes each
        returned dataset id, and stops when the server returns the
        "No unprocessed datasets" ``ApiException`` (matched by message string,
        same as Java's ``fetchUnprocessed``)."""
        client = self._get_client()
        while True:
            try:
                dids = client.data_unprocessed(EVALUATION_ENGINE_ID, self.mode)
            except OpenmlApiError as e:
                if "No unprocessed" in e.message:
                    return
                raise
            if not dids:
                return
            for did in dids:
                self.process(did)
