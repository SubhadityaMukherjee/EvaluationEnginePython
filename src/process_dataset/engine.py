"""Python port of ``org.openml.webapplication.ProcessDataset``.

Two entry points:
  * ``ProcessDataset(dataset_id=...)`` — process one dataset (or, without an
    id, poll the server for unprocessed datasets). Currently the polling loop
    raises ``NotImplementedError``; see module docstring of
    ``src.evaluate_run`` for the same pattern.
  * ``ProcessDataset().process_and_print(did)`` — local-only feature
    extraction, prints the feature XML. Mirrors Java's ``process_dataset_print``.

Out of scope (marked with TODOs):
  * ``dataFeaturesUpload`` — feature upload.
  * ``dataStatusUpdate`` — flipping dataset status from PREP to ACTIVE.
  * ``dataQualitiesUpload`` — qualities upload.
  * ``dataUnprocessed`` polling.
"""

from __future__ import annotations

from typing import Optional

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


class ProcessDataset:
    """Port of ``ProcessDataset``. Construct with a ``dataset_id`` to process
    one dataset immediately (mirrors Java's 3-arg constructor); construct
    without one to get an instance ready for ``process_and_print`` or
    ``poll``. The ``DataFeature`` / ``DataQuality`` from the last call are
    kept on ``self.last_features`` / ``self.last_qualities`` for inspection
    (Java uploads them as a side effect — TODO here)."""

    def __init__(self, dataset_id: Optional[int] = None, mode: str = "normal") -> None:
        self.mode = mode
        self.last_features: Optional[DataFeature] = None
        self.last_qualities: Optional[DataQuality] = None

        if dataset_id is not None:
            self.process(dataset_id)

    # ----------------------------------------------------------------------
    # Single-dataset path
    # ----------------------------------------------------------------------

    def process(self, did: int) -> tuple[DataFeature, DataQuality]:
        """Port of ``ProcessDataset.process``. Returns ``(features, qualities)``.

        Mirrors Java's structure: download → extract features → (TODO upload
        features, TODO status update) → compute qualities → (TODO upload
        qualities). Errors are captured into ``DataFeature.error`` /
        ``DataQuality.error`` rather than raised, matching the Java
        ``processDatasetWithError`` fallthrough."""
        try:
            dsd = get_dataset_description_xml(did)
            default_target = dsd.get("oml:default_target_attribute")
            status = dsd.get("oml:status")

            info = get_data_and_meta_information_from_did(did)

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

            # TODO: apiconnector.dataFeaturesUpload(features)
            # (Java swallows ApiException code 441 — already uploaded.)

            # TODO: if status == DATA_STATUS_PREP:
            #     apiconnector.dataStatusUpdate(did, DATA_STATUS_ACTIVE)

            qualities = load_arff_qualities(
                info,
                did=did,
                evaluation_engine_id=EVALUATION_ENGINE_ID,
            )
            # TODO: apiconnector.dataQualitiesUpload(qualities)

            self.last_features = features
            self.last_qualities = qualities
            return features, qualities

        except Exception as exc:  # noqa: BLE001 — matches Java's catch-all
            err = str(exc)
            self.last_features = DataFeature(
                did=did,
                evaluation_engine_id=EVALUATION_ENGINE_ID,
                error=err,
            )
            self.last_qualities = DataQuality(
                did=did,
                evaluation_engine_id=EVALUATION_ENGINE_ID,
                error=err,
            )
            # TODO: apiconnector.dataFeaturesUpload(self.last_features)
            return self.last_features, self.last_qualities

    # ----------------------------------------------------------------------
    # Print-only path (Java: process_dataset_print)
    # ----------------------------------------------------------------------

    def process_and_print(self, did: int) -> None:
        """Port of Java's ``processAndPrint`` — compute features locally and
        print them as XML, without any upload. Java only prints features here
        (no qualities), so we match that."""
        info = get_data_and_meta_information_from_did(did)
        features = load_arff_features(
            info,
            did=did,
            evaluation_engine_id=EVALUATION_ENGINE_ID,
        )
        self.last_features = features
        print(features_to_xml(features))

    # ----------------------------------------------------------------------
    # Polling loop (TODO)
    # ----------------------------------------------------------------------

    def poll(self) -> None:
        """Port of ProcessDataset.java:35-43.

        Original behaviour: call ``dataUnprocessed(engine_id, mode)`` in a
        loop, process each returned dataset id, stop when the server returns
        the "No unprocessed datasets" ApiException.
        """
        # TODO: implement once uploads land. Without uploads the loop has no
        # observable side effect, so we refuse to spin rather than burn the
        # server.
        raise NotImplementedError(
            "ProcessDataset polling loop is not implemented yet — pass a dataset_id."
        )
