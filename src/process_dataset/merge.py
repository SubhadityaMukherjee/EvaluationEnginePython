"""Python port of ``org.openml.webapplication.MergeDataset``.

Given a MultiTask task id, downloads every dataset listed in the task's
``source_data_list`` input, prepends an ``openml-dataset-id`` nominal column
to each, verifies the attribute names agree across datasets, and concatenates
the rows into a single ARFF. Writes to stdout or ``--output``.

Mirrors Java's behaviour point for point, including:
  * Rejecting learning-curve tasks (``Settings.LEARNING_CURVE_TASK_IDS = {3}``).
  * Using only attribute *names* for the compatibility check (types are taken
    from the first dataset; mismatches produce undefined output, as in Java).
  * Relation name ``merged_<sorted ids joined by _>``.

The Java version converts nominal → STRING mid-merge then re-applies a
``StringToNominal`` filter to collect the union of category values; we skip
the round-trip and build the unioned nominal declaration directly.
"""

from __future__ import annotations

import json
from typing import Optional

import arff

from src.client import OpenmlClient
from src.helpers import download_to_temp_file, get_task_inputs_xml

# Java's Settings.LEARNING_CURVE_TASK_IDS.
LEARNING_CURVE_TASK_IDS: tuple[int, ...] = (3,)

# Attribute name prepended to every row identifying the source dataset.
# Mirrors MergeDataset.java:67.
DATASET_ID_ATTRIBUTE = "openml-dataset-id"


class MergeDataset:
    """Port of ``MergeDataset``. Construct with a ``task_id`` and a client;
    call ``merge()`` to get the merged ARFF string."""

    def __init__(self, task_id: int, client: OpenmlClient) -> None:
        self.task_id = task_id
        self.client = client
        # dataset_id (str) → parsed liac-arff payload, each with the
        # openml-dataset-id column already prepended.
        self.datasets: dict[str, dict] = {}

        ti = get_task_inputs_xml(task_id, self.client.base_url)
        task_type_id = int(ti["oml:task_type_id"])
        if task_type_id in LEARNING_CURVE_TASK_IDS:
            raise ValueError(
                f"MergeDataset requires a MultiTask task, but task {self.task_id} "
                f"has task_type_id {task_type_id}."
            )

        data_ids = self._parse_source_data_list(ti)
        self._download_datasets(data_ids)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_source_data_list(task_inputs: dict) -> list[int]:
        """Extract the ``source_data_list`` JSON array from the task inputs
        XML and return a sorted list of unique dataset ids. Mirrors
        MergeDataset.java:54-62."""
        inputs = task_inputs.get("oml:input", [])
        if isinstance(inputs, dict):  # single-input edge case
            inputs = [inputs]
        for inp in inputs:
            if inp.get("@name") == "source_data_list":
                raw = inp.get("#text") or inp.get("oml:value") or ""
                ids = sorted({int(x) for x in json.loads(raw)})
                if not ids:
                    raise ValueError(
                        "Task 'source_data_list' is empty; nothing to merge."
                    )
                return ids
        raise ValueError(
            f"Task has no source_data_list input — is task_type_id MultiTask?"
        )

    def _download_datasets(self, dataset_ids: list[int]) -> None:
        """Download each dataset's ARFF via the client (server-aware), parse
        with liac-arff, and prepend the ``openml-dataset-id`` nominal column.
        Mirrors MergeDataset.downloadDatasets."""
        for did in dataset_ids:
            desc = self.client.data_get(did)
            url = desc["oml:url"]
            file_path = download_to_temp_file(url, suffix=".arff")
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                payload = arff.load(f)

            # Prepend the openml-dataset-id nominal attribute. Its category
            # list contains just this dataset's id (Java parity).
            payload["attributes"] = [
                (DATASET_ID_ATTRIBUTE, [str(did)])
            ] + payload["attributes"]
            for row in payload["data"]:
                row.insert(0, str(did))

            self.datasets[str(did)] = payload

    # ------------------------------------------------------------------
    # Merge (Java: merge())
    # ------------------------------------------------------------------

    def verify_attribute_set(self) -> bool:
        """All datasets share the same attribute *names*. Java checks names
        only (MergeDataset.java:84-92)."""
        if not self.datasets:
            return False
        names_sets = [
            {name for name, _ in ds["attributes"]} for ds in self.datasets.values()
        ]
        first = names_sets[0]
        return all(names == first for names in names_sets[1:])

    def merge(self) -> str:
        """Merge all datasets into one ARFF string. Mirrors MergeDataset.merge
        — verifies attributes, builds a relation name from sorted ids, unions
        nominal categories across datasets, concatenates rows."""
        if not self.verify_attribute_set():
            raise ValueError(
                f"Cannot merge: attribute names differ across the "
                f"{len(self.datasets)} source datasets."
            )

        sorted_ids = sorted(self.datasets.keys())
        # Java: "merged" + keys.toString() munged to remove brackets/spaces.
        relation = "merged_" + "_".join(sorted_ids)

        # First dataset (by sorted id) dictates attribute names + order.
        # Nominal categories are unioned across all datasets so every value
        # has a home in the merged declaration.
        first = self.datasets[sorted_ids[0]]
        merged_attributes: list[tuple[str, object]] = []
        for attr_idx, (name, type_spec) in enumerate(first["attributes"]):
            if isinstance(type_spec, list):
                # Nominal — union categories across datasets, deterministic order.
                union: list[str] = []
                seen: set[str] = set()
                for did in sorted_ids:
                    other_spec = self.datasets[did]["attributes"][attr_idx][1]
                    if isinstance(other_spec, list):
                        for value in other_spec:
                            if value not in seen:
                                seen.add(value)
                                union.append(value)
                merged_attributes.append((name, union))
            else:
                merged_attributes.append((name, type_spec))

        merged_data: list = []
        for did in sorted_ids:
            merged_data.extend(self.datasets[did]["data"])

        return arff.dumps(
            {
                "relation": relation,
                "attributes": merged_attributes,
                "data": merged_data,
            }
        )
