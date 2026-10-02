import json
from unittest.mock import MagicMock

import arff as arff_lib
import pytest

from src.process_dataset.merge import (
    DATASET_ID_ATTRIBUTE,
    LEARNING_CURVE_TASK_IDS,
    MergeDataset,
)


def make_task_inputs(task_type_id, ids):
    return {
        "oml:task_type_id": str(task_type_id),
        "oml:input": [{"@name": "source_data_list", "#text": json.dumps(ids)}],
    }


class TestParseSourceDataList:
    def test_sorted_unique_ids(self):
        ids = MergeDataset._parse_source_data_list(make_task_inputs(8, [3, 1, 3]))
        assert ids == [1, 3]

    def test_value_via_oml_value_field(self):
        ti = {
            "oml:input": [
                {
                    "@name": "source_data_list",
                    "oml:value": json.dumps([7, 2]),
                }
            ]
        }

        assert MergeDataset._parse_source_data_list(ti) == [2, 7]

    def test_empty_list_raises(self):
        ti = {"oml:input": [{"@name": "source_data_list", "#text": "[]"}]}

        with pytest.raises(ValueError, match="nothing to merge"):
            MergeDataset._parse_source_data_list(ti)

    def test_missing_input_raises(self):
        with pytest.raises(ValueError, match="source_data_list"):
            MergeDataset._parse_source_data_list({"oml:input": []})

    def test_single_input_dict_form(self):
        ti = {
            "oml:input": {
                "@name": "source_data_list",
                "#text": json.dumps([5, 4]),
            }
        }

        assert MergeDataset._parse_source_data_list(ti) == [4, 5]


class TestConstructor:
    def test_rejects_learning_curve_tasks(self, mocker):
        mocker.patch(
            "src.process_dataset.merge.get_task_inputs_xml",
            return_value=make_task_inputs(LEARNING_CURVE_TASK_IDS[0], [1]),
        )

        with pytest.raises(ValueError, match="MultiTask"):
            MergeDataset(1, MagicMock())

    def _wire_downloads(self, mocker, tmp_path, datasets):
        """Patch get_task_inputs_xml + client.data_get + download_to_temp_file
        to serve the given {did: arff_text} datasets from tmp_path."""
        ids = sorted(datasets)
        mocker.patch(
            "src.process_dataset.merge.get_task_inputs_xml",
            return_value=make_task_inputs(8, ids),
        )
        client = MagicMock()
        client.base_url = "https://test.openml.org/api/v1/"
        client.data_get.side_effect = lambda did: {
            "oml:url": f"https://example.org/{did}.arff"
        }
        paths = {}
        for did, text in datasets.items():
            p = tmp_path / f"{did}.arff"
            p.write_text(text)
            paths[f"https://example.org/{did}.arff"] = str(p)
        mocker.patch(
            "src.process_dataset.merge.download_to_temp_file",
            side_effect=lambda url, suffix=".arff": paths[url],
        )
        return client

    def test_downloads_and_prepends_dataset_id_column(self, mocker, tmp_path):
        client = self._wire_downloads(
            mocker,
            tmp_path,
            {
                1: "@relation a\n@attribute x numeric\n@data\n1\n2\n",
                2: "@relation b\n@attribute x numeric\n@data\n3\n",
            },
        )

        md = MergeDataset(99, client)

        assert set(md.datasets) == {"1", "2"}
        for did, payload in md.datasets.items():
            assert payload["attributes"][0] == (
                DATASET_ID_ATTRIBUTE,
                [did],
            )
            assert all(row[0] == did for row in payload["data"])


def make_payload(did, nominal=("a", "b"), rows=None):
    return {
        "relation": f"ds{did}",
        "attributes": [
            (DATASET_ID_ATTRIBUTE, [str(did)]),
            ("cat", list(nominal)),
            ("x", "NUMERIC"),
        ],
        "data": [[str(did), c, v] for c, v in (rows or [])],
    }


@pytest.fixture
def merged_two_dataset_instance():
    md = MergeDataset.__new__(MergeDataset)
    md.task_id = 99
    md.client = MagicMock()
    md.datasets = {
        "1": make_payload(1, ("a", "b"), [("a", 1), ("b", 2)]),
        "2": make_payload(2, ("b", "c"), [("c", 3), ("b", 4)]),
    }
    return md


class TestVerifyAttributeSet:
    def test_same_names_pass(self, merged_two_dataset_instance):
        assert merged_two_dataset_instance.verify_attribute_set() is True

    def test_different_names_fail(self, merged_two_dataset_instance):
        md = merged_two_dataset_instance
        md.datasets["2"]["attributes"][1] = ("other", ["b", "c"])

        assert md.verify_attribute_set() is False

    def test_empty_datasets_fail(self):
        md = MergeDataset.__new__(MergeDataset)
        md.datasets = {}

        assert md.verify_attribute_set() is False


class TestMerge:
    def test_relation_name_from_sorted_ids(self, merged_two_dataset_instance):
        text = merged_two_dataset_instance.merge()

        assert arff_lib.loads(text)["relation"] == "merged_1_2"

    def test_nominal_categories_are_unioned(self, merged_two_dataset_instance):
        payload = arff_lib.loads(merged_two_dataset_instance.merge())

        cat_attr = payload["attributes"][1]
        assert cat_attr == ("cat", ["a", "b", "c"])

    def test_rows_concatenated_in_sorted_id_order(self, merged_two_dataset_instance):
        payload = arff_lib.loads(merged_two_dataset_instance.merge())

        assert payload["data"] == [
            ["1", "a", 1],
            ["1", "b", 2],
            ["2", "c", 3],
            ["2", "b", 4],
        ]

    def test_numeric_attributes_kept_from_first_dataset(
        self, merged_two_dataset_instance
    ):
        payload = arff_lib.loads(merged_two_dataset_instance.merge())

        assert payload["attributes"][2] == ("x", "NUMERIC")

    def test_mismatched_attributes_raise(self, merged_two_dataset_instance):
        md = merged_two_dataset_instance
        md.datasets["2"]["attributes"][1] = ("other", ["b", "c"])

        with pytest.raises(ValueError, match="attribute names differ"):
            md.merge()

    def test_end_to_end_merge_via_downloads(self, mocker, tmp_path):
        mocker.patch(
            "src.process_dataset.merge.get_task_inputs_xml",
            return_value=make_task_inputs(8, [5, 4]),
        )
        client = MagicMock()
        client.base_url = "https://test.openml.org/api/v1/"
        client.data_get.side_effect = lambda did: {
            "oml:url": f"https://example.org/{did}.arff"
        }
        texts = {
            4: "@relation four\n@attribute cat {x, y}\n@data\nx\ny\n",
            5: "@relation five\n@attribute cat {y, z}\n@data\nz\n",
        }
        paths = {}
        for did, text in texts.items():
            p = tmp_path / f"{did}.arff"
            p.write_text(text)
            paths[f"https://example.org/{did}.arff"] = str(p)
        mocker.patch(
            "src.process_dataset.merge.download_to_temp_file",
            side_effect=lambda url, suffix=".arff": paths[url],
        )

        md = MergeDataset(99, client)
        merged = arff_lib.loads(md.merge())

        assert merged["relation"] == "merged_4_5"
        assert merged["attributes"][1] == ("cat", ["x", "y", "z"])
        assert merged["data"] == [["4", "x"], ["4", "y"], ["5", "z"]]
