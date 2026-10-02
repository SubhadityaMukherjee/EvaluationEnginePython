import arff as arff_lib
import pandas as pd
import pytest

from src.process_dataset.arff import arff_head, save_splits_arff, splits_to_arff


@pytest.fixture
def splits():
    return pd.DataFrame(
        [
            ("TRAIN", 0, 0, 0),
            ("TEST", 1, 0, 0),
        ],
        columns=["type", "rowid", "repeat", "fold"],
    )


@pytest.fixture
def splits_with_sample():
    return pd.DataFrame(
        [
            ("TRAIN", 0, 0, 0, 0),
            ("TEST", 1, 0, 0, 1),
        ],
        columns=["type", "rowid", "repeat", "fold", "sample"],
    )


class TestSplitsToArff:
    def test_four_column_schema(self, splits):
        text = splits_to_arff(splits)

        payload = arff_lib.loads(text)
        assert payload["relation"] == "splits"
        assert payload["attributes"] == [
            ("type", ["TRAIN", "TEST"]),
            ("rowid", "NUMERIC"),
            ("repeat", "NUMERIC"),
            ("fold", "NUMERIC"),
        ]
        assert payload["data"] == [["TRAIN", 0, 0, 0], ["TEST", 1, 0, 0]]

    def test_sample_column_included_when_present(self, splits_with_sample):
        payload = arff_lib.loads(splits_to_arff(splits_with_sample))

        assert [name for name, _ in payload["attributes"]] == [
            "type",
            "rowid",
            "repeat",
            "fold",
            "sample",
        ]

    def test_column_order_is_canonical_regardless_of_df_order(self):
        scrambled = pd.DataFrame(
            {
                "fold": [0, 0],
                "type": ["TRAIN", "TEST"],
                "repeat": [0, 0],
                "rowid": [0, 1],
            }
        )

        payload = arff_lib.loads(splits_to_arff(scrambled))

        assert [name for name, _ in payload["attributes"]] == [
            "type",
            "rowid",
            "repeat",
            "fold",
        ]
        assert payload["data"] == [["TRAIN", 0, 0, 0], ["TEST", 1, 0, 0]]

    def test_custom_relation(self, splits):
        text = splits_to_arff(splits, relation="weka_generated_folds")

        assert arff_lib.loads(text)["relation"] == "weka_generated_folds"


class TestSaveSplitsArff:
    def test_writes_parseable_file(self, splits, tmp_path):
        path = tmp_path / "splits.arff"

        save_splits_arff(splits, str(path))

        assert path.exists()
        payload = arff_lib.loads(path.read_text())
        assert len(payload["data"]) == 2


class TestArffHead:
    def test_first_n_lines(self):
        text = "\n".join(f"line{i}" for i in range(10))

        head = arff_head(text, n=3)

        assert head == "line0\nline1\nline2"
