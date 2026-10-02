import numpy as np
import pandas as pd
import pytest

from src.data_loader import DataLoader, _df_to_attributes_and_rows
from src.models import DatasetDownloadInfo

ARFF_TEXT = """\
@relation test

@attribute outlook {sunny, overcast, rainy}
@attribute temperature numeric
@attribute name STRING

@data
sunny,75.0,alice
rainy,?,bob
?,65.0,?
"""


class TestFormatValidation:
    def test_unsupported_format_raises(self):
        bad_format = "csv"
        with pytest.raises(ValueError, match="Unsupported data format"):
            DataLoader(bad_format)

    def test_format_is_case_insensitive(self):
        assert DataLoader("ARFF").fmt == "arff"
        assert DataLoader("Parquet").fmt == "parquet"


class TestArffBackend:
    def test_loads_attributes_and_rows(self, tmp_path):
        path = tmp_path / "ds.arff"
        path.write_text(ARFF_TEXT)
        dataset = DatasetDownloadInfo(
            file_path=str(path), default_target_attribute="outlook"
        )

        attributes, rows = DataLoader("arff").load(dataset)

        assert attributes == [
            ("outlook", ["sunny", "overcast", "rainy"]),
            ("temperature", "NUMERIC"),
            ("name", "STRING"),
        ]
        assert rows[0] == ["sunny", 75.0, "alice"]
        assert rows[1][1] is None
        assert rows[2][2] is None


class TestParquetBackend:
    def test_round_trip_type_mapping(self, tmp_path):
        df = pd.DataFrame(
            {
                "cat": pd.Categorical(["a", "b", None], categories=["a", "b"]),
                "num": [1.5, None, 3.0],
                "when": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
                "label": ["x", "y", "z"],
            }
        )
        path = tmp_path / "ds.parquet"
        df.to_parquet(path)
        dataset = DatasetDownloadInfo(
            file_path=str(path), default_target_attribute="label"
        )

        attributes, _rows = DataLoader("parquet").load(dataset)

        attr_map = dict(attributes)
        assert attr_map["cat"] == ["a", "b"]
        assert attr_map["num"] == "NUMERIC"
        assert attr_map["when"] == "DATE"
        assert attr_map["label"] == "STRING"

    def test_missing_values_become_none(self):
        df = pd.DataFrame({"num": [1.0, None, 3.0]})

        _, rows = _df_to_attributes_and_rows(df)

        assert rows == [[1.0], [None], [3.0]]


class TestDfToAttributesAndRows:
    def test_categorical_columns_keep_declared_categories(self):
        df = pd.DataFrame({"c": pd.Categorical(["b", "a"], categories=["b", "a"])})

        attributes, _ = _df_to_attributes_and_rows(df)

        assert attributes == [("c", ["b", "a"])]

    def test_numeric_columns_tagged_numeric(self):
        df = pd.DataFrame({"i": [1, 2], "f": [0.5, 1.5]})

        attributes, _ = _df_to_attributes_and_rows(df)

        assert attributes == [("i", "NUMERIC"), ("f", "NUMERIC")]

    def test_datetime_columns_tagged_date(self):
        df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-06-01"])})

        attributes, _ = _df_to_attributes_and_rows(df)

        assert attributes == [("d", "DATE")]

    def test_string_columns_tagged_string(self):
        df = pd.DataFrame({"s": ["hello", "world"]})

        attributes, _ = _df_to_attributes_and_rows(df)

        assert attributes == [("s", "STRING")]

    def test_column_names_stringified(self):
        df = pd.DataFrame({42: [1.0]})

        attributes, _ = _df_to_attributes_and_rows(df)

        assert attributes == [("42", "NUMERIC")]

    def test_categorical_values_come_through_as_labels(self):
        df = pd.DataFrame({"c": pd.Categorical(["b", "a"], categories=["b", "a"])})

        _, rows = _df_to_attributes_and_rows(df)

        assert rows == [["b"], ["a"]]

    def test_nan_replaced_by_none_everywhere(self):
        df = pd.DataFrame(
            {
                "num": [np.nan, 1.0],
                "cat": pd.Categorical([None, "a"], categories=["a"]),
            }
        )

        _, rows = _df_to_attributes_and_rows(df)

        assert rows == [[None, None], [1.0, "a"]]
