import xmltodict

from src.features.serialization import (
    feature_to_oml_dict,
    features_to_oml_dict,
    features_to_xml,
    parse_features_xml,
)
from src.models import DataFeature, Feature


class TestFeatureToOmlDict:
    def test_minimal_feature(self):
        feat = Feature(index=0, name="age", data_type="numeric")

        d = feature_to_oml_dict(feat)

        assert d == {
            "oml:index": "0",
            "oml:name": "age",
            "oml:data_type": "numeric",
            # bool fields default to False and are always emitted
            "oml:is_target": "false",
            "oml:is_ignore": "false",
            "oml:is_row_identifier": "false",
        }

    def test_nominal_values_included(self):
        feat = Feature(
            index=1,
            name="outlook",
            data_type="nominal",
            nominal_values=["overcast", "rainy", "sunny"],
        )

        d = feature_to_oml_dict(feat)

        assert d["oml:nominal_value"] == ["overcast", "rainy", "sunny"]

    def test_booleans_are_lowercased(self):
        feat = Feature(
            index=0,
            name="class",
            data_type="nominal",
            is_target=True,
            is_ignore=False,
            is_row_identifier=True,
        )

        d = feature_to_oml_dict(feat)

        assert d["oml:is_target"] == "true"
        assert d["oml:is_ignore"] == "false"
        assert d["oml:is_row_identifier"] == "true"

    def test_numeric_stat_fields(self):
        feat = Feature(
            index=0,
            name="age",
            data_type="numeric",
            number_of_missing_values=2,
            number_of_distinct_values=5,
            minimum_value=1.5,
            maximum_value=9.0,
        )

        d = feature_to_oml_dict(feat)

        assert d["oml:number_of_missing_values"] == "2"
        assert d["oml:number_of_distinct_values"] == "5"
        assert d["oml:minimum_value"] == "1.5"
        assert d["oml:maximum_value"] == "9.0"

    def test_class_distribution_included(self):
        feat = Feature(
            index=0,
            name="play",
            data_type="nominal",
            class_distribution="no:2,yes:4",
        )

        assert feature_to_oml_dict(feat)["oml:class_distribution"] == "no:2,yes:4"

    def test_none_fields_omitted(self):
        feat = Feature(index=0, name="age", data_type="numeric")

        d = feature_to_oml_dict(feat)

        assert "oml:number_of_missing_values" not in d
        assert "oml:class_distribution" not in d
        assert "oml:nominal_value" not in d


class TestFeaturesToOmlDict:
    def test_envelope(self):
        df = DataFeature(
            did=61,
            features=[
                Feature(index=0, name="age", data_type="numeric"),
                Feature(index=1, name="class", data_type="nominal"),
            ],
        )

        d = features_to_oml_dict(df)

        inner = d["oml:data_features"]
        assert inner["@xmlns:oml"] == "http://openml.org/openml"
        assert [f["oml:name"] for f in inner["oml:feature"]] == [
            "age",
            "class",
        ]


class TestFeaturesXmlRoundTrip:
    def test_round_trip(self):
        df = DataFeature(
            did=61,
            features=[
                Feature(index=0, name="age", data_type="numeric"),
                Feature(
                    index=1,
                    name="outlook",
                    data_type="nominal",
                    nominal_values=["sunny", "rainy"],
                    is_target=False,
                ),
            ],
        )

        xml = features_to_xml(df)
        parsed = parse_features_xml(xml)

        feats = parsed["oml:data_features"]["oml:feature"]
        assert isinstance(feats, list) and len(feats) == 2
        assert feats[1]["oml:nominal_value"] == ["sunny", "rainy"]

    def test_parse_forces_feature_list_for_single_element(self):
        xml = """<?xml version="1.0" encoding="utf-8"?>
<oml:data_features xmlns:oml="http://openml.org/openml">
  <oml:feature>
    <oml:index>0</oml:index>
    <oml:name>age</oml:name>
    <oml:data_type>numeric</oml:data_type>
  </oml:feature>
</oml:data_features>"""

        parsed = parse_features_xml(xml)

        assert isinstance(parsed["oml:data_features"]["oml:feature"], list)

    def test_unparse_parse_minimal(self):
        df = DataFeature(features=[Feature(index=0, name="a", data_type="string")])
        xml = features_to_xml(df, pretty=False)

        reparsed = xmltodict.parse(xml)["oml:data_features"]["oml:feature"]
        assert reparsed["oml:name"] == "a"
        assert reparsed["oml:data_type"] == "string"
