import xmltodict

from src.models import DataQuality, Quality
from src.qualities.serialization import (
    parse_qualities_xml,
    qualities_to_oml_dict,
    qualities_to_xml,
    quality_to_oml_dict,
)


class TestQualityToOmlDict:
    def test_with_value(self):
        d = quality_to_oml_dict(Quality(name="NumberOfInstances", value=14.0))

        assert d == {"oml:name": "NumberOfInstances", "oml:value": "14.0"}

    def test_none_value(self):
        d = quality_to_oml_dict(Quality(name="NumberOfClasses", value=None))

        assert d == {"oml:name": "NumberOfClasses", "oml:value": None}


class TestQualitiesToOmlDict:
    def test_envelope(self):
        dq = DataQuality(
            did=61,
            qualities=[Quality(name="a", value=1.0), Quality(name="b")],
        )

        d = qualities_to_oml_dict(dq)

        inner = d["oml:data_qualities"]
        assert inner["@xmlns:oml"] == "http://openml.org/openml"
        assert len(inner["oml:quality"]) == 2


class TestQualitiesXmlRoundTrip:
    def test_round_trip(self):
        dq = DataQuality(
            did=61,
            qualities=[
                Quality(name="NumberOfInstances", value=14.0),
                Quality(name="NumberOfClasses", value=None),
            ],
        )

        xml = qualities_to_xml(dq)
        parsed = parse_qualities_xml(xml)

        quals = parsed["oml:data_qualities"]["oml:quality"]
        assert isinstance(quals, list) and len(quals) == 2
        assert quals[0]["oml:name"] == "NumberOfInstances"
        assert quals[0]["oml:value"] == "14.0"
        assert quals[1]["oml:value"] is None

    def test_unparse_parse_minimal(self):
        dq = DataQuality(qualities=[Quality(name="x", value=0.5)])

        reparsed = xmltodict.parse(qualities_to_xml(dq, pretty=False))
        assert reparsed["oml:data_qualities"]["oml:quality"]["oml:name"] == "x"
