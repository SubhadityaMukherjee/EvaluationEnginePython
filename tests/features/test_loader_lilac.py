from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest

from src.features.loader import _liac_type


@pytest.mark.parametrize(
    "type_spec, expected",
    [
        (["yes", "no"], ("nominal", ("yes", "no"))),
        # numeric variants (case-insensitive, matched against NUMERIC_TYPES)
        ("numeric", ("numeric", None)),
        ("NUMERIC", ("numeric", None)),
        ("real", ("numeric", None)),
        ("integer", ("numeric", None)),
        # explicitly special-cased strings
        ("STRING", ("string", None)),
        ("string", ("string", None)),
        ("DATE", ("date", None)),
        ("date", ("date", None)),
        # anything unrecognized just gets lowercased and passed through
        ("relational", ("relational", None)),
    ],
)
def test_lilac_type(type_spec, expected):
    result = _liac_type(type_spec=type_spec)
    assert result == expected
