import pytest

from src.exceptions import (
    OpenmlApiError,
    OpenmlConfigError,
    OpenmlError,
    PredictionValidationError,
)


class TestOpenmlApiError:
    def test_message_includes_code(self):
        err = OpenmlApiError(431, "Dataset already processed")
        assert str(err) == "[431] Dataset already processed"

    def test_attributes(self):
        err = OpenmlApiError(1013, "no runs left")
        assert err.code == 1013
        assert err.message == "no runs left"

    def test_is_openml_error(self):
        assert isinstance(OpenmlApiError(1, "x"), OpenmlError)


class TestPredictionValidationError:
    def test_dual_inheritance(self):
        err = PredictionValidationError("bad predictions")
        assert isinstance(err, OpenmlError)
        assert isinstance(err, ValueError)

    def test_catchable_as_value_error(self):
        with pytest.raises(ValueError):
            raise PredictionValidationError("missing row_id")


class TestHierarchy:
    @pytest.mark.parametrize(
        "exc",
        [
            OpenmlApiError(1, "x"),
            OpenmlConfigError("no key"),
            PredictionValidationError("bad"),
        ],
    )
    def test_all_derive_from_openml_error(self, exc):
        assert isinstance(exc, OpenmlError)
