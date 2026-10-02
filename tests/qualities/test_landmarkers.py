import numpy as np
import pytest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from src.qualities.landmarkers import (
    _MEASURES,
    ALL_LANDMARKERS,
    compute_all_landmarkers,
    expected_landmarker_ids,
    generic_landmarker,
)


@pytest.fixture
def separable():
    X = np.array([[0.0], [0.1], [0.2], [0.3], [2.0], [2.1], [2.2], [2.3]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    return X, y


class TestExpectedLandmarkerIds:
    def test_count(self):
        ids = expected_landmarker_ids()

        assert len(ids) == len(ALL_LANDMARKERS) * len(_MEASURES) == 36

    def test_suffixes(self):
        ids = expected_landmarker_ids()

        for name, _ in ALL_LANDMARKERS:
            for suffix in _MEASURES:
                assert f"{name}{suffix}" in ids


class TestGenericLandmarker:
    def test_perfectly_separable_data(self, separable):
        X, y = separable

        result = generic_landmarker("TestNB", GaussianNB(), X, y)

        assert set(result) == {
            "TestNBAUC",
            "TestNBErrRate",
            "TestNBKappa",
        }
        assert result["TestNBAUC"] == pytest.approx(1.0)
        assert result["TestNBErrRate"] == pytest.approx(0.0)
        assert result["TestNBKappa"] == pytest.approx(1.0)

    def test_single_class_returns_all_none(self):
        X = np.array([[0.0], [1.0]])
        y = np.array([5, 5])

        result = generic_landmarker("X", DecisionTreeClassifier(), X, y)

        assert result == {"XAUC": None, "XErrRate": None, "XKappa": None}

    def test_noisy_data_has_nonzero_error(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 2))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        result = generic_landmarker("Stump", DecisionTreeClassifier(max_depth=1), X, y)

        assert result["StumpErrRate"] is not None
        assert result["StumpKappa"] is not None
        assert result["StumpErrRate"] > 0.0
        assert result["StumpKappa"] < 1.0


class TestComputeAllLandmarkers:
    def test_returns_every_expected_id(self, separable):
        X, y = separable

        result = compute_all_landmarkers(X, y)

        assert sorted(result) == sorted(expected_landmarker_ids())
