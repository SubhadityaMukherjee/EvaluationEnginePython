from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

# Dataset file formats supported throughout the engine. Picked once at the
# edge (CLI flag / notebook) and threaded through download + parse.
DataFormat = Literal["arff", "parquet"]

# ============================================================================
# Dataset metadata
# ============================================================================


@dataclass(slots=True)
class Quality:
    name: str
    value: float | None = None

    def __str__(self) -> str:
        return f"{self.name} - {self.value}"


@dataclass()
class DatasetDownloadInfo:
    file_path: str
    default_target_attribute: str | None


# ============================================================================
# Feature models
# ============================================================================


@dataclass()
class Feature:
    index: int
    name: str
    data_type: str

    nominal_values: list[str] = field(default_factory=list)

    is_target: bool = False
    is_ignore: bool = False
    is_row_identifier: bool = False

    number_of_distinct_values: int | None = None
    number_of_unique_values: int | None = None
    number_of_missing_values: int | None = None
    number_of_integer_values: int | None = None
    number_of_real_values: int | None = None
    number_of_nominal_values: int | None = None
    number_of_values: int | None = None

    maximum_value: float | None = None
    minimum_value: float | None = None
    mean_value: float | None = None
    standard_deviation: float | None = None

    class_distribution: str | None = None

    def __str__(self) -> str:
        return f"{self.index} - {self.name}"


@dataclass()
class DataFeature:
    did: int | None = None
    evaluation_engine_id: int | None = None
    features: list[Feature] = field(default_factory=list)
    error: str | None = None

    def feature_map(
        self,
        *,
        sorted_names: bool = False,
    ) -> dict[str, Feature]:
        result = {f.name: f for f in self.features}
        return dict(sorted(result.items())) if sorted_names else result


@dataclass()
class DataQuality:
    did: int | None = None
    evaluation_engine_id: int | None = None
    qualities: list[Quality] = field(default_factory=list)
    error: str | None = None

    def quality_map(
        self,
        *,
        sorted_names: bool = False,
    ) -> dict[str, Quality]:
        result: dict[str, Quality] = {f.name: f for f in self.qualities}
        return dict(sorted(result.items())) if sorted_names else result


# ============================================================================
# Run evaluation models
# ============================================================================


@dataclass
class EvaluationScore:
    """One computed metric. Mirrors ``org.openml.apiconnector.xml.EvaluationScore``."""

    function: str
    value: float | None = None
    stdev: float | None = None
    array: list | None = None
    repeat: int | None = None
    fold: int | None = None
    sample: int | None = None
    sample_size: int | None = None


@dataclass
class RunEvaluation:
    """Aggregated result of evaluating one run. Mirrors ``RunEvaluation``."""

    run_id: int | None = None
    # Canonical evaluation-engine id; mirrors ``EVALUATION_ENGINE_ID`` in
    # ``src.runs.evaluators`` (kept there to avoid a circular import).
    evaluation_engine_id: int = 1
    scores: list[EvaluationScore] = field(default_factory=list)
    error: str | None = None
    warning: str | None = None

    def add_scores(self, scores: Iterable[EvaluationScore]) -> None:
        self.scores.extend(scores)


# ============================================================================
# Constants
# ============================================================================

_NUMERIC_TYPES = frozenset({"NUMERIC", "REAL", "INTEGER"})

_OML_STR_FIELDS = "name"
_OML_BOOL_FIELDS = (
    "is_target",
    "is_ignore",
    "is_row_identifier",
)

_OML_INT_FIELDS = (
    "number_of_missing_values",
    "number_of_distinct_values",
    "number_of_unique_values",
    "number_of_integer_values",
    "number_of_real_values",
    "number_of_nominal_values",
    "number_of_values",
)

_OML_FLOAT_FIELDS = (
    "maximum_value",
    "minimum_value",
    "mean_value",
    "standard_deviation",
)


class EstimationProcedureType(str, Enum):
    CROSSVALIDATION = "CROSSVALIDATION"
    HOLDOUT = "HOLDOUT"
    HOLDOUT_ORDERED = "HOLDOUT_ORDERED"
    LEAVEONEOUT = "LEAVEONEOUT"
    TESTONTRAININGDATA = "TESTONTRAININGDATA"
    LEARNINGCURVE_CV = "LEARNINGCURVE_CV"


@dataclass(frozen=True)
class EstimationProcedure:
    """Estimation-procedure configuration read by the Java dispatcher.

    ``folds`` / ``repeats`` / ``percentage`` correspond to the procedure fields
    consumed by ``GenerateFolds.java``; ``percentage`` is a test-set size in the
    range 0..100.
    """

    type: EstimationProcedureType
    folds: int | None = None
    repeats: int | None = None
    percentage: float | None = None
