from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


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
    default_target_attribute: Optional[str]


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

    number_of_distinct_values: Optional[int] = None
    number_of_unique_values: Optional[int] = None
    number_of_missing_values: Optional[int] = None
    number_of_integer_values: Optional[int] = None
    number_of_real_values: Optional[int] = None
    number_of_nominal_values: Optional[int] = None
    number_of_values: Optional[int] = None

    maximum_value: Optional[float] = None
    minimum_value: Optional[float] = None
    mean_value: Optional[float] = None
    standard_deviation: Optional[float] = None

    class_distribution: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.index} - {self.name}"


@dataclass()
class DataFeature:
    did: Optional[int] = None
    evaluation_engine_id: Optional[int] = None
    features: list[Feature] = field(default_factory=list)
    error: Optional[str] = None

    def feature_map(
        self,
        *,
        sorted_names: bool = False,
    ) -> dict[str, Feature]:
        result = {f.name: f for f in self.features}
        return dict(sorted(result.items())) if sorted_names else result


@dataclass()
class DataQuality:
    did: Optional[int] = None
    evaluation_engine_id: Optional[int] = None
    qualities: list[Quality] = field(default_factory=list)
    error: Optional[str] = None

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
    value: Optional[float] = None
    stdev: Optional[float] = None
    array: Optional[list] = None
    repeat: Optional[int] = None
    fold: Optional[int] = None
    sample: Optional[int] = None
    sample_size: Optional[int] = None


@dataclass
class RunEvaluation:
    """Aggregated result of evaluating one run. Mirrors ``RunEvaluation``."""

    run_id: Optional[int] = None
    # Canonical evaluation-engine id; mirrors ``EVALUATION_ENGINE_ID`` in
    # ``src.runs.evaluators`` (kept there to avoid a circular import).
    evaluation_engine_id: int = 1
    scores: list[EvaluationScore] = field(default_factory=list)
    error: Optional[str] = None
    warning: Optional[str] = None

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


@dataclass(slots=True)
class Quality:
    name: str
    value: float | None = None

    def __str__(self) -> str:
        return f"{self.name} - {self.value}"


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
    folds: Optional[int] = None
    repeats: Optional[int] = None
    percentage: Optional[float] = None
