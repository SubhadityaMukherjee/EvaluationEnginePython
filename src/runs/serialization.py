"""XML serialization for ``RunEvaluation`` — the body of the
``POST /run/evaluate`` upload.

Mirrors the XStream annotations on Java's
``org.openml.apiconnector.xml.RunEvaluation`` /
``EvaluationScore``. Specifically:
  * ``repeat`` / ``fold`` / ``sample`` are XML *attributes* on
    ``<oml:evaluation>`` (Java marks them ``@XStreamAsAttribute``); the rest
    are child elements.
  * ``EvaluationScore.array`` (a Python list) becomes the comma-joined
    ``oml:array_data`` string Java expects.
  * Fields that are ``None`` are omitted entirely (XStream's default for null
    fields), so an upload with only ``function`` set produces a minimal element.
"""

import xmltodict

from src.models import EvaluationScore, RunEvaluation


def evaluation_score_to_oml_dict(score: EvaluationScore) -> dict:
    """One ``<oml:evaluation>`` element. ``@repeat`` / ``@fold`` / ``@sample``
    are xmltodict attribute keys (the leading ``@``); the remaining fields
    are child elements, included only when non-None."""
    result: dict = {}

    # Attributes — Java's @XStreamAsAttribute on repeat/fold/sample.
    for attr in ("repeat", "fold", "sample"):
        v = getattr(score, attr)
        if v is not None:
            result[f"@{attr}"] = str(v)

    # Child elements.
    result["oml:name"] = score.function

    if score.value is not None:
        result["oml:value"] = repr(float(score.value))
    if score.stdev is not None:
        result["oml:stdev"] = repr(float(score.stdev))
    if score.array:
        # Java's array_data is a single comma-separated string, not repeated
        # elements — see EvaluationScore.getArray_data().
        result["oml:array_data"] = ",".join(str(x) for x in score.array)
    if score.sample_size is not None:
        result["oml:sample_size"] = str(score.sample_size)

    return result


def run_evaluation_to_oml_dict(run_eval: RunEvaluation) -> dict:
    """``<oml:run_evaluation>`` envelope. Order of children mirrors
    RunEvaluation.java's field declaration order so the output is diff-stable
    against the Java serializer."""
    inner: dict = {
        "@xmlns:oml": "http://openml.org/openml",
        "oml:run_id": str(run_eval.run_id),
        "oml:evaluation_engine_id": str(run_eval.evaluation_engine_id),
    }
    if run_eval.error is not None:
        inner["oml:error"] = run_eval.error
    if run_eval.warning is not None:
        inner["oml:warning"] = run_eval.warning
    if run_eval.scores:
        inner["oml:evaluation"] = [
            evaluation_score_to_oml_dict(s) for s in run_eval.scores
        ]
    return {"oml:run_evaluation": inner}


def run_evaluation_to_xml(
    run_eval: RunEvaluation,
    *,
    pretty: bool = True,
) -> str:
    return xmltodict.unparse(run_evaluation_to_oml_dict(run_eval), pretty=pretty)
