"""Domain exceptions for the OpenML EvaluationEngine.

Design policy
-------------
This module is intentionally *small*. Generic argument errors — bad types,
unsupported enum strings, out-of-range values — keep using the Python
builtins (``ValueError``, ``TypeError``, ``KeyError``) so callers do not have
to learn engine-specific names for ordinary mistakes. Engine-specific
exceptions are reserved for failure modes that are:

* worth catching by name (e.g. the polling loops that stop on
  "no unprocessed datasets"),
* distinct from a plain bad-argument ``ValueError`` (e.g. local
  misconfiguration), or
* produced by an external system (the OpenML server).

Third-party exceptions are deliberately **not** wrapped: ``requests`` raises
``requests.HTTPError`` on transport failures, ``xmltodict`` / dict access raise
``KeyError`` on unexpected XML, and sklearn raises its own errors during
metric computation. Those propagate unchanged — re-wrapping them at every
call site would hide useful tracebacks without adding information.

All engine exceptions derive from :class:`OpenmlError`, so a caller can catch
every engine-specific failure with a single ``except OpenmlError`` while still
letting standard library / third-party exceptions bubble up.

Hierarchy
---------

::

    OpenmlError
    +-- OpenmlApiError            # server returned an oml:api_error envelope
    +-- OpenmlConfigError         # local misconfiguration (missing API key, ...)
    +-- PredictionValidationError # predictions file is invalid for the task
        (also subclasses ValueError, so ``except ValueError`` still catches it)
"""

from __future__ import annotations


class OpenmlError(Exception):
    """Base class for every engine-specific exception.

    Subclass it (or one of its children) when a failure mode needs a stable,
    catchable identity. Do not raise ``OpenmlError`` directly — pick the
    specific subclass that describes the problem.

    Catching ``OpenmlError`` covers all three engine failure categories below
    without touching the unrelated standard-library / third-party exceptions.
    """


class OpenmlApiError(OpenmlError):
    """Raised when the OpenML server returns an ``oml:api_error`` envelope.

    Mirrors ``org.openml.apiconnector.io.ApiException``: ``code`` is the
    OpenML numeric error code (e.g. 431 = dataset already processed,
    441 = features already uploaded, 1013 = no unevaluated runs remaining) so
    callers can branch on it. ``message`` carries the server's human-readable
    text, including any ``additional_information`` the server appended.

    Example
    -------
    >>> try:
    ...     client.evaluation_request(1, "normal", 1000)
    ... except OpenmlApiError as err:
    ...     if err.code == 1013:
    ...         ...  # nothing left to evaluate — stop polling
    """

    def __init__(self, code: int, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class OpenmlConfigError(OpenmlError):
    """Raised when the engine is misconfigured locally (not a server error).

    The usual cause is an unset ``OPENML_API_KEY`` with no key passed to the
    client. It is always fixable by the caller — no input data needs to
    change — which is why it is separated from the data/prediction errors and
    from server-side :class:`OpenmlApiError`.
    """


class PredictionValidationError(OpenmlError, ValueError):
    """Raised when a predictions file is invalid for its task/dataset.

    Covers every way a submitted set of predictions can be wrong: a required
    column (``row_id``, ``prediction``, ``confidence.<class>``) is missing, a
    ``row_id`` is out of range, the per-fold prediction counts do not match
    the task's splits, predictions are out of order, or a confidence value is
    missing.

    This consolidates what was previously an inconsistent mix of
    ``RuntimeError`` and ``ValueError`` into one catchable type. It
    deliberately also subclasses :class:`ValueError` so that handlers written
    against the standard library still catch it, while engine code can branch
    on the engine-specific type::

        try:
            evaluate_batch(...)
        except PredictionValidationError as err:
            log.warning("skipping run %s: %s", run_id, err)
    """
