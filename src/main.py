"""Python port of ``org.openml.webapplication.Main``.

CLI dispatcher mirroring Main.java. Run as ``python -m src.main --help``.

Supported functions (``-f`` / ``--function``):
  * ``evaluate_run``         — port of EvaluateRun; needs ``--id``.
  * ``process_dataset``      — port of ProcessDataset; ``--id`` to process one
                               dataset, omit to poll (currently raises
                               NotImplementedError).
  * ``process_dataset_print``— local feature extraction, prints XML.
  * ``generate_folds``       — wraps src.process_dataset.generate_folds.

Unsupported functions (matching Main.java's fallthrough branch):
  * ``extract_features_all``, ``extract_features_simple`` — FantailConnector
    port not in scope; the Python qualities module uses pymfe, not Weka's
    fantail characterizers, so the mapping is not 1:1.
  * ``merge_datasets``       — MergeDataset.java not ported.
  * ``all_wrong``, ``different_predictions`` — InstanceBased.java not ported.
  * ``challenge``            — ChallengeSets.java not ported.

Each prints a clear NotImplementedError when invoked. The Java options are
mapped 1:1 (``-id`` → ``--id``, ``-u`` → ``--user``, etc.) with both short and
long forms accepted.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

# Maps Java's Settings.SUPPORTED_TASK_TYPES_EVALUATION.
from src.runs import SUPPORTED_TASK_TYPES_EVALUATION

# Java's Main.FOLD_GENERATION_SEED.
FOLD_GENERATION_SEED = 0


# ============================================================================
# Argument parser
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="src.main",
        description="OpenML evaluation engine (Python port of Main.java).",
    )

    # Identity / scope options.
    p.add_argument("-id", "--id", type=int, default=None,
                   help="The id of the dataset/run used.")
    p.add_argument("-u", "--user", type=int, default=None,
                   help="The user id (uploader filter for evaluate_run).")
    p.add_argument("-t", "--task", type=str, default=None,
                   help="The task id (or comma-separated list).")
    p.add_argument("-r", "--run", type=str, default=None,
                   help="The run id (comma-separated for some functions).")
    p.add_argument("-f", "--function", type=str, required=True,
                   help="The function to invoke (see module docstring).")

    # Behavior flags.
    p.add_argument("-x", "--random", action="store_true",
                   help="Pick a random id rather than the next one in order.")
    p.add_argument("-reverse", "--reverse", action="store_true",
                   help="Start evaluating from the last runs.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose output.")
    p.add_argument("-m", "--md5", action="store_true",
                   help="Present the splits file output as an md5 hash.")

    # String / numeric modifiers.
    p.add_argument("-config", "--config", type=str, default=None,
                   help="Config string describing the settings for API interaction.")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="The output file path (or offset, for challenge).")
    p.add_argument("-test", "--test", type=str, default=None,
                   help="A list of rowids for a holdout set (fold generation).")
    p.add_argument("-tag", "--tag", type=str, default=None,
                   help="A tag that will get priority in processing fantail features.")
    p.add_argument("-mode", "--mode", type=str, default=None,
                   help="{train,test} for challenge; ttid override for evaluate_run.")
    p.add_argument("-size", "--size", type=int, default=None,
                   help="Desired size of train/test set.")

    return p


# ============================================================================
# Function dispatchers (one per -f value)
# ============================================================================


def _cmd_evaluate_run(args: argparse.Namespace) -> None:
    """Port of Main.java:98-112. ``--mode`` overrides the supported task-type
    set with a single ttid; otherwise all of SUPPORTED_TASK_TYPES_EVALUATION."""
    from src.evaluate_run import EvaluateRun

    if args.id is None:
        raise SystemExit("evaluate_run requires --id <run_id>.")

    if args.mode is not None:
        ttids = {int(args.mode)}
    else:
        ttids = set(SUPPORTED_TASK_TYPES_EVALUATION)

    evaluation_mode = "reverse" if args.reverse else ("random" if args.random else "normal")

    # Note: constructor evaluates the run and stores the result on
    # ``.last_result``. Java uploads the result server-side; that path is TODO.
    er = EvaluateRun(
        run_id=args.id,
        evaluation_mode=evaluation_mode,
        task_type_ids=ttids,
        task_ids=args.task,
        tag=args.tag,
        uploader_id=args.user,
    )
    r = er.last_result
    if r is None:
        return  # Polling path (no run_id) — TODO.
    if r.error:
        print(f"run {r.run_id}: error - {r.error}", file=sys.stderr)
    else:
        per_cell = sum(1 for s in r.scores if s.fold is not None)
        glob = sum(1 for s in r.scores if s.fold is None)
        print(f"run {r.run_id}: {len(r.scores)} scores "
              f"({per_cell} per-cell, {glob} global).")
        for s in (s for s in r.scores if s.fold is None):
            v = f"{s.value:.6f}" if s.value is not None else "None"
            print(f"  {s.function:35s} = {v}")


def _cmd_process_dataset(args: argparse.Namespace) -> None:
    """Port of Main.java:114-117."""
    from src.process_dataset import ProcessDataset

    mode = "random" if args.random else "normal"
    if args.id is None:
        # Java constructor would poll. We expose it as an explicit .poll().
        ProcessDataset(mode=mode).poll()
        return

    pd = ProcessDataset(dataset_id=args.id, mode=mode)
    f, q = pd.last_features, pd.last_qualities
    if f and f.error:
        print(f"dataset {args.id}: features error - {f.error}", file=sys.stderr)
    elif f:
        print(f"dataset {args.id}: {len(f.features)} features extracted.")
    if q and q.error:
        print(f"dataset {args.id}: qualities error - {q.error}", file=sys.stderr)
    elif q:
        print(f"dataset {args.id}: {len(q.qualities)} qualities extracted.")


def _cmd_process_dataset_print(args: argparse.Namespace) -> None:
    """Port of Main.java:118-120. Local-only feature extraction → stdout."""
    from src.process_dataset import ProcessDataset

    if args.id is None:
        raise SystemExit("process_dataset_print requires --id <dataset_id>.")
    ProcessDataset().process_and_print(args.id)


def _cmd_generate_folds(args: argparse.Namespace) -> None:
    """Port of Main.java:141-149. Writes splits ARFF to ``--output`` or stdout.

    The Java GenerateFolds uses the dataset's task-default procedure; we don't
    have a Python equivalent of that lookup, so we default to 10-fold CV with
    1 repeat and ``FOLD_GENERATION_SEED``. Override via the env-var-style
    procedure name parsed from ``--mode`` if given (one of: crossvalidation,
    holdout, leaveoneout, testontrainingdata, learningcurve).
    """
    from src.models import EstimationProcedure, EstimationProcedureType
    from src.process_dataset import generate_folds
    from src.process_dataset.arff import splits_to_arff

    if args.id is None:
        raise SystemExit("generate_folds requires --id <dataset_id>.")

    procedure_name = (args.mode or "crossvalidation").lower()
    try:
        ep_type = EstimationProcedureType[procedure_name.upper()]
    except KeyError:
        raise SystemExit(
            f"Unknown procedure {procedure_name!r}. Try one of: "
            "crossvalidation, holdout, holdout_ordered, leaveoneout, "
            "testontrainingdata, learningcurve_cv."
        )

    procedure_kwargs = {}
    if ep_type in (EstimationProcedureType.CROSSVALIDATION,
                   EstimationProcedureType.LEARNINGCURVE_CV):
        procedure_kwargs = {"folds": 10, "repeats": 1}
    elif ep_type is EstimationProcedureType.HOLDOUT:
        procedure_kwargs = {"percentage": 33, "repeats": 1}

    procedure = EstimationProcedure(type=ep_type, **procedure_kwargs)
    splits, _, _ = generate_folds(did=args.id, procedure=procedure, seed=FOLD_GENERATION_SEED)
    text = splits_to_arff(splits)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {len(splits)} split rows to {args.output}")
    else:
        print(text)


def _not_implemented(function: str) -> None:
    raise NotImplementedError(
        f"Function {function!r} is not ported yet. See src/main.py docstring."
    )


# ============================================================================
# Entry point
# ============================================================================

_DISPATCH: dict[str, callable] = {
    "evaluate_run": _cmd_evaluate_run,
    "process_dataset": _cmd_process_dataset,
    "process_dataset_print": _cmd_process_dataset_print,
    "generate_folds": _cmd_generate_folds,
    # Functions below are recognised (matching Main.java's option parsing) but
    # unimplemented. Each call site logs the missing functionality.
    "extract_features_all": lambda a: _not_implemented("extract_features_all"),
    "extract_features_simple": lambda a: _not_implemented("extract_features_simple"),
    "merge_datasets": lambda a: _not_implemented("merge_datasets"),
    "all_wrong": lambda a: _not_implemented("all_wrong"),
    "different_predictions": lambda a: _not_implemented("different_predictions"),
    "challenge": lambda a: _not_implemented("challenge"),
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    handler = _DISPATCH.get(args.function)
    if handler is None:
        # Mirrors Main.java's "call to unknown function" branch.
        print(f"Error: call to unknown function: {args.function}", file=sys.stderr)
        return 1

    try:
        handler(args)
        return 0
    except NotImplementedError as e:
        print(f"Not implemented: {e}", file=sys.stderr)
        return 0  # Java exits 0 on LegacyWarning; we mirror for NIY.
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 — top-level catch, matches Java.
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
