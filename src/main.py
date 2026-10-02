"""
CLI dispatcher mirroring Main.java. Run as ``python -m src.main --help``.

Supported functions (``-f`` / ``--function``):
  * ``evaluate_run``         — port of EvaluateRun; needs ``--id``.
  * ``process_dataset``      — port of ProcessDataset; ``--id`` to process one
                               dataset, omit to poll.
  * ``process_dataset_print``— local feature extraction, prints XML.
  * ``extract_features_simple``— port of FantailConnector (simple set); ``--id``
                               for one dataset, omit to poll.
  * ``extract_features_all`` — port of FantailConnector (all set, including
                               the sklearn landmarker port); ``--id`` for one
                               dataset, omit to poll.
  * ``merge_datasets``       — port of MergeDataset; needs ``--id`` (MultiTask
                               task id); writes merged ARFF to ``--output`` or
                               stdout.
  * ``generate_folds``       — wraps src.process_dataset.generate_folds.

Unsupported functions (matching Main.java's fallthrough branch):
  * ``all_wrong``, ``different_predictions`` — InstanceBased.java not ported.
  * ``challenge``            — ChallengeSets.java not ported.

Each prints a clear NotImplementedError when invoked. The Java options are
mapped 1:1 (``-id`` → ``--id``, ``-u`` → ``--user``, etc.) with both short and
long forms accepted. Note: Java's ``-test`` (holdout rowids for fold
generation) is NOT ported — ``-test`` / ``--test`` here targets
``test.openml.org`` (see ``OpenmlClient``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from src.runs import SUPPORTED_TASK_TYPES_EVALUATION

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
    p.add_argument(
        "-id", "--id", type=int, default=None, help="The id of the dataset/run used."
    )
    p.add_argument(
        "-u",
        "--user",
        type=int,
        default=None,
        help="The user id (uploader filter for evaluate_run).",
    )
    p.add_argument(
        "-t",
        "--task",
        type=str,
        default=None,
        help="The task id (or comma-separated list).",
    )
    p.add_argument(
        "-r",
        "--run",
        type=str,
        default=None,
        help="The run id (comma-separated for some functions).",
    )
    p.add_argument(
        "-f",
        "--function",
        type=str,
        required=True,
        help="The function to invoke (see module docstring).",
    )

    # Behavior flags.
    p.add_argument(
        "-x",
        "--random",
        action="store_true",
        help="Pick a random id rather than the next one in order.",
    )
    p.add_argument(
        "-reverse",
        "--reverse",
        action="store_true",
        help="Start evaluating from the last runs.",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose output.")
    p.add_argument(
        "-m",
        "--md5",
        action="store_true",
        help="Present the splits file output as an md5 hash.",
    )

    # String / numeric modifiers.
    p.add_argument(
        "-config",
        "--config",
        type=str,
        default=None,
        help="Config string describing the settings for API interaction.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="The output file path (or offset, for challenge).",
    )
    p.add_argument(
        "-tag",
        "--tag",
        type=str,
        default=None,
        help="A tag that will get priority in processing features.",
    )
    p.add_argument(
        "-mode",
        "--mode",
        type=str,
        default=None,
        help="{train,test} for challenge; ttid override for evaluate_run.",
    )
    p.add_argument(
        "-size",
        "--size",
        type=int,
        default=None,
        help="Desired size of train/test set.",
    )
    p.add_argument(
        "-test",
        "--test",
        action="store_true",
        help="Target test.openml.org instead of production (default api key "
        "'normaluser'; used by process_dataset/evaluate_run/extract_features/"
        "merge_datasets).",
    )

    p.add_argument(
        "-df",
        "--dataset-format",
        type=str,
        default="arff",
        choices=["arff", "parquet"],
        help="Dataset file format to download and parse (default: arff). "
        "Parquet is being phased in as ARFF support is retired.",
    )

    p.add_argument(
        "-no-upload",
        "--no-upload",
        action="store_true",
        dest="no_upload",
        help="Compute results locally but skip the upload step. Currently "
        "honored by evaluate_run, which prints the evaluation XML to stdout "
        "instead of POSTing it.",
    )

    return p


# ============================================================================
# Function dispatchers (one per -f value)
# ============================================================================


def _cmd_evaluate_run(args: argparse.Namespace) -> None:
    """``--mode`` overrides the supported task-type
    set with a single ttid; otherwise all of SUPPORTED_TASK_TYPES_EVALUATION."""
    from src.client import OpenmlClient
    from src.evaluate_run import EvaluateRun

    if args.id is None:
        raise SystemExit("evaluate_run requires --id <run_id>.")

    if args.mode is not None:
        try:
            ttids = {int(args.mode)}
        except ValueError:
            raise SystemExit(
                "evaluate_run --mode must be an integer task-type id "
                f"(got {args.mode!r})."
            ) from None
    else:
        ttids = set(SUPPORTED_TASK_TYPES_EVALUATION)

    evaluation_mode = (
        "reverse" if args.reverse else ("random" if args.random else "normal")
    )

    # Constructor evaluates the run and stores the result on ``.last_result``;
    # the upload to /run/evaluate happens as a side effect of evaluate().
    upload = not args.no_upload

    er = EvaluateRun(
        run_id=args.id,
        evaluation_mode=evaluation_mode,
        task_type_ids=ttids,
        task_ids=args.task,
        tag=args.tag,
        uploader_id=args.user,
        client=OpenmlClient(test=args.test),
        dataset_format=args.dataset_format,
        upload=upload,
    )
    r = er.last_result
    if r is None:
        return  # Polling path (no run_id) — TODO.
    if not upload:
        # Dry-run: emit the evaluation XML the engine *would* have uploaded so
        # callers (e.g. the comparison notebook) can read the computed scores
        # without anything being written to the server.
        from src.runs.serialization import run_evaluation_to_xml

        sys.stdout.write(run_evaluation_to_xml(r, pretty=False))
        sys.stdout.write("\n")
        return
    if r.error:
        print(f"run {r.run_id}: error - {r.error}", file=sys.stderr)
    else:
        per_cell = sum(1 for s in r.scores if s.fold is not None)
        glob = sum(1 for s in r.scores if s.fold is None)
        print(
            f"run {r.run_id}: {len(r.scores)} scores "
            f"({per_cell} per-cell, {glob} global)."
        )
        for s in (s for s in r.scores if s.fold is None):
            v = f"{s.value:.6f}" if s.value is not None else "None"
            print(f"  {s.function:35s} = {v}")


def _cmd_process_dataset(args: argparse.Namespace) -> None:
    from src.client import OpenmlClient
    from src.process_dataset import ProcessDataset

    mode = "random" if args.random else "normal"
    client = OpenmlClient(test=args.test)
    if args.id is None:
        # Java constructor would poll. We expose it as an explicit .poll().
        ProcessDataset(
            mode=mode, client=client, dataset_format=args.dataset_format
        ).poll()
        return

    pd = ProcessDataset(
        dataset_id=args.id,
        mode=mode,
        client=client,
        dataset_format=args.dataset_format,
    )
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
    from src.client import OpenmlClient
    from src.process_dataset import ProcessDataset

    if args.id is None:
        raise SystemExit("process_dataset_print requires --id <dataset_id>.")
    ProcessDataset(
        client=OpenmlClient(test=args.test), dataset_format=args.dataset_format
    ).process_and_print(args.id)


def _cmd_generate_folds(args: argparse.Namespace) -> None:
    """Writes splits ARFF to ``--output`` or stdout.

    Mirrors Java's ``GenerateFolds``: ``--id`` is a TASK id. The source
    dataset, estimation-procedure type, and folds/repeats/percentage are all
    read from that task (Java's Main.java:138-146 / GenerateFolds.java); the
    seed is ``FOLD_GENERATION_SEED`` (Java's Main.java:46).
    """
    from src.client import OpenmlClient
    from src.process_dataset.arff import splits_to_arff
    from src.process_dataset.module import generate_folds_for_task

    if args.id is None:
        raise SystemExit("generate_folds requires --id <task_id>.")

    splits, _, _ = generate_folds_for_task(
        task_id=args.id,
        base_url=OpenmlClient(test=args.test).base_url,
        seed=FOLD_GENERATION_SEED,
        data_format=args.dataset_format,
    )
    text = splits_to_arff(splits)

    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            raise SystemExit(f"Could not write to {args.output}: {e}") from None
        print(f"wrote {len(splits)} split rows to {args.output}")
    else:
        print(text)


def _cmd_extract_features(args: argparse.Namespace, characterizer_set: str) -> None:
    """Shared handler for ``extract_features_simple`` / ``extract_features_all``.
    ``--id`` processes one dataset; omitting it polls the qualities-unprocessed
    endpoint. ``--tag`` becomes the priority tag (Java parity)."""
    from src.client import OpenmlClient
    from src.qualities.extract import ExtractFeatures

    mode = "random" if args.random else "normal"
    client = OpenmlClient(test=args.test)
    ef = ExtractFeatures(
        client=client,
        mode=mode,
        characterizer_set=characterizer_set,
        priority_tag=args.tag,
        dataset_format=args.dataset_format,
    )
    if args.id is None:
        ef.poll()
        return

    dq = ef.process(args.id)
    if dq.error:
        print(f"dataset {args.id}: qualities error - {dq.error}", file=sys.stderr)
    else:
        print(f"dataset {args.id}: {len(dq.qualities)} qualities extracted.")


def _cmd_merge_datasets(args: argparse.Namespace) -> None:
    """``--id`` is a MultiTask task id. Writes the merged ARFF to ``--output``
    or stdout (Java: ``Output.instances2file``)."""
    from src.client import OpenmlClient
    from src.process_dataset.merge import MergeDataset

    if args.id is None:
        raise SystemExit("merge_datasets requires --id <task_id>.")

    client = OpenmlClient(test=args.test)
    md = MergeDataset(task_id=args.id, client=client)
    text = md.merge()
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
        except OSError as e:
            raise SystemExit(f"Could not write to {args.output}: {e}") from None
        print(f"wrote merged ARFF to {args.output}", file=sys.stderr)
    else:
        print(text)


def _not_implemented(function: str) -> None:
    raise NotImplementedError(
        f"Function {function!r} is not ported yet. See src/main.py docstring."
    )


# ============================================================================
# Entry point
# ============================================================================

_DISPATCH: dict[str, Callable[[argparse.Namespace], None]] = {
    "evaluate_run": _cmd_evaluate_run,
    "process_dataset": _cmd_process_dataset,
    "process_dataset_print": _cmd_process_dataset_print,
    "generate_folds": _cmd_generate_folds,
    "extract_features_all": lambda a: _cmd_extract_features(a, "all"),
    "extract_features_simple": lambda a: _cmd_extract_features(a, "simple"),
    "merge_datasets": _cmd_merge_datasets,
    "all_wrong": lambda a: _not_implemented("all_wrong"),
    "different_predictions": lambda a: _not_implemented("different_predictions"),
    "challenge": lambda a: _not_implemented("challenge"),
}


def main(argv: Sequence[str] | None = None) -> int:
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
