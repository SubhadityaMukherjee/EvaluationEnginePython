# OpenML EvaluationEngine (Python)

Python port of the OpenML [EvaluationEngine](https://github.com/openml/OpenML/tree/main/evaluationengine)
— the service that processes datasets (features & meta-features), generates
folds, merges multitask datasets, and evaluates submitted runs on the OpenML
server.

[![Tests](https://github.com/SubhadityaMukherjee/EvaluationEnginePython/actions/workflows/tests.yml/badge.svg)](https://github.com/SubhadityaMukherjee/EvaluationEnginePython/actions/workflows/tests.yml)
[![Docker](https://github.com/SubhadityaMukherjee/EvaluationEnginePython/actions/workflows/docker.yml/badge.svg)](https://github.com/SubhadityaMukherjee/EvaluationEnginePython/actions/workflows/docker.yml)

## What it does

| Function (`-f`)          | Java original                | What it does |
|--------------------------|------------------------------|--------------|
| `evaluate_run`           | `EvaluateRun`                | Download a run's predictions/splits/dataset, compute all evaluation scores, upload them (`--no-upload` prints the XML locally) |
| `process_dataset`        | `ProcessDataset`             | Extract + upload dataset features and qualities; `--id` for one dataset, omit to poll for unprocessed ones |
| `process_dataset_print`  | `processAndPrint`            | Extract features locally and print the XML (no upload) |
| `extract_features_simple`| `FantailConnector` (simple)  | Compute + upload base qualities (SimpleMetaFeatures + pymfe groups); polls when no `--id` |
| `extract_features_all`   | `FantailConnector` (all)     | Same, plus sklearn landmarker meta-features |
| `generate_folds`         | `GenerateFolds`              | Generate the splits table (cross-validation, holdout, learning-curve, ...) for a task and write it as ARFF |
| `merge_datasets`         | `MergeDataset`               | Merge all datasets of a MultiTask task into one ARFF |
| `all_wrong`, `different_predictions`, `challenge` | —   | Not ported (see `src/main.py` docstring) |

## Setup

Requires Python >= 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                # create .venv and install dependencies
uv run pytest          # run the test suite (offline; ~400 tests)
```

API authentication: set `OPENML_API_KEY` in the environment, or rely on the
test server's default demo key when using `--test` (below).

## CLI usage

Entry point: `python -m src.main` (`-f/--function` is required; run with
`--help` for all flags).

```bash
# Evaluate one run on the test server (test.openml.org, demo API key)
uv run python -m src.main -f evaluate_run -id 1 -test

# Same, but only compute and print the evaluation XML without uploading
uv run python -m src.main -f evaluate_run -id 1 -test -no-upload

# Process a single dataset / start the processing poll loop
uv run python -m src.main -f process_dataset -id 61 -test
uv run python -m src.main -f process_dataset -test

# Print a dataset's features as XML (no upload)
uv run python -m src.main -f process_dataset_print -id 61 -test

# Generate the folds ARFF for a task (writes to --output or stdout)
uv run python -m src.main -f generate_folds -id 1720 -test -o splits.arff

# Meta-features (qualities) for one dataset, or poll the server
uv run python -m src.main -f extract_features_simple -id 61 -test
uv run python -m src.main -f extract_features_all -test

# Merge the datasets of a MultiTask task
uv run python -m src.main -f merge_datasets -id 1901 -test -o merged.arff
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `-test`          | Target `test.openml.org` (default key `normaluser`) instead of production |
| `-id`            | Dataset / run / task id, depending on the function |
| `-df`            | Dataset file format: `arff` (default) or `parquet` |
| `-no-upload`     | Compute locally, skip uploading (evaluate_run prints the XML) |
| `-o/--output`    | Output file (generate_folds, merge_datasets) |
| `-tag`, `-mode`, `-u`, `-t`, `-x`, `-reverse` | Priority tag, mode, uploader/task filters, random/reverse ordering |

## Docker

The image runs the same CLI as its entrypoint:

```bash
# Build
docker build -t evaluation-engine .

# Prints usage
docker run --rm evaluation-engine

# Evaluate a run on the test server
docker run --rm evaluation-engine -f evaluate_run -id 1 -test

# Dry run (no upload), mounting an output directory for generated files
docker run --rm -v "$PWD/out:/out" evaluation-engine \
  -f generate_folds -id 1720 -test -o /out/splits.arff

# Production server: pass your API key through the environment
docker run --rm -e OPENML_API_KEY evaluation-engine -f process_dataset -id 61
```

The image installs only runtime dependencies (`uv sync --no-dev`) and does not
contain the test suite.

## Documentation

API reference and usage docs are built with MkDocs:

```bash
uv run mkdocs serve      # live preview at http://localhost:8000
uv run mkdocs build      # static site in site/
```

This README is the docs home page; the full API reference lives under
`docs/api/` (see `mkdocs.yml`).

## Project layout

```
src/
├── main.py               # CLI dispatcher (Main.java)
├── client.py             # OpenML REST client (OpenmlConnector subset)
├── evaluate_run.py       # run evaluation driver (EvaluateRun.java)
├── helpers.py            # HTTP/XML + ARFF + prediction math helpers
├── data_loader.py        # ARFF/Parquet -> normalized (attributes, rows)
├── models.py             # dataclasses shared across the engine
├── exceptions.py         # OpenmlError hierarchy
├── runs/                 # run evaluation: metrics, fold counting, XML
├── features/             # dataset feature extraction + XML
├── qualities/            # meta-features (pymfe), landmarkers, polling
└── process_dataset/      # fold generation, dataset processing, merging
tests/                    # pytest suite mirroring src/ layout
docs/                     # MkDocs sources (API reference)
reference_and_sanity_checks/   # manual parity notebooks vs the Java engine
```

## Parity notes

The Python engine mirrors the Java behaviour (metric definitions, seed usage,
error-code handling such as 431/441), with documented deviations where a Weka
component has no Python equivalent — e.g. landmarker classifiers are sklearn
approximations and the `CfsSubsetEval_*` meta-features are omitted. The
`reference_and_sanity_checks/` notebooks compare outputs against the live test
server.
