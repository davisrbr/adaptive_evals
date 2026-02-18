# Adaptive Evals

Code release for the EMNLP 2025 paper:
**"Adaptively profiling models with task elicitation"** ([arXiv:2503.01986](https://arxiv.org/abs/2503.01986)).

## Release Status

This repository is in release state for the paper artifact (`v1.0.0-emnlp2025`).
The cleanup phase is complete; this branch is not a work-in-progress snapshot.

## Included Scope

The released codebase includes:

- Adaptive evaluation runners for:
  - TruthfulQA (`runners/truthfulqa.py`)
  - LegalBench/MAUD tasks (`runners/legal.py`)
  - Politeness (`runners/politeness.py`)
- Core task/solver/scorer implementations (`tasks/`, `solvers/`, `scorers/`)
- Consistency/adaptive generation utilities (`utils_consistency/`, `tasks/task_adaptive_consistency.py`)
- PRESS report-card task (`tasks/task_press.py`, `solvers/solver_press.py`)
- TruthfulQA featurization utilities (`featurization/`)
- Release hygiene checks (`scripts/release_check.py`) and smoke tests (`tests/test_release_smoke.py`)
- External dependencies via submodules:
  - `legalbench`
  - `dataset-featurization`

Generated logs, caches, plots, notebooks, and large result dumps are intentionally excluded from git tracking.

## Repository Layout

Top-level root is intentionally minimal (metadata + folders):

- Project metadata: `README.md`, `LICENSE`, `CITATION.cff`, `pyproject.toml`, `requirements.txt`
- Environment bootstrap: `env_create.sh`
- Repro and release scripts: `scripts/`
- Core source folders: `runners/`, `tasks/`, `solvers/`, `scorers/`, `config/`, `data/`, `prompting/`, `utils_*`, `featurization/`

## Quick Start

```bash
git clone --recurse-submodules <REPO_URL>
cd adaptive_evals
bash env_create.sh
source .venv/bin/activate
```

Set provider credentials as needed, for example:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## Reproducing Paper Runs

Use the wrapper script:

```bash
bash scripts/run_paper_repro.sh          # dry-run: writes command script only
bash scripts/run_paper_repro.sh --execute
```

This orchestrates:

- `python runners/truthfulqa.py`
- `python runners/legal.py`
- `python runners/politeness.py`

Outputs are written to timestamped directories under `artifacts/`.

## Running Pipelines Directly

Examples:

```bash
python runners/truthfulqa.py --models-for-transfer openai/gpt-4o-mini --models-for-generation openai/gpt-4o-mini
python runners/legal.py --models-for-transfer openai/gpt-4o-mini --models-for-generation openai/gpt-4o-mini
python runners/politeness.py --models-for-transfer openai/gpt-4o-mini --models-for-generation openai/gpt-4o-mini
```

Use `--help` on each runner for all options.

## Release Hygiene

Before publishing a release or tag:

```bash
python scripts/release_check.py --mode ci
pytest -q tests/test_release_smoke.py
```

CI (`.github/workflows/ci.yml`) enforces these checks on `main`, `release/**`, and `public/**`.

## Citation

See `CITATION.cff`.
