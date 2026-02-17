# Adaptive Evals

Code release for the EMNLP 2025 paper:  
**"Adaptively profiling models with task elicitation"** ([arXiv:2503.01986](https://arxiv.org/abs/2503.01986)).

## What This Repo Contains

This release keeps the core implementation for adaptive evaluation workflows:

- task definitions (`tasks/`)
- solver/scorer implementations (`solvers/`, `scorers/`)
- experiment runners (`experiment_runner_*`)
- featurization and analysis helpers (`featurization/`, `utils_*`, `data/`)
- external dependencies via submodules (`legalbench`, `dataset-featurization`)

Generated logs, caches, large result dumps, and plotting artifacts are intentionally excluded from git tracking.

## Quick Start

```bash
git clone --recurse-submodules <REPO_URL>
cd adaptive_evals
bash env_create.sh
source .venv/bin/activate
```

Set provider credentials as needed (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) before running experiments.

## Reproducing Paper Runs

Use the paper runner wrapper:

```bash
bash scripts/run_paper_repro.sh          # dry-run (prints commands)
bash scripts/run_paper_repro.sh --execute
```

## Release Policy

- CI enforces release hygiene via `scripts/release_check.py`.

## Citation

See `CITATION.cff`.
