# Adaptive Evals

Code release for our EMNLP 2025 paper: **"Adaptively profiling models with task elicitation"** ([arXiv:2503.01986](https://arxiv.org/abs/2503.01986)).

## Included Scope

The released codebase includes:

- Adaptive evaluation runners for:
  - TruthfulQA (`runners/truthfulqa.py`)
  - LegalBench/MAUD tasks (`runners/legal.py`)
  - Politeness (`runners/politeness.py`)
  - PAIR jailbreak evaluation (`runners/pair.py`)
  - Cyberbullying adaptive evaluation (`runners/cyberbullying.py`)
  - Forecasting evaluation runner (`runners/forecasting.py`) and consistency/adaptive generation utilities (`utils_consistency/`, `tasks/task_adaptive_consistency.py`)
- Core task/solver/scorer implementations (`tasks/`, `solvers/`, `scorers/`)
- [PRESS](https://arxiv.org/abs/2409.00844v1) report-cards, implemented in inspect-ai (`tasks/task_press.py`, `solvers/solver_press.py`)
- TruthfulQA featurization utilities (`experiments/featurization/`)
- External dependencies via submodules:
  - `legalbench`
  - `dataset-featurization`

Generated logs, caches, plots, notebooks, and large result dumps are intentionally excluded from git tracking.

## Repository Layout

Top-level root is intentionally minimal (metadata + folders):

- Project metadata: `README.md`, `LICENSE`, `CITATION.cff`, `pyproject.toml`, `requirements.txt`
- Environment bootstrap: `env_create.sh`
- Repro and release scripts: `scripts/`
- Core source folders: `runners/`, `tasks/`, `solvers/`, `scorers/`, `config/`, `data/`, `prompting/`, `utils_*`
- Experiment bundles and analysis assets: `experiments/`

## Quick Start

```bash
git clone --recurse-submodules [https://github.com/davisrbr/adaptive_evals.git](https://github.com/davisrbr/adaptive_evals.git)
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
bash scripts/run_paper_repro.sh --execute
```
Outputs are written to timestamped directories under `artifacts/`.

## Running Pipelines Directly

Examples:

```bash
python runners/truthfulqa.py --models-for-transfer openai/gpt-4o-mini --models-for-generation openai/gpt-4o-mini
python runners/legal.py --models-for-transfer openai/gpt-4o-mini --models-for-generation openai/gpt-4o-mini
python runners/politeness.py --models-for-transfer openai/gpt-4o-mini --models-for-generation openai/gpt-4o-mini
python runners/pair.py --target-models together/meta-llama/Llama-2-7b-chat-hf --attack-models openai/gpt-4o
python runners/cyberbullying.py --target-models openai/gpt-4o-mini --attack-models openai/gpt-4o
python runners/forecasting.py --models openai/gpt-4o-mini
```

Use `--help` on each runner for all options.

## Citation

See `CITATION.cff`.
