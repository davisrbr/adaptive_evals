# Release Manifest (EMNLP 2025)

This document defines what is intentionally included in the public code release.

## Included

- Core code:
  - `tasks/`
  - `solvers/`
  - `scorers/`
  - `config/`
  - `prompting/`
  - `utils_consistency/`, `utils_elicitation/`, `utils_forecasting/`, `utils_plotting/`
  - `featurization/`
  - root-level experiment runners and helper scripts (`experiment_runner_*.py`, `featurize_*.py`, `visualize_featurization.py`, `legal_test_runner.py`)
- Lightweight input/config data required by code:
  - selected files under `data/`
  - `model_info/`
- External repos as submodules:
  - `dataset-featurization`
  - `legalbench`
- Release infrastructure:
  - `README.md`, `REPRODUCIBILITY.md`, `DATA.md`, `ARTIFACTS.md`, `LICENSE`, `CITATION.cff`
  - `pyproject.toml`, `requirements.txt`, `.github/workflows/ci.yml`
  - `scripts/release_check.py`, `scripts/run_paper_repro.sh`, `scripts/export_public_release.sh`

## Excluded

- Generated logs/caches/results (for example `logs/`, `results/`, `cache/`, `legal_*_results/`, `politeness_200*/`, `cyberbullying_100*/`)
- Notebook-first exploratory artifacts and plotting exports (`*.ipynb`, generated `*.pdf`, `*.png`, `*.svg`)
- Local/editor files (`.DS_Store`, `.vscode/`, `__pycache__/`)

## Rationale

This repository release is code-first and reproducibility-focused. Large generated artifacts are published separately and referenced in `ARTIFACTS.md`.
