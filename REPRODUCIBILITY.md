# Reproducibility Guide

## 1. Environment

```bash
git clone --recurse-submodules <REPO_URL>
cd adaptive_evals
bash env_create.sh
source .venv/bin/activate
```

## 2. Credentials

Set API keys needed for your selected model providers.

Example:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

## 3. Run the paper pipeline

Dry-run (recommended first):

```bash
bash scripts/run_paper_repro.sh
```

Execute:

```bash
bash scripts/run_paper_repro.sh --execute
```

Outputs are written under `artifacts/repro_<timestamp>/`.

## 4. Verify repository hygiene

```bash
python scripts/release_check.py
```

## Notes

- Full model evaluations can be expensive and provider-dependent.
- The release intentionally excludes raw experiment logs and large generated result bundles from git tracking.
