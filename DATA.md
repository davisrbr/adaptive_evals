# Data Policy

## In-Repo Data

The repository includes small configuration/input files required to run code paths (for example selected CSV/JSON files in `data/` and `model_info/`).

## Out-of-Repo Artifacts

The following are not tracked in git for release quality and repository size control:

- raw Inspect `.eval` logs
- generated result CSV dumps from large sweeps
- cache files
- notebook-generated plots and PDFs

These are documented and distributed separately (see `ARTIFACTS.md`).
