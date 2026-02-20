#!/usr/bin/env bash
set -euo pipefail

EXECUTE=0
if [[ "${1:-}" == "--execute" ]]; then
  EXECUTE=1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="artifacts/repro_${STAMP}"
mkdir -p "$OUTDIR"

cat > "$OUTDIR/commands.sh" <<CMDS
#!/usr/bin/env bash
set -euo pipefail

python runners/truthfulqa.py \
  --models-for-transfer openai/gpt-4o-mini \
  --models-for-generation openai/gpt-4o-mini \
  --results-path "$OUTDIR/truthfulqa_results.csv" \
  --cache-path "$OUTDIR/truthfulqa_cache.csv"

python runners/legal.py \
  --models-for-transfer openai/gpt-4o-mini \
  --models-for-generation openai/gpt-4o-mini \
  --results-path "$OUTDIR/legal_results.csv" \
  --cache-path "$OUTDIR/legal_cache.csv"

python runners/politeness.py \
  --models-for-transfer openai/gpt-4o-mini \
  --models-for-generation openai/gpt-4o-mini \
  --results-path "$OUTDIR/politeness_results.csv" \
  --cache-path "$OUTDIR/politeness_cache.csv"
CMDS
chmod +x "$OUTDIR/commands.sh"

python scripts/release_check.py

if [[ "$EXECUTE" -eq 0 ]]; then
  echo "Dry-run mode. Generated command script at: $OUTDIR/commands.sh"
  echo "Review it, then run: bash scripts/run_paper_repro.sh --execute"
  exit 0
fi

if [[ -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "Set model-provider API keys before --execute (e.g., OPENAI_API_KEY or ANTHROPIC_API_KEY)." >&2
  exit 1
fi

bash "$OUTDIR/commands.sh"
echo "Reproduction run complete. Outputs saved in: $OUTDIR"
