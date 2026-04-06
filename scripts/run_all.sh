#!/usr/bin/env bash
# Run all three eval rounds sequentially.
# Usage: ./scripts/run_all.sh [MODEL]
# Example: ./scripts/run_all.sh openai/gpt-4o

set -euo pipefail

MODEL="${1:-openai/gpt-4o}"
LOG_DIR="./logs"

echo "=== Round 1: Baseline (4 samples) ==="
inspect eval tasks/baseline/task.py --model "$MODEL" --log-dir "$LOG_DIR"

echo ""
echo "=== Round 2: Hypothesis Test (4 samples) ==="
inspect eval tasks/hypothesis_test/task.py --model "$MODEL" --log-dir "$LOG_DIR"

echo ""
echo "=== Round 3: Regex Depth (3 samples) ==="
inspect eval tasks/regex_depth/task.py --model "$MODEL" --log-dir "$LOG_DIR"

echo ""
echo "All rounds complete. Logs in $LOG_DIR/"
