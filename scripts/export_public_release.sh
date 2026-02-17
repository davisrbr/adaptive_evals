#!/usr/bin/env bash
set -euo pipefail

BRANCH="${1:-public/emnlp2025}"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

git archive --format=tar HEAD | tar -xf - -C "$TMPDIR"

CURRENT_BRANCH="$(git branch --show-current)"
git switch --orphan "$BRANCH"
git rm -rf . >/dev/null 2>&1 || true
cp -a "$TMPDIR"/. .
git add .
git commit -m "Public EMNLP 2025 release snapshot"

echo "Created orphan branch: $BRANCH"
echo "Previous branch was: $CURRENT_BRANCH"
