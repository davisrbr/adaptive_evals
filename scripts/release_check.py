#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

BANNED_PREFIXES = (
    "logs/",
    "logs_politeness/",
    "results/",
    "results_truthfulqa/",
    "results_truthfulqa_nocot/",
    "cache/",
    "plots/",
    "legal_150_results/",
    "legal_150_results_nocot/",
    "legal_150_cache/",
    "legal_38_results/",
    "legal_38_cache/",
    "legal_2_results/",
    "legal_2_cache/",
    "politeness_200",
    "cyberbullying_100",
    "cyberbullying_test/",
    "test_cyberbullying_attack/",
    "tasks/cond_100/",
    "tasks/logs/",
    "tasks/adaptive_consistency_checkpoints/",
)

BANNED_SUFFIXES = {
    ".eval",
    ".log",
    ".key",
    ".pdf",
    ".png",
    ".svg",
    ".ipynb",
}

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".csv",
    ".sh",
    ".ini",
    ".cfg",
    ".cff",
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"),
]

MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024


def tracked_files() -> list[Path]:
    out = subprocess.check_output(["git", "ls-files", "-z"], text=False)
    paths = [p.decode("utf-8") for p in out.split(b"\0") if p]
    return [Path(p) for p in paths]


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    return path.suffix == ""


def scan_text_for_secrets(path: Path) -> list[str]:
    try:
        data = path.read_bytes()
    except OSError:
        return []
    if b"\x00" in data[:8192]:
        return []
    try:
        text = data.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        return []

    matches: list[str] = []
    for pat in SECRET_PATTERNS:
        if pat.search(text):
            matches.append(pat.pattern)
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Repository release hygiene checks")
    parser.add_argument("--mode", choices=["local", "ci"], default="local")
    args = parser.parse_args()

    root = Path.cwd()
    violations: list[str] = []

    for rel in tracked_files():
        rel_posix = rel.as_posix()

        if any(rel_posix.startswith(prefix) for prefix in BANNED_PREFIXES):
            violations.append(f"Banned tracked path prefix: {rel_posix}")

        if rel.suffix.lower() in BANNED_SUFFIXES:
            violations.append(f"Banned tracked extension ({rel.suffix}): {rel_posix}")

        full_path = root / rel
        if full_path.is_file():
            size = full_path.stat().st_size
            if size > MAX_FILE_SIZE_BYTES:
                violations.append(
                    f"Tracked file exceeds {MAX_FILE_SIZE_BYTES} bytes: {rel_posix} ({size} bytes)"
                )

            if is_text_file(rel):
                for pat in scan_text_for_secrets(full_path):
                    violations.append(f"Possible secret pattern '{pat}' found in {rel_posix}")

    if violations:
        print("Release check FAILED:")
        for issue in violations:
            print(f"- {issue}")
        return 1

    print(f"Release check passed ({args.mode} mode).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
