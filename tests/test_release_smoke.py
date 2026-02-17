from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_release_files_exist() -> None:
    required = [
        "README.md",
        "RELEASE_MANIFEST.md",
        "REPRODUCIBILITY.md",
        "DATA.md",
        "ARTIFACTS.md",
        "CITATION.cff",
        "pyproject.toml",
        "scripts/release_check.py",
        "scripts/run_paper_repro.sh",
    ]
    for rel in required:
        assert Path(rel).exists(), f"Missing required release file: {rel}"


def test_release_check_passes() -> None:
    proc = subprocess.run(
        [sys.executable, "scripts/release_check.py", "--mode", "ci"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
