from pathlib import Path


def find_configs(directory):
    """Find all .yaml files in directory and subdirectories."""
    root = Path(directory)
    # BUG: glob("*.yaml") only matches top-level, not recursive
    return sorted([str(p) for p in root.glob("*.yaml")])
