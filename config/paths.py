"""
Central path configuration for adaptive evaluation experiments.
Consolidates hardcoded paths and provides consistent directory structure.
"""
import os
from pathlib import Path

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent

# Base directories
LOGS_DIR = PROJECT_ROOT / "logs"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Task-specific base paths
LEGALBENCH_DIR = PROJECT_ROOT / "legalbench"
LEGALBENCH_TASKS_DIR = LEGALBENCH_DIR / "tasks"

# Default directory names for different experiment types
EXPERIMENT_DIRS = {
    "cyberbullying": "cyberbullying",
    "legal": "legalbench", 
    "truthfulqa": "truthfulqa",
    "politeness": "politeness",
    "forecasting": "forecasting",
    "pair": "pair",
    "consistency": "consistency",
}

def get_log_dir(task_type: str, experiment_stage: str = "initial", experiment_id: str = "") -> Path:
    """
    Get standardized log directory path.
    
    Args:
        task_type: Type of task (cyberbullying, legal, etc.)
        experiment_stage: Stage of experiment (initial, adaptive, re_eval)
        experiment_id: Unique experiment identifier
    
    Returns:
        Path to log directory
    """
    base_dir = LOGS_DIR / EXPERIMENT_DIRS.get(task_type, task_type)
    if experiment_stage and experiment_id:
        return base_dir / experiment_stage / experiment_id
    elif experiment_stage:
        return base_dir / experiment_stage
    else:
        return base_dir


def get_cache_path(task_type: str, experiment_id: str = "") -> Path:
    """
    Get standardized cache file path.
    
    Args:
        task_type: Type of task
        experiment_id: Unique experiment identifier
        
    Returns:
        Path to cache CSV file
    """
    if experiment_id:
        filename = f"{task_type}_cache_{experiment_id}.csv"
    else:
        filename = f"{task_type}_cache.csv"
    return CACHE_DIR / filename


def get_results_path(task_type: str, experiment_id: str = "", filename: str = "") -> Path:
    """
    Get standardized results file path.
    
    Args:
        task_type: Type of task
        experiment_id: Unique experiment identifier
        filename: Specific filename (optional)
        
    Returns:
        Path to results file
    """
    base_dir = RESULTS_DIR / task_type
    
    if filename:
        return base_dir / filename
    elif experiment_id:
        return base_dir / f"experiment_results_{experiment_id}.csv"
    else:
        return base_dir / f"experiment_results.csv"


def get_legalbench_prompt_path(task_name: str, prompt_type: str = "base_prompt") -> Path:
    """
    Get path to LegalBench prompt template.
    
    Args:
        task_name: Name of the legal task
        prompt_type: Type of prompt (base_prompt, claude_prompt, base_prompt_wo_example)
        
    Returns:
        Path to prompt template file
    """
    return LEGALBENCH_TASKS_DIR / task_name / f"{prompt_type}.txt"


def ensure_dir(path: Path) -> None:
    """Ensure directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def ensure_dirs(*paths: Path) -> None:
    """Ensure multiple directories exist."""
    for path in paths:
        ensure_dir(path)


# Ensure base directories exist
ensure_dirs(LOGS_DIR, CACHE_DIR, RESULTS_DIR, DATA_DIR)