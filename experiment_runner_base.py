"""
Base experiment runner with shared functionality for all evaluation experiments.
Consolidates common patterns from cyberbullying, legal, politeness, and truthfulqa runners.
"""
import os
import csv
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from inspect_ai import Epochs, eval
from inspect_ai.log import EvalLog, read_eval_log


@dataclass
class ExperimentConfig:
    """Base configuration for all experiments"""
    use_example: bool = True
    use_cot_target: bool = True
    use_cot_attacker: bool = False
    num_epochs: int = 100
    randomize_sampling: bool = False
    task_type: str = "base"
    
    def __post_init__(self):
        # Configure logging to be quiet by default
        self._setup_logging()
    
    def _setup_logging(self):
        """Disable noisy logging output"""
        logging.getLogger().setLevel(logging.ERROR)
        logging.getLogger('inspect_ai').setLevel(logging.ERROR)
        logging.getLogger('httpx').setLevel(logging.ERROR)
        logging.getLogger('httpcore').setLevel(logging.ERROR)
    
    @property
    def experiment_id(self) -> str:
        """Generate a readable experiment identifier"""
        components = []
        if self.use_example:
            components.append("with_examples")
        if self.use_cot_target:
            components.append("cot_target")
        if self.use_cot_attacker:
            components.append("cot_attacker")
        if self.num_epochs:
            components.append(f"num_epochs_{self.num_epochs}")
        return "_".join(components) or "base"
    
    def get_cache_path(self) -> str:
        """Get the path for the cache CSV"""
        return os.path.join("cache", f"initial_eval_{self.experiment_id}.csv")
    
    def get_initial_log_dir(self) -> str:
        """Get the directory for initial evaluation logs"""
        return os.path.join("logs", self.task_type, "initial", self.experiment_id)
    
    def get_adaptive_log_dir(self) -> str:
        """Get the directory for adaptive evaluation logs"""
        return os.path.join("logs", self.task_type, "adaptive", self.experiment_id)
    
    def get_experiment_csv_path(self) -> str:
        """Get the path for experiment results CSV"""
        return os.path.join("results", self.task_type, f"experiment_results_{self.experiment_id}.csv")


class BaseExperimentRunner(ABC):
    """Base class for all experiment runners with shared functionality"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.cache_csv = config.get_cache_path()
        self.experiment_csv = config.get_experiment_csv_path()
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.cache_csv), exist_ok=True)
        os.makedirs(os.path.dirname(self.experiment_csv), exist_ok=True)
        os.makedirs(config.get_initial_log_dir(), exist_ok=True)
        os.makedirs(config.get_adaptive_log_dir(), exist_ok=True)
    
    def read_eval_cache(self) -> Dict[str, str]:
        """
        Returns a dict of {model_name: log_path} previously saved.
        """
        if not os.path.exists(self.cache_csv):
            return {}
        output = {}
        with open(self.cache_csv, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                model_name = row["model_name"]
                log_path = row["log_path"]
                output[model_name] = log_path
        return output
    
    def write_eval_cache(self, model_name: str, log_path: str) -> None:
        """
        Appends the given (model_name, log_path) to the CSV cache.
        Creates the file with headers if it does not exist.
        """
        file_exists = os.path.exists(self.cache_csv)
        with open(self.cache_csv, mode="a", newline="") as f:
            fieldnames = ["model_name", "log_path"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({"model_name": model_name, "log_path": log_path})
    
    def extract_accuracy_metrics(self, eval_log: EvalLog) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """
        Extract basic accuracy metrics from an evaluation log.
        Subclasses can override for task-specific metrics.
        """
        if not eval_log or not hasattr(eval_log, 'results') or not eval_log.results:
            return None, None, None
        
        # Extract accuracy from results
        accuracy = None
        std_err = None
        
        # Try to get accuracy from scores
        if hasattr(eval_log.results, 'scores'):
            for score_name, score_data in eval_log.results.scores.items():
                if 'accuracy' in score_name.lower():
                    accuracy = score_data.value if hasattr(score_data, 'value') else score_data
                    if hasattr(score_data, 'std_err'):
                        std_err = score_data.std_err
                    break
        
        # Fallback: calculate from samples
        if accuracy is None and hasattr(eval_log, 'samples'):
            correct = sum(1 for sample in eval_log.samples 
                         if hasattr(sample, 'scores') and any(
                             score.value == 1.0 for score in sample.scores.values()
                         ))
            accuracy = correct / len(eval_log.samples) if eval_log.samples else 0.0
        
        model_name = eval_log.eval.model if hasattr(eval_log, 'eval') and hasattr(eval_log.eval, 'model') else "unknown"
        return accuracy, std_err, model_name
    
    def check_experiment_already_run(self, **kwargs) -> bool:
        """
        Check if an experiment with the given parameters has already been run.
        Subclasses should override to define their specific parameter matching logic.
        """
        if not os.path.exists(self.experiment_csv):
            return False
        
        with open(self.experiment_csv, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Default implementation - subclasses should override
                # with their specific parameter matching
                return False
        return False
    
    def write_experiment_log(self, **kwargs) -> None:
        """
        Write experiment results to CSV log.
        Subclasses should override to define their specific fields.
        """
        file_exists = os.path.exists(self.experiment_csv)
        
        # Default fields - subclasses should override
        fieldnames = ["timestamp", "experiment_id"] + list(kwargs.keys())
        
        with open(self.experiment_csv, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            
            row = {
                "timestamp": datetime.now().isoformat(),
                "experiment_id": self.config.experiment_id,
                **kwargs
            }
            writer.writerow(row)
    
    @abstractmethod
    def get_default_models(self) -> Dict[str, List[str]]:
        """Return default model configurations for this experiment type"""
        pass
    
    @abstractmethod
    def create_initial_task(self, **kwargs):
        """Create the initial evaluation task"""
        pass
    
    @abstractmethod
    def create_adaptive_task(self, **kwargs):
        """Create the adaptive evaluation task"""
        pass
    
    @abstractmethod
    def run_experiment(self, **kwargs):
        """Run the full experiment pipeline"""
        pass


class TransferExperimentMixin:
    """Mixin for experiments that support transfer learning between models"""
    
    def run_transfer_experiment(self, source_models: List[str], target_models: List[str], **kwargs):
        """
        Run transfer experiment where models trained on source_models 
        are evaluated on target_models.
        """
        # Common transfer logic that can be shared across experiment types
        pass


# Utility functions that were duplicated across experiment runners
def ensure_directory_exists(path: str) -> None:
    """Ensure directory exists, creating it if necessary"""
    os.makedirs(path, exist_ok=True)


def safe_read_json_from_string(json_str: str) -> Optional[Dict]:
    """Safely parse JSON string, returning None if invalid"""
    try:
        return eval(json_str) if json_str else None
    except:
        return None


def get_log_path_with_timestamp(base_dir: str, task_name: str, model_name: str) -> str:
    """Generate timestamped log path"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    return os.path.join(base_dir, f"{task_name}_{safe_model_name}_{timestamp}.json")