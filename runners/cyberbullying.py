"""
Refactored cyberbullying experiment runner using the new base class.
This demonstrates the cleanup approach for all experiment runners.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import click
from datetime import datetime

from runners.base import BaseExperimentRunner, ExperimentConfig
from config.models import get_models_for_task
from config.constants import DEFAULT_CONFIGS, CSV_FIELDNAMES
from tasks.task_cyberbullying import cyberbullying_task, cyberbullying_jailbreak_task, cyberbullying_adaptive_task
from inspect_ai import Epochs, eval
from inspect_ai.log import EvalLog


@dataclass
class CyberbullyingExperimentConfig(ExperimentConfig):
    """Cyberbullying-specific experiment configuration"""
    
    def __init__(self, max_streams: int = 1, **kwargs):
        super().__init__(task_type="cyberbullying", **kwargs)
        self.max_streams = max_streams
        # Apply cyberbullying-specific defaults
        config_defaults = DEFAULT_CONFIGS["cyberbullying"]
        for key, value in config_defaults.items():
            if not hasattr(self, key):
                setattr(self, key, value)


class CyberbullyingExperimentRunner(BaseExperimentRunner):
    """Cyberbullying experiment runner using the base class"""
    
    def __init__(self, config: CyberbullyingExperimentConfig):
        super().__init__(config)
        self.config: CyberbullyingExperimentConfig = config
    
    def get_default_models(self) -> Dict[str, List[str]]:
        """Get default models for cyberbullying experiments"""
        return {
            "target_models": get_models_for_task("cyberbullying", "target_models"),
            "attack_models": get_models_for_task("cyberbullying", "attack_models"),
            "judge_models": get_models_for_task("cyberbullying", "judge_models"),
        }
    
    def create_initial_task(self, **kwargs):
        """Create initial cyberbullying evaluation task"""
        return cyberbullying_task()
    
    def create_adaptive_task(self, **kwargs):
        """Create adaptive cyberbullying task"""
        return cyberbullying_adaptive_task(
            initial_log_path=kwargs.get("initial_log_path"),
            **kwargs
        )
    
    def create_jailbreak_task(self, **kwargs):
        """Create cyberbullying jailbreak task"""
        return cyberbullying_jailbreak_task(**kwargs)
    
    def extract_cyberbullying_metrics(self, eval_log: EvalLog) -> Tuple[Optional[float], Optional[int], Optional[int]]:
        """Extract cyberbullying-specific metrics"""
        if not eval_log or not hasattr(eval_log, 'samples') or not eval_log.samples:
            return None, None, None
        
        total_score = 0
        success_count = 0
        total_count = len(eval_log.samples)
        
        for sample in eval_log.samples:
            if not hasattr(sample, 'store') or not sample.store:
                continue
                
            best_score = sample.store.get("best_score", 0)
            jailbreak_success = sample.store.get("jailbreak_success", False)
            
            total_score += best_score
            if jailbreak_success:
                success_count += 1
        
        mean_score = total_score / total_count if total_count > 0 else 0
        return mean_score, success_count, total_count
    
    def write_experiment_log(self, **kwargs) -> None:
        """Write cyberbullying-specific experiment log"""
        fieldnames = CSV_FIELDNAMES["cyberbullying"]
        super().write_experiment_log(**kwargs)
    
    def run_experiment(self, target_models: List[str], attack_models: List[str], **kwargs):
        """Run cyberbullying experiments"""
        for target_model in target_models:
            for attack_model in attack_models:
                print(f"Running cyberbullying experiment: target={target_model}, attack={attack_model}")
                
                # Create and run initial task
                initial_task = self.create_initial_task()
                log_dir = self.config.get_initial_log_dir()
                
                initial_logs = eval(
                    initial_task,
                    epochs=Epochs(1, "max"),
                    max_connections=10000,
                    log_dir=str(log_dir),
                    model=target_model,
                    log_level="error"
                )
                
                if not initial_logs or initial_logs[0].status != "success":
                    print(f"Initial experiment failed for {target_model}")
                    continue
                
                initial_log = initial_logs[0]
                self.write_eval_cache(target_model, initial_log.location)
                
                # Extract metrics
                mean_score, success_count, total_count = self.extract_cyberbullying_metrics(initial_log)
                
                # Log results
                self.write_experiment_log(
                    model_name=target_model,
                    log_path=initial_log.location,
                    mean_score=mean_score,
                    success_count=success_count,
                    total_count=total_count,
                    experiment_config=self.config.experiment_id
                )


@click.command()
@click.option("--target-models", multiple=True, help="Target models to evaluate")
@click.option("--attack-models", multiple=True, help="Attack models to use")
@click.option("--use-cot", is_flag=True, help="Use chain of thought")
@click.option("--max-streams", default=1, type=int, help="Maximum streams")
@click.option("--num-epochs", default=100, type=int, help="Number of epochs")
def main(target_models: List[str], attack_models: List[str], use_cot: bool, 
         max_streams: int, num_epochs: int):
    """
    Refactored cyberbullying experiment runner using base class.
    Demonstrates the cleanup approach for consolidating experiment runners.
    """
    
    # Use defaults if no models specified
    if not target_models:
        target_models = get_models_for_task("cyberbullying", "target_models")
    if not attack_models:
        attack_models = get_models_for_task("cyberbullying", "attack_models")
    
    # Create configuration
    config = CyberbullyingExperimentConfig(
        use_cot_target=use_cot,
        max_streams=max_streams,
        num_epochs=num_epochs
    )
    
    # Create and run experiment
    runner = CyberbullyingExperimentRunner(config)
    runner.run_experiment(
        target_models=list(target_models),
        attack_models=list(attack_models)
    )


if __name__ == "__main__":
    main()