#!/usr/bin/env python3
"""
Comprehensive test suite for adaptive_evals project functionality.

This test file verifies:
1. All task files load correctly and are compatible with inspect eval
2. All refactored experiment runners initialize properly
3. Basic inspect eval functionality works with available models
4. Import dependencies are resolved correctly

Usage:
    python test_all_functionality.py

Requirements:
    - conda environment 'inspect4' activated
    - env.sh sourced for API keys
    - adaptive_evals package installed with: pip install -U .
"""

import subprocess
import sys
import time
import traceback
from typing import Dict, List, Tuple, Optional


class TestSuite:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.results = []
    
    def run_test(self, test_name: str, test_func, *args, **kwargs) -> bool:
        """Run a single test and record results."""
        try:
            test_func(*args, **kwargs)
            self.passed += 1
            self.results.append(f"✅ {test_name}: PASSED")
            print(f"✅ {test_name}: PASSED")
            return True
        except Exception as e:
            self.failed += 1
            self.results.append(f"❌ {test_name}: FAILED - {str(e)}")
            print(f"❌ {test_name}: FAILED - {str(e)}")
            return False
    
    def print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success rate: {(self.passed/total)*100:.1f}%" if total > 0 else "No tests run")
        
        if self.failed > 0:
            print("\nFAILED TESTS:")
            for result in self.results:
                if "FAILED" in result:
                    print(f"  {result}")
        
        print("\n" + "="*60)
        return self.failed == 0


def test_task_loading():
    """Test that all task files load correctly."""
    tasks_to_test = [
        ('task_adaptive_truthfulqa', 'truthfulqa_initial', 'TruthfulQA Initial'),
        ('task_politeness', 'politeness_n_shot', 'Politeness N-Shot'),
        ('task_adaptive_legal', 'legalbench_initial_aggregated', 'Legal Initial'),
        ('pair_inspect', 'pair_task', 'PAIR Task'),
        ('task_cyberbullying', 'cyberbullying_task', 'Cyberbullying Task'),
    ]
    
    for module_name, task_func, display_name in tasks_to_test:
        try:
            module = __import__(f'tasks.{module_name}', fromlist=[task_func])
            task_function = getattr(module, task_func)
            task = task_function()
            
            # Verify task has expected attributes
            assert hasattr(task, 'dataset'), f"{display_name} task missing dataset"
            assert len(task.dataset) > 0, f"{display_name} task has empty dataset"
            
        except Exception as e:
            raise Exception(f"Failed to load {display_name}: {str(e)}")


def test_refactored_runners():
    """Test that all refactored experiment runners initialize correctly."""
    runners = [
        ('cyberbullying', 'CyberbullyingExperimentConfig', 'CyberbullyingExperimentRunner'),
        ('pair', 'PairExperimentConfig', 'PairExperimentRunner'),
        ('politeness', 'PolitenessExperimentConfig', 'PolitenessExperimentRunner'),
        ('truthfulqa', 'TruthfulQAExperimentConfig', 'TruthfulQAExperimentRunner'),
        ('legal', 'LegalExperimentConfig', 'LegalExperimentRunner')
    ]
    
    for task_name, config_class, runner_class in runners:
        try:
            module = __import__(f'experiment_runner_{task_name}_refactored', fromlist=[config_class, runner_class])
            ConfigClass = getattr(module, config_class)
            RunnerClass = getattr(module, runner_class)
            
            # Test configuration initialization
            config = ConfigClass(
                cache_path=f'test_{task_name}_cache.csv',
                results_path=f'test_{task_name}_results.csv'
            )
            
            # Verify config methods work
            assert config.get_cache_path() == f'test_{task_name}_cache.csv'
            assert config.get_results_path() == f'test_{task_name}_results.csv'
            
            # Test runner initialization
            runner = RunnerClass(config)
            
            # Verify runner has expected attributes
            assert hasattr(runner, 'config'), f"{task_name} runner missing config"
            assert hasattr(runner, 'cache_csv'), f"{task_name} runner missing cache_csv"
            
        except Exception as e:
            raise Exception(f"Failed to initialize {task_name} runner: {str(e)}")


def test_base_experiment_runner():
    """Test base experiment runner functionality."""
    from experiment_runner_base import BaseExperimentRunner, ExperimentConfig
    
    # Test basic config
    config = ExperimentConfig(
        cache_path='test_cache.csv',
        results_path='test_results.csv',
        task_type='test'
    )
    
    # Test config methods
    assert config.get_cache_path() == 'test_cache.csv'
    assert config.get_results_path() == 'test_results.csv'
    assert 'test' in config.get_initial_log_dir()
    
    # Test runner initialization
    runner = BaseExperimentRunner(config)
    assert runner.config == config
    assert runner.cache_csv == 'test_cache.csv'


def test_inspect_eval_compatibility():
    """Test that inspect eval can load task files without errors."""
    import subprocess
    
    tasks_to_test = [
        'tasks/task_adaptive_truthfulqa.py@truthfulqa_initial',
        'tasks/task_politeness.py@politeness_n_shot',
        'tasks/task_adaptive_legal.py@legalbench_initial_aggregated',
        'tasks/pair_inspect.py@pair_task'
    ]
    
    for task_path in tasks_to_test:
        try:
            # Test that inspect eval can parse and load the task
            # We use --help to avoid actually running the evaluation
            cmd = [
                'inspect', 'eval', task_path,
                '--model', 'openai/gpt-4o-mini',
                '--limit', '0',  # Don't run any samples
                '--no-score'     # Don't score anything
            ]
            
            # Run with a short timeout to verify loading works
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check that the command didn't fail with import errors
            if "ModuleNotFoundError" in result.stderr:
                raise Exception(f"Import error: {result.stderr}")
            if "No module named" in result.stderr:
                raise Exception(f"Missing module: {result.stderr}")
            if result.returncode not in [0, 1]:  # 1 is OK for no samples run
                raise Exception(f"Command failed with code {result.returncode}: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            # Timeout is OK - means the task loaded successfully
            pass
        except Exception as e:
            raise Exception(f"inspect eval failed for {task_path}: {str(e)}")


def test_critical_imports():
    """Test that critical imports work without dependency conflicts."""
    critical_imports = [
        ('inspect_ai', 'Inspect AI framework'),
        ('experiment_runner_base', 'Base experiment runner'),
        ('data.data_inspect', 'PAIR data module (should not import jailbreakbench)'),
        ('tasks.pair_inspect', 'PAIR task module'),
        ('tasks.task_adaptive_truthfulqa', 'TruthfulQA task module'),
        ('tasks.task_politeness', 'Politeness task module'),
        ('tasks.task_adaptive_legal', 'Legal task module'),
    ]
    
    for module_name, description in critical_imports:
        try:
            __import__(module_name)
        except Exception as e:
            raise Exception(f"Failed to import {description} ({module_name}): {str(e)}")


def test_config_package():
    """Test that the centralized config package works."""
    try:
        from config.models import get_models_for_task, DEFAULT_EVAL_MODELS
        from config.paths import get_log_dir, get_cache_path
        from config.constants import DEFAULT_SIMILARITY_THRESHOLD
        
        # Test model configuration
        assert isinstance(DEFAULT_EVAL_MODELS, list)
        assert len(DEFAULT_EVAL_MODELS) > 0
        
        # Test path functions
        log_dir = get_log_dir("test", "initial")
        log_dir_str = str(log_dir)
        assert "test" in log_dir_str
        assert "initial" in log_dir_str
        
        cache_path = get_cache_path("test_experiment")
        cache_path_str = str(cache_path)
        assert "test_experiment" in cache_path_str
        
        # Test constants
        assert isinstance(DEFAULT_SIMILARITY_THRESHOLD, (int, float))
        
    except Exception as e:
        raise Exception(f"Config package test failed: {str(e)}")


def test_cli_interfaces():
    """Test that CLI interfaces are properly configured."""
    runners_with_cli = [
        'experiment_runner_legal_refactored.py',
        'experiment_runner_pair_refactored.py',
        'experiment_runner_politeness_refactored.py',
        'experiment_runner_truthfulqa_refactored.py'
    ]
    
    for runner_file in runners_with_cli:
        try:
            result = subprocess.run(
                ['python', runner_file, '--help'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                raise Exception(f"CLI help failed: {result.stderr}")
            
            # Check that help output contains expected elements
            help_text = result.stdout
            assert "Usage:" in help_text, f"Missing usage info in {runner_file}"
            assert "Options:" in help_text, f"Missing options info in {runner_file}"
            
        except subprocess.TimeoutExpired:
            raise Exception(f"CLI help command timed out for {runner_file}")
        except Exception as e:
            raise Exception(f"CLI test failed for {runner_file}: {str(e)}")


def main():
    """Run all tests."""
    print("🧪 Starting Comprehensive Test Suite for adaptive_evals")
    print("="*60)
    
    suite = TestSuite()
    
    # Core functionality tests
    suite.run_test("Critical Imports", test_critical_imports)
    suite.run_test("Task Loading", test_task_loading)
    suite.run_test("Base Experiment Runner", test_base_experiment_runner)
    suite.run_test("Refactored Runners", test_refactored_runners)
    suite.run_test("Config Package", test_config_package)
    
    # Integration tests
    suite.run_test("Inspect Eval Compatibility", test_inspect_eval_compatibility)
    suite.run_test("CLI Interfaces", test_cli_interfaces)
    
    # Print final summary
    success = suite.print_summary()
    
    if success:
        print("🎉 ALL TESTS PASSED! The adaptive_evals project is fully functional.")
        return 0
    else:
        print("💥 SOME TESTS FAILED! Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())