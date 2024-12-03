# Adaptive Evals

A framework for *adaptively* evaluating language models by using language models to generate test datasets based on previous evaluations. Built using [Inspect AI](https://inspect.ai-safety-institute.org.uk/).

## Overview

This project implements adaptive:
- Jailbreaks for HarmBench/JailbreakBench by building on PAIR by using previous successful jailbreaks to inform new jailbreak attempts.
- Adaptive evaluations for hallucinations with [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- Adaptive evaluations for legal judgement with [LegalBench](https://github.com/HazyResearch/legalbench/tree/main)
- [In progress] Adaptive evaluations for forecasting with [llm_forecasting](https://github.com/dannyallover/llm_forecasting) 

## Project Structure

- `tasks/`: Contains task definitions and evaluation configurations. The general set-up here is standard evaluation run -> adaptive evaluation run using logs from previous -> ...
  - `pair_inspect.py`: Implements PAIR tasks and configurations
  - `task_adaptive_truthfulqa.py`: TruthfulQA adaptive evaluation tasks
  - `task_adaptive_legal.py`: LegalBench adaptive evaluation tasks
  - `eval_adaptive_mc.py`: Standard multiple choice evaluation, on which adaptive evaluations are built

- `solvers/`: Contains the core logic for different evaluation approaches
  - `solvers_inspect.py`: Main solver implementations including PAIR and decomposition
  - `solver_adaptive_truthfulqa.py`: TruthfulQA-specific adaptive solver

## Task Implementations

### Adaptive jailbreaks
- Basic PAIR implementation with configurable parameters:
  - Max iterations
  - Target/Judge/Attack model selection
  - Historical context length
  - Scoring mechanisms (custom hierarchical/strong reject)
- Builds on basic PAIR with:
  - Embedding-based similar example retrieval
  - Configurable percentile-based sampling
  - TO DO: different diversity terms
  - TO DO: implement Maksym's template jailbreak

### TruthfulQA Adaptive
- Multiple choice evaluation framework
- Supports both single (mc1) and multiple (mc2) correct answer formats
- Adaptive generation of new questions based on model performance
- Configurable positive/negative sampling ratios

### LegalBench Adaptive
- Task-specific legal reasoning evaluation
- Support for chain-of-thought prompting in both generation and evaluation
- Separate judge model scoring
- Randomized sampling options

## Generic adaptive multiple choice testing
- Uses embeddings for similar example retrieval for in-context prompting
  - Configurable percentile-based sampling
- Supports multiple model configurations for generation and evaluation
- Flexible scoring mechanisms with judge model options

## Getting Started

1. Install Inspect AI following the [official documentation](https://inspect.ai-safety-institute.org.uk/)
2. Clone this repository
3. Configure your model access and API keys as required by Inspect AI