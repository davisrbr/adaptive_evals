"""
Central model configuration for all adaptive evaluation experiments.
Consolidates hardcoded model lists that were scattered across experiment runners.
"""

# Default evaluation models used across experiments
DEFAULT_EVAL_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini", 
    "anthropic/claude-3-5-sonnet-latest",
    "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "together/deepseek-ai/DeepSeek-V3",
]

# Default generator models for adaptive question generation
DEFAULT_GENERATOR_MODELS = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "anthropic/claude-3-5-sonnet-latest",
    "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
]

# Default judge models for evaluation
DEFAULT_JUDGE_MODELS = [
    "anthropic/claude-3-5-sonnet-latest",
    "openai/gpt-4o",
    "together/deepseek-ai/DeepSeek-V3",
]

# Task-specific model configurations
TASK_SPECIFIC_MODELS = {
    "cyberbullying": {
        "target_models": [
            "together/meta-llama/Llama-2-7b-chat-hf",
            "together/meta-llama/Llama-3-1-turbo",
            "openai/gpt-4o-mini",
            "anthropic/claude-3-5-haiku-latest",
        ],
        "attack_models": [
            "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
            "openai/gpt-4o",
            "anthropic/claude-3-5-sonnet-latest",
        ],
        "judge_models": [
            "anthropic/claude-3-5-sonnet-latest",
            "openai/gpt-4o",
        ]
    },
    "legal": {
        "eval_models": [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "together/deepseek-ai/DeepSeek-V3",
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "anthropic/claude-3-5-sonnet-latest",
        ],
        "generator_models": [
            "openai/gpt-4o-mini",
            "openai/gpt-4o",
            "anthropic/claude-3-5-sonnet-latest",
        ],
        "judge_models": [
            "anthropic/claude-3-5-sonnet-latest",
        ]
    },
    "truthfulqa": {
        "adaptive_eval_models": [
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "openai/gpt-4o-mini",
            "together/deepseek-ai/DeepSeek-V3",
        ],
        "generator_models": [
            "openai/gpt-4o",
        ],
        "re_eval_models": [
            "openai/gpt-4o-mini",
            "together/meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "together/deepseek-ai/DeepSeek-V3",
            "anthropic/claude-3-5-sonnet-latest",
        ],
        "judge_models": [
            "anthropic/claude-3-5-sonnet-latest",
        ]
    },
    "politeness": {
        "eval_models": [
            "openai/gpt-4o-mini",
        ],
        "generator_models": [
            "openai/gpt-4o-mini",
        ],
        "judge_models": [
            "together/deepseek-ai/DeepSeek-V3",
        ]
    },
    "forecasting": {
        "eval_models": [
            "openai/gpt-4",
            "openai/gpt-4o",
            "anthropic/claude-3-5-sonnet-latest",
        ]
    },
    "pair": {
        "target_models": [
            "together/meta-llama/Llama-2-7b-chat-hf",
            "together/meta-llama/Llama-3-1-turbo",
        ],
        "judge_models": [
            "openai/gpt-4o",
            "anthropic/claude-3-5-sonnet-latest",
        ],
        "attack_models": [
            "together/mistralai/Mixtral-8x22B-Instruct-v0.1",
            "anthropic/claude-3-5-sonnet-latest",
        ]
    }
}

def get_models_for_task(task_type: str, model_role: str = "eval_models") -> list[str]:
    """
    Get model list for specific task and role.
    
    Args:
        task_type: Type of task (cyberbullying, legal, truthfulqa, etc.)
        model_role: Role of models (eval_models, generator_models, judge_models, etc.)
    
    Returns:
        List of model names for the specified task and role
    """
    if task_type in TASK_SPECIFIC_MODELS:
        return TASK_SPECIFIC_MODELS[task_type].get(model_role, [])
    
    # Fallback to defaults
    if model_role == "eval_models":
        return DEFAULT_EVAL_MODELS
    elif model_role == "generator_models":
        return DEFAULT_GENERATOR_MODELS
    elif model_role == "judge_models":
        return DEFAULT_JUDGE_MODELS
    else:
        return DEFAULT_EVAL_MODELS


def get_all_unique_models() -> list[str]:
    """Get all unique model names across all tasks and roles."""
    all_models = set()
    
    # Add defaults
    all_models.update(DEFAULT_EVAL_MODELS)
    all_models.update(DEFAULT_GENERATOR_MODELS)
    all_models.update(DEFAULT_JUDGE_MODELS)
    
    # Add task-specific models
    for task_config in TASK_SPECIFIC_MODELS.values():
        for model_list in task_config.values():
            all_models.update(model_list)
    
    return sorted(list(all_models))