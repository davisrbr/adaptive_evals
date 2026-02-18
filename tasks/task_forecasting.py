""" 
Approaching Human-Level Forecasting with Language Models

Danny Halawi, Fred Zhang, Chen Yueh-Han, Jacob Steinhardt

https://arxiv.org/abs/2402.18563 

Based on https://github.com/dannyallover/llm_forecasting
Dataset: https://huggingface.co/datasets/YuehHanChen/forecasting

advanced_forecasting_solver uses Halawi's retrieval, reasoning, and ensembling setup with their best scratchpad prompts

- Point to your own chromedriver path in util_forecastings/information_retrieval.py
- Add keys.py in util_forecastings

Examples:
inspect eval task_forecasting.py --model openai/gpt-4
inspect eval task_forecasting.py --solver=advanced_forecasting_solver --model=openai/gpt-4o-2024-08-06
"""

from inspect_ai import Task, task 
from inspect_ai.dataset import hf_dataset, Sample
from inspect_ai.model import GenerateConfig 
from solvers.solvers_forecasting import zero_shot_forecasting_solver
from scorers.scorers_forecasting import brier_score


@task
def forecasting() -> Task:
    """
    Task implementation for the Forecasting benchmark using YuehHanChen/forecasting dataset
    """
    def record_to_sample(record):
        """Convert dataset record to Sample with input, target, and metadata fields"""
        return Sample(
            input=record["question"],  # Primary input
            target=str(record["resolution"]),  # Target value for scoring
            metadata={
                # Fields needed for the zero-shot prompt
                "background": record["background"],
                "resolution_criteria": record["resolution_criteria"],
                "date_begin": record["date_begin"],
                "date_close": record["date_close"],
                # Additional metadata not used in prompt
                #"gpt_3p5_category": record["gpt_3p5_category"],
                "url": record["url"],
                "community_predictions": record["community_predictions"],
                "question_type": record["question_type"],
                "extracted_urls": record["extracted_urls"],
                "date_resolve_at": record["date_resolve_at"],
                "data_source": record["data_source"],
                "is_resolved": record["is_resolved"], 
                "retrieval_date": record["sampled_retrieval_date"]
            }
        )

    # Load dataset
    # dataset = hf_dataset(
    #     "YuehHanChen/forecasting",
    #     split="test",
    #     trust=True,
    #     sample_fields=record_to_sample,
    #     auto_id=True,
    #     shuffle=True,
    # )
    dataset = hf_dataset(
        "prithvi3/forecast_sample_test_filtered_earlyres",
        split="test",
        trust=True,
        sample_fields=record_to_sample,
        auto_id=True,
        shuffle=True,
    )

    # Filter for resolved binary questions
    dataset = dataset.filter(
        lambda x: x.metadata["is_resolved"] is True and x.metadata["question_type"].lower() == "binary"
    )

    # Select only the first 2 samples for testing
    dataset = dataset[:5]

    return Task(
        dataset=dataset,
        solver=zero_shot_forecasting_solver(),
        scorer=[brier_score()],
        #config=GenerateConfig(temperature=0.5),
    )


