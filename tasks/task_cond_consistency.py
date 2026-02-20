from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, hf_dataset
from scorers.scorers_consistency import cond_consistency_scorer
from solvers.solvers_cond_consistency import cond_consistency_solver
import logging
import os
from inspect_ai.model import get_model

@task
def cond_initial_consistency() -> Task:
    """Initial conditional consistency evaluation task using HuggingFace dataset."""
    def record_to_sample(record: dict) -> Sample:
        return Sample(
            input=record['P_title'],  # Just the P_title in input
            target="",  # No target needed for consistency evaluation
            metadata={
                'P_title': record['P_title'],
                'P_body': record['P_body'],
                'Q_given_P_title': record['Q_given_P_title'],
                'Q_given_P_body': record['Q_given_P_body'],
                'P_and_Q_title': record['P_and_Q_title'],
                'P_and_Q_body': record['P_and_Q_body']
            }
        )

    dataset = hf_dataset(
        "prithvi3/cond_100_test", 
        split="test",
        trust=True,
        sample_fields=record_to_sample,
        auto_id=True,
        shuffle=True,
    )


    return Task(
        dataset=dataset,
        solver=cond_consistency_solver(),
        scorer=cond_consistency_scorer()
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    log_dir = "logs/cond_consistency"
    os.makedirs(log_dir, exist_ok=True)
    #"together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    try:
        task = cond_initial_consistency()
        initial_log = eval(
            task, 
            model = "openai/o1-mini",
            # temperature=0,
            start_log=True,
            log_format="eval"
        )[0]
                    # model=["together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K", "openai/gpt-4o", "openai/gpt-4o-mini", "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "anthropic/claude-3-5-sonnet-20241022"], 

        print("\nEvaluation complete")
        
    except Exception as e:
        logging.error(f"Error in evaluation pipeline: {str(e)}")
        raise