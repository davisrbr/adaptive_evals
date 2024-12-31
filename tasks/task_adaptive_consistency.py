from inspect_ai import Task, task, eval
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from solvers.solvers_consistency import consistency_solver, adaptive_consistency_solver, ConsistencyType, adaptive_consistency_judge_solver, temp_adaptive_consistency_judge_solver
from scorers.scorers_consistency import consistency_scorer, adaptive_consistency_judge_scorer, temp_adaptive_consistency_judge_scorer
import logging
import os
import random
from typing import Optional, List



@task
def initial_consistency() -> Task:
    """Initial consistency evaluation task."""
    
    def record_to_sample(record: dict) -> Sample:
        return Sample(
            input=record['question'],
            target=str(record['resolution']),
            metadata={
                'id': str(random.randint(10000, 99999)),
                'question_type': record['question_type'],
                'data_source': record['data_source'],
                'is_resolved': record.get('is_resolved', False),
                'community_predictions': record.get('community_predictions', []),
                'original_data': {
                    'title': record['question'],
                    'body': record['background'],
                    'resolution_criteria': record['resolution_criteria'],
                    'resolution_date': record['date_resolve_at'],
                    'created_date': record.get('date_begin', '')
                }
            }
        )

    dataset = hf_dataset(
        "prithvi3/filtered_forecast_sample_test",
        split="test",
        trust=True,
        sample_fields=record_to_sample,
        auto_id=True,
        shuffle=True,
    )

    dataset = dataset.filter(
        lambda x: x.metadata["is_resolved"] is True and 
                 x.metadata["question_type"].lower() == "binary"
    )

    #dataset_questions = [sample.input for sample in dataset]

    dataset = dataset[:2]
    
    return Task(
        dataset=dataset,
        solver=consistency_solver(),  # Single solver that handles all types
        scorer=consistency_scorer()
    )


@task
def adaptive_consistency(
    initial_log_path: str,
    consistency_types: List[str] = [ct.value for ct in ConsistencyType],
    use_embeddings: bool = False, 
    dataset_path: Optional[str] = None,
) -> Task:
    """Creates adversarial consistency questions using metrics from previous evaluations"""

    def record_to_sample(record: dict) -> Sample:
        return Sample(
            input=record['question'],
            target=str(record['resolution']),
            metadata={
                'id': str(random.randint(10000, 99999)),
                'question_type': record['question_type'],
                'data_source': record['data_source'],
                'is_resolved': record.get('is_resolved', False),
                'community_predictions': record.get('community_predictions', []),
                'original_data': {
                    'title': record['question'],
                    'body': record['background'],
                    'resolution_criteria': record['resolution_criteria'],
                    'resolution_date': record['date_resolve_at'],
                    'created_date': record.get('date_begin', '')
                }
            }
        )
    
    dataset_questions = None
    if use_embeddings and dataset_path:
        dataset = hf_dataset(
            dataset_path,
            split="test",
            trust=True,
            sample_fields=record_to_sample,
            auto_id=True,
            shuffle=True,
        )
        dataset = dataset.filter(
            lambda x: x.metadata["is_resolved"] is True and 
                     x.metadata["question_type"].lower() == "binary"
        )
        # dataset = dataset[:25]
        dataset_questions = [sample.input for sample in dataset]
    
    # return Task(
    #     dataset=MemoryDataset(name="adaptive_consistency", samples=[]),
    #     solver=[
    #         adaptive_consistency_solver(
    #             initial_log_path=initial_log_path,
    #             consistency_types=consistency_types,
    #             use_embeddings=use_embeddings,
    #             dataset_questions=dataset_questions,
    #         ), 
    #         adaptive_consistency_judge_solver(
    #             initial_log_path=initial_log_path,
    #             judge_model_name="openai/o1-mini",
    #         )
    #     ],
    #     scorer=[consistency_scorer(), adaptive_consistency_judge_scorer()]
    # )
    return Task(
        dataset=MemoryDataset(name="adaptive_consistency", samples=[]),
        solver=[
            adaptive_consistency_solver(
                initial_log_path=initial_log_path,
                consistency_types=consistency_types,
                use_embeddings=use_embeddings,
                dataset_questions=dataset_questions,
            )
        ],
        scorer=[consistency_scorer()]
    )

@task
def temp_adaptive_judge(
    initial_log_path: str,
    adaptive_log_path: str,
) -> Task:
    """Creates a task to judge the adaptive consistency questions"""
    
    return Task(
        dataset=MemoryDataset(name="temp_adaptive_judge", samples=[]),
        solver=[
            temp_adaptive_consistency_judge_solver(
                initial_log_path=initial_log_path,
                adaptive_log_path=adaptive_log_path,
                judge_model_name="openai/gpt-4o", #placeholder
            )
        ],
        scorer=[temp_adaptive_consistency_judge_scorer()]
    )


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     log_dir = "logs/initial_consistency"
#     os.makedirs(log_dir, exist_ok=True)
    
#     try:
#         task = initial_consistency()
#         initial_log = eval(
#             task, 
#             # epochs=Epochs(1, "max"),
#             # max_connections=1000,
#             # log_dir=log_dir,
#             model="openai/gpt-4o",
#             temperature=0,
#             start_log=True
#         )[0]

#         print("\nEvaluation complete")
#         # print(f"Initial log path: {initial_log.location}")
        
#     except Exception as e:
#         logging.error(f"Error in evaluation pipeline: {str(e)}")
#         raise


# if __name__ == "__main__":
#    logging.basicConfig(
#        level=logging.INFO,
#        format='%(asctime)s - %(levelname)s - %(message)s'
#    )
#    #initial_log_path = "logs/2024-12-26T11-19-29+05-30_initial-consistency_cExeriowwKWrcS4rBbUc9f.eval"
#    initial_log_path = "logs/2024-12-31T18-27-08+05-30_initial-consistency_njAobmajK8z8AMLiJo7BJS.eval"
   
#    dataset_path="prithvi3/filtered_forecast_sample_test"

#    task = adaptive_consistency(
#        initial_log_path=initial_log_path,
#        use_embeddings=False,  # Enable embeddings if needed
#        dataset_path=None  # Provide dataset path when using embeddings
#    )
   
#    result = eval(
#        task,
#        model="openai/gpt-4o",
#        temperature=0,
#        start_log=True
#    )[0]
   
#    print("\nEvaluation complete")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Paths to evaluation logs
    initial_log_path = "logs/2024-12-31T18-27-08+05-30_initial-consistency_njAobmajK8z8AMLiJo7BJS.eval"
    adaptive_log_path = "logs/2024-12-31T18-28-15+05-30_adaptive-consistency_aDE5pPq8oawANKvx28FtX6.eval"  # This would be your adaptive eval log
    
    task = temp_adaptive_judge(
        initial_log_path=initial_log_path,
        adaptive_log_path=adaptive_log_path,
    )
    
    result = eval(
        task,
        model="openai/gpt-4o",
        temperature=0,
        start_log=True
    )[0]
    
    #print("\nEvaluation complete")
    #print(f"Judge accuracy: {result.scores[0].value:.3f}")  # Access accuracy score