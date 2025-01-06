from inspect_ai import Task, task, eval, Epochs
from inspect_ai.dataset import Sample, hf_dataset, MemoryDataset
from solvers.solvers_consistency import consistency_solver, adaptive_consistency_solver, ConsistencyType, adaptive_consistency_judge_solver, temp_adaptive_consistency_judge_solver
from scorers.scorers_consistency import consistency_scorer, adaptive_consistency_judge_scorer, temp_adaptive_consistency_judge_scorer
import logging
import os
from pathlib import Path
import random
from typing import Optional, List, Dict 
import pandas as pd

def get_questions_from_type(consistency_type: str) -> List[str]:
    """Get list of questions needed for a consistency type."""
    clean_type = consistency_type.lower().replace('checker', '').strip()
    
    mappings = {
        'expevidence': ["P", "Q", "P_given_Q", "P_given_not_Q"],
        'andor': ["P", "Q", "P_and_Q", "P_or_Q"],
        'but': ["P", "Q_and_not_P", "P_or_Q"],
        'cond': ["P", "Q_given_P", "P_and_Q"],
        'condcond': ["P", "Q_given_P", "R_given_P_and_Q", "P_and_Q_and_R"],
        'consequence': ["P", "cons_P"],
        'paraphrase': ["P", "para_P"],
        'and': ["P", "Q", "P_and_Q"],
        'or': ["P", "Q", "P_or_Q"],
        'not': ["P", "not_P"]
    }
    

    return mappings.get(clean_type, [])

def create_csv_sample(row: pd.Series) -> Sample:
    """Create a sample from a CSV row."""
    consistency_type = row['consistency_type'].lower().replace('checker', '').strip()
    question_names = get_questions_from_type(row['consistency_type'])
    
    
    questions = {}
    for q in question_names:
        if pd.notna(row[f'{q}_title']):
            questions[q] = {
                'question': {
                    'title': row[f'{q}_title'],
                    'body': row[f'{q}_body'] if pd.notna(row[f'{q}_body']) else '',
                    'question_type': 'binary'
                }
            }
    
    return Sample(
        input=row['P_title'],
        target='',
        metadata={
            'id': str(random.randint(100000, 999999)),
            'consistency_type': consistency_type,
            'questions': questions
        }
    )

@task
def initial_consistency(csv_path: Optional[str] = None) -> Task:
    """Initial consistency evaluation task."""
    if csv_path:
        # Process CSV input
        df = pd.read_csv(csv_path)
        df = df[df['consistency_type']!='condcond']
        # df = df.sample(10)
        samples = [create_csv_sample(row) for _, row in df.iterrows()]
        dataset = MemoryDataset(name="initial_consistency", samples=samples)
        return Task(
            dataset=dataset,
            solver=consistency_solver(use_pretransformed=True),
            scorer=consistency_scorer()
        )
        
    else:
        # Original HuggingFace dataset processing
        def record_to_sample(record: dict) -> Sample:
            return Sample(
                input=record['question'],
                target=str(record['resolution']),
                metadata={
                    'id': str(random.randint(10000, 99999)),
                    'question_type': record['question_type'],
                    'is_resolved': record.get('is_resolved', False),
                    'original_data': {
                        'title': record['question'],
                        'body': record['background'],
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
            lambda x: (
                x.metadata["is_resolved"] is True and 
                x.metadata["question_type"].lower() == "binary" and 
                "sqrt" not in x.input.lower()
            )
        )
        dataset = dataset[:3]

        return Task(
            dataset=dataset,
            solver=consistency_solver(),
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
                # 'data_source': record['data_source'],
                'is_resolved': record.get('is_resolved', False),
                # 'community_predictions': record.get('community_predictions', []),
                'original_data': {
                    'title': record['question'],
                    'body': record['background'],
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

# INITIAL CONSISTENCY

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
    
#     log_dir = "logs/initial_consistency"
#     os.makedirs(log_dir, exist_ok=True)
    
#     try:
#         # Add argument parser to handle CSV path
#         import argparse
#         parser = argparse.ArgumentParser()
#         parser.add_argument('--csv_path', type=str, help='Path to CSV file with pretransformed data')
#         args = parser.parse_args()
        
#         # Create task based on whether CSV path is provided
#         task = initial_consistency(args.csv_path)
        
#         initial_log = eval(
#             task, 
#             model="openai/gpt-4o",
#             temperature=0,
#             start_log=True
#         )[0]

#         print("\nEvaluation complete")
        
#     except Exception as e:
#         logging.error(f"Error in evaluation pipeline: {str(e)}")
#         raise

#ADAPTIVE CONSISTENCY

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    #initial_log_path = "logs/2024-12-26T11-19-29+05-30_initial-consistency_cExeriowwKWrcS4rBbUc9f.eval"
    #initial_log_path = "logs/2024-12-31T20-20-44+05-30_initial-consistency_CtRhvzAAkDnGJgXJ868DUB.eval" #Inital log for 100 samples, forecasting dataset
    
    
    # Construct the path relative to project root
    project_root = Path(__file__).parent.parent
    initial_log_path =str(project_root  / "logs" / '2025-01-05T18-27-01+05-30_initial-consistency_3f8K9pe7m7TXieyh4yodUe.eval') # Paleka full dataset

    
    #dataset_path = "prithvi3/filtered_forecast_sample_test"
    
    tasks = [
        adaptive_consistency(
            initial_log_path=initial_log_path,
            consistency_types=[ct.value],  # Pass single consistency type as list
            use_embeddings=False,
            dataset_path=None
        )
        for ct in ConsistencyType
    ]

    # tasks = [
    #     adaptive_consistency(
    #         initial_log_path=initial_log_path,
    #         consistency_types=['paraphrase', 'consequence'],  # Pass single consistency type as list
    #         use_embeddings=False,
    #         dataset_path=None
    #     )
    # ]


    # Evaluate all tasks
    results = eval(
        tasks,
        max_connections=1,
        model="openai/gpt-4o",
        temperature=0,
        start_log=True,
        max_tasks=len(ConsistencyType)  
    )[0]

    print("\nEvaluation complete")

#JUDGE

# if __name__ == "__main__":
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
    
#     # Paths to evaluation logs
#     initial_log_path = "logs/2024-12-31T18-27-08+05-30_initial-consistency_njAobmajK8z8AMLiJo7BJS.eval"
#     adaptive_log_path = "logs/2024-12-31T18-28-15+05-30_adaptive-consistency_aDE5pPq8oawANKvx28FtX6.eval"  # This would be your adaptive eval log
    
#     task = temp_adaptive_judge(
#         initial_log_path=initial_log_path,
#         adaptive_log_path=adaptive_log_path,
#     )
    
#     result = eval(
#         task,
#         model="openai/gpt-4o",
#         temperature=0,
#         start_log=True
#     )[0]
    
    #print("\nEvaluation complete")
    #print(f"Judge accuracy: {result.scores[0].value:.3f}")  # Access accuracy score