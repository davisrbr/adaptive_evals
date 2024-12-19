# tasks/run_adaptive_evals.py
from typing import List, Dict, Optional
from datetime import datetime
import os
import logging
from utils_consistency.consistency_question_generator import generate_consistency_checks
from utils_consistency.forecaster import main_parallel_forecast_processing
from utils_consistency.metrics import calculate_consistency_metrics
from utils_consistency.adaptive_consistency_generator import (
    main_adversarial_generation,
    GenerationConfig
)
import random
import pandas as pd

def prepare_questions(df: pd.DataFrame, sample_size: int = 10) -> List[Dict]:
    """
    Filter and sample questions from the dataset.
    
    Args:
        df: Raw DataFrame from HuggingFace dataset
        sample_size: Number of questions to sample
        
    Returns:
        List of formatted question dictionaries
    """
    # Filter for resolved binary questions
    filtered_df = df[
        (df['is_resolved'] == True) & 
        (df['question_type'].str.lower() == 'binary')
    ].copy()
    
    # Adjust sample size if it exceeds available data
    actual_sample_size = min(sample_size, len(filtered_df))
    logging.info(f"Sampling {actual_sample_size} questions from {len(filtered_df)} available questions")
    
    # Randomly sample questions
    sampled_df = filtered_df.sample(n=actual_sample_size, random_state=0)
    
    # Format questions in the required structure
    formatted_questions = []
    for _, row in sampled_df.iterrows():
        question = {
            'id': str(random.randint(10000, 99999)),
            'title': row['question'],
            'body': f"{row['background']}\n\nResolution Criteria:\n{row['resolution_criteria']}",
            'resolution_date': row['date_resolve_at'],
            'created_date': row['date_begin'],
            'question_type': 'binary',
            'data_source': row['data_source']
        }
        formatted_questions.append(question)
    
    return formatted_questions

def run_adaptive_evaluation(
    input_dataset: str,
    consistency_types: List[str],
    sample_size: int = 100,
    output_dir: str = "outputs",
    adversarial_config: Optional[GenerationConfig] = None,
    **kwargs  # Supports: max_workers, pre_request_sleep, batch_size for parallel processing
) -> Dict:
    """
    Run complete adaptive evaluation pipeline.
    
    Args:
        input_dataset: Input dataset (DataFrame or path)
        consistency_types: List of consistency types to generate
        sample_size: Number of questions to sample
        output_dir: Directory for outputs and checkpoints
        adversarial_config: Optional config for adversarial generation
        **kwargs: Additional arguments for parallel processing
    
    Returns:
        Dictionary containing all generated DataFrames and analysis
    """
    # Setup
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.info(f"Starting adaptive evaluation pipeline with {sample_size} samples")
    
    try:
        # Convert input if string
        df = pd.read_csv(input_dataset) if isinstance(input_dataset, str) else input_dataset
        formatted_questions = prepare_questions(df, sample_size=sample_size)
        logging.info(f"Prepared {len(formatted_questions)} formatted questions")

        # 1. Generate consistency checks
        consistency_df = generate_consistency_checks(
            formatted_questions=formatted_questions,  
            consistency_types=consistency_types,
            output_dir=output_dir,
            **kwargs
        )
        consistency_path = os.path.join(output_dir, f"consistency_checks_{timestamp}.csv")
        consistency_df.to_csv(consistency_path)
        logging.info(f"Generated {len(consistency_df)} consistency checks")
        
        # 2. Run forecasting
        forecast_path = os.path.join(output_dir, f"forecasts_{timestamp}.csv")
        forecast_df = main_parallel_forecast_processing(
            input_csv=consistency_path,
            output_csv=forecast_path,
            **kwargs
        )
        logging.info(f"Completed forecasting for {len(forecast_df)} questions")
        
        # 3. Calculate metrics
        metrics_df = calculate_consistency_metrics(forecast_df)
        metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.csv")
        metrics_df.to_csv(metrics_path)
        logging.info(f"Calculated consistency metrics for {len(metrics_df)} questions")
        
        # 4. Generate adversarial questions
        if adversarial_config is None:
            adversarial_config = GenerationConfig(
                n_questions=10,
                samples_per_bucket={
                    'worst': 3,
                    'poor': 3,
                    'medium': 2,
                    'best': 2
                },
                samples_per_consistency={
                    'not': 10,
                    'consequence': 10,
                    'paraphrase': 10,
                    'and': 10,
                    'or': 10,
                },
                model="gpt-4o",
                temperature=0.7,
                checkpoint_dir="adversarial_checkpoints"
            )
        
        analysis, generated_df = main_adversarial_generation(
            metrics_path,
            config=adversarial_config
        )
        
        # Save analysis
        analysis_path = os.path.join(output_dir, f"analysis_{timestamp}.txt")
        with open(analysis_path, 'w') as f:
            f.write(analysis)
        logging.info(f"Generated {len(generated_df)} adversarial questions")
        
        return {
            'consistency_df': consistency_df,
            'forecast_df': forecast_df,
            'metrics_df': metrics_df,
            'adversarial_df': generated_df,
            'analysis': analysis
        }
        
    except Exception as e:
        logging.error(f"Error in adaptive evaluation: {str(e)}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    input_dataset = pd.read_csv("hf://datasets/prithvi3/filtered_forecast_sample_test/test_data_12_05_24_6_filtered.csv")
    results = run_adaptive_evaluation(
        input_dataset=input_dataset,
        consistency_types=['not', 'paraphrase', 'consequence', 'and', 'or'],
        sample_size=15
    )