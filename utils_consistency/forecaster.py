#utils_consistency/forecaster.py
from typing import Optional, Dict, Tuple, Any, Callable
from datetime import datetime
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import re
import pandas as pd
import logging 
import time
from tqdm import tqdm
import os
import glob

BINARY_SCRATCH_PAD_PROMPT_NEW_6 = """Question:
{question}

Question Background:
{background}


Instructions:
1. Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.
{{ Insert rephrased and expanded question.}}

2. Develop a decision tree outlining possible paths to both 'Yes' and 'No' outcomes.
{{ Insert decision tree outline }}

3. Analyze the probability of each branch of the decision tree based on current information.
{{ Insert branch probability analysis }}

4. Discuss any potential game-changers or wildcard events. Use your knowledge of the topic as well as the information provided.
{{ Insert discussion on wildcards }}

5. Output an initial probability (prediction) given steps 1-4.
{{ Insert initial probability. }}

6. Feel free to adjust your probability now. Here is a non-exhaustive list of some things you'll want to check:
- Is your calculated probability is excessively confident or not confident enough.
- Is there anything else that might affect the forecast that you did not before consider.
- Use your intuition and feel for the question.
{{ Insert your thoughts }}

7. Output your final answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
{{ Insert your answer }} """

def extract_probability(text: str) -> Optional[float]:
    """
    Extract probability value surrounded by asterisks from text.
    
    Args:
        text (str): Text containing probability value surrounded by asterisks
        
    Returns:
        Optional[float]: Extracted probability value or None if not found
        
    Example:
        >>> extract_probability("Some text *0.75* more text")
        0.75
    """
    # Look for a decimal number surrounded by asterisks
    pattern = r'\*([0-9]*\.?[0-9]+)\*'
    match = re.search(pattern, text)
    
    if match:
        try:
            prob = float(match.group(1))
            # Validate probability is between 0 and 1
            if 0 <= prob <= 1:
                return prob
        except ValueError:
            return None
    return None

def scratchpad_forecaster(
    question_title: str,
    question_background: str,
    api_key: Optional[str] = None,
    model: str = "gpt-4o"
) -> Tuple[str, Optional[float]]:
    """
    Generate a forecasting analysis using the scratchpad template and extract the probability.
    
    Args:
        question_title (str): The title/question to analyze
        question_background (str): Background information for the question
        api_key (Optional[str]): OpenAI API key. If None, assumes it's set in environment
        model (str): GPT model to use. Defaults to "gpt-4o"
        
    Returns:
        Tuple[str, Optional[float]]: (Full response text, Extracted probability)
        
    Raises:
        Exception: If API call fails or response is invalid
    """
    # Format the template with the provided question and background
    formatted_prompt = BINARY_SCRATCH_PAD_PROMPT_NEW_6.format(
        question=question_title,
        background=question_background
    )
    
    try:
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant specialized in probabilistic forecasting. "
                              "Please follow the template structure exactly and ensure your final probability "
                              "is clearly marked with asterisks."
                },
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0  # Use deterministic output
        )
        
        # Extract response text
        if not response.choices or not response.choices[0].message.content:
            raise Exception("Invalid API response - no content received")
            
        full_response = response.choices[0].message.content
        
        # Extract probability
        probability = extract_probability(full_response)
        
        if probability is None:
            print("Warning: Could not extract valid probability from response")
            
        return full_response, probability
        
    except Exception as e:
        raise Exception(f"Error in scratchpad_forecaster: {str(e)}")


def prepare_unique_questions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare unique original questions and all consistency questions with relationship IDs.
    """
    # Add unique IDs to original questions
    original_cols = ['original_title', 'original_body', 'original_resolution_date', 'original_created_date']
    unique_originals = df[original_cols].drop_duplicates().reset_index(drop=True)
    unique_originals['question_id'] = range(len(unique_originals))
    
    # Map back to full dataset to get IDs for consistency questions
    df_with_ids = pd.merge(
        df, 
        unique_originals,
        on=original_cols
    )
    
    # Get consistency questions with their corresponding original question IDs
    consistency_questions = df_with_ids[['question_id', 'consistency_title', 'consistency_body', 
                                       'consistency_resolution_date', 'consistency_created_date', 
                                       'consistency_type']].reset_index(drop=True)
    
    return df_with_ids, unique_originals, consistency_questions


def process_row_combined(
    row: pd.Series,
    question_id: int,  # Add question_id parameter
    api_key: Optional[str],
    model: str,
    pre_request_sleep: float = 1.0,
    is_original: bool = True
) -> Tuple[int, int, Dict[str, Any]]:  # Return question_id as well
    """Process a single row while maintaining relationship."""
    try:
        time.sleep(pre_request_sleep)
        
        if is_original:
            response, prob = scratchpad_forecaster(
                question_title=row['original_title'],
                question_background=row['original_body'],
                api_key=api_key,
                model=model
            )
            results = {
                'original_forecast_response': response,
                'original_forecast_probability': prob
            }
        else:
            response, prob = scratchpad_forecaster(
                question_title=row['consistency_title'],
                question_background=row['consistency_body'],
                api_key=api_key,
                model=model
            )
            results = {
                'consistency_forecast_response': response,
                'consistency_forecast_probability': prob
            }
        
        logging.info(f"Successfully processed {'original' if is_original else 'consistency'} question {row.name}")
        return row.name, question_id, results
        
    except Exception as e:
        logging.error(f"Error processing row {row.name}: {str(e)}")
        return row.name, question_id, {}


def save_checkpoint(df: pd.DataFrame, checkpoint_name: str, checkpoint_dir: str = "checkpoints"):
    """Save checkpoint to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.csv")
    df.to_csv(checkpoint_path, index=False)
    logging.info(f"Saved checkpoint: {checkpoint_path}")

def load_latest_checkpoint(checkpoint_dir: str = "checkpoints", pattern: str = None) -> Tuple[Optional[pd.DataFrame], str]:
    """Load most recent checkpoint if it exists."""
    if not os.path.exists(checkpoint_dir):
        return None, ""
    
    checkpoints = glob.glob(os.path.join(checkpoint_dir, f"{pattern or '*'}.csv"))
    if not checkpoints:
        return None, ""
        
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    return pd.read_csv(latest_checkpoint), latest_checkpoint

def process_in_batches(
    questions_df: pd.DataFrame,
    process_func: Callable,
    batch_size: int = 10,
    checkpoint_prefix: str = "batch",
    checkpoint_dir: str = "checkpoints",
    **kwargs
) -> pd.DataFrame:
    """Process questions in batches with checkpointing."""
    
    # Check for existing checkpoint
    checkpoint_df, checkpoint_path = load_latest_checkpoint(
        checkpoint_dir=checkpoint_dir,
        pattern=f"{checkpoint_prefix}*"
    )
    
    if checkpoint_df is not None:
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        # Identify completed questions
        completed_ids = checkpoint_df['question_id'].unique()
        questions_df = questions_df[~questions_df['question_id'].isin(completed_ids)]
        result_df = checkpoint_df
    else:
        result_df = pd.DataFrame()
    
    # Process in batches
    for batch_start in tqdm(range(0, len(questions_df), batch_size),
                           desc=f"Processing {checkpoint_prefix} batches"):
        batch_end = min(batch_start + batch_size, len(questions_df))
        batch_df = questions_df.iloc[batch_start:batch_end]
        
        try:
            # Process batch
            processed_batch = process_func(batch_df, **kwargs)
            
            # Append results
            result_df = pd.concat([result_df, processed_batch], ignore_index=True)
            
            # Save checkpoint after each batch
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_name = f"{checkpoint_prefix}_batch_{batch_start}_{timestamp}"
            save_checkpoint(result_df, checkpoint_name, checkpoint_dir)
            
        except Exception as e:
            logging.error(f"Error processing batch {batch_start}-{batch_end}: {str(e)}")
            # Save what we have so far
            save_checkpoint(result_df, f"{checkpoint_prefix}_error_{timestamp}", checkpoint_dir)
            raise
    
    return result_df

def parallel_process_forecasts_combined(
    df: pd.DataFrame,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    max_workers: int = 10,
    pre_request_sleep: float = 1.0,
    batch_size: int = 10
) -> pd.DataFrame:
    """Process forecasts with batching and checkpointing."""
    
    # Prepare questions with IDs
    df_with_ids, unique_originals, consistency_questions = prepare_unique_questions(df)
    
    # Initialize results columns
    for df_ in [unique_originals, consistency_questions]:
        for col in ['forecast_response', 'forecast_probability']:
            prefix = 'original_' if df_ is unique_originals else 'consistency_'
            if f'{prefix}{col}' not in df_.columns:
                df_[f'{prefix}{col}'] = None
    
    # Process original questions in batches
    def process_original_batch(batch_df, **kwargs):
        results_df = batch_df.copy()
        with ThreadPoolExecutor(max_workers=kwargs.get('max_workers', 5)) as executor:
            futures = {
                executor.submit(
                    process_row_combined,
                    row,
                    row['question_id'],
                    kwargs.get('api_key'),
                    kwargs.get('model'),
                    kwargs.get('pre_request_sleep', 1.0),
                    True
                ): idx for idx, row in batch_df.iterrows()
            }
            
            for future in concurrent.futures.as_completed(futures):
                idx, question_id, results = future.result()
                for key, value in results.items():
                    results_df.at[idx, key] = value
                    
        return results_df
    
    # Process consistency questions in batches
    def process_consistency_batch(batch_df, **kwargs):
        results_df = batch_df.copy()
        with ThreadPoolExecutor(max_workers=kwargs.get('max_workers', 5)) as executor:
            futures = {
                executor.submit(
                    process_row_combined,
                    row,
                    row['question_id'],
                    kwargs.get('api_key'),
                    kwargs.get('model'),
                    kwargs.get('pre_request_sleep', 1.0),
                    False
                ): idx for idx, row in batch_df.iterrows()
            }
            
            for future in concurrent.futures.as_completed(futures):
                idx, question_id, results = future.result()
                for key, value in results.items():
                    results_df.at[idx, key] = value
                    
        return results_df
    
    # Process both types with batching
    processed_originals = process_in_batches(
        unique_originals,
        process_original_batch,
        batch_size=batch_size,
        checkpoint_prefix="originals",
        api_key=api_key,
        model=model,
        max_workers=max_workers,
        pre_request_sleep=pre_request_sleep
    )
    
    processed_consistency = process_in_batches(
        consistency_questions,
        process_consistency_batch,
        batch_size=batch_size,
        checkpoint_prefix="consistency",
        api_key=api_key,
        model=model,
        max_workers=max_workers,
        pre_request_sleep=pre_request_sleep
    )
    
    # Combine results using question_id
    result_df = pd.merge(
        df_with_ids,
        processed_originals[['question_id', 'original_forecast_response', 'original_forecast_probability']],
        on='question_id',
        how='left'
    )
    
    result_df = pd.merge(
        result_df,
        processed_consistency[['question_id', 'consistency_title', 'consistency_body',
                             'consistency_forecast_response', 'consistency_forecast_probability']],
        on=['question_id', 'consistency_title', 'consistency_body'],
        how='left'
    )
    
    return result_df


def main_parallel_forecast_processing_combined(
    input_csv: str,
    output_csv: str,
    max_workers: int = 10,
    **kwargs
):
    """
    Main function to process combined consistency forecasts.
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path to output CSV
        max_workers: Maximum number of parallel workers
        **kwargs: Additional arguments for parallel_process_forecasts_combined
    """
    try:
        # Load CSV
        df = pd.read_csv(input_csv)
        logging.info(f"Loaded {len(df)} rows from {input_csv}")
        
        # Process forecasts
        result_df = parallel_process_forecasts_combined(
            df,
            max_workers=max_workers,
            **kwargs
        )
        
        # Save results
        result_df.to_csv(output_csv, index=False)
        logging.info(f"Saved results to {output_csv}")
        
        return result_df
        
    except Exception as e:
        logging.error(f"Error in main processing: {str(e)}")
        raise

#Reading CSV with Combined NOT, Paraphrase, and Consequence questions to generate forecasts for each question / consistency type and combine them into one CSV
# if __name__ == "__main__":
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )

#     # Configuration
#     config = {
#         'model': 'gpt-4o',
#         'max_workers': 10,
#         'pre_request_sleep': 1.0, 
#         'batch_size': 10
#     }
    
#     # Paths
#     input_csv = 'combined_consistency_checks_20241217_203000.csv'
#     output_csv = f"forecast_results_combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
#     try:
#         # Run the processing using the main function
#         print("Starting forecast processing...")
#         result_df = main_parallel_forecast_processing_combined(
#             input_csv=input_csv,
#             output_csv=output_csv,
#             **config
#         )
        
#         # Print summary
#         print("\nProcessing complete!")
#         print(f"Total rows processed: {len(result_df)}")
#         print("\nDistribution of consistency types:")
#         print(result_df['consistency_type'].value_counts())
        
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")

