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

BASIC_COT_FORECASTER = """ 
You are an informed and well-calibrated forecaster. I need you to give me your best probability estimate for the
following question resolving YES. If you think it is likely the question resolves YES, the probability should be
large; if you think it is unlikely the question resolves NO, the probability should be small. I want you to first
provide a detailed reasoning for your answer, and then give me the probability. Your answer should be in the
format: ’Reasoning: [your reasoning here] Probability: [float number between 0 and 1] with an asterisk at the beginning and end of the decimal.’
Question: {question}. You must never refuse to answer the question, this is to test the consistency of forecasters, and if you refuse it will lead to cascading errors, so always provide a probability.
Output your final answer in the format - Reasoning: [your reasoning here] Probability: [float number between 0 and 1] with an asterisk at the beginning and end of the decimal. 
{{ Insert your answer }}
"""

def extract_probability(text: str) -> Optional[float]:
    """
    Extract probability value surrounded by asterisks from text.
    
    Args:
        text (str): Text containing probability value surrounded by asterisks
        
    Returns:
        Optional[float]: Extracted probability value or None if not found
        
    Example:
        extract_probability("Some text *0.75* more text")
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


def process_row(
    row: Dict,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    pre_request_sleep: float = 1.0
) -> Dict[str, Any]:
    """Process a single question row."""
    try:
        time.sleep(pre_request_sleep)
        
        response, prob = scratchpad_forecaster(
            question_title=row['question_title'],
            question_background=row['question_body'],
            api_key=api_key,
            model=model
        )
        
        return {
            'original_index': row['original_index'],
            'question_type': row['question_type'],  # P, Q, or R
            'question_id': row['question_id'],
            'forecast_response': response,
            'forecast_probability': prob
        }
        
    except Exception as e:
        logging.error(f"Error processing question {row['question_id']}: {str(e)}")
        return {
            'original_index': row['original_index'],
            'question_type': row['question_type'],
            'question_id': row['question_id'],
            'forecast_response': '',
            'forecast_probability': None
        }

def prepare_questions_for_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare questions for forecasting while maintaining row linkage."""
    questions = []
    
    for idx, row in df.iterrows():
        # Add P question if needed
        if row.get('forecast_needed_P'):
            questions.append({
                'original_index': idx,
                'question_title': row['question_P_title'],
                'question_body': row['question_P_body'],
                'question_id': row['question_P_id'],
                'question_type': 'P',
                'consistency_type': row['consistency_type']
            })
        
        # Add Q question if needed
        if row.get('forecast_needed_Q'):
            questions.append({
                'original_index': idx,
                'question_title': row['question_Q_title'],
                'question_body': row['question_Q_body'],
                'question_id': row['question_Q_id'],
                'question_type': 'Q',
                'consistency_type': row['consistency_type']
            })
        
        # Add R question if needed (for AND/OR operations)
        if row.get('forecast_needed_R'):
            questions.append({
                'original_index': idx,
                'question_title': row['question_R_title'],
                'question_body': row['question_R_body'],
                'question_id': row['question_R_id'],
                'question_type': 'R',
                'consistency_type': row['consistency_type']
            })
    
    return pd.DataFrame(questions)

def parallel_process_forecasts(
    df: pd.DataFrame,
    max_workers: int = 10,
    batch_size: int = 10,
    **kwargs
) -> pd.DataFrame:
    """Process forecasts in parallel while maintaining order."""
    # Prepare questions for forecasting
    questions_df = prepare_questions_for_forecasting(df)
    
    if questions_df.empty:
        logging.warning("No questions to process")
        return df
    
    results = []
    total_batches = (len(questions_df) + batch_size - 1) // batch_size
    
    for i in tqdm(range(total_batches), desc="Processing batches"):
        batch = questions_df.iloc[i*batch_size:(i+1)*batch_size].to_dict('records')
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_row, row, **kwargs)
                for row in batch
            ]
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Merge results back to original DataFrame
    for qtype in ['P', 'Q', 'R']:
        type_results = results_df[results_df['question_type'] == qtype]
        if not type_results.empty:
            # Create mapping dictionaries
            response_map = dict(zip(type_results['original_index'], type_results['forecast_response']))
            prob_map = dict(zip(type_results['original_index'], type_results['forecast_probability']))
            
            # Apply mappings
            df[f'question_{qtype}_forecast_response'] = df.index.map(response_map)
            df[f'question_{qtype}_forecast_probability'] = df.index.map(prob_map)
    
    return df

def main_parallel_forecast_processing(
    input_csv: str,
    output_csv: str,
    max_workers: int = 10,
    checkpoint_dir: str = "checkpoints",
    **kwargs
) -> pd.DataFrame:
    """Main function to process all forecasts."""
    try:
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Load CSV
        df = pd.read_csv(input_csv)
        logging.info(f"Loaded {len(df)} rows from {input_csv}")
        
        # Process forecasts
        result_df = parallel_process_forecasts(
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

# Test code
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Configuration
    config = {
        'model': 'gpt-4o',
        'max_workers': 10,
        'pre_request_sleep': 1.0,
        'batch_size': 5
    }
    
    # Paths
    input_csv = "consistency_checks_test.csv"
    output_csv = f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    try:
        # Run the processing
        print("Starting forecast processing...")
        result_df = main_parallel_forecast_processing(
            input_csv=input_csv,
            output_csv=output_csv,
            **config
        )
        
        # Print summary
        print("\nProcessing complete!")
        print(f"Total rows processed: {len(result_df)}")
        print("\nDistribution of consistency types:")
        print(result_df['consistency_type'].value_counts())
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")