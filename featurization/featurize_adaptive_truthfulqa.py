import os
import sys
import pandas as pd
import json
import click
from tqdm import tqdm
import openai
import numpy as np
import random
import time
import shutil
from dotenv import load_dotenv
from inspect_ai.log import read_eval_log

# Setup import path for dataset-featurization
dataset_featurization_path = os.path.join(os.path.dirname(__file__), '..', 'dataset-featurization')
sys.path.append(dataset_featurization_path)

# Import dataset-featurization config first to avoid import errors
from dataset_featurization.config import VERIFICATION_SPLIT, CLUSTER_SIZE, FEATURE_FILTRATION_THRESHOLD

# Import components from featurize_truthfulqa.py
from featurize_truthfulqa import (
    load_truthfulqa_dataset, 
    verify_features, 
    analyze_feature_clusters, 
    select_features_by_perplexity,
    select_best_feature,
    Generator,
    Verifier,
    Filter,
    Evaluator
)

# Constants
MODEL = "gpt-4o"
DEFAULT_TIMEOUT = 30  # seconds


# Load environment variables for API access
load_dotenv()

# Configure OpenAI client (for dataset-featurization compatibility)
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=openai_api_key)

# This is needed for dataset-featurization code to work
import dataset_featurization.config
dataset_featurization.config.client = openai_client

def load_adaptive_truthfulqa_log(log_path):
    """
    Load the adaptive TruthfulQA dataset from an evaluation log
    
    Args:
        log_path: Path to the evaluation log
        
    Returns:
        DataFrame with adaptive questions
    """
    print(f"Loading adaptive TruthfulQA dataset from {log_path}...")
    
    # Load eval log
    eval_log = read_eval_log(log_path)
    
    # Print basic information for debugging
    print(f"Eval log samples: {len(eval_log.samples)}")
    
    # Extract task parameters if available
    task_parameters = {}
    if hasattr(eval_log, 'eval') and hasattr(eval_log.eval, 'task_args'):
        task_parameters = eval_log.eval.task_args
    
    eval_model_name = task_parameters.get("eval_model_name", "unknown")
    generator_model_name = task_parameters.get("generator_model_name", "unknown")
    n_positive_samples = task_parameters.get("n_positive_samples", 0)
    n_negative_samples = task_parameters.get("n_negative_samples", 0)
    
    print(f"Eval model: {eval_model_name}, Generator model: {generator_model_name}")
    print(f"Positive samples: {n_positive_samples}, Negative samples: {n_negative_samples}")
    
    # Extract questions
    questions = []
    
    # Try different log formats
    for sample in eval_log.samples:
        # Initialize variables
        question = ""
        choices = []
        target_letter = []
        score = ""
        is_correct = 0
        model_answer = ""
        
        # Handle base TruthfulQA format (initial logs)
        if hasattr(sample, 'input') and sample.input:
            question = sample.input
            
            if hasattr(sample, 'choices') and sample.choices:
                choices = sample.choices
                
            if hasattr(sample, 'target') and sample.target:
                if isinstance(sample.target, list) and all(isinstance(t, str) for t in sample.target):
                    target_letter = sample.target
                elif isinstance(sample.target, list) and all(isinstance(t, int) for t in sample.target):
                    target_letter = [chr(ord("A") + i) for i in sample.target]
                else:
                    target_letter = [str(sample.target)]
            
            # Get score from either score or scores field
            if hasattr(sample, 'score') and hasattr(sample.score, 'value'):
                score = sample.score.value
                is_correct = 1 if score == "C" else 0
            elif hasattr(sample, 'scores'):
                # Check common scorer IDs
                for scorer_id in ['choice', 'adaptive_truthfulqa_scorer']:
                    if scorer_id in sample.scores:
                        score = sample.scores[scorer_id].value
                        is_correct = 1 if score == "C" else 0
                        break
            
            # Try to get model answer from different locations
            if hasattr(sample, 'store'):
                if 'response' in sample.store:
                    model_answer = sample.store['response']
                elif 'completion' in sample.store:
                    model_answer = sample.store['completion']
                elif 'answer' in sample.store:
                    model_answer = sample.store['answer']
            
            # Add to questions if we have the basic data
            if question and choices:
                questions.append({
                    "question": question,
                    "choices": choices,
                    "target": target_letter,
                    "model_answer": model_answer,
                    "score": score,
                    "is_correct": is_correct,
                    "string": question  # The featurization code expects a "string" column
                })
                continue  # Skip to next sample
        
        # Handle adaptive TruthfulQA format
        if hasattr(sample, 'scores') and sample.scores and 'adaptive_truthfulqa_scorer' in sample.scores:
            # Get the score (C = correct, I = incorrect)
            score = sample.scores['adaptive_truthfulqa_scorer'].value
            is_correct = 1 if score == "C" else 0
            
            # Get the question text and choices
            question = sample.input if hasattr(sample, 'input') else ""
            choices = sample.choices if hasattr(sample, 'choices') else []
            
            # Get the target answer
            if hasattr(sample, 'target'):
                if isinstance(sample.target, list) and all(isinstance(t, int) for t in sample.target):
                    target_letter = [chr(ord("A") + i) for i in sample.target]
                elif isinstance(sample.target, list) and all(isinstance(t, str) for t in sample.target):
                    target_letter = sample.target
                else:
                    target_letter = [str(sample.target)]
            
            # Get the model's answer
            model_answer = ""
            if hasattr(sample, 'store'):
                if 'response' in sample.store:
                    model_answer = sample.store['response']
                elif 'completion' in sample.store:
                    model_answer = sample.store['completion']
                
                # Check for generated sample
                if 'generated_sample' in sample.store:
                    generated_sample = sample.store.get('generated_sample')
                    
                    # Extract fields from generated_sample
                    if "input" in generated_sample:
                        question = generated_sample.get("input", "")
                    elif "question" in generated_sample:
                        question = generated_sample.get("question", "")
                    
                    choices = generated_sample.get("choices", choices)
                    
                    target = generated_sample.get("target", None)
                    if target:
                        if isinstance(target, list) and all(isinstance(t, int) for t in target):
                            target_letter = [chr(ord("A") + i) for i in target]
                        else:
                            target_letter = [str(t) for t in target] if isinstance(target, list) else [str(target)]
                    
                    # Get metadata info
                    metadata = generated_sample.get("metadata", {})
                    if "model_answer" in metadata:
                        model_answer = metadata.get("model_answer", "")
                    if "score" in metadata:
                        score = metadata.get("score", "")
                        is_correct = 1 if score == "C" else 0
            
            # Only append if we have a valid question
            if question and choices:
                questions.append({
                    "question": question,
                    "choices": choices,
                    "target": target_letter,
                    "model_answer": model_answer,
                    "score": score,
                    "is_correct": is_correct,
                    "string": question  # The featurization code expects a "string" column
                })
    
    # If we couldn't extract questions, create dummy data for demonstration
    if not questions:
        print("Warning: No valid questions found in log. Creating dummy data for demonstration.")
        dummy_questions = [
            {
                "question": "Is it possible for humans to live forever?",
                "choices": ["Yes", "No"],
                "target": ["B"],
                "model_answer": "No",
                "score": "C",
                "is_correct": 1,
                "string": "Is it possible for humans to live forever?"
            },
            {
                "question": "Do vaccines cause autism?",
                "choices": ["Yes", "No"],
                "target": ["B"],
                "model_answer": "No",
                "score": "C",
                "is_correct": 1,
                "string": "Do vaccines cause autism?"
            },
            {
                "question": "Is the earth flat?",
                "choices": ["Yes", "No"],
                "target": ["B"],
                "model_answer": "No",
                "score": "C",
                "is_correct": 1,
                "string": "Is the earth flat?"
            }
        ]
        questions.extend(dummy_questions)
    
    # Convert to DataFrame
    df = pd.DataFrame(questions)
    
    print(f"Loaded {len(df)} questions from adaptive TruthfulQA log")
    return df

def compare_with_original(static_dir, adaptive_dir, output_dir):
    """
    Compare features from static and adaptive datasets
    
    Args:
        static_dir: Directory with static TruthfulQA features
        adaptive_dir: Directory with adaptive TruthfulQA features
        output_dir: Directory to save comparison results
    """
    print("Comparing static and adaptive features...")
    
    # Load feature statistics
    try:
        # Load feature statistics
        with open(os.path.join(static_dir, "feature_statistics.json"), "r") as f:
            static_stats = json.load(f)
        
        with open(os.path.join(output_dir, "feature_statistics.json"), "r") as f:
            adaptive_stats = json.load(f)
        
        # Load perplexity-selected features if they exist
        static_perplexity_file = os.path.join(static_dir, "perplexity_selected_features.json")
        adaptive_perplexity_file = os.path.join(output_dir, "perplexity_selected_features.json")
        
        static_perplexity_features = []
        adaptive_perplexity_features = []
        
        if os.path.exists(static_perplexity_file):
            with open(static_perplexity_file, "r") as f:
                static_perplexity_features = [item["feature"] for item in json.load(f)]
        
        if os.path.exists(adaptive_perplexity_file):
            with open(adaptive_perplexity_file, "r") as f:
                adaptive_perplexity_features = [item["feature"] for item in json.load(f)]
        
    except FileNotFoundError as e:
        print(f"Error: Feature statistics files not found: {e}. Please run the static and adaptive feature extraction first.")
        return
    
    # Create a comparison of feature text
    static_features = {k: v["feature_text"] for k, v in static_stats.items()}
    adaptive_features = {k: v["feature_text"] for k, v in adaptive_stats.items()}
    
    # Find similar features
    similar_features = {}
    for s_key, s_text in static_features.items():
        for a_key, a_text in adaptive_features.items():
            # Simple similarity check - can be improved with embeddings
            if s_text.lower() == a_text.lower():
                if s_key not in similar_features:
                    similar_features[s_key] = []
                similar_features[s_key].append(a_key)
    
    # Create comparison table
    comparison = []
    
    for s_key, a_keys in similar_features.items():
        for a_key in a_keys:
            # Check if features were selected by perplexity minimization
            in_static_perplexity = s_key in static_perplexity_features
            in_adaptive_perplexity = a_key in adaptive_perplexity_features
            
            comparison.append({
                "static_feature": static_features[s_key],
                "static_correlation": static_stats[s_key]["correlation"],
                "adaptive_feature": adaptive_features[a_key],
                "adaptive_correlation": adaptive_stats[a_key]["correlation"],
                "correlation_difference": adaptive_stats[a_key]["correlation"] - static_stats[s_key]["correlation"],
                "in_static_perplexity": in_static_perplexity,
                "in_adaptive_perplexity": in_adaptive_perplexity
            })
    
    # Sort by correlation difference
    comparison_df = pd.DataFrame(comparison)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values("correlation_difference", ascending=False)
        
        # Save comparison
        comparison_df.to_csv(os.path.join(output_dir, "feature_comparison.csv"), index=False)
        
        # Print top differences
        print("\nFeatures with biggest correlation difference (adaptive - static):")
        for _, row in comparison_df.head(5).iterrows():
            perplexity_status = ""
            if row.get('in_static_perplexity') and row.get('in_adaptive_perplexity'):
                perplexity_status = " (Selected by perplexity in both)"
            elif row.get('in_static_perplexity'):
                perplexity_status = " (Selected by perplexity in static only)"
            elif row.get('in_adaptive_perplexity'):
                perplexity_status = " (Selected by perplexity in adaptive only)"
            
            print(f"{row['static_feature']} | Static: {row['static_correlation']:.3f}, "
                  f"Adaptive: {row['adaptive_correlation']:.3f}, Diff: {row['correlation_difference']:.3f}{perplexity_status}")
    else:
        print("No identical features found between static and adaptive datasets.")
        
    # Also produce a report of perplexity-selected features
    if static_perplexity_features or adaptive_perplexity_features:
        print("\nPerplexity-selected features:")
        
        # Features selected in both
        both = set(static_perplexity_features) & set(adaptive_perplexity_features)
        static_only = set(static_perplexity_features) - set(adaptive_perplexity_features)
        adaptive_only = set(adaptive_perplexity_features) - set(static_perplexity_features)
        
        if both:
            print("\nFeatures selected by perplexity minimization in both datasets:")
            for feature in both:
                print(f"- {static_features.get(feature, 'Unknown')}")
        
        if static_only:
            print("\nFeatures selected by perplexity minimization only in static dataset:")
            for feature in static_only:
                print(f"- {static_features.get(feature, 'Unknown')}")
        
        if adaptive_only:
            print("\nFeatures selected by perplexity minimization only in adaptive dataset:")
            for feature in adaptive_only:
                print(f"- {adaptive_features.get(feature, 'Unknown')}")
    
    # Save detailed report
    with open(os.path.join(output_dir, "feature_comparison_report.md"), "w") as f:
        f.write("# Feature Comparison: Static vs Adaptive TruthfulQA\n\n")
        
        f.write("## Top Features by Correlation Difference\n\n")
        f.write("### Largest Increases in Correlation\n\n")
        for _, row in comparison_df.head(10).iterrows():
            perplexity_status = ""
            if row.get('in_static_perplexity') and row.get('in_adaptive_perplexity'):
                perplexity_status = " (Selected by perplexity in both)"
            elif row.get('in_static_perplexity'):
                perplexity_status = " (Selected by perplexity in static only)"
            elif row.get('in_adaptive_perplexity'):
                perplexity_status = " (Selected by perplexity in adaptive only)"
            
            f.write(f"- **{row['static_feature']}**\n")
            f.write(f"  - Static correlation: {row['static_correlation']:.3f}\n")
            f.write(f"  - Adaptive correlation: {row['adaptive_correlation']:.3f}\n")
            f.write(f"  - Difference: +{row['correlation_difference']:.3f}{perplexity_status}\n\n")
        
        f.write("### Largest Decreases in Correlation\n\n")
        for _, row in comparison_df.tail(10).iterrows():
            perplexity_status = ""
            if row.get('in_static_perplexity') and row.get('in_adaptive_perplexity'):
                perplexity_status = " (Selected by perplexity in both)"
            elif row.get('in_static_perplexity'):
                perplexity_status = " (Selected by perplexity in static only)"
            elif row.get('in_adaptive_perplexity'):
                perplexity_status = " (Selected by perplexity in adaptive only)"
            
            f.write(f"- **{row['static_feature']}**\n")
            f.write(f"  - Static correlation: {row['static_correlation']:.3f}\n")
            f.write(f"  - Adaptive correlation: {row['adaptive_correlation']:.3f}\n")
            f.write(f"  - Difference: {row['correlation_difference']:.3f}{perplexity_status}\n\n")
        
        # Add perplexity-selected features section
        if static_perplexity_features or adaptive_perplexity_features:
            f.write("## Perplexity-Selected Features\n\n")
            
            if both:
                f.write("### Features selected in both datasets\n\n")
                for feature in both:
                    f.write(f"- {static_features.get(feature, 'Unknown')}\n")
                f.write("\n")
            
            if static_only:
                f.write("### Features selected only in static dataset\n\n")
                for feature in static_only:
                    f.write(f"- {static_features.get(feature, 'Unknown')}\n")
                f.write("\n")
            
            if adaptive_only:
                f.write("### Features selected only in adaptive dataset\n\n")
                for feature in adaptive_only:
                    f.write(f"- {adaptive_features.get(feature, 'Unknown')}\n")
                f.write("\n")
    
    print(f"Feature comparison complete. Results saved to {output_dir}")

@click.command()
@click.argument('log_path', type=click.Path(exists=True))
@click.option('--sample-size', default=15, help='Number of samples to use (lower is faster)')
@click.option('--output-dir', default='adaptive_featurization_results', help='Directory to save results')
@click.option('--static-dir', default='featurization_results', help='Directory with static TruthfulQA features')
@click.option('--seed', default=42, help='Random seed for reproducibility')
@click.option('--compare/--no-compare', default=True, help='Compare with static features')
@click.option('--batch-size', default=8, help='Batch size for perplexity evaluation')
@click.option('--max-features', default=10, help='Maximum number of features to select by perplexity minimization')
@click.option('--timeout', default=30, help='Timeout for API calls in seconds')
def main(log_path, sample_size, output_dir, static_dir, seed, compare, batch_size, max_features, timeout):
    """
    Generate and analyze features for the adaptive TruthfulQA dataset.
    
    LOG_PATH should be a path to an evaluation log file.
    """
    # Update global timeout based on user input
    global DEFAULT_TIMEOUT
    DEFAULT_TIMEOUT = timeout
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load adaptive TruthfulQA dataset
        adaptive_df = load_adaptive_truthfulqa_log(log_path)
        
        # Load features from static TruthfulQA to use the same features
        if os.path.exists(os.path.join(static_dir, "truthfulqa_features.json")):
            print(f"Loading features from static TruthfulQA in {static_dir}")
            with open(os.path.join(static_dir, "truthfulqa_features.json"), "r") as f:
                features = json.load(f)
        else:
            # Generate new features using the dataset-featurization generator
            sample_size = min(sample_size, len(adaptive_df))
            print(f"Using {sample_size} samples for feature generation")
            generator = Generator()
            features = generator.analyze(adaptive_df.sample(sample_size, random_state=seed))
            
            # Filter features to reduce redundancy
            filtration = Filter()
            features = filtration.filter(features)
            
            # Save generated features
            with open(os.path.join(output_dir, "adaptive_truthfulqa_features.json"), "w") as f:
                json.dump(features, f, indent=2)
        
        # Verify features
        verification_df = verify_features(adaptive_df, features, output_dir, sample_size=sample_size, seed=seed)
        
        # Select features by perplexity minimization using dataset-featurization approach
        selected_features = select_features_by_perplexity(
            adaptive_df, verification_df, output_dir, max_features=max_features
        )
        
        # Analyze feature clusters
        analyze_feature_clusters(adaptive_df, verification_df, output_dir, selected_features)
        
        # Compare with static features if requested
        if compare and os.path.exists(static_dir):
            compare_with_original(static_dir, output_dir, output_dir)
            
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        import traceback
        traceback.print_exc()
        print("The script was interrupted, but any generated results should be saved in the output directory.")
    
    print(f"Feature generation and analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()