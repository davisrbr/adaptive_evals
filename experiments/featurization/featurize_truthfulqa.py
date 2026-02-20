import os
import sys
import pandas as pd
import json
from datasets import load_dataset
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import numpy as np
import random
import click
import statistics
import time

# Setup import path for dataset-featurization
dataset_featurization_path = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset-featurization')
sys.path.append(dataset_featurization_path)

dataset_featurization_path_relative = os.path.join(
    os.path.dirname(__file__), '..', '..', 'dataset-featurization', 'dataset_featurization'
)
sys.path.append(dataset_featurization_path_relative)

# # Now you can import from dataset_featurization
# from dataset_featurization.config import VERIFICATION_SPLIT, CLUSTER_SIZE, FEATURE_FILTRATION_THRESHOLD, MODEL, NUM_EXAMPLE_SAMPLED, GENERATOR_SYSTEM_PROMPT, GENERATOR_USER_PROMPT

# Import the necessary utilities from dataset-featurization
from dataset_featurization.utils.generator import Generator
from dataset_featurization.utils.verifier import Verifier
from dataset_featurization.utils.filtration import Filter
from dataset_featurization.utils.perplexity_evaluator import Evaluator

# Set up constants
MODEL = "gpt-4o"

# Define prompt template for TruthfulQA and configure in dataset-featurization
def prompt_template(row, features):
    system_message = "You are a helpful, truthful AI assistant."
    user_message = f"Question: {row['string']}"
    
    if features and features.strip():
        user_message += f"\n\nThe question has these features:\n{features}"
    
    assistant_message = row.get('model_answer', '')
    
    return system_message, user_message, assistant_message

# Configure dataset-featurization's prompt template
import dataset_featurization.config
dataset_featurization.config.prompt_template = prompt_template

# Load environment variables for API access
load_dotenv()

# Configure OpenAI client (for dataset-featurization compatibility)
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_client = openai.OpenAI(api_key=openai_api_key)

# This is needed for dataset-featurization code to work
import dataset_featurization.config
dataset_featurization.config.client = openai_client

def load_truthfulqa_dataset():
    """
    Load the TruthfulQA dataset from Hugging Face using the same method as in adaptive_evals
    """
    print("Loading TruthfulQA dataset...")
    
    def record_to_sample(record):
        return {
            "question": record["question"],
            "choices": record["mc1_targets"]["choices"],
            "labels": record["mc1_targets"]["labels"],
            "target": [chr(ord("A") + i) for i, label in enumerate(record["mc1_targets"]["labels"]) if label == 1],
            "embedding": record.get("embedding", None)
        }

    dataset = load_dataset("davisrbr/truthfulqa-embeddings", name="default", split="validation")
    processed_data = []
    
    for record in dataset:
        processed_data.append(record_to_sample(record))
    
    # Convert to pandas DataFrame format needed for featurization
    df = pd.DataFrame(processed_data)
    df["string"] = df["question"]  # The featurization code expects a "string" column
    
    print(f"Loaded {len(df)} questions from TruthfulQA dataset")
    return df

def generate_features(df, sample_size=50, seed=42):
    """
    Generate features for the TruthfulQA dataset using the dataset-featurization toolkit
    
    Args:
        df: DataFrame containing the TruthfulQA dataset
        sample_size: Number of samples to use (for faster processing)
        seed: Random seed for reproducibility
    
    Returns:
        List of generated features
    """
    print("Generating features...")
    
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Sample the dataset for faster processing
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=seed).reset_index(drop=True)
    else:
        df_sample = df.reset_index(drop=True)
    
    # Initialize the feature generator from dataset-featurization
    generator = Generator()
    
    # Generate raw features
    all_features = generator.analyze(df_sample)
    
    # Filter features to reduce redundancy
    filtration = Filter()
    filtered_features = filtration.filter(all_features)
    
    # Remove duplicates
    unique_features = list(set(filtered_features))
    print(f"Generated {len(unique_features)} unique features after filtering")
    
    return unique_features
    

def verify_features(df, features, output_dir, sample_size=50, seed=42):
    """
    Verify which features apply to each sample in the dataset
    
    Args:
        df: DataFrame containing the TruthfulQA dataset
        features: List of features to verify
        output_dir: Directory to save the results
        sample_size: Number of samples to use (for faster processing)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with boolean columns for each feature
    """
    print("Verifying features...")
    
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Sample the dataset for faster processing
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=seed).reset_index(drop=True)
    else:
        df_sample = df.reset_index(drop=True)
        
    # Initialize the feature verifier
    verifier = Verifier()
    
    # Verify features using the original implementation
    verification_df = verifier.process(df_sample["string"].to_list(), features)
    
    # Save feature mappings and add prefixes
    feature_df = pd.DataFrame()
    feature_df["string"] = verification_df["string"]
    
    # Create column names with "feature_" prefix and save the mappings
    for i, feature in enumerate(features):
        feature_name = f"feature_{i}"
        feature_df[feature_name] = verification_df[feature]
        
        # Save the mapping from feature_name to feature text
        with open(os.path.join(output_dir, f"{feature_name}.txt"), "w") as f:
            f.write(feature)
    
    # Save the full verification results
    feature_df.to_csv(os.path.join(output_dir, "truthfulqa_features_verification.csv"), index=False)
    
    return feature_df

def select_features_by_perplexity(df, features_df, output_dir, max_features=10):
    """
    Select features based on perplexity minimization using dataset-featurization's exact implementation
    
    Args:
        df: Original dataset DataFrame
        features_df: DataFrame with feature verification results 
        output_dir: Directory to save analysis results
        max_features: Maximum number of features to select
    
    Returns:
        List of selected features
    """
    print("Selecting features using dataset-featurization's exact perplexity minimization...")
    
    # Prepare data for evaluation
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare DataFrame for the evaluator - adding _property suffix to features
    eval_df = df.iloc[features_df.index].copy()
    
    # Get feature column names (excluding 'string')
    feature_cols = [col for col in features_df.columns if col.startswith("feature_")]
    
    # Dictionary to store feature text
    feature_texts = {}
    for feature in feature_cols:
        with open(os.path.join(output_dir, f"{feature}.txt"), "r") as f:
            feature_texts[feature] = f.read().strip()
    
    # Add feature properties to the evaluation dataframe
    for feature in feature_cols:
        eval_df[f"{feature}_property"] = features_df[feature]
    
    evaluator = Evaluator(batch_size=8)
    eval_df.reset_index(drop=True, inplace=True)
    evaluator.init_cached_prompts(eval_df)
    
    # Prepare verified_df in format expected by dataset-featurization
    verified_df = features_df.copy()
    verified_df["string"] = eval_df["string"]
    
    # Initialize tracking variables
    best_features = []
    losses = []
    
    # Create output files matching dataset-featurization approach
    best_features_file = os.path.join(output_dir, f"best_features.txt")
    with open(best_features_file, "w") as bf_file:
        bf_file.write("")
    
    with open(os.path.join(output_dir, f"perplexities.txt"), "w") as losses_file:
        losses_file.write("")
    
    # Run feature selection for specified number of iterations
    
    # FOR TESTING: Mock feature selection is commented out but available if needed
    """
    print("Using mock feature selection for testing purposes...")
    feature_cols = [col for col in features_df.columns if col.startswith("feature_")]
    feature_texts = {}
    for feature in feature_cols:
        with open(os.path.join(output_dir, f"{feature}.txt"), "r") as f:
            feature_texts[feature] = f.read().strip()
            
    # Take the first max_features features as our "best" features
    selected_features = feature_cols[:min(max_features, len(feature_cols))]
    
    with open(os.path.join(output_dir, "perplexity_selected_features.json"), "w") as f:
        json.dump([
            {
                "feature": feature,
                "feature_text": feature_texts[feature],
                "perplexity": 10.0 - i,  # Mock perplexity that decreases
                "perplexity_reduction": i * 0.5  # Mock reduction that increases
            }
            for i, feature in enumerate(selected_features)
        ], f, indent=2)
    
    # Also save as a simple text file
    with open(os.path.join(output_dir, "perplexity_selected_features.txt"), "w") as f:
        for i, feature in enumerate(selected_features):
            f.write(f"{feature}: {feature_texts[feature]} (perplexity: {10.0 - i:.3f})\n")
    
    return selected_features
    """

    for iteration in tqdm(range(max_features), desc=f"Selecting features"):
        # Evaluating feature perplexities
        if best_features:
            best_feature = best_features[-1]
            sub_df = eval_df[eval_df[f"{best_feature}_property"] == True]
            
            if len(sub_df) > 0:
                temp_evaluated_df = evaluator.evaluate(
                    verified_df[verified_df["string"].isin(sub_df["string"])].reset_index(drop=True),
                    sub_df.reset_index(drop=True),
                    list(sub_df.index),
                    feature_names=best_features
                )
                temp_evaluated_df.index = verified_df[verified_df["string"].isin(sub_df["string"])].index
                evaluated_df.loc[verified_df[verified_df["string"].isin(sub_df["string"])].index] = temp_evaluated_df
            else:
                # If no samples match this feature, re-evaluate all
                evaluated_df = evaluator.evaluate(verified_df, eval_df, list(eval_df.index), feature_names=best_features)
        else:
            evaluated_df = evaluator.evaluate(verified_df, eval_df, list(eval_df.index), feature_names=best_features)
        
        # Calculate mean perplexity (loss)
        loss = statistics.mean(evaluated_df["empty"].to_list())
        losses.append(evaluated_df["empty"].to_list())
        
        # Select best feature using dataset-featurization approach
        best_feature = select_best_feature(evaluated_df, best_features)
        
        # Exit if no improvement is possible
        if best_feature == "empty":
            print("Terminating feature selection because no additional features that lower perplexity can be found.")
            break
        
        # Mark the selected feature in the dataframe
        eval_df[f"{best_feature}_property"] = verified_df[best_feature]
        best_features.append(best_feature)
        
        # Log the results
        print(f"Selected feature: {feature_texts.get(best_feature, best_feature)}")
        remaining_features = [col for col in evaluated_df.columns[:-1] if col not in best_features]
        remaining_scores = evaluated_df[remaining_features].sum().sort_values()
        print(f"Top 5 remaining features: {[feature_texts.get(f, f) for f in remaining_scores.index[:5].tolist()]}")
        
        # Update tracking files
        with open(best_features_file, "w") as bf_file:
            bf_file.write("\n".join(best_features) + "\n")
        
        with open(os.path.join(output_dir, f"perplexities.txt"), "a") as losses_file:
            losses_file.write(f"{loss}\n")
    
    # Convert results to the expected format for this codebase
    selected_features = []
    for i, feature in enumerate(best_features):
        perplexity = float(statistics.mean(losses[i])) if i < len(losses) else 0
        prev_perplexity = float(statistics.mean(losses[i-1])) if i > 0 and i-1 < len(losses) else 0
        
        selected_features.append({
            "feature": feature,
            "feature_text": feature_texts.get(feature, feature),
            "perplexity": perplexity,
            "perplexity_reduction": prev_perplexity - perplexity if i > 0 else 0
        })
    
    # Save selected features in the expected format
    with open(os.path.join(output_dir, "perplexity_selected_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)
    
    # Also save as a simple text file
    with open(os.path.join(output_dir, "perplexity_selected_features.txt"), "w") as f:
        for feature_info in selected_features:
            f.write(f"{feature_info['feature']}: {feature_info['feature_text']} (perplexity: {feature_info['perplexity']:.3f})\n")
    
    return [feature_info["feature"] for feature_info in selected_features]


def select_best_feature(evaluated_df, best_features):
    """
    Select the best feature based on evaluation scores (exact implementation from dataset-featurization)
    """
    remaining_features = [col for col in evaluated_df.columns if col not in best_features and col != "empty"]
    if not remaining_features:
        return "empty"
        
    feature_scores = evaluated_df[remaining_features].sum().sort_values()
    return feature_scores.index[0] if len(feature_scores) > 0 else "empty"

def analyze_feature_clusters(df, features_df, output_dir, selected_features=None):
    """
    Analyze feature clusters to identify common patterns
    
    Args:
        df: Original dataset DataFrame
        features_df: DataFrame with feature verification results
        output_dir: Directory to save analysis results
        selected_features: List of features selected by perplexity minimization
    """
    print("Analyzing feature clusters...")
    
    # Map correct/incorrect answers
    merged_df = features_df.copy()
    merged_df["answer_correct"] = df.iloc[features_df.index]["target"].apply(lambda x: 1 if x else 0)
    
    # Get just the feature columns
    feature_cols = [col for col in features_df.columns if col.startswith("feature_")]
    
    # Calculate feature occurrence rates
    feature_stats = {}
    for feature in feature_cols:
        # Overall occurrence rate
        occurrence_rate = features_df[feature].mean()
        
        # Occurrence rate in correct vs incorrect answers
        if merged_df["answer_correct"].sum() > 0:
            correct_rate = merged_df[merged_df["answer_correct"] == 1][feature].mean()
            incorrect_rate = merged_df[merged_df["answer_correct"] == 0][feature].mean()
            
            # Calculate correlation with correctness
            correlation = merged_df[feature].corr(merged_df["answer_correct"])
            
            feature_stats[feature] = {
                "occurrence_rate": float(occurrence_rate),
                "correct_rate": float(correct_rate) if not pd.isna(correct_rate) else 0.0,
                "incorrect_rate": float(incorrect_rate) if not pd.isna(incorrect_rate) else 0.0,
                "correlation": float(correlation) if not pd.isna(correlation) else 0.0,
                "feature_text": open(os.path.join(output_dir, f"{feature}.txt"), "r").read().strip(),
                "selected_by_perplexity": feature in (selected_features or [])
            }
    
    # Save feature statistics
    with open(os.path.join(output_dir, "feature_statistics.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)
    
    # Find top correlating features
    correlations = [(feature, stats["correlation"]) for feature, stats in feature_stats.items()]
    top_positive = sorted(correlations, key=lambda x: x[1], reverse=True)[:10]
    top_negative = sorted(correlations, key=lambda x: x[1])[:10]
    
    # Print top correlating features
    print("\nTop positively correlated features with correctness:")
    for feature, corr in top_positive:
        print(f"{feature_stats[feature]['feature_text']}: {corr:.3f}")
    
    print("\nTop negatively correlated features with correctness:")
    for feature, corr in top_negative:
        print(f"{feature_stats[feature]['feature_text']}: {corr:.3f}")
    
    # Save summary of top features
    with open(os.path.join(output_dir, "top_correlating_features.txt"), "w") as f:
        f.write("Top positively correlated features with correctness:\n")
        for feature, corr in top_positive:
            f.write(f"{feature_stats[feature]['feature_text']}: {corr:.3f}\n")
        
        f.write("\nTop negatively correlated features with correctness:\n")
        for feature, corr in top_negative:
            f.write(f"{feature_stats[feature]['feature_text']}: {corr:.3f}\n")

@click.command()
@click.option('--sample-size', default=50, help='Number of samples to use')
@click.option('--output-dir', default='featurization_results', help='Directory to save results')
@click.option('--seed', default=42, help='Random seed for reproducibility')
@click.option('--max-features', default=10, help='Maximum number of features to select by perplexity minimization')
@click.option('--batch-size', default=8, help='Batch size for evaluation')
def main(sample_size, output_dir, seed, max_features, batch_size):
    """Generate and analyze features for the TruthfulQA dataset using dataset-featurization's exact logic"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Load TruthfulQA dataset
    truthfulqa_df = load_truthfulqa_dataset()
    
    # Generate features
    print("Generating features...")
    features = generate_features(truthfulqa_df, sample_size=sample_size, seed=seed)
    
    # Save generated features
    with open(os.path.join(output_dir, "truthfulqa_features.json"), "w") as f:
        json.dump(features, f, indent=2)
    
    # Verify features
    print("Verifying features...")
    verification_df = verify_features(truthfulqa_df, features, output_dir, sample_size=sample_size, seed=seed)
    
    # Select features by perplexity minimization using dataset-featurization's exact approach
    print("Selecting features using perplexity minimization...")
    selected_features = select_features_by_perplexity(
        truthfulqa_df, verification_df, output_dir, max_features=max_features
    )
    
    # Analyze feature clusters
    print("Analyzing feature clusters...")
    analyze_feature_clusters(truthfulqa_df, verification_df, output_dir, selected_features)
    
    print(f"Feature generation and analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
