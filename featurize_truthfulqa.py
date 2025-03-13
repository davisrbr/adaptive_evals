import os
import sys
import pandas as pd
import torch
import json
from datasets import load_dataset
from tqdm import tqdm
import openai
from dotenv import load_dotenv
import numpy as np
import random
import click

# Import necessary components from dataset-featurization
sys.path.append(os.path.join(os.getcwd(), "dataset-featurization"))
from dataset_featurization.utils.generator import Generator
from dataset_featurization.utils.verifier import Verifier
from dataset_featurization.config import VERIFICATION_SYSTEM_PROMPT, VERIFICATION_USER_PROMPT, VERIFICATION_SPLIT, CLUSTER_SIZE, MODEL

# Load environment variables for API access
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Configure OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    
    # Initialize the feature generator
    generator = Generator()
    
    # Generate raw features
    all_features = []
    
    # Process in batches to avoid rate limiting
    batch_size = 5
    for i in tqdm(range(0, len(df_sample), batch_size), desc="Generating features"):
        batch_df = df_sample.iloc[i:i+batch_size].reset_index(drop=True)
        batch_features = generator.analyze(batch_df)
        all_features.extend(batch_features)
    
    # Remove duplicates
    unique_features = list(set(all_features))
    print(f"Generated {len(unique_features)} unique features")
    
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
    
    # Split features into chunks for better processing
    feature_chunks = [features[i:i+VERIFICATION_SPLIT] for i in range(0, len(features), VERIFICATION_SPLIT)]
    
    # Create dataframe to store verification results
    verification_df = pd.DataFrame()
    verification_df["string"] = df_sample["string"]
    
    # Verify features in chunks
    for chunk_idx, feature_chunk in enumerate(feature_chunks):
        print(f"Verifying feature chunk {chunk_idx+1}/{len(feature_chunks)}")
        feature_results = verifier.verify(df_sample, feature_chunk)
        
        # Add verified features to dataframe
        for i, feature in enumerate(feature_chunk):
            feature_name = f"feature_{chunk_idx * VERIFICATION_SPLIT + i}"
            verification_df[feature_name] = feature_results[:, i]
            
            # Also save the mapping from feature_name to feature text
            with open(os.path.join(output_dir, f"{feature_name}.txt"), "w") as f:
                f.write(feature)
    
    # Save the full verification results
    verification_df.to_csv(os.path.join(output_dir, "truthfulqa_features_verification.csv"), index=False)
    
    return verification_df

def analyze_feature_clusters(df, features_df, output_dir):
    """
    Analyze feature clusters to identify common patterns
    
    Args:
        df: Original dataset DataFrame
        features_df: DataFrame with feature verification results
        output_dir: Directory to save analysis results
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
                "feature_text": open(os.path.join(output_dir, f"{feature}.txt"), "r").read().strip()
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
def main(sample_size, output_dir, seed):
    """Generate and analyze features for the TruthfulQA dataset"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load TruthfulQA dataset
    truthfulqa_df = load_truthfulqa_dataset()
    
    # Generate features
    features = generate_features(truthfulqa_df, sample_size=sample_size, seed=seed)
    
    # Save generated features
    with open(os.path.join(output_dir, "truthfulqa_features.json"), "w") as f:
        json.dump(features, f, indent=2)
    
    # Verify features
    verification_df = verify_features(truthfulqa_df, features, output_dir, sample_size=sample_size, seed=seed)
    
    # Analyze feature clusters
    analyze_feature_clusters(truthfulqa_df, verification_df, output_dir)
    
    print(f"Feature generation and analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()