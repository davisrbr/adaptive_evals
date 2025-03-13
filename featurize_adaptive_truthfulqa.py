import os
import sys
import pandas as pd
import json
import click
from tqdm import tqdm
import openai
import numpy as np
import random
from dotenv import load_dotenv
from inspect_ai.log import read_eval_log

# Import necessary components from dataset-featurization
sys.path.append(os.path.join(os.getcwd(), "dataset-featurization"))
from dataset_featurization.utils.generator import Generator
from dataset_featurization.utils.verifier import Verifier
from dataset_featurization.config import VERIFICATION_SPLIT

# Load environment variables for API access
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set")
    sys.exit(1)

# Configure OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    
    # Extract questions
    questions = []
    
    for sample in eval_log.samples:
        # Get the generated sample
        generated_sample = sample.store.get("generated_sample")
        if not generated_sample:
            continue
        
        # Get question and choices
        question = generated_sample.get("input", "")
        choices = generated_sample.get("choices", [])
        target = generated_sample.get("target", [])
        
        if not question or not choices or not target:
            continue
            
        # Convert target index to letter
        target_letter = [chr(ord("A") + i) for i in target]
        
        # Get model answer and score
        model_answer = generated_sample.get("metadata", {}).get("model_answer", "")
        score = generated_sample.get("metadata", {}).get("score", "")
        
        questions.append({
            "question": question,
            "choices": choices,
            "target": target_letter,
            "model_answer": model_answer,
            "score": score,
            "is_correct": 1 if score == "C" else 0,
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(questions)
    df["string"] = df["question"]  # The featurization code expects a "string" column
    
    print(f"Loaded {len(df)} questions from adaptive TruthfulQA log")
    return df

def generate_features(df, sample_size=50, seed=42):
    """
    Generate features for the adaptive TruthfulQA dataset
    
    Args:
        df: DataFrame containing the adaptive TruthfulQA dataset
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
        df: DataFrame containing the adaptive TruthfulQA dataset
        features: List of features to verify
        output_dir: Directory to save the results
        sample_size: Number of samples to use
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
    verification_df["is_correct"] = df_sample["is_correct"]
    
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
    verification_df.to_csv(os.path.join(output_dir, "adaptive_truthfulqa_features_verification.csv"), index=False)
    
    return verification_df

def analyze_features(verification_df, output_dir):
    """
    Analyze features to identify correlations with correctness
    
    Args:
        verification_df: DataFrame with feature verification results
        output_dir: Directory to save analysis results
    """
    print("Analyzing features...")
    
    # Get just the feature columns
    feature_cols = [col for col in verification_df.columns if col.startswith("feature_")]
    
    # Calculate feature statistics
    feature_stats = {}
    for feature in feature_cols:
        # Overall occurrence rate
        occurrence_rate = verification_df[feature].mean()
        
        # Occurrence rate in correct vs incorrect answers
        if verification_df["is_correct"].sum() > 0:
            correct_rate = verification_df[verification_df["is_correct"] == 1][feature].mean()
            incorrect_rate = verification_df[verification_df["is_correct"] == 0][feature].mean()
            
            # Calculate correlation with correctness
            correlation = verification_df[feature].corr(verification_df["is_correct"])
            
            # Calculate odds ratio between feature presence and correctness
            feature_present_correct = ((verification_df[feature] == 1) & (verification_df["is_correct"] == 1)).sum()
            feature_present_incorrect = ((verification_df[feature] == 1) & (verification_df["is_correct"] == 0)).sum()
            feature_absent_correct = ((verification_df[feature] == 0) & (verification_df["is_correct"] == 1)).sum()
            feature_absent_incorrect = ((verification_df[feature] == 0) & (verification_df["is_correct"] == 0)).sum()
            
            if feature_present_incorrect > 0 and feature_absent_correct > 0:
                odds_ratio = (feature_present_correct * feature_absent_incorrect) / (feature_present_incorrect * feature_absent_correct)
            else:
                odds_ratio = float('nan')
            
            feature_stats[feature] = {
                "occurrence_rate": float(occurrence_rate),
                "correct_rate": float(correct_rate) if not pd.isna(correct_rate) else 0.0,
                "incorrect_rate": float(incorrect_rate) if not pd.isna(incorrect_rate) else 0.0,
                "correlation": float(correlation) if not pd.isna(correlation) else 0.0,
                "odds_ratio": float(odds_ratio) if not pd.isna(odds_ratio) else 0.0,
                "feature_text": open(os.path.join(output_dir, f"{feature}.txt"), "r").read().strip()
            }
    
    # Save feature statistics
    with open(os.path.join(output_dir, "adaptive_feature_statistics.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)
    
    # Find most predictive features
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
    with open(os.path.join(output_dir, "adaptive_top_correlating_features.txt"), "w") as f:
        f.write("Top positively correlated features with correctness:\n")
        for feature, corr in top_positive:
            f.write(f"{feature_stats[feature]['feature_text']}: {corr:.3f}\n")
        
        f.write("\nTop negatively correlated features with correctness:\n")
        for feature, corr in top_negative:
            f.write(f"{feature_stats[feature]['feature_text']}: {corr:.3f}\n")

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
        with open(os.path.join(static_dir, "feature_statistics.json"), "r") as f:
            static_stats = json.load(f)
        
        with open(os.path.join(adaptive_dir, "adaptive_feature_statistics.json"), "r") as f:
            adaptive_stats = json.load(f)
    except FileNotFoundError:
        print("Error: Feature statistics files not found. Please run the static and adaptive feature extraction first.")
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
            comparison.append({
                "static_feature": static_features[s_key],
                "static_correlation": static_stats[s_key]["correlation"],
                "adaptive_feature": adaptive_features[a_key],
                "adaptive_correlation": adaptive_stats[a_key]["correlation"],
                "correlation_difference": adaptive_stats[a_key]["correlation"] - static_stats[s_key]["correlation"]
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
            print(f"{row['static_feature']} | Static: {row['static_correlation']:.3f}, Adaptive: {row['adaptive_correlation']:.3f}, Diff: {row['correlation_difference']:.3f}")
    else:
        print("No identical features found between static and adaptive datasets.")

@click.command()
@click.argument('log_path', type=click.Path(exists=True))
@click.option('--sample-size', default=50, help='Number of samples to use')
@click.option('--output-dir', default='adaptive_featurization_results', help='Directory to save results')
@click.option('--static-dir', default='featurization_results', help='Directory with static TruthfulQA features')
@click.option('--seed', default=42, help='Random seed for reproducibility')
@click.option('--compare/--no-compare', default=True, help='Compare with static features')
def main(log_path, sample_size, output_dir, static_dir, seed, compare):
    """
    Generate and analyze features for the adaptive TruthfulQA dataset.
    
    LOG_PATH should be a path to an evaluation log file.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load adaptive TruthfulQA dataset
    adaptive_df = load_adaptive_truthfulqa_log(log_path)
    
    # Generate features
    features = generate_features(adaptive_df, sample_size=sample_size, seed=seed)
    
    # Save generated features
    with open(os.path.join(output_dir, "adaptive_truthfulqa_features.json"), "w") as f:
        json.dump(features, f, indent=2)
    
    # Verify features
    verification_df = verify_features(adaptive_df, features, output_dir, sample_size=sample_size, seed=seed)
    
    # Analyze features
    analyze_features(verification_df, output_dir)
    
    # Compare with static features if requested
    if compare and os.path.exists(static_dir):
        compare_with_original(static_dir, output_dir, output_dir)
    
    print(f"Feature generation and analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()