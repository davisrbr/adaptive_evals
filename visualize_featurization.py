import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import click

def load_feature_data(static_dir, adaptive_dir):
    """
    Load feature data from static and adaptive results
    
    Args:
        static_dir: Directory with static TruthfulQA features
        adaptive_dir: Directory with adaptive TruthfulQA features
        
    Returns:
        Tuple of static and adaptive feature statistics
    """
    # Load static feature statistics
    try:
        with open(os.path.join(static_dir, "feature_statistics.json"), "r") as f:
            static_stats = json.load(f)
    except FileNotFoundError:
        print(f"Warning: No static feature statistics found in {static_dir}")
        static_stats = {}
    
    # Load adaptive feature statistics
    try:
        with open(os.path.join(adaptive_dir, "adaptive_feature_statistics.json"), "r") as f:
            adaptive_stats = json.load(f)
    except FileNotFoundError:
        print(f"Warning: No adaptive feature statistics found in {adaptive_dir}")
        adaptive_stats = {}
    
    return static_stats, adaptive_stats

def plot_correlation_distribution(static_stats, adaptive_stats, output_dir):
    """
    Plot distribution of feature correlations with correctness
    
    Args:
        static_stats: Dictionary of static feature statistics
        adaptive_stats: Dictionary of adaptive feature statistics
        output_dir: Directory to save visualization results
    """
    plt.figure(figsize=(12, 6))
    
    # Extract correlation values
    static_corrs = [stats["correlation"] for stats in static_stats.values()]
    adaptive_corrs = [stats["correlation"] for stats in adaptive_stats.values()]
    
    # Create histogram
    sns.histplot(static_corrs, alpha=0.5, label='Static TruthfulQA', kde=True, color='blue')
    sns.histplot(adaptive_corrs, alpha=0.5, label='Adaptive TruthfulQA', kde=True, color='red')
    
    plt.title('Distribution of Feature Correlations with Correctness')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "correlation_distribution.png"), dpi=300)
    plt.close()

def plot_top_features(static_stats, adaptive_stats, output_dir):
    """
    Plot top positively and negatively correlated features
    
    Args:
        static_stats: Dictionary of static feature statistics
        adaptive_stats: Dictionary of adaptive feature statistics
        output_dir: Directory to save visualization results
    """
    # Create dataframe for static features
    static_df = pd.DataFrame([
        {
            "feature": stats["feature_text"],
            "correlation": stats["correlation"],
            "type": "Static"
        }
        for feature, stats in static_stats.items()
    ])
    
    # Create dataframe for adaptive features
    adaptive_df = pd.DataFrame([
        {
            "feature": stats["feature_text"],
            "correlation": stats["correlation"],
            "type": "Adaptive"
        }
        for feature, stats in adaptive_stats.items()
    ])
    
    # Combine dataframes
    combined_df = pd.concat([static_df, adaptive_df])
    
    # Get top positive and negative correlations
    top_positive = combined_df.sort_values("correlation", ascending=False).head(10)
    top_negative = combined_df.sort_values("correlation").head(10)
    
    # Plot top positive correlations
    plt.figure(figsize=(12, 8))
    bars = sns.barplot(
        data=top_positive,
        y="feature",
        x="correlation",
        hue="type",
        palette={"Static": "blue", "Adaptive": "red"},
        dodge=False
    )
    
    plt.title('Top Positively Correlated Features with Correctness')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_positive_features.png"), dpi=300)
    plt.close()
    
    # Plot top negative correlations
    plt.figure(figsize=(12, 8))
    bars = sns.barplot(
        data=top_negative,
        y="feature",
        x="correlation",
        hue="type",
        palette={"Static": "blue", "Adaptive": "red"},
        dodge=False
    )
    
    plt.title('Top Negatively Correlated Features with Correctness')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_negative_features.png"), dpi=300)
    plt.close()

def plot_feature_occurrence_comparison(static_stats, adaptive_stats, output_dir):
    """
    Plot comparison of feature occurrence rates
    
    Args:
        static_stats: Dictionary of static feature statistics
        adaptive_stats: Dictionary of adaptive feature statistics
        output_dir: Directory to save visualization results
    """
    # Create dataframe with feature occurrence rates
    data = []
    
    for feature_key, stats in static_stats.items():
        feature_text = stats["feature_text"]
        
        # Check if this feature exists in adaptive stats
        adaptive_key = None
        for a_key, a_stats in adaptive_stats.items():
            if a_stats["feature_text"].lower() == feature_text.lower():
                adaptive_key = a_key
                break
        
        if adaptive_key:
            data.append({
                "feature": feature_text,
                "static_occurrence": stats["occurrence_rate"],
                "adaptive_occurrence": adaptive_stats[adaptive_key]["occurrence_rate"],
                "static_correlation": stats["correlation"],
                "adaptive_correlation": adaptive_stats[adaptive_key]["correlation"],
                "correlation_diff": adaptive_stats[adaptive_key]["correlation"] - stats["correlation"]
            })
    
    # If we found matching features, create visualization
    if data:
        df = pd.DataFrame(data)
        
        # Sort by the absolute difference in correlation
        df["abs_corr_diff"] = df["correlation_diff"].abs()
        df = df.sort_values("abs_corr_diff", ascending=False).head(15)
        
        # Plot occurrence rates
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(df))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 10))
        rects1 = ax.bar(x - width/2, df["static_occurrence"], width, label='Static Occurrence', color='lightblue')
        rects2 = ax.bar(x + width/2, df["adaptive_occurrence"], width, label='Adaptive Occurrence', color='lightcoral')
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Occurrence Rate')
        ax.set_title('Feature Occurrence Rates: Static vs Adaptive')
        ax.set_xticks(x)
        ax.set_xticklabels(df["feature"], rotation=90)
        ax.legend()
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_occurrence_comparison.png"), dpi=300)
        plt.close()
        
        # Plot correlation values
        plt.figure(figsize=(14, 10))
        
        rects1 = ax.bar(x - width/2, df["static_correlation"], width, label='Static Correlation', color='blue')
        rects2 = ax.bar(x + width/2, df["adaptive_correlation"], width, label='Adaptive Correlation', color='red')
        
        ax.set_xlabel('Features')
        ax.set_ylabel('Correlation with Correctness')
        ax.set_title('Feature Correlations: Static vs Adaptive')
        ax.set_xticks(x)
        ax.set_xticklabels(df["feature"], rotation=90)
        ax.legend()
        
        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, "feature_correlation_comparison.png"), dpi=300)
        plt.close()
    else:
        print("No matching features found between static and adaptive datasets for comparison.")

def plot_feature_space(static_dir, adaptive_dir, output_dir):
    """
    Create a 2D visualization of feature space
    
    Args:
        static_dir: Directory with static TruthfulQA features
        adaptive_dir: Directory with adaptive TruthfulQA features
        output_dir: Directory to save visualization results
    """
    # Load verification results
    try:
        static_df = pd.read_csv(os.path.join(static_dir, "truthfulqa_features_verification.csv"))
        adaptive_df = pd.read_csv(os.path.join(adaptive_dir, "adaptive_truthfulqa_features_verification.csv"))
    except FileNotFoundError:
        print("Warning: Feature verification CSV files not found. Skipping feature space visualization.")
        return
    
    # Get feature columns
    static_feature_cols = [col for col in static_df.columns if col.startswith("feature_")]
    adaptive_feature_cols = [col for col in adaptive_df.columns if col.startswith("feature_")]
    
    if not static_feature_cols or not adaptive_feature_cols:
        print("Warning: No feature columns found in verification data. Skipping feature space visualization.")
        return
    
    # Create feature matrices
    static_features = static_df[static_feature_cols].values
    adaptive_features = adaptive_df[adaptive_feature_cols].values
    
    # Create labels
    static_labels = static_df["is_correct"] if "is_correct" in static_df.columns else np.zeros(len(static_df))
    adaptive_labels = adaptive_df["is_correct"] if "is_correct" in adaptive_df.columns else np.zeros(len(adaptive_df))
    
    # Combine features and normalize
    combined_features = np.vstack([static_features, adaptive_features])
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)
    
    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    combined_tsne = tsne.fit_transform(combined_features_scaled)
    
    # Split back into static and adaptive
    static_tsne = combined_tsne[:len(static_features)]
    adaptive_tsne = combined_tsne[len(static_features):]
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot static features
    plt.scatter(
        static_tsne[:, 0], 
        static_tsne[:, 1], 
        c=static_labels, 
        cmap='Blues', 
        edgecolor='k', 
        alpha=0.7, 
        s=100,
        label='Static'
    )
    
    # Plot adaptive features
    plt.scatter(
        adaptive_tsne[:, 0], 
        adaptive_tsne[:, 1], 
        c=adaptive_labels, 
        cmap='Reds', 
        edgecolor='k', 
        alpha=0.7, 
        s=100,
        label='Adaptive'
    )
    
    plt.title('t-SNE Visualization of Feature Space')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_space_tsne.png"), dpi=300)
    plt.close()

@click.command()
@click.option('--static-dir', default='featurization_results', help='Directory with static TruthfulQA features')
@click.option('--adaptive-dir', default='adaptive_featurization_results', help='Directory with adaptive TruthfulQA features')
@click.option('--output-dir', default='visualization_results', help='Directory to save visualization results')
def main(static_dir, adaptive_dir, output_dir):
    """
    Visualize and compare static and adaptive feature results.
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load feature data
    static_stats, adaptive_stats = load_feature_data(static_dir, adaptive_dir)
    
    if not static_stats or not adaptive_stats:
        print("Error: Could not load feature statistics. Please run featurization first.")
        return
    
    # Create visualizations
    print("Creating correlation distribution plot...")
    plot_correlation_distribution(static_stats, adaptive_stats, output_dir)
    
    print("Creating top features plot...")
    plot_top_features(static_stats, adaptive_stats, output_dir)
    
    print("Creating feature occurrence comparison plot...")
    plot_feature_occurrence_comparison(static_stats, adaptive_stats, output_dir)
    
    print("Creating feature space visualization...")
    plot_feature_space(static_dir, adaptive_dir, output_dir)
    
    print(f"Visualization complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()