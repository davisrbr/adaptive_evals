# TruthfulQA Featurization

This project contains scripts for featurizing both static and adaptive TruthfulQA datasets using the dataset-featurization framework. The goal is to identify natural language features that correlate with model performance and understand how these features differ between static and adaptive datasets.

## Setup

1. Make sure you're on the labeling branch:
   ```bash
   git checkout labeling
   ```

2. Ensure the dataset-featurization submodule is initialized:
   ```bash
   git submodule update --init --recursive
   ```

3. Install required packages:
   ```bash
   pip install -r dataset-featurization/requirements.txt
   pip install tqdm matplotlib seaborn scikit-learn
   ```

4. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY=your-api-key
   ```

## Usage

### Step 1: Featurize the static TruthfulQA dataset

```bash
python featurization/featurize_truthfulqa.py --sample-size 50 --output-dir featurization/featurization_results
```

Options:
- `--sample-size`: Number of samples to use (default: 50)
- `--output-dir`: Directory to save results (default: featurization_results)
- `--seed`: Random seed for reproducibility (default: 42)

### Step 2: Featurize an adaptive TruthfulQA evaluation log

```bash
python featurization/featurize_adaptive_truthfulqa.py /path/to/eval_log.json --sample-size 50 --output-dir featurization/adaptive_featurization_results
```

Arguments:
- `log_path`: Path to an evaluation log file

Options:
- `--sample-size`: Number of samples to use (default: 50)
- `--output-dir`: Directory to save results (default: adaptive_featurization_results)
- `--static-dir`: Directory with static TruthfulQA features (default: featurization_results)
- `--seed`: Random seed for reproducibility (default: 42)
- `--compare/--no-compare`: Compare with static features (default: True)

### Step 3: Visualize the results

```bash
python featurization/visualize_featurization.py --static-dir featurization/featurization_results --adaptive-dir featurization/adaptive_featurization_results --output-dir featurization/visualization_results
```

Options:
- `--static-dir`: Directory with static TruthfulQA features (default: featurization_results)
- `--adaptive-dir`: Directory with adaptive TruthfulQA features (default: adaptive_featurization_results)
- `--output-dir`: Directory to save visualization results (default: visualization_results)

## Output

The scripts generate several outputs:

### Static TruthfulQA Featurization

- `truthfulqa_features.json`: List of generated features
- `truthfulqa_features_verification.csv`: CSV file with feature verification results
- `feature_statistics.json`: Statistics about each feature
- `top_correlating_features.txt`: Summary of top correlating features

### Adaptive TruthfulQA Featurization

- `adaptive_truthfulqa_features.json`: List of generated features
- `adaptive_truthfulqa_features_verification.csv`: CSV file with feature verification results
- `adaptive_feature_statistics.json`: Statistics about each feature
- `adaptive_top_correlating_features.txt`: Summary of top correlating features
- `feature_comparison.csv`: Comparison between static and adaptive features

### Visualizations

- `correlation_distribution.png`: Comparison of feature correlation distributions
- `top_positive_features.png`: Top positively correlated features
- `top_negative_features.png`: Top negatively correlated features
- `feature_occurrence_comparison.png`: Comparison of feature occurrence rates
- `feature_correlation_comparison.png`: Comparison of feature correlations
- `feature_space_tsne.png`: t-SNE visualization of feature space

## How It Works

1. **Feature Generation**: The scripts use an LLM to generate candidate features by analyzing the dataset.
2. **Feature Verification**: Each candidate feature is verified against all samples in the dataset to create a feature occurrence matrix.
3. **Statistical Analysis**: The scripts calculate correlations between features and correctness, as well as other statistical measures.
4. **Comparison**: The adaptive featurization script compares features from the static and adaptive datasets to identify differences.
5. **Visualization**: The visualization script creates various plots to help understand the differences between static and adaptive features.

This approach helps identify natural language features that characterize the dataset and correlate with model performance, providing insights into the differences between static and adaptive datasets.