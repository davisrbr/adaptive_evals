# Script to process the results of adaptive_truthfulqa experiments and plot accuracy

from collections import defaultdict
from inspect_ai.log import list_eval_logs, read_eval_log
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the base directory where your logs are stored
log_base_dir = "new_logs"

# Get all log directories that contain the adaptive_truthfulqa logs
log_dirs = [
    os.path.join(log_base_dir, d)
    for d in os.listdir(log_base_dir)
    if d.startswith("initial_adaptive_truthfulqa_log")
]

# Initialize a list to collect data
data = []

for log_dir in log_dirs:
    # Get all log files in the directory
    log_files = list_eval_logs(log_dir)

    for log_file in log_files:
        # Read the log file
        eval_log = read_eval_log(log_file)

        if eval_log.status == "success":
            # Try to extract parameters from eval_log
            try:
                task_parameters = eval_log.eval.task_args
            except AttributeError:
                task_parameters = {}

            # Extract the parameters we need
            eval_model_name = task_parameters.get("eval_model_name", "unknown")
            generator_model_name = task_parameters.get("generator_model_name", "unknown")
            n_positive_samples = task_parameters.get("n_positive_samples", 0)
            n_negative_samples = task_parameters.get("n_negative_samples", 0)
            use_cot = task_parameters.get("use_cot", False)

            # Compute accuracy
            total_samples = 0
            correct_samples = 0
            for sample in eval_log.samples:
                # Check if 'scores' field is available
                if sample.scores and sample.scores['adaptive_truthfulqa_scorer'].value in {"C", "I"}:
                    total_samples += 1
                    if sample.scores['adaptive_truthfulqa_scorer'].value == "C":
                        correct_samples += 1
            if total_samples == 0:
                continue  # Skip if no valid samples
            accuracy = correct_samples / total_samples

            # Collect data
            data.append({
                "eval_model": eval_model_name,
                "generator_model": generator_model_name,
                "n_positive_samples": n_positive_samples,
                "n_negative_samples": n_negative_samples,
                "use_cot": "CoT" if use_cot else "No CoT",
                "accuracy": accuracy,
            })

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Print the first few rows of the DataFrame for inspection
print(df.head())

# save df to csv
df.to_csv("adaptive_truthfulqa_results.csv", index=False)

# Set the visual style
sns.set(style="whitegrid")

# Get the list of unique evaluation models
eval_models = df['eval_model'].unique()
print(eval_models)

# Create a separate plot for each evaluation model
for eval_model in eval_models:
    plt.figure(figsize=(12, 8))
    subset = df[df['eval_model'] == eval_model]
    scatter = sns.scatterplot(
        data=subset,
        x="n_negative_samples",
        y="accuracy",
        hue="generator_model",
        style="use_cot",
        size="n_positive_samples",
        sizes=(50, 200),
        alpha=0.7,
    )
    plt.title(f"Accuracy vs. Negative Examples (Eval Model: {eval_model})", fontsize=16)
    plt.xlabel("Negative In-Context Examples", fontsize=14)
    plt.ylabel("Accuracy", fontsize=14)
    plt.legend(title="Generator Model", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"adaptive_truthfulqa_accuracy_vs_negative_examples_{eval_model.replace('/', '_')}.png", dpi=300)
    plt.show()
