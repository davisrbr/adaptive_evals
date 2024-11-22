# Script to process and plot combined results from adaptive_truthfulqa and adaptive_legal experiments
from inspect_ai.log import read_eval_log
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import concurrent.futures

sns.set(style="whitegrid")

@dataclass(frozen=True)
class AdaptiveLogKey:
    eval_model_name: str
    generator_model_name: str
    n_positive_samples: int
    n_negative_samples: int
    use_cot_generator: bool
    randomize_sampling: bool
    dataset: str  # 'truthfulqa' or 'legal'

@dataclass
class AdaptiveLogValue:
    log_path: str
    accuracy: float
    total_samples: int
    original_accuracy: float
    original_log_model: str

# Set the base directories where your logs are stored
truthfulqa_base_dir = "new_logs"
legal_base_dir = "legal_experiment_logs"
base_dirs = [truthfulqa_base_dir, legal_base_dir]

# Load initial logs from both datasets
initial_logs = {}
for base_dir in base_dirs:
    for d in os.listdir(base_dir):
        if "initial" in d:
            log_path = os.path.join(base_dir, d)
            log = read_eval_log(log_path)
            dataset = 'truthfulqa' if 'truthfulqa' in base_dir else 'legal'
            # Compute initial accuracy
            initial_logs[(log.eval.model, dataset)] = AdaptiveLogValue(
                log_path=log_path,
                accuracy=log.results.metrics['accuracy'].value,
                total_samples=log.results.total_samples,
                original_accuracy=log.results.metrics['accuracy'].value,
                original_log_model=log.eval.model
            )

adaptive_logs = {}

def process_log(log_path, dataset):
    try:
        log = read_eval_log(log_path)
        if log.status == "success":
            initial_log_path = log.eval.task_args.get('initial_log_path', None)
            if initial_log_path:
                initial_base_log = os.path.join("..", initial_log_path)
                initial_log = read_eval_log(initial_base_log)
            else:
                initial_log = None
            eval_model_name = log.eval.task_args.get('eval_model_name', 'unknown')
            generator_model_name = log.eval.task_args.get('generator_model_name', 'unknown')
            key = AdaptiveLogKey(
                eval_model_name=eval_model_name,
                generator_model_name=generator_model_name,
                n_positive_samples=log.eval.task_args.get('n_positive_samples', 0),
                n_negative_samples=log.eval.task_args.get('n_negative_samples', 0),
                use_cot_generator=log.eval.task_args.get('use_cot_generator', False),
                randomize_sampling=log.eval.task_args.get('randomize_sampling', False),
                dataset=dataset
            )
            # Compute accuracy based on "C" vs "I"
            correct_samples = 0
            total_samples = 0
            scorer_name = 'adaptive_truthfulqa_scorer' if dataset == 'truthfulqa' else 'adaptive_legal_scorer'
            for sample in log.samples:
                if sample.scores and scorer_name in sample.scores:
                    score = sample.scores[scorer_name]
                    if score.value == "C" and score.answer != "[ERROR]":
                        correct_samples += 1
                    if score.answer != "[ERROR]":
                        total_samples += 1
            if total_samples == 0:
                return None  # Skip if no valid samples
            accuracy = correct_samples / total_samples

            original_accuracy = initial_logs.get((eval_model_name, dataset), AdaptiveLogValue('', 0.0, 0, 0.0, '')).accuracy

            value = AdaptiveLogValue(
                log_path=log_path,
                accuracy=accuracy,
                total_samples=log.results.total_samples,
                original_accuracy=original_accuracy,
                original_log_model=eval_model_name
            )
            return (key, value)
        else:
            # Optionally, delete the log file to save time in the future
            # os.remove(log_path)
            return None
    except Exception as e:
        print(f"Error processing {log_path}: {e}")
        return None

# Gather all adaptive log paths
adaptive_log_paths = []
for base_dir in base_dirs:
    dataset = 'truthfulqa' if 'truthfulqa' in base_dir else 'legal'
    paths = [
        (os.path.join(base_dir, d), dataset)
        for d in os.listdir(base_dir) if "adaptive" in d
    ]
    adaptive_log_paths.extend(paths)

# Process logs concurrently
def process_wrapper(args):
    return process_log(*args)

with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    results = executor.map(process_wrapper, adaptive_log_paths)
    for result in results:
        if result:
            key, value = result
            # Check if the key is already in adaptive_logs
            if key in adaptive_logs:
                existing_value = adaptive_logs[key]
                # Replace if the new value has more samples
                if value.total_samples > existing_value.total_samples:
                    adaptive_logs[key] = value
            else:
                adaptive_logs[key] = value

# Evaluation model for plotting
eval_model_name = 'openai/gpt-4o'

# Prepare data for plotting
data = []

# Collect adaptive run accuracies for different generator models
for key, value in adaptive_logs.items():
    if key.eval_model_name == eval_model_name and key.n_negative_samples == 10 and key.n_positive_samples == 2 and "70B" not in key.generator_model_name and not key.randomize_sampling:
        generator_model_name = key.generator_model_name.split('/')[-1]
        if "Instruct-Turbo" in key.generator_model_name:
            generator_model_name = generator_model_name.replace("-Instruct-Turbo", "")
        elif "claude" in generator_model_name:
            generator_model_name = "claude-3-5-sonnet"

        # Append accuracy
        entry = {
            'Generator Model': generator_model_name,
            'use_cot': 'CoT' if key.use_cot_generator else 'No CoT',
            'Dataset': 'TruthfulQA' if key.dataset == 'truthfulqa' else 'LegalBench',
            'Accuracy': value.accuracy,
            'num_samples': value.total_samples,
            'initial_accuracy': value.original_accuracy
        }
        data.append(entry)
        print(f"Appending entry: {entry}")

# Create a DataFrame for plotting
df = pd.DataFrame(data)
# Filter to get a single instance per group, taking the one with most samples if there are multiple
def filter_group(group):
    return group.sort_values('num_samples', ascending=False).head(1)
df = df.groupby(['Generator Model', 'use_cot', 'Dataset']).apply(filter_group).reset_index(drop=True)

# Print DataFrame contents
print("DataFrame contents:")
print(df)

# Set the seaborn style
sns.set_style("whitegrid")

# Plotting the data
plt.figure(figsize=(12, 7))

# Create the bar plot
sns.barplot(
    data=df,
    x='Generator Model',
    y='Accuracy',
    hue='Dataset',
    hue_order=['LegalBench', 'TruthfulQA'],
    order=df['Generator Model'].unique(),
    palette='dark',
    edgecolor='black'
)

# Adjust labels and title
plt.xlabel('Question Writer Model', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
# plt.title(f'Adaptive evaluation for {eval_model_name.split("/")[-1]}', fontsize=22)

# Increase fontsize of labels and ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Plot the baseline performance as dashed horizontal lines
datasets = df['Dataset'].unique()
colors = {'TruthfulQA': 'blue', 'LegalBench': 'red'}
for dataset in datasets:
    initial_accuracy = initial_logs.get((eval_model_name, dataset.lower()), AdaptiveLogValue('', 0.0, 0, 0.0, '')).accuracy
    plt.axhline(y=initial_accuracy, color=colors[dataset], linestyle='--', linewidth=2, label=f'{dataset} Baseline')

plt.ylim(0, 1)
plt.legend(title='Dataset', title_fontsize=16, fontsize=14)
plt.tight_layout()
plt.savefig(f"adaptive_combined_accuracy_{eval_model_name.split('/')[-1]}_bars.png", dpi=300)
plt.show() 