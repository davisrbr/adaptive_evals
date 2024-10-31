from inspect_ai.log import read_eval_log
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import concurrent.futures
import seaborn as sns


sns.set_style("white")

@dataclass(frozen=True)
class AdaptiveLogKey:
    eval_model_name: str
    generator_model_name: str
    n_positive_samples: int
    n_negative_samples: int
    use_cot_generator: bool
    randomize_sampling: bool
    use_o1_judge: bool
        
@dataclass
class AdaptiveLogValue:
    log_path: str
    accuracy: float
    total_samples: int
    original_accuracy: float
    original_log_model: str
    accuracy_with_judge: float = None
    total_samples_with_judge: int = None  # Added to store total_samples_with_judge

# Set the base directories where your logs are stored
log_base_dir = "/Users/davisbrown/inspect_attacks/legal_experiment_logs"
base_dirs = ["/Users/davisbrown/inspect_attacks/legal_experiment_logs", "/Users/davisbrown/inspect_attacks/logs"]

# Get all log directories that contain the initial aggregated legalbench run
initial_logs = {}
for d in os.listdir(log_base_dir):
    if "initial" in d:
        log_path = os.path.join(log_base_dir, d)
        log = read_eval_log(log_path)
        # Compute initial accuracy
        initial_logs[log.eval.model] = AdaptiveLogValue(
            log_path=log_path,
            accuracy=log.results.metrics['accuracy'].value,
            total_samples=log.results.total_samples,
            original_accuracy=log.results.metrics['accuracy'].value,
            original_log_model=log.eval.model
        )

adaptive_logs = {}

def process_log(log_path):
    try:
        log = read_eval_log(log_path)
        if log.status == "success":
            initial_log_path = log.eval.task_args['initial_log_path']
            initial_base_log = os.path.join("..", initial_log_path)
            initial_log = read_eval_log(initial_base_log)
            eval_model_name = log.eval.task_args['eval_model_name']
            generator_model_name = log.eval.task_args['generator_model_name']
            key = AdaptiveLogKey(
                eval_model_name=eval_model_name,
                generator_model_name=generator_model_name,
                n_positive_samples=log.eval.task_args['n_positive_samples'],
                n_negative_samples=log.eval.task_args['n_negative_samples'],
                use_cot_generator=log.eval.task_args['use_cot_generator'],
                randomize_sampling=log.eval.task_args.get('randomize_sampling', False),
                use_o1_judge=log.eval.task_args.get('judge_model_name', None) is not None
            )
            # Compute accuracy based on "C" vs "I"
            correct_samples = sum(
                1 for sample in log.samples 
                if (
                    sample.scores and 
                    sample.scores['adaptive_legal_scorer'].value == "C" and 
                    sample.scores['adaptive_legal_scorer'].answer != "[ERROR]"
                )
            )
            total_samples = sum(
                1 for sample in log.samples 
                if sample.scores and 
                sample.scores['adaptive_legal_scorer'].answer != "[ERROR]"
            )
            assert total_samples > 8, f"Total samples is less than 8 for {log_path}"
            accuracy = correct_samples / total_samples

            # Check if 'adaptive_legal_judge_scorer' exists in any sample
            if any('adaptive_legal_judge_scorer' in sample.scores for sample in log.samples):
                # Filter samples where 'adaptive_legal_judge_scorer' is valid
                filtered_samples = [
                    sample for sample in log.samples
                    if (
                        sample.scores and
                        sample.scores['adaptive_legal_judge_scorer'].value == "C"
                    )
                ]
                total_samples_with_judge = len(filtered_samples)
                correct_samples_with_judge = sum(
                    1 for sample in filtered_samples
                    if (
                        sample.scores and 
                        sample.scores['adaptive_legal_scorer'].value == "C" and 
                        sample.scores['adaptive_legal_scorer'].answer != "[ERROR]"
                    )
                )
                if total_samples_with_judge > 0:
                    accuracy_with_judge = correct_samples_with_judge / total_samples_with_judge
                else:
                    print(f"No samples with judge for {log_path}")
                    accuracy_with_judge = None
            else:
                accuracy_with_judge = None
                total_samples_with_judge = 0  # Set to zero if no judge samples

            value = AdaptiveLogValue(
                log_path=log_path,
                accuracy=accuracy,
                total_samples=log.results.total_samples,
                original_accuracy=initial_log.results.metrics['accuracy'].value,
                original_log_model=initial_log.eval.model,
                accuracy_with_judge=accuracy_with_judge,
                total_samples_with_judge=total_samples_with_judge  # Store total_samples_with_judge
            )
            if value.accuracy_with_judge is not None:
                print(key)
                print(value)
            return (key, value)
        else:
            # delete the log file to save time in the future
            os.remove(log_path)
            return None
    except Exception:
        return None

# Gather all adaptive log paths
adaptive_log_paths = [
    os.path.join(base_dir, d)
    for base_dir in base_dirs
    for d in os.listdir(base_dir) if "adaptive" in d
]

# process logs concurrently
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    results = executor.map(process_log, adaptive_log_paths)
    for result in results:
        if result:
            key, value = result
            # Check if the key is already in adaptive_logs
            if key in adaptive_logs:
                existing_value = adaptive_logs[key]
                # Only replace if the new value has more samples with judge
                if (value.accuracy_with_judge is not None and 
                    existing_value.accuracy_with_judge is not None and
                    value.total_samples_with_judge > existing_value.total_samples_with_judge):
                    adaptive_logs[key] = value
            else:
                adaptive_logs[key] = value

# Evaluation model for plotting
eval_model_name = 'openai/gpt-4o-mini'

# Get the initial accuracy for this evaluation model
initial_accuracy = initial_logs[eval_model_name].accuracy

# Prepare data for plotting
data = []

# Collect adaptive run accuracies for different generator models
for key, value in adaptive_logs.items():
    # Adjusted condition to include entries even when accuracy_with_judge is None
    if key.eval_model_name == eval_model_name and key.n_negative_samples == 10 and key.n_positive_samples == 2 and "70B" not in key.generator_model_name and not key.randomize_sampling and key.eval_model_name == value.original_log_model:
        generator_model_name = key.generator_model_name.split('/')[-1]
        if "Instruct-Turbo" in key.generator_model_name:
            generator_model_name = generator_model_name.replace("-Instruct-Turbo", "")
        elif "claude" in generator_model_name:
            generator_model_name = "claude-3-5-sonnet"

        # Append accuracy without judge
        entry_no_judge = {
            'Generator Model': generator_model_name,
            'use_cot': 'CoT' if key.use_cot_generator else 'No CoT',
            'Accuracy Type': '(no judge)',
            'Accuracy': value.accuracy
        }
        data.append(entry_no_judge)
        print(f"Appending entry: {entry_no_judge}")

        # Append accuracy with judge if available
        if value.accuracy_with_judge is not None:
            entry_with_judge = {
                'Generator Model': generator_model_name,
                'use_cot': 'CoT' if key.use_cot_generator else 'No CoT',
                'Accuracy Type': '(o1 judge)',
                'Accuracy': value.accuracy_with_judge
            }
            data.append(entry_with_judge)
            print(f"Appending entry: {entry_with_judge}")
        else:
            print(f"No accuracy_with_judge for {generator_model_name}, use_cot: {'CoT' if key.use_cot_generator else 'No CoT'}")

# Create a DataFrame for plotting
df = pd.DataFrame(data)

# Print DataFrame contents
print("DataFrame contents:")
print(df)

# Create a new column combining 'use_cot' and 'Accuracy Type' for grouping
df['Category'] = df['use_cot'] + ' ' + df['Accuracy Type']

# Ensure consistent category ordering
category_order = ['CoT (o1 judge)', 'CoT (no judge)', 'No CoT (o1 judge)', 'No CoT (no judge)']

# Set the seaborn style to 'white'
sns.set_style("white")

# Plotting the data with four bars for each generator model
plt.figure(figsize=(12, 7))

# Define custom color palette
palette = {
    'CoT (no judge)': 'lightblue',
    'CoT (o1 judge)': 'blue',
    'No CoT (no judge)': 'lightcoral',
    'No CoT (o1 judge)': 'red'
}

# Create the bar plot
sns.barplot(
    data=df,
    x='Generator Model',
    y='Accuracy',
    hue='Category',
    order=df['Generator Model'].unique(),
    hue_order=category_order,
    palette=palette
)

# Adjust labels and title
plt.xlabel('Question Writer Model', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.title(f'Adaptive LegalBench evaluation for {eval_model_name.split("/")[-1]}', fontsize=22)

# Increase fontsize of labels and ticks
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# Plot the baseline performance as a dashed horizontal line
plt.axhline(y=initial_accuracy, color='gray', linestyle='--', linewidth=2)

# Annotate the accuracy value above the line
plt.text(1.5, initial_accuracy + 0.01, f'Accuracy on LegalBench: {initial_accuracy:.2f}', color='gray', fontsize=20)

plt.ylim(0, 1)
plt.legend(title='Configuration', title_fontsize=16, fontsize=14)
plt.tight_layout()
plt.savefig(f"adaptive_legal_accuracy_{eval_model_name.split('/')[-1]}_four_bars.png", dpi=300)
plt.show()

######################################################## Second Plot ########################################################

# For 'openai/gpt-4o', examine runs with different n_negative_samples
# Plot these on the x-axis with different bars for different generator models

# Specify the evaluation model name
eval_model_name = 'openai/gpt-4o'

# Get the initial accuracy for this evaluation model
initial_accuracy = initial_logs[eval_model_name].accuracy

# Prepare data for plotting
data = []

# Collect adaptive run accuracies for different generator models, n_negative_samples, use_cot, and randomize_sampling
for key, value in adaptive_logs.items():
    if key.eval_model_name == eval_model_name and key.n_negative_samples != 15 and key.n_negative_samples != 10 and ("70B" not in key.generator_model_name and "mini" not in key.generator_model_name):
        data.append({
            'Generator Model': key.generator_model_name.split('/')[-1],
            'n_negative_samples': key.n_negative_samples if key.n_negative_samples < 60 else 64,
            'Accuracy': value.accuracy,
            'use_cot': key.use_cot_generator,
            'randomize_sampling': key.randomize_sampling,
            'num_samples': value.total_samples,
            'config_label': f"CoT={'Yes' if key.use_cot_generator else 'No'}, Random={'Yes' if key.randomize_sampling else 'No'}"
        })

# Create a DataFrame for plotting
df = pd.DataFrame(data)

# Get the list of generator models
generator_models = ["Meta-Llama-3.1-405B-Instruct-Turbo", "gpt-4o"] # df['Generator Model'].unique()
print(df)

# Set up the plot with subplots for each generator model
num_models = len(generator_models) # because we are not using claude-3-5-sonnet :()
fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 6), sharey=True)

# Ensure axes is iterable
if num_models == 1:
    axes = [axes]

for i, gen_model in enumerate(generator_models):
    ax = axes[i]

    # Filter data for the current generator model
    df_model = df[df['Generator Model'] == gen_model]

    # Pivot the data to have combinations of 'use_cot' and 'randomize_sampling' as configurations
    df_pivot = df_model.pivot_table(
        index=['n_negative_samples', 'use_cot'],
        columns='randomize_sampling',
        values='Accuracy'
    ).reset_index()

    df_pivot.columns.name = None
    # Rename columns for clarity
    df_pivot.rename(columns={False: 'Adaptive', True: 'Randomized'}, inplace=True)

    available_columns = df_pivot.columns

    # Add a small offset to x positions based on 'use_cot'
    use_cot_offset = {False: -0.05, True: 0.05}

    # Initialize a set to keep track of labels we've added to the legend
    legend_labels = set()

    # Plot lines and points for each configuration
    for idx, row in df_pivot.iterrows():
        x_pos = np.log2(row['n_negative_samples']) + use_cot_offset[row['use_cot']]

        y_values = []
        labels_list = []
        facecolors = []
        edgecolors = []

        # Select marker style and color based on 'use_cot'
        if row['use_cot']:
            marker_style = 'o'
            color = 'red'
        else:
            marker_style = 's'  # square
            color = 'blue'

        # Check for 'Adaptive' data
        if 'Adaptive' in available_columns and pd.notna(row.get('Adaptive')):
            y_values.append(row['Adaptive'])
            labels_list.append('Adaptive')
            facecolors.append(color)
            edgecolors.append(color)

        # Check for 'Randomized' data
        if 'Randomized' in available_columns and pd.notna(row.get('Randomized')):
            y_values.append(row['Randomized'])
            labels_list.append('Randomized')
            facecolors.append('white')
            edgecolors.append(color)

        # If both exist, plot the line connecting them
        if len(y_values) == 2:
            ax.plot([x_pos, x_pos], y_values, color='gray', linewidth=8, zorder=1, alpha=0.4)

        # Plot each point
        for y, label_text, fc, ec in zip(y_values, labels_list, facecolors, edgecolors):
            # only plot CoT for now
            if not row['use_cot']:
                continue
            # Construct the label for this point
            point_label = f"{'CoT' if row['use_cot'] else 'Without CoT'}, {label_text}"

            # Only assign the label if we haven't already used it
            if point_label not in legend_labels:
                legend_labels.add(point_label)
                plot_label = point_label
            else:
                plot_label = None  # No label assigned to avoid duplicates

            # Plot the point
            ax.scatter(
                x_pos, y,
                facecolors=fc, edgecolors=ec,
                marker=marker_style, s=150, zorder=2,
                label=plot_label
            )

    # Set x-ticks to log2 scale
    x_ticks = sorted(df_model['n_negative_samples'].unique())
    ax.set_xticks(np.log2(x_ticks))
    ax.set_xticklabels([f"$2^{{{int(np.log2(n))}}}$" for n in x_ticks], fontsize=16)

    # Plot the baseline performance as a horizontal dashed line
    ax.axhline(y=initial_accuracy, color='gray', linestyle='--', linewidth=2)

    # Annotate the baseline accuracy for the leftmost plot
    if i == 0:
        ax.text(np.log2(x_ticks[0]) - 0.2, initial_accuracy - 0.08,
                f'Accuracy on LegalBench: {initial_accuracy:.2f}', color='gray', fontsize=18)
        ax.set_ylabel('Accuracy', fontsize=18)
    else:
        ax.set_ylabel('')

    # Set titles and labels
    model_name = gen_model.split('/')[-1]
    if "claude" in model_name:
        model_name = "claude-3-5-sonnet"
    elif "Instruct-Turbo" in model_name:
        model_name = model_name.replace("-Instruct-Turbo", "")
    ax.set_title(f'{model_name}', fontsize=18)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(ax.get_yticks(), fontsize=16)

    # Adjust legend
    if i == num_models - 1:
        ax.legend(fontsize=12)
    else:
        ax.legend().set_visible(False)

    ax.set_xlabel('Number of examples', fontsize=16)

# Adjust layout and add a super title
plt.suptitle(f'Adaptive LegalBench evaluation for {eval_model_name.split("/")[-1]}', fontsize=22)
plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig(f"adaptive_legal_accuracy_{eval_model_name.split('/')[-1]}_dumbb.png", dpi=300)
plt.show()
