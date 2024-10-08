import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    # Data
    models = ['Mistral 8x22B', 'Mistral 8x7B', 'Mistral 8x7B (json parsing)']
    conditions = ['Baseline', '1 example', '2 examples', '3 examples']

    mistral_8x22b = [
        [13/30, 13/30],
        [14/30, 7/30],
        [13/30, 9/30],
        [13/30, 7/30]
    ]

    mistral_8x7b = [
        [13/30],
        [3/30],
        [4/30],
        [9/30]
    ]

    mistral_8x7b_json = [
        [13/30],
        [3/30],
        [5/30],
        [9/30]
    ]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set the width of each bar and the positions of the bars
    width = 0.2
    x = np.arange(len(conditions))

    # Create the bars for each model
    ax.bar(x - width, [d[0] for d in mistral_8x22b], width, label='Mistral 8x22B (PAIR judge)', color='blue')
    ax.bar(x, [d[1] if len(d) > 1 else 0 for d in mistral_8x22b], width, label='Mistral 8x22B (StrongReject judge)', color='lightblue')
    ax.bar(x + width, [d[0] for d in mistral_8x7b], width, label='Mistral 8x7B', color='green')
    ax.bar(x + 2*width, [d[0] for d in mistral_8x7b_json], width, label='Mistral 8x7B (with json parsing)', color='lightgreen')

    # Customize the plot
    ax.set_ylabel('Score')
    ax.set_title('HarmBench Performance Comparison (30 examples)')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(conditions)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Add value labels on top of each bar
    for i, model_data in enumerate([mistral_8x22b, mistral_8x22b, mistral_8x7b, mistral_8x7b_json]):
        for j, value in enumerate(model_data):
            if len(value) > 1:  # For Mistral 8x22B with two judges
                ax.text(x[j] + (i-1.5)*width, value[0], f'{value[0]:.2f}', ha='center', va='bottom')
                ax.text(x[j] + (i-0.5)*width, value[1], f'{value[1]:.2f}', ha='center', va='bottom')
            else:
                ax.text(x[j] + (i-0.5)*width, value[0], f'{value[0]:.2f}', ha='center', va='bottom')

    # Adjust the layout and display the plot
    plt.tight_layout()
    plt.savefig("plots/plot_random_adaptive_results.png")