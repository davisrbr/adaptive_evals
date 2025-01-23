"""
Generates heatmaps for adaptive_legal experiment results, with optional filters:
  1) Filter by number of epochs.
  2) Filter by whether cot_in_context was used.

For each generator model, a separate heatmap is produced where:
    - X-axis = original_eval_model_name
    - Y-axis = eval_model_name
    - Cell color = aggregated accuracy


Example usage:
    python plots/plot_adaptive_results.py \
        --logs-dir logs/adaptive_legal_test \
        --out-dir plots/adaptive_legal_test \
        --filter-cot True \
        --filter-examples True \
        --model-cache-csv .csv \
        --epochs 50
"""

import os
import json
import argparse
import csv
from typing import Optional, Dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from inspect_ai.log import read_eval_log


def read_eval_cache(cache_csv: str) -> Dict[str, str]:
    """
    Returns a dict of {model_name: log_path} previously saved.
    """
    if not os.path.exists(cache_csv):
        return {}
    output = {}
    with open(cache_csv, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model_name"]
            log_path = row["log_path"]
            output[model_name] = log_path
    return output


def gather_baseline_results(cache_csv: str) -> pd.DataFrame:
    """
    Loads baseline logs from a CSV cache, extracts each model's baseline accuracy,
    and returns them in a DataFrame with columns:
        - model_name
        - use_example
        - use_cot_target
        - use_cot_in_context_attacker
        - baseline_accuracy
    The naming conventions in experiment_runner_legal.py are assumed.
    """
    if not cache_csv or not os.path.isfile(cache_csv):
        print(f"No cache CSV provided or file not found: {cache_csv}")
        return pd.DataFrame()

    cache_data = read_eval_cache(cache_csv)
    records = []
    for model_name, log_path in cache_data.items():
        try:
            log = read_eval_log(log_path)
        except Exception:
            continue

        if log.status != "success" or not hasattr(log, "eval"):
            continue

        evaluation = getattr(log, "eval", None)
        if not evaluation or not hasattr(evaluation, "task_args"):
            continue

        task_args = evaluation.task_args
        use_example = task_args.get("use_example")
        use_cot_target = task_args.get("use_cot")
        use_cot_in_context_attacker = task_args.get("use_cot_in_context_attacker")
        results = getattr(log, "results", None)
        if not results or not hasattr(results, "scores"):
            continue

        baseline_accuracy = None
        for scorer in results.scores:
            # Adjust the scorer name or key to whatever the initial scoring was (e.g., "accuracy").
            if "accuracy" in scorer.metrics:
                baseline_accuracy = scorer.metrics["accuracy"].value
                break

        if baseline_accuracy is None:
            continue

        records.append({
            "model_name": model_name,
            "use_example": use_example,
            "use_cot_target": use_cot_target,
            "use_cot_in_context_attacker": use_cot_in_context_attacker,
            "baseline_accuracy": baseline_accuracy,
        })

    return pd.DataFrame(records)


def gather_adaptive_results(log_dir: str) -> pd.DataFrame:
    """
    Recursively searches log_dir for Inspect JSON logs from adaptive_legal runs,
    extracts relevant fields, and returns them in a DataFrame. We look for
    completed_samples instead of epochs, and parse accuracy from
    'adaptive_legal_scorer_judged'.

    Expected columns:
        - original_eval_model_name
        - generator_model_name
        - eval_model_name
        - accuracy
        - cot_in_context (bool)
        - use_example (bool)
        - completed_samples
        - n_positive_samples
        - n_negative_samples
        - log_path
    """
    records = []
    for root, _, files in os.walk(log_dir):
        for file in files:
            if not (file.endswith(".json") or file.endswith(".eval")):
                continue

            full_path = os.path.join(root, file)
            try:
                log = read_eval_log(full_path)
            except Exception:
                # If parse fails, skip
                continue

            evaluation = getattr(log, "eval", None)
            if not evaluation:
                print(f"Warning: No evaluation object found in {full_path}")
                continue

            task_args = getattr(evaluation, "task_args", {})
            if not task_args:
                print(f"Warning: No task arguments found in {full_path}")
                continue

            original_eval_model_name = task_args.get("original_eval_model_name")
            generator_model_name = task_args.get("generator_model_name")
            eval_model_name = task_args.get("eval_model_name")
            n_pos_samples = task_args.get("n_positive_samples")
            n_neg_samples = task_args.get("n_negative_samples")
            cot_in_context = task_args.get("cot_in_context", None)
            use_example = task_args.get("use_example", None)

            completed_samples = None
            results = getattr(log, "results", None)
            if results is not None:
                completed_samples = getattr(results, "completed_samples", None)

            # Extract accuracy from 'adaptive_legal_scorer_judged'
            accuracy = None
            if results and hasattr(results, "scores"):
                scorer = next((s for s in results.scores if s.name == "adaptive_legal_scorer_judged"), None)
                if scorer and "accuracy_judged" in scorer.metrics:
                    accuracy_metric = scorer.metrics["accuracy_judged"]
                    accuracy = accuracy_metric.value

            if not (original_eval_model_name and generator_model_name and eval_model_name):
                continue

            records.append({
                "original_eval_model_name": original_eval_model_name,
                "generator_model_name": generator_model_name,
                "eval_model_name": eval_model_name,
                "n_positive_samples": n_pos_samples,
                "n_negative_samples": n_neg_samples,
                "cot_in_context": cot_in_context,
                "use_example": use_example,
                "completed_samples": completed_samples,
                "accuracy": accuracy,
                "log_path": full_path,
            })

    return pd.DataFrame(records)


def plot_adaptive_heatmaps(
    results_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    out_dir: str,
    filter_cot: Optional[bool] = None,
    filter_examples: Optional[bool] = None,
    filter_cot_in_context: Optional[bool] = None,
    epochs: Optional[int] = None
) -> None:
    """
    Creates one heatmap per generator_model_name with:
    - X-axis: "Baseline" + original_eval_model_name
    - Y-axis: eval_model_name
    - Cell: Mean accuracy (or baseline accuracy for baseline column)
    """
    if results_df.empty:
        print("No results to plot.")
        return

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Construct filter tag for filename
    filter_tag_parts = []
    if filter_cot is not None:
        filter_tag_parts.append(f"cot_{filter_cot}")
    if filter_examples is not None:
        filter_tag_parts.append(f"ex_{filter_examples}")
    if filter_cot_in_context is not None:
        filter_tag_parts.append(f"cotInContext_{filter_cot_in_context}")
    if epochs is not None:
        filter_tag_parts.append(f"epochs_{epochs}")

    filter_tag = ""
    if filter_tag_parts:
        filter_tag = "_" + "_".join(filter_tag_parts)

    generator_models = results_df["generator_model_name"].unique()

    for gen_model in generator_models:
        subset = results_df[results_df["generator_model_name"] == gen_model]
        if subset.empty:
            continue

        # Create pivot table for adaptive results
        pivot_df = (
            subset
            .groupby(["eval_model_name", "original_eval_model_name"], as_index=False)
            .agg({"accuracy": "mean"})
        )
        pivot_table = pivot_df.pivot(
            index="eval_model_name",
            columns="original_eval_model_name",
            values="accuracy",
        )

        # Add baseline accuracy column if we have baseline data
        if not baseline_df.empty:
            baseline_accuracies = {}
            for model in pivot_table.index:
                model_baseline = baseline_df[baseline_df["model_name"] == model]
                if not model_baseline.empty:
                    baseline_accuracies[model] = model_baseline["baseline_accuracy"].iloc[0]
                else:
                    baseline_accuracies[model] = None
            
            # Create new DataFrame with baseline as first column
            baseline_series = pd.Series(baseline_accuracies, name="Baseline")
            pivot_table = pd.concat([baseline_series, pivot_table], axis=1)

        plt.figure(figsize=(12, 6))  # Wider to accommodate rotated labels
        ax = sns.heatmap(
            pivot_table,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            cbar=True,
            vmin=0,
            vmax=1,
        )
        plt.title(f"Adaptive Accuracy\n(Generator: {gen_model})", pad=20, size=14, weight='bold')
        ax.set_xlabel("Adaptive evaluation based on this model (or baseline)")
        ax.set_ylabel("Adaptively Evaluated Model")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Create filename with all filters
        model_name = gen_model.replace('/', '_')
        out_path = os.path.join(out_dir, f"heatmap_{model_name}{filter_tag}.png")
        print(f"Saving plot to: {out_path}")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()


def infer_cache_csv_name(filter_cot_in_context: Optional[bool], filter_examples: Optional[bool], filter_cot: Optional[bool]) -> Optional[str]:
    """
    Infers the cache CSV name using the same convention as experiment_runner_legal.py.
    Returns None if any of the required filters are not set.
    """
    if filter_examples is not None and filter_cot is not None and filter_cot_in_context is not None:
        return f"initial_eval_cache_{filter_examples}_{filter_cot}_{filter_cot_in_context}.csv"
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot adaptive_legal results (including baseline) as heatmaps.")
    parser.add_argument("--logs-dir", type=str, default="logs/", help="Directory containing Inspect .json logs.")
    parser.add_argument("--out-dir", type=str, default="plots/adaptive_legal", help="Directory to store generated heatmaps.")
    parser.add_argument("--filter-cot", action="store_true", default=False,
                        help="If set, filter logs by cot_in_context=True")
    parser.add_argument("--filter-examples", action="store_true", default=False,
                        help="If set, filter logs by use_example=True")
    parser.add_argument("--filter-cot-in-context", action="store_true", default=False,
                        help="If set, filter logs by use_cot_in_context_attacker=True")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Optional. If set, only plot logs with this many completed_samples.")
    parser.add_argument("--model-cache-csv", type=str, default=None,
                        help="Path to CSV used to store initial baseline logs, for plotting baseline accuracy.")

    args = parser.parse_args()

    # If model_cache_csv is not provided, try to infer it from the filters
    if not args.model_cache_csv:
        inferred_cache = infer_cache_csv_name(
            filter_cot_in_context=args.filter_cot_in_context,
            filter_examples=args.filter_examples,
            filter_cot=args.filter_cot
        )
        if inferred_cache:
            args.model_cache_csv = inferred_cache
            print(f"Using inferred cache CSV: {inferred_cache}")
        else:
            print("Warning: Could not infer cache CSV name. Please provide all filters or specify --model-cache-csv directly.")

    # Gather adaptive results
    df = gather_adaptive_results(args.logs_dir)
    print(f"Found {len(df)} total adaptive_legal runs in {args.logs_dir}.")

    # Filter by cot_in_context
    if args.filter_cot is not None:
        if "cot_in_context" in df.columns:
            initial_count = len(df)
            df = df[df["cot_in_context"] == args.filter_cot]
            print(f"Filtered by cot_in_context={args.filter_cot}, {initial_count} -> {len(df)} logs.")
        else:
            print("Warning: 'cot_in_context' column not found in DataFrame, skipping filter.")

    # Filter by use_example
    if args.filter_examples is not None:
        if "use_example" in df.columns:
            initial_count = len(df)
            df = df[df["use_example"] == args.filter_examples]
            print(f"Filtered by use_example={args.filter_examples}, {initial_count} -> {len(df)} logs.")
        else:
            print("Warning: 'use_example' column not found in DataFrame, skipping filter.")

    # Filter by use_cot_in_context_attacker
    if args.filter_cot_in_context is not None:
        col_name = "use_cot_in_context_attacker"
        if col_name in df.columns:
            initial_count = len(df)
            df = df[df[col_name] == args.filter_cot_in_context]
            print(f"Filtered by {col_name}={args.filter_cot_in_context}, {initial_count} -> {len(df)} logs.")
        else:
            print(f"Warning: '{col_name}' column not found in DataFrame, skipping filter.")

    # Filter by completed_samples (using --epochs)
    if args.epochs is not None:
        if "completed_samples" in df.columns:
            initial_count = len(df)
            df = df[df["completed_samples"] == args.epochs]
            print(f"Filtered by completed_samples={args.epochs}, {initial_count} -> {len(df)} logs.")
        else:
            print("Warning: 'completed_samples' column not found in DataFrame, skipping filter.")

    # Load baseline results and plot combined heatmaps
    baseline_df = pd.DataFrame()
    if args.model_cache_csv:
        baseline_df = gather_baseline_results(args.model_cache_csv)
    
    plot_adaptive_heatmaps(
        results_df=df,
        baseline_df=baseline_df,
        out_dir=args.out_dir,
        filter_cot=args.filter_cot,
        filter_examples=args.filter_examples,
        filter_cot_in_context=args.filter_cot_in_context,
        epochs=args.epochs,
    )

    print(f"Plots saved to {args.out_dir}") 