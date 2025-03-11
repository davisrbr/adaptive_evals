#!/usr/bin/env python
import os
import glob
import pandas as pd
import click
from typing import List, Optional, Set, Dict, Tuple
from inspect_ai.log import read_eval_log
from data.multdataset_emeddings import embed_and_plot_dataset

def find_experiment_csvs(base_dir: str) -> Dict[str, List[str]]:
    """
    Find all experiment CSV files in the given directory.
    Categorizes them as either 'legal' or 'truthfulqa' based on filename.
    
    Args:
        base_dir: Directory to search for experiment CSVs
        
    Returns:
        Dictionary with keys 'legal' and 'truthfulqa', each containing a list of CSV paths
    """
    result = {"legal": [], "truthfulqa": []}
    
    # Recursively find all CSV files
    csv_files = glob.glob(os.path.join(base_dir, "**", "*.csv"), recursive=True)
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if "legal_experiment" in filename or "_experiment_results_" in filename:
            result["legal"].append(csv_file)
        elif "experiment_results" in filename:
            # Inspect file contents to determine if it's TruthfulQA
            try:
                df = pd.read_csv(csv_file, nrows=1)
                if "score_threshold" in df.columns:  # Specific to TruthfulQA
                    result["truthfulqa"].append(csv_file)
            except (pd.errors.EmptyDataError, KeyError):
                pass  # Skip files that can't be determined
    
    return result

def get_logs_from_csv(csv_path: str) -> Dict[str, Set[str]]:
    """
    Extract all log paths from an experiment CSV file.
    
    Args:
        csv_path: Path to the experiment CSV
    
    Returns:
        Dictionary with keys 'initial', 'adaptive', and 're_eval', 
        each containing a set of log paths
    """
    logs = {
        "initial": set(),
        "adaptive": set(),
        "re_eval": set()
    }
    
    try:
        df = pd.read_csv(csv_path)
        
        # Extract log paths from each column
        if "initial_log_path" in df.columns:
            logs["initial"].update(df["initial_log_path"].dropna().unique())
        
        if "adaptive_log_path" in df.columns:
            logs["adaptive"].update(df["adaptive_log_path"].dropna().unique())
            
        if "re_eval_log_path" in df.columns:
            logs["re_eval"].update(df["re_eval_log_path"].dropna().unique())
    
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
    
    return logs

def visualize_logs(
    dataset_type: str,
    logs: Dict[str, Set[str]],
    model_name: str,
    output_dir: str,
    compare_stages: bool = False,
    prompt_only: bool = False,
    response_only: bool = False,
    force_recompute: bool = False,
) -> List[str]:
    """
    Generate visualizations for all logs of a particular dataset type.
    
    Args:
        dataset_type: 'legal' or 'truthfulqa'
        logs: Dict with keys 'initial', 'adaptive', 're_eval', each containing log paths
        model_name: Embedding model name
        output_dir: Directory to save plots
        compare_stages: If True, create comparison plots between initial/adaptive/re-eval logs
        prompt_only: If True, only use prompts for embeddings
        response_only: If True, only use responses for embeddings
        force_recompute: If True, recompute embeddings even if cache exists
        
    Returns:
        List of paths to generated plot files
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_plots = []
    
    # Create individual visualizations for each log
    for stage, log_paths in logs.items():
        for log_path in log_paths:
            if not os.path.exists(log_path):
                print(f"Log file not found: {log_path}")
                continue
                
            try:
                log_data = read_eval_log(log_path)
                
                # Extract model name from log path
                path_parts = log_path.split('/')
                model_identifier = next((p for p in path_parts if 'gpt' in p.lower() or 
                                       'claude' in p.lower() or 
                                       'llama' in p.lower() or
                                       'deepseek' in p.lower()), 
                                     os.path.basename(os.path.dirname(log_path)))
                
                plot_title = f"{dataset_type.capitalize()} {stage.capitalize()} - {model_identifier}"
                plot_dir = os.path.join(output_dir, dataset_type, stage)
                os.makedirs(plot_dir, exist_ok=True)
                
                plot_df = embed_and_plot_dataset(
                    dataset_type=dataset_type,
                    log_data=log_data,
                    model_name=model_name,
                    color_by="color",
                    prompt_only=prompt_only,
                    response_only=response_only,
                    cache_dir=os.path.join(output_dir, "cache"),
                    output_dir=plot_dir,
                    force_recompute=force_recompute,
                    plot_title=plot_title
                )
                
                # Record the generated plot path
                plot_filename = f"{plot_title.replace(' ', '_').lower()}.html"
                plot_path = os.path.join(plot_dir, plot_filename)
                if os.path.exists(plot_path):
                    generated_plots.append(plot_path)
                    
            except Exception as e:
                print(f"Error processing log {log_path}: {e}")
    
    # Create comparison plots between stages if requested
    if compare_stages:
        # Compare initial vs adaptive
        if logs["initial"] and logs["adaptive"]:
            for initial_log in logs["initial"]:
                for adaptive_log in logs["adaptive"]:
                    try:
                        initial_data = read_eval_log(initial_log)
                        adaptive_data = read_eval_log(adaptive_log)
                        
                        initial_model = os.path.basename(os.path.dirname(initial_log))
                        adaptive_model = os.path.basename(os.path.dirname(adaptive_log))
                        
                        plot_title = f"{dataset_type.capitalize()} Initial vs Adaptive"
                        plot_dir = os.path.join(output_dir, dataset_type, "comparisons")
                        os.makedirs(plot_dir, exist_ok=True)
                        
                        plot_df = embed_and_plot_dataset(
                            dataset_type=dataset_type,
                            log_data=initial_data,
                            model_name=model_name,
                            color_by="color",
                            prompt_only=prompt_only,
                            response_only=response_only,
                            secondary_log_data=adaptive_data,
                            primary_label="Initial Dataset",
                            secondary_label="Adaptive Dataset",
                            cache_dir=os.path.join(output_dir, "cache"),
                            output_dir=plot_dir,
                            force_recompute=force_recompute,
                            plot_title=plot_title
                        )
                        
                        # Record the generated plot path
                        plot_filename = f"{plot_title.replace(' ', '_').lower()}.html"
                        plot_path = os.path.join(plot_dir, plot_filename)
                        if os.path.exists(plot_path):
                            generated_plots.append(plot_path)
                            
                    except Exception as e:
                        print(f"Error creating comparison plot: {e}")
    
    return generated_plots

@click.command()
@click.argument("experiment_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--output-dir", type=str, default="experiment_plots", 
              help="Directory to save visualization plots")
@click.option("--dataset-type", type=click.Choice(["legal", "truthfulqa", "both"]), 
              default="both", help="Type of dataset to visualize")
@click.option("--model-name", type=str, 
              default="sentence-transformers/all-mpnet-base-v2",
              help="SentenceTransformer model to use for embeddings")
@click.option("--prompt-only", is_flag=True, help="Only use prompts for embeddings")
@click.option("--response-only", is_flag=True, help="Only use responses for embeddings")
@click.option("--compare-stages", is_flag=True, 
              help="Create comparison plots between initial/adaptive/re-eval logs")
@click.option("--force-recompute", is_flag=True, 
              help="Force recomputation of embeddings and t-SNE")
def main(
    experiment_dir: str,
    output_dir: str,
    dataset_type: str,
    model_name: str,
    prompt_only: bool,
    response_only: bool,
    compare_stages: bool,
    force_recompute: bool,
):
    """
    Generate t-SNE visualizations for all experiment logs in the given directory.
    
    This script will:
    1. Find all experiment CSV files in the directory
    2. Extract log paths from those CSVs
    3. Generate visualizations for each log
    4. Optionally create comparison plots between stages
    
    EXPERIMENT_DIR: Directory containing experiment CSVs and logs
    """
    print(f"Scanning directory: {experiment_dir}")
    
    # Find experiment CSVs
    experiment_csvs = find_experiment_csvs(experiment_dir)
    
    # Determine which dataset types to process
    dataset_types = []
    if dataset_type == "both" or dataset_type == "legal":
        if experiment_csvs["legal"]:
            dataset_types.append("legal")
    if dataset_type == "both" or dataset_type == "truthfulqa":
        if experiment_csvs["truthfulqa"]:
            dataset_types.append("truthfulqa")
    
    if not dataset_types:
        print(f"No {'legal or truthfulqa' if dataset_type == 'both' else dataset_type} experiment CSVs found in {experiment_dir}")
        return
    
    for dtype in dataset_types:
        print(f"Processing {dtype} experiments...")
        all_logs = {"initial": set(), "adaptive": set(), "re_eval": set()}
        
        # Extract all log paths from experiment CSVs
        for csv_path in experiment_csvs[dtype]:
            print(f"Reading log paths from {csv_path}")
            logs = get_logs_from_csv(csv_path)
            for stage, paths in logs.items():
                all_logs[stage].update(paths)
        
        print(f"Found {sum(len(paths) for paths in all_logs.values())} log files to process")
        
        # Generate visualizations
        generated_plots = visualize_logs(
            dataset_type=dtype,
            logs=all_logs,
            model_name=model_name,
            output_dir=output_dir,
            compare_stages=compare_stages,
            prompt_only=prompt_only,
            response_only=response_only,
            force_recompute=force_recompute
        )
        
        print(f"Generated {len(generated_plots)} plots for {dtype} experiments")
        
        # Print paths to a few plots as examples
        if generated_plots:
            print("Sample plot paths:")
            for plot in generated_plots[:3]:
                print(f"  - {plot}")
            if len(generated_plots) > 3:
                print(f"  - ... and {len(generated_plots) - 3} more")

if __name__ == "__main__":
    main()