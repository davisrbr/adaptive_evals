from typing import Optional, List, Dict, Any
import os
import hashlib
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import argparse
from inspect_ai.log import read_eval_log

def filter_top_samples_legal(log_data) -> list:
    """
    Filter and return only the incorrectly answered samples from a 'legal' dataset log.
    For legal datasets, incorrect answers are determined by the score in metadata.
    
    Args:
        log_data: A log object containing a .samples list.

    Returns:
        A filtered list of samples incorrectly answered by the model.
    """
    if not hasattr(log_data, "samples"):
        return []

    filtered = []
    for sample in log_data.samples:
        # For legal datasets, a non-zero score typically indicates an incorrect answer
        # This matches what we see in solver_adaptive_legal_refactor.py where scores
        # are stored directly in metadata
        score = sample.metadata.get("score", 0)
        if score != 0:
            filtered.append(sample)

    return filtered


def filter_top_samples_truthfulqa(log_data) -> list:
    """
    Filter and return only the incorrectly answered samples from a 'truthfulqa' dataset log.
    For TruthfulQA, incorrect answers are determined by the score in metadata.

    Args:
        log_data: A log object containing a .samples list.

    Returns:
        A filtered list of samples incorrectly answered by the model.
    """
    if not hasattr(log_data, "samples"):
        return []

    filtered = []
    for sample in log_data.samples:
        # For TruthfulQA, similar to legal, a non-zero score typically indicates
        # an incorrect answer based on the scoring in the solver
        score = sample.metadata.get("score", 0)
        if score != 0:
            filtered.append(sample)

    return filtered



def get_best_prompt_legal(sample):
    """
    Retrieve what we'll consider the 'best' prompt/response pair for a legal sample.
    In this placeholder, we pick the highest iteration number from sample.store,
    and treat that as the best iteration.

    Args:
        sample: A single log sample object containing a .store with iteration data.

    Returns:
        (best_prompt, best_response) or (None, None) if not found.
    """
    iteration_keys = [
        k for k in sample.store.keys()
        if k.startswith("iteration_") and k.endswith("_prompt")
    ]
    if not iteration_keys:
        return None, None

    # Pick the iteration with the largest iteration number
    iteration_nums = sorted(
        [int(k.split("_")[1]) for k in iteration_keys], reverse=True
    )
    best_iter = iteration_nums[0]
    best_prompt = sample.store.get(f"iteration_{best_iter}_prompt", None)
    best_response = sample.store.get(f"iteration_{best_iter}_response", None)

    return best_prompt, best_response


def get_best_prompt_truthfulqa(sample):
    """
    Retrieve the 'best' prompt/response pair for a TruthfulQA sample.
    Similar logic as legal: pick the highest iteration number from sample.store
    as the best iteration.

    Args:
        sample: A single log sample object containing a .store with iteration data.

    Returns:
        (best_prompt, best_response) or (None, None) if not found.
    """
    iteration_keys = [
        k for k in sample.store.keys()
        if k.startswith("iteration_") and k.endswith("_prompt")
    ]
    if not iteration_keys:
        return None, None

    iteration_nums = sorted(
        [int(k.split("_")[1]) for k in iteration_keys], reverse=True
    )
    best_iter = iteration_nums[0]
    best_prompt = sample.store.get(f"iteration_{best_iter}_prompt", None)
    best_response = sample.store.get(f"iteration_{best_iter}_response", None)

    return best_prompt, best_response


def load_legal_data(
    log_data: Any,
    include_intermediate: bool = True
) -> List[Dict[str, Any]]:
    """
    Load and parse legal samples from a log-like object.

    Args:
        log_data: Raw legal dataset object or log (format may differ from other datasets).
        include_intermediate: If True, include intermediate prompts/responses, else only best.

    Returns:
        A list of dictionaries, each containing fields:
          - "prompt"
          - "response"
          - "color"
          - "score"
          - "name"
          - "source"
          - "iteration"
    """
    parsed_samples = []
    if not hasattr(log_data, "samples"):
        return parsed_samples

    samples = log_data.samples
    if not include_intermediate:
        # Use only incorrect samples for top samples
        samples = filter_top_samples_legal(log_data)

    for sample in samples:
        # Retrieve the best iteration or all iterations
        if include_intermediate:
            iteration_keys = [
                k for k in sample.store.keys()
                if k.startswith("iteration_") and k.endswith("_prompt")
            ]
            iteration_keys = sorted(iteration_keys, key=lambda x: int(x.split("_")[1]))
            prompts_to_process = []
            for key in iteration_keys:
                iteration_num = int(key.split("_")[1])
                prompt = sample.store[key]
                response_key = f"iteration_{iteration_num}_response"
                if response_key in sample.store:
                    response = sample.store[response_key]
                    prompts_to_process.append((prompt, response, iteration_num))
        else:
            # Get best prompt based on highest iteration number
            best_prompt, best_response = get_best_prompt_legal(sample)
            if best_prompt and best_response:
                prompts_to_process = [(best_prompt, best_response, "best")]
            else:
                prompts_to_process = []

        for prompt, response, iteration in prompts_to_process:
            # Extract metadata from the sample
            # Legal tasks often have a task_name in the metadata
            task_name = sample.metadata.get("task_name", "Unknown")
            
            # Get score - in legal dataset it's typically in metadata directly
            score_val = sample.metadata.get("score", 0)
            
            # For color categorization, try to get something meaningful from the sample
            # This could be the task_name or another attribute
            color_val = task_name
            
            # Get sample identifier
            name_val = sample.metadata.get("id", f"Sample-{id(sample)}")
            
            # Source info
            source_val = "Legal dataset"
            
            # Format iteration
            iter_val = f"Iteration {iteration}" if isinstance(iteration, int) else iteration

            parsed_samples.append({
                "prompt": prompt,
                "response": response,
                "color": color_val,
                "score": score_val,
                "name": name_val,
                "source": source_val,
                "iteration": iter_val
            })

    return parsed_samples


def load_truthfulqa_data(
    log_data: Any,
    include_intermediate: bool = True
) -> List[Dict[str, Any]]:
    """
    Load and parse TruthfulQA samples from a log-like object.

    Args:
        log_data: Raw TruthfulQA dataset object or log (format may differ from other datasets).
        include_intermediate: If True, include intermediate prompts/responses, else only best.

    Returns:
        A list of dictionaries, each containing fields:
          - "prompt"
          - "response"
          - "color"
          - "score"
          - "name"
          - "source"
          - "iteration"
    """
    parsed_samples = []
    if not hasattr(log_data, "samples"):
        return parsed_samples

    samples = log_data.samples
    if not include_intermediate:
        # Use only incorrect samples for top samples
        samples = filter_top_samples_truthfulqa(log_data)

    for sample in samples:
        # Retrieve either best iteration or all iterations
        if include_intermediate:
            iteration_keys = [
                k for k in sample.store.keys()
                if k.startswith("iteration_") and k.endswith("_prompt")
            ]
            iteration_keys = sorted(iteration_keys, key=lambda x: int(x.split("_")[1]))
            prompts_to_process = []
            for key in iteration_keys:
                iteration_num = int(key.split("_")[1])
                prompt = sample.store[key]
                response_key = f"iteration_{iteration_num}_response"
                if response_key in sample.store:
                    response = sample.store[response_key]
                    prompts_to_process.append((prompt, response, iteration_num))
        else:
            # Get best prompt based on highest iteration number
            best_prompt, best_response = get_best_prompt_truthfulqa(sample)
            if best_prompt and best_response:
                prompts_to_process = [(best_prompt, best_response, "best")]
            else:
                prompts_to_process = []

        for prompt, response, iteration in prompts_to_process:
            # Extract metadata from the sample
            # For TruthfulQA, try to get question category if available
            category = sample.metadata.get("category", "Unknown")
            
            # Get score from metadata
            score_val = sample.metadata.get("score", 0)
            
            # Use category for color
            color_val = category
            
            # Get sample identifier
            name_val = sample.metadata.get("id", f"Sample-{id(sample)}")
            
            # Source info
            source_val = "TruthfulQA dataset"
            
            # Format iteration
            iter_val = f"Iteration {iteration}" if isinstance(iteration, int) else iteration

            parsed_samples.append({
                "prompt": prompt,
                "response": response,
                "color": color_val,
                "score": score_val,
                "name": name_val,
                "source": source_val,
                "iteration": iter_val
            })

    return parsed_samples


def prepare_embedding_data_for_dataset(
    dataset_type: str,
    log_data: Any,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    color_by: str = "color",
    prompt_only: bool = False,
    response_only: bool = False,
    secondary_log_data: Any = None,
    primary_label: str = "Dataset A",
    secondary_label: str = "Dataset B",
    include_intermediate: bool = True,
    cache_dir: str = "multidataset_cache",
    force_recompute: bool = False
) -> pd.DataFrame:
    """
    Prepare text embeddings and compute t-SNE for a given dataset type.

    Args:
        dataset_type: Either "legal" or "truthfulqa" (or any other recognized dataset).
        log_data: Primary data object or log.
        model_name: SentenceTransformers model to use for embeddings.
        color_by: Which dictionary field to use for color grouping, defaults to "color".
        prompt_only: If True, embed only the prompt text.
        response_only: If True, embed only the response text.
        secondary_log_data: Optional second data object to compare.
        primary_label: Label for the primary data.
        secondary_label: Label for the secondary data.
        include_intermediate: If True, include intermediate samples.
        cache_dir: Where to cache processed data and embeddings.
        force_recompute: If True, skip caches and recompute embeddings.
    Returns:
        A DataFrame with t-SNE coordinates and metadata for plotting.
    """
    if prompt_only and response_only:
        raise ValueError("Cannot set both prompt_only and response_only to True")

    os.makedirs(cache_dir, exist_ok=True)

    # Pick the appropriate data loader
    if dataset_type.lower() == "legal":
        primary_samples = load_legal_data(log_data, include_intermediate=include_intermediate)
        if secondary_log_data is not None:
            secondary_samples = load_legal_data(secondary_log_data, include_intermediate=include_intermediate)
        else:
            secondary_samples = []
    elif dataset_type.lower() == "truthfulqa":
        primary_samples = load_truthfulqa_data(log_data, include_intermediate=include_intermediate)
        if secondary_log_data is not None:
            secondary_samples = load_truthfulqa_data(secondary_log_data, include_intermediate=include_intermediate)
        else:
            secondary_samples = []
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # Combine data for both primary and secondary
    for sample in primary_samples:
        sample["source"] = primary_label
    for sample in secondary_samples:
        sample["source"] = secondary_label

    all_samples = primary_samples + secondary_samples
    if not all_samples:
        print("No samples found in logs.")
        return pd.DataFrame()

    # Build a cache key from relevant parameters
    cache_key_parts = [
        dataset_type,
        primary_label,
        secondary_label,
        "prompt_only" if prompt_only else ("response_only" if response_only else "full"),
        "intermediate" if include_intermediate else "best_only"
    ]
    data_cache_key = hashlib.md5("_".join(cache_key_parts).encode()).hexdigest()
    data_cache_file = os.path.join(cache_dir, f"{data_cache_key}_data.pkl")
    embedding_cache_file = os.path.join(cache_dir, f"{data_cache_key}_embeddings.npz")

    tsne_cache_key_parts = cache_key_parts + [model_name, color_by, "tsne"]
    tsne_cache_key = hashlib.md5("_".join(tsne_cache_key_parts).encode()).hexdigest()
    tsne_cache_file = os.path.join(cache_dir, f"{tsne_cache_key}_tsne_cache.csv")

    # If we have a t-SNE cache and not forcing recompute, load and return it
    if not force_recompute and os.path.isfile(tsne_cache_file):
        print(f"Loading t-SNE data from cache: {tsne_cache_file}")
        return pd.read_csv(tsne_cache_file)

    # Attempt to load cached data
    all_data = None
    if not force_recompute and os.path.isfile(data_cache_file):
        try:
            with open(data_cache_file, "rb") as f:
                all_data = pickle.load(f)
            print(f"Loaded cached sample data with {len(all_data['raw_texts'])} entries.")
        except Exception as e:
            print(f"Error loading cached sample data: {e}. Reprocessing logs...")
            all_data = None

    # If still no data, process logs to create our structure
    if all_data is None:
        print("Processing samples for embedding...")
        all_data = {
            "texts": [],
            "raw_texts": [],
            "color_values": [],
            "scores": [],
            "names": [],
            "sources": [],
            "iterations": []
        }

        for entry in all_samples:
            prompt = entry["prompt"]
            response = entry["response"]

            # Build the hover text
            hover_text_parts = []
            if not response_only:
                hover_text_parts.append(prompt.replace("\n", "<br>"))
            if not prompt_only:
                if not response_only:
                    hover_text_parts.append("<br><b>Response:</b><br>")
                hover_text_parts.append(response.replace("\n", "<br>"))
            hover_text = "<br>".join(hover_text_parts)

            all_data["texts"].append(hover_text)

            # Raw text for embedding
            if prompt_only:
                raw_text = prompt
            elif response_only:
                raw_text = response
            else:
                raw_text = prompt + "\n\n" + response
            all_data["raw_texts"].append(raw_text)

            all_data["color_values"].append(entry.get(color_by, "Unknown"))
            all_data["scores"].append(entry["score"])
            all_data["names"].append(entry["name"])
            all_data["sources"].append(entry["source"])
            all_data["iterations"].append(entry["iteration"])

        # Cache processed sample data
        if all_data["raw_texts"]:
            with open(data_cache_file, "wb") as f:
                pickle.dump(all_data, f)
        else:
            print("No valid data to embed. Returning empty DataFrame.")
            return pd.DataFrame()

    # Check if we have cached embeddings
    embeddings = None
    if not force_recompute and os.path.isfile(embedding_cache_file):
        try:
            cached_data = np.load(embedding_cache_file, allow_pickle=True)
            embeddings = cached_data["embeddings"]
            if len(embeddings) != len(all_data["raw_texts"]):
                print("Mismatch between cached embeddings count and current data. Recomputing...")
                embeddings = None
        except Exception as e:
            print(f"Error loading cached embeddings: {e}. Recomputing...")
            embeddings = None

    # Compute embeddings if needed
    if embeddings is None:
        print(f"Computing embeddings for {len(all_data['raw_texts'])} entries using {model_name}...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(all_data["raw_texts"], convert_to_tensor=False, show_progress_bar=True)
        np.savez_compressed(embedding_cache_file, embeddings=embeddings)

    # Finally perform t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(embeddings) - 1),
        random_state=42,
        n_iter=1000,
        learning_rate="auto",
        init="pca"
    )
    coords = tsne.fit_transform(embeddings)

    # Create a DataFrame
    plot_df = pd.DataFrame({
        "TSNE1": coords[:, 0],
        "TSNE2": coords[:, 1],
        "Color": all_data["color_values"],
        "Score": all_data["scores"],
        "Name": all_data["names"],
        "Source": all_data["sources"],
        "Iteration": all_data["iterations"],
        "Text": all_data["texts"]
    })

    # Cache the results
    plot_df.to_csv(tsne_cache_file, index=False)
    return plot_df


def plot_embedding_data_for_dataset(
    plot_df: pd.DataFrame,
    color_by: str = "Color",
    output_dir: str = "multidataset_plots",
    title: str = "Multidataset t-SNE",
) -> None:
    """
    Create an interactive plot from the processed embedding data.

    Args:
        plot_df: DataFrame with columns TSNE1, TSNE2, Color, Score, Name, Source, Iteration, Text
        color_by: Column name in plot_df to color data by (usually "Color").
        output_dir: Directory to save plots.
        title: Plot title.
    """
    if plot_df.empty:
        print("Empty DataFrame. Nothing to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Determine if color_by is categorical
    is_categorical = isinstance(plot_df[color_by].iloc[0], str)

    fig = go.Figure()
    fig.update_layout(title=title)

    if is_categorical:
        categories = plot_df[color_by].unique()
        # Optional: define or generate a color map
        # For simplicity, just do one trace per category
        for cat in categories:
            subset_df = plot_df[plot_df[color_by] == cat]
            sizes = [max(10, 15 * s) for s in subset_df["Score"]]
            fig.add_trace(go.Scatter(
                x=subset_df["TSNE1"],
                y=subset_df["TSNE2"],
                mode="markers",
                marker=dict(size=sizes),
                text=subset_df["Text"],
                hoverinfo="text",
                name=str(cat),
            ))
    else:
        # If color_by is continuous, use a single scatter with a colorscale
        sizes = [max(10, 15 * s) for s in plot_df["Score"]]
        fig.add_trace(go.Scatter(
            x=plot_df["TSNE1"],
            y=plot_df["TSNE2"],
            mode="markers",
            marker=dict(
                size=sizes,
                color=plot_df[color_by],
                colorscale="Viridis",
                showscale=True
            ),
            text=plot_df["Text"],
            hoverinfo="text",
            name=color_by,
        ))

    fig.update_layout(
        xaxis_title="TSNE1",
        yaxis_title="TSNE2",
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor="right",
            borderwidth=1,
            bgcolor="rgba(255, 255, 255, 0.5)"
        )
    )

    # Show the plot
    fig.show()

    # Save outputs
    html_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.html")
    fig.write_html(html_path)
    print(f"Interactive HTML plot saved to {html_path}")

    try:
        pdf_path = os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.pdf")
        fig.write_image(pdf_path, width=900, height=700)
        print(f"Static PDF image saved to {pdf_path}")
    except Exception as e:
        print(f"Failed to save PDF; install kaleido if needed: {e}")


def embed_and_plot_dataset(
    dataset_type: str,
    log_data: Any,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    color_by: str = "Color",
    prompt_only: bool = False,
    response_only: bool = False,
    secondary_log_data: Any = None,
    primary_label: str = "Dataset A",
    secondary_label: str = "Dataset B",
    include_intermediate: bool = True,
    cache_dir: str = "multidataset_cache",
    output_dir: str = "multidataset_plots",
    force_recompute: bool = False,
    plot_title: str = "Multidataset t-SNE"
) -> pd.DataFrame:
    """
    Convenience function that prepares embedding data for 'legal' or 'truthfulqa' dataset,
    then calls the plotting routine.

    Args:
        dataset_type: "legal" or "truthfulqa".
        log_data: Primary data object or log.
        model_name: Name of the SentenceTransformers model.
        color_by: Which field to color by (default "Color").
        prompt_only: If True, embed only prompt text.
        response_only: If True, embed only response text.
        secondary_log_data: Optional secondary data object/log for comparison.
        primary_label: Label for primary data.
        secondary_label: Label for secondary data.
        include_intermediate: Include intermediate steps.
        cache_dir: Where to store caches.
        output_dir: Where to store output plots.
        force_recompute: Skip caches if True.
        plot_title: Title of the plot.

    Returns:
        pd.DataFrame with t-SNE coordinates, color, text, etc.
    """
    plot_df = prepare_embedding_data_for_dataset(
        dataset_type=dataset_type,
        log_data=log_data,
        model_name=model_name,
        color_by=color_by,
        prompt_only=prompt_only,
        response_only=response_only,
        secondary_log_data=secondary_log_data,
        primary_label=primary_label,
        secondary_label=secondary_label,
        include_intermediate=include_intermediate,
        cache_dir=cache_dir,
        force_recompute=force_recompute,
    )

    if not plot_df.empty:
        plot_embedding_data_for_dataset(
            plot_df=plot_df,
            color_by="Color" if color_by == "Color" else color_by,
            output_dir=output_dir,
            title=plot_title
        )

    return plot_df

def main():
    parser = argparse.ArgumentParser(description="Generate t-SNE visualizations for adaptive datasets")
    parser.add_argument("--dataset-type", type=str, required=True, choices=["legal", "truthfulqa"], 
                        help="Type of dataset to process")
    parser.add_argument("--log-path", type=str, required=True, 
                        help="Path to the primary log file")
    parser.add_argument("--secondary-log-path", type=str, default=None, 
                        help="Optional path to secondary log file for comparison")
    parser.add_argument("--model-name", type=str, 
                        default="sentence-transformers/all-mpnet-base-v2",
                        help="SentenceTransformer model to use for embeddings")
    parser.add_argument("--color-by", type=str, default="color",
                        help="Which field to use for coloring points")
    parser.add_argument("--prompt-only", action="store_true", 
                        help="Only use prompts for embeddings")
    parser.add_argument("--response-only", action="store_true", 
                        help="Only use responses for embeddings")
    parser.add_argument("--primary-label", type=str, default="Primary Dataset",
                        help="Label for primary dataset")
    parser.add_argument("--secondary-label", type=str, default="Secondary Dataset",
                        help="Label for secondary dataset (if provided)")
    parser.add_argument("--no-intermediate", action="store_true",
                        help="Exclude intermediate iterations, use only best")
    parser.add_argument("--cache-dir", type=str, default="multidataset_cache",
                        help="Directory for caching embeddings")
    parser.add_argument("--output-dir", type=str, default="multidataset_plots",
                        help="Directory for saving output plots")
    parser.add_argument("--force-recompute", action="store_true",
                        help="Force recomputation of embeddings and t-SNE")
    parser.add_argument("--plot-title", type=str, default=None,
                        help="Title for the plot (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the log files
    print(f"Loading primary log from {args.log_path}")
    primary_log = read_eval_log(args.log_path)
    
    secondary_log = None
    if args.secondary_log_path:
        print(f"Loading secondary log from {args.secondary_log_path}")
        secondary_log = read_eval_log(args.secondary_log_path)
    
    # Generate default plot title if not provided
    if not args.plot_title:
        dataset_name = "Legal Benchmark" if args.dataset_type == "legal" else "TruthfulQA"
        embedding_type = "Prompt-Only" if args.prompt_only else ("Response-Only" if args.response_only else "Full")
        args.plot_title = f"{dataset_name} {embedding_type} Embeddings"
    
    # Run the embedding and plotting
    print(f"Processing {args.dataset_type} dataset...")
    plot_df = embed_and_plot_dataset(
        dataset_type=args.dataset_type,
        log_data=primary_log,
        model_name=args.model_name,
        color_by=args.color_by,
        prompt_only=args.prompt_only,
        response_only=args.response_only,
        secondary_log_data=secondary_log,
        primary_label=args.primary_label,
        secondary_label=args.secondary_label,
        include_intermediate=not args.no_intermediate,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        force_recompute=args.force_recompute,
        plot_title=args.plot_title
    )
    
    print(f"Processed {len(plot_df) if plot_df is not None else 0} data points")
    print(f"Visualization saved to {args.output_dir}")

if __name__ == "__main__":
    main()