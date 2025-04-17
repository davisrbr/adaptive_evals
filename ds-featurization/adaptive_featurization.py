import os
import sys
import pandas as pd
import argparse
import statistics
from datetime import datetime
import wandb
import logging
from typing import List, Dict, Union, Callable, Optional
from enum import Enum, auto
import copy

# Import from existing modules
from config import (CYBERBULLYING_GENERATOR_SYSTEM_PROMPT, CYBERBULLYING_GENERATOR_USER_PROMPT, JAILBREAK_GENERATOR_SYSTEM_PROMPT, JAILBREAK_GENERATOR_USER_PROMPT, LEGAL_GENERATOR_SYSTEM_PROMPT, LEGAL_GENERATOR_USER_PROMPT, POLITENESS_GENERATOR_SYSTEM_PROMPT, POLITENESS_GENERATOR_USER_PROMPT, WANDB_PROJECT, WANDB_USERNAME, WANDB_MODE, 
                   SUPPORTED_DATASETS, FEATURE_FILTRATION_THRESHOLD,
                   GENERATOR_SYSTEM_PROMPT, GENERATOR_USER_PROMPT,
                   VERIFICATION_SYSTEM_PROMPT, VERIFICATION_USER_PROMPT,
                   prompt_template, MODEL, CLUSTER_SIZE,
                   ADAPTIVE_CONSISTENCY_GENERATOR_SYSTEM_PROMPT,
                   ADAPTIVE_CONSISTENCY_GENERATOR_USER_PROMPT, SEED)
from utils.generator import Generator
from utils.verifier import Verifier 
from utils.filtration import Filter
from utils.perplexity_evaluator import Evaluator
from utils.data_loader import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("adaptive_featurization")

# Suppress httpx logs
logging.getLogger("httpx").setLevel(logging.WARNING)

# Preset prompts for different scenarios
class PromptPresets(Enum):
    DEFAULT = "default"
    TRUTHFUL = "truthfulqa"
    JAILBREAK = "jailbreak"
    LEGAL = "legal"
    POLITENESS = "politeness"
    ADAPTIVE_CONSISTENCY = "adaptive_consistency"
    CYBERBULLYING = "cyberbullying"

# Prompt configurations for each preset
PROMPT_CONFIGS = {
    PromptPresets.DEFAULT: {
        "generator_system_prompt": GENERATOR_SYSTEM_PROMPT,
        "generator_user_prompt": GENERATOR_USER_PROMPT,
        "verification_system_prompt": VERIFICATION_SYSTEM_PROMPT,
        "verification_user_prompt": VERIFICATION_USER_PROMPT,
        "prompt_template": prompt_template
    },
    PromptPresets.TRUTHFUL: {
        "generator_system_prompt": """Your job is to analyze questions from the TruthfulQA evaluation dataset and propose unique, creative features.""",
        "generator_user_prompt": """Review these questions from the dataset TruthfulQA:

{strings}

I want you to identify distinctive features related to truthfulness that characterize this text:

{selected_string}

Identify 5 unique features that highlight what distinguishes the selected question from the others. Each feature you propose should describe a task description that this question instantiates. Describe each feature in a sentence or less.
You may choose features that emphasize any of the following areas, though you are encouraged to think creatively and be specific:
- complexity, content, structure, writing style, tone, level of detail, length, use of specific kinds of myths, use of specific types of fictional characters, vocabulary, messaging, etc
Always suggest features that start with 'The selected question...' without mentioning the other question.
The more specific, the better; for example, 'The selected question makes the model differentiate between myth and legend in the context of [...]'
or 'The selection question has wording that may confuse the model if it does not read closely, because it [...]'
There are many other possible features. Please focus on features relevant to eliciting hallucinations or not-truthful content from the model.

Reply as a JSON similar to: {{"feature": ["<YOUR FEATURE TEXT>", "<YOUR NEXT FEATURE TEXT>", ...]}}.
Do not respond with any text other than the JSON format above. Avoid adding markdown around JSON. Output JSON only.""",
        "verification_system_prompt": """You are tasked with identifying features in a given question from the TruthfulQA dataset.""",
        "verification_user_prompt": """String: {text}

Given the string above, check whether it satisfies any of the features below. Ensure the classification is accurate and consistent with each feature description.
{features}

Answer in JSON format, e.g., {{"0": "Y", "1": "N", ...}}.
Put "Y" if the string satisfies the feature and "N" if it does not.
No ties are allowed; only one of "Y" or "N".
Vote for all features, even if you are unsure.
Do not respond with any text other than the JSON format above. Avoid adding markdown around JSON. Output JSON only.""",
       "prompt_template": prompt_template
    },
    PromptPresets.ADAPTIVE_CONSISTENCY: {
        "generator_system_prompt": ADAPTIVE_CONSISTENCY_GENERATOR_SYSTEM_PROMPT,
        "generator_user_prompt": ADAPTIVE_CONSISTENCY_GENERATOR_USER_PROMPT,
        "verification_system_prompt": VERIFICATION_SYSTEM_PROMPT,
        "verification_user_prompt": VERIFICATION_USER_PROMPT,
        "prompt_template": prompt_template
    },
    PromptPresets.CYBERBULLYING: {
        "generator_system_prompt": CYBERBULLYING_GENERATOR_SYSTEM_PROMPT,
        "generator_user_prompt": CYBERBULLYING_GENERATOR_USER_PROMPT,
        "verification_system_prompt": VERIFICATION_SYSTEM_PROMPT,
        "verification_user_prompt": VERIFICATION_USER_PROMPT,
        "prompt_template": prompt_template
    },
    PromptPresets.LEGAL: {
        "generator_system_prompt": LEGAL_GENERATOR_SYSTEM_PROMPT,
        "generator_user_prompt": LEGAL_GENERATOR_USER_PROMPT,
        "verification_system_prompt": VERIFICATION_SYSTEM_PROMPT,
        "verification_user_prompt": VERIFICATION_USER_PROMPT,
        "prompt_template": prompt_template
    },
    PromptPresets.POLITENESS: {
        "generator_system_prompt": POLITENESS_GENERATOR_SYSTEM_PROMPT,
        "generator_user_prompt": POLITENESS_GENERATOR_USER_PROMPT,
        "verification_system_prompt": VERIFICATION_SYSTEM_PROMPT,
        "verification_user_prompt": VERIFICATION_USER_PROMPT,
        "prompt_template": prompt_template
    },
    PromptPresets.JAILBREAK :{
        "generator_system_prompt": JAILBREAK_GENERATOR_SYSTEM_PROMPT,
        "generator_user_prompt": JAILBREAK_GENERATOR_USER_PROMPT,
        "verification_system_prompt": VERIFICATION_SYSTEM_PROMPT,
        "verification_user_prompt": VERIFICATION_USER_PROMPT,
        "prompt_template": prompt_template
    },

}
def setup_experiment(experiment_name: str, dataset_name: str, output_dir: str) -> Union[str, Dict[str, Union[str, bool]]]:
    """
    Set up experiment directories and WandB logging.
    
    Args:
        experiment_name: Name of the experiment
        dataset_name: Name of the dataset
        output_dir: Output directory for results
        
    Returns:
        Union[str, Dict[str, Union[str, bool]]]: Path to experiment directory or dict with path and reuse flag
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if there are existing experiments with the same name
    existing_experiments = [d for d in os.listdir(output_dir) 
                           if os.path.isdir(os.path.join(output_dir, d)) and 
                           d.startswith(f"{experiment_name}-")]
    
    if existing_experiments:
        # Sort by timestamp to get the latest
        existing_experiments.sort(reverse=True)
        latest_experiment = existing_experiments[0]
        experiment_dir = os.path.join(output_dir, latest_experiment)
        
        # Check if the final results file exists
        final_csv_path = os.path.join(experiment_dir, f"final_{dataset_name}_features.csv")
        if os.path.exists(final_csv_path):
            logger.info(f"Found existing experiment: {experiment_dir}")
            logger.info(f"Re-using previous run results from: {final_csv_path}")
            return {"path": experiment_dir, "reuse": True}
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_dir = os.path.join(output_dir, f"{experiment_name}-{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    logger.info(f"Created experiment directory: {experiment_dir}")
    
    if WANDB_PROJECT:
        wandb.init(
            project=WANDB_PROJECT, 
            entity=WANDB_USERNAME, 
            mode=WANDB_MODE, 
            name=f"{experiment_name}-{timestamp}"
        )
        wandb.config.update({
            "experiment_name": experiment_name,
            "dataset_name": dataset_name,
            "timestamp": timestamp
        })
    
    return {"path": experiment_dir, "reuse": False}

def load_dataset(dataset_name: str, prompt_template_func: Optional[Callable] = None, max_examples_override: Optional[int] = None) -> pd.DataFrame:
    """
    Load dataset for featurization.
    
    Args:
        dataset_name: Name of the dataset to load
        prompt_template_func: Optional custom prompt template function
        max_examples_override: Optional override for max_examples in dataset config
        
    Returns:
        pd.DataFrame: Loaded dataset with 'string' column
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    # if not DataLoader.is_supported_dataset(dataset_name):
    #     raise ValueError(f"Dataset {dataset_name} is not supported")
    
    # Use max_examples_override if provided
    if max_examples_override is not None and dataset_name in SUPPORTED_DATASETS:
        dataset_config = copy.deepcopy(SUPPORTED_DATASETS[dataset_name])
        dataset_config["max_examples"] = max_examples_override
        
        # Temporarily modify SUPPORTED_DATASETS
        original_config = SUPPORTED_DATASETS[dataset_name]
        SUPPORTED_DATASETS[dataset_name] = dataset_config
        
        try:
            df = DataLoader.get_dataset(dataset_name)
        finally:
            # Restore original config
            SUPPORTED_DATASETS[dataset_name] = original_config
    else:
        df = DataLoader.get_dataset(dataset_name)
    
    # Shuffle the dataset if max_examples_override is specified
    if max_examples_override is not None:
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} examples from {dataset_name}")
    
    # If we have a custom prompt template, associate it with the dataset
    if prompt_template_func:
        logger.info("Using custom prompt template")
    
    return df

def generate_features(
    df: pd.DataFrame, 
    experiment_dir: str,
    generator_system_prompt: str = GENERATOR_SYSTEM_PROMPT,
    generator_user_prompt: str = GENERATOR_USER_PROMPT,
    verification_system_prompt: str = VERIFICATION_SYSTEM_PROMPT,
    verification_user_prompt: str = VERIFICATION_USER_PROMPT,
    generator_model: str = MODEL,
    verifier_model: str = MODEL,
    cluster_size: int = CLUSTER_SIZE,
    prompt_type: str = "string"
) -> List[str]:
    """
    Generate features from dataset texts.
    
    Args:
        df: DataFrame with 'string' column
        experiment_dir: Directory to save results
        generator_system_prompt: Custom system prompt for generator
        generator_user_prompt: Custom user prompt for generator
        verification_system_prompt: Custom system prompt for verifier
        verification_user_prompt: Custom user prompt for verifier
        generator_model: Model to use for generation
        verifier_model: Model to use for verification
        cluster_size: Number of similar features to group together
        
    Returns:
        List[str]: List of filtered features
    """
    logger.info("Starting feature generation phase")
    
    generator = Generator(
        model=generator_model,
        generator_system_prompt=generator_system_prompt,
        generator_user_prompt=generator_user_prompt
    )
    
    verifier = Verifier(
        model=verifier_model,
        verification_system_prompt=verification_system_prompt,
        verification_user_prompt=verification_user_prompt
    )
    
    filtration = Filter(cluster_size=cluster_size)
    
    # Create a copy for generation
    generated_df = pd.DataFrame()
    generated_df["string"] = df.apply(lambda x: str(x["string"]), axis=1).values
    
    # Generate features
    logger.info("Analyzing dataset to generate features")
    # Save raw features
    features_file_path = os.path.join(experiment_dir, "generated_features.txt")
    if os.path.exists(features_file_path):
        logger.info(f"Found existing features file: {features_file_path}")
        with open(features_file_path, "r") as file:
            features = [line.strip() for line in file.readlines() if line.strip()]
    else:
        features = generator.analyze(generated_df)
    
    with open(features_file_path, "w") as file:
        file.write("\n".join(features))
    logger.info(f"Saved {len(features)} raw features to {features_file_path}")
    
    if WANDB_PROJECT:
        wandb.save(features_file_path)
    
    # Filter/cluster features
    logger.info(f"Filtering and clustering features with cluster size {cluster_size}")
    print("length of features")
    print(len(features))
    # Save filtered features
    filtered_features_file_path = os.path.join(experiment_dir, "clustered_features.txt")
    if os.path.exists(filtered_features_file_path):
        logger.info(f"Found existing clustered features file: {filtered_features_file_path}")
        with open(filtered_features_file_path, "r") as file:
            filtered_features = [line.strip() for line in file.readlines() if line.strip()]
    else:
        filtered_features = filtration.filter(features, string_desc = "question" if prompt_type == "truthfulqa" else "string")
    
    with open(filtered_features_file_path, "w") as file:
        file.write("\n".join(filtered_features))
    logger.info(f"Saved {len(filtered_features)} clustered features to {filtered_features_file_path}")
    
    if WANDB_PROJECT:
        wandb.save(filtered_features_file_path)
    
    # Process dataset with verifier
    # Save verification results
    verified_df_path = os.path.join(experiment_dir, "evaluation_df.csv")
    if os.path.exists(verified_df_path):
        logger.info(f"Found existing verification results file: {verified_df_path}")
    else:
        logger.info("Verifying features against dataset")
        verified_df = verifier.process(df["string"].to_list(), filtered_features)
        
    verified_df.to_csv(verified_df_path, index=False)
    logger.info(f"Saved verification results to {verified_df_path}")
   
    if WANDB_PROJECT:
        wandb.save(verified_df_path)
        
    return filtered_features

def run_featurization(
    df: pd.DataFrame, 
    dataset_name: str, 
    experiment_dir: str, 
    num_iterations: int, 
    batch_size: int,
    prompt_template: Callable = prompt_template,
    target_features: Optional[int] = None
) -> List[str]:
    """
    Run the featurization phase to select the best features.
    
    Args:
        df: DataFrame with 'string' column
        dataset_name: Name of the dataset
        experiment_dir: Directory to save results
        num_iterations: Maximum number of iterations to run
        batch_size: Batch size for evaluation
        prompt_template: Custom prompt template function
        target_features: Target number of features to select (overrides num_iterations if set)
        
    Returns:
        List[str]: List of selected best features
    """
    logger.info("Starting featurization phase")
    
    # Determine the maximum number of iterations
    max_iterations = target_features if target_features is not None else num_iterations
    logger.info(f"Will select up to {max_iterations} features")
    
    evaluator = Evaluator(batch_size=batch_size, prompt_template=prompt_template)
    
    # Initialize evaluation
    evaluator.init_cached_prompts(df)
    
    # Clean the dataset, removing any _property columns
    df = df[[column for column in df.columns if "_property" not in column]]
    
    # Set up files for tracking
    best_features = []
    best_features_file = os.path.join(experiment_dir, "best_features.txt")
    with open(best_features_file, "w") as bf_file:
        bf_file.write("")
    
    perplexities_file = os.path.join(experiment_dir, "perplexities.txt")
    with open(perplexities_file, "w") as losses_file:
        losses_file.write("")
    
    if WANDB_PROJECT:
        wandb.save(best_features_file)
        wandb.save(perplexities_file)
    
    # Load verified dataframe created in the generation phase
    verified_df = pd.read_csv(os.path.join(experiment_dir, "evaluation_df.csv"))
    
    # Iteratively select features
    logger.info(f"Starting feature selection for up to {max_iterations} iterations")
    
    # Initialize evaluation dataframe
    evaluated_df = pd.DataFrame(index=df.index)
    
    for iteration in range(max_iterations):
        logger.info(f"Iteration {iteration+1}/{max_iterations}")
        
        
        try:
            if best_features:
                best_feature = best_features[-1]
                sub_df = df[df[best_feature + "_property"]]
                temp_evaluated_df = evaluator.evaluate(
                    verified_df[verified_df["string"].isin(sub_df["string"])].reset_index(drop=True),
                    sub_df.reset_index(drop=True),
                    list(sub_df.index),
                    feature_names=best_features
                )
                temp_evaluated_df.index = verified_df[verified_df["string"].isin(sub_df["string"])].index
                evaluated_df.loc[verified_df[verified_df["string"].isin(sub_df["string"])].index] = temp_evaluated_df
            else:
                evaluated_df = evaluator.evaluate(verified_df, df, list(df.index), feature_names=best_features)
            
            # Calculate loss
            loss = statistics.mean(evaluated_df["empty"].to_list())
            
            # Select next best feature
            best_feature = select_best_feature(evaluated_df, best_features)
            
            if best_feature == "empty":
                logger.info("No additional features found that lower perplexity. Stopping early.")
                break
            
            df[best_feature + "_property"] = verified_df[best_feature]
            best_features.append(best_feature)
            
            # Log results
            log_results(best_feature, evaluated_df, best_features, best_features_file, loss, experiment_dir)
        except Exception as e:
            logger.error(f"Error in iteration {iteration+1}: {e}")
    
    # Save final CSV 
    final_csv_path = os.path.join(experiment_dir, f"final_{dataset_name}_features.csv")
    
    # Create a clean dataframe for final output
    clean_df = df.copy()
    
    # Clean up column names by removing the "_property" suffix
    column_mapping = {col: col.replace("_property", "") for col in clean_df.columns if "_property" in col}
    clean_df = clean_df.rename(columns=column_mapping)
    
    # Save the cleaned dataframe
    clean_df.to_csv(final_csv_path, index=False)
    logger.info(f"Saved final features CSV to {final_csv_path}")
    
    return best_features

def select_best_feature(evaluated_df, best_features):
    """Select the best feature based on evaluation scores"""
    remaining_features = [col for col in evaluated_df.columns if col not in best_features and col != "empty"]
    feature_scores = evaluated_df[remaining_features].sum().sort_values()
    if len(feature_scores) == 0:
        return "empty"
    return feature_scores.index[0]

def log_results(best_feature, evaluated_df, best_features, best_features_file, loss, output_dir):
    """Log results to files and wandb"""
    logger.info(f"Selected feature: {best_feature}")
    
    remaining_features = [col for col in evaluated_df.columns[:-1] if col not in best_features]
    remaining_scores = evaluated_df[remaining_features].sum().sort_values()
    top_remaining = remaining_scores.index[:5].tolist() if len(remaining_scores) >= 5 else remaining_scores.index.tolist()
    logger.info(f"Top remaining features: {top_remaining}")
    
    with open(best_features_file, "w") as bf_file:
        bf_file.write("\n".join(best_features) + "\n")
    
    with open(os.path.join(output_dir, f"perplexities.txt"), "a") as losses_file:
        losses_file.write(f"{loss}\n")
    
    if WANDB_PROJECT:
        wandb.log({"mean_perplexity": loss})

def adaptive_featurization(
    experiment_name: str,
    dataset_name: str,
    output_dir: str = "data",
    num_iterations: int = 10,
    batch_size: int = 16,
    prompt_preset: str = PromptPresets.DEFAULT.value,
    custom_prompts: Dict[str, str] = None,
    generator_model: str = MODEL,
    verifier_model: str = MODEL,
    target_features: Optional[int] = None,
    cluster_size: int = CLUSTER_SIZE,
    max_examples_override: Optional[int] = None
):
    """
    Run the entire adaptive featurization pipeline.
    
    Args:
        experiment_name: Name of the experiment
        dataset_name: Name of the dataset
        output_dir: Output directory for results
        num_iterations: Number of iterations for feature selection
        batch_size: Batch size for evaluation
        prompt_preset: Name of the prompt preset to use
        custom_prompts: Optional dict with custom prompts overriding the preset
        generator_model: Model to use for generation
        verifier_model: Model to use for verification
        target_features: Target number of features to select
        cluster_size: Number of similar features to group together
        max_examples_override: Override max_examples in dataset config
    """
    # Get prompt configuration
    preset = PromptPresets(prompt_preset)
    logger.info(f"Using prompt preset: {preset.name}")
    prompt_config = PROMPT_CONFIGS[preset].copy()
    
    # Override with any custom prompts
    if custom_prompts:
        for key, value in custom_prompts.items():
            if key in prompt_config:
                prompt_config[key] = value
                logger.info(f"Overriding {key} with custom prompt")
    
    # Setup experiment
    experiment_setup = setup_experiment(experiment_name, dataset_name, output_dir)
    
    if isinstance(experiment_setup, dict):
        experiment_dir = experiment_setup["path"]
        reuse_experiment = experiment_setup["reuse"]
    else:
        # For backward compatibility with previous code
        experiment_dir = experiment_setup
        reuse_experiment = False
    
    # If we're reusing an experiment, load the best features from the file
    if reuse_experiment:
        best_features_file = os.path.join(experiment_dir, "best_features.txt")
        with open(best_features_file, "r") as bf_file:
            best_features = [line.strip() for line in bf_file.readlines() if line.strip()]
        
        logger.info(f"Reusing existing experiment. Found {len(best_features)} features.")
        
        return {
            "experiment_dir": experiment_dir,
            "dataset_name": dataset_name,
            "best_features": best_features,
            "reused": True
        }
    
    # Load dataset
    df = load_dataset(dataset_name, prompt_config.get("prompt_template"), max_examples_override)
    
    # Generate features
    generate_features(
        df, 
        experiment_dir,
        generator_system_prompt=prompt_config.get("generator_system_prompt"),
        generator_user_prompt=prompt_config.get("generator_user_prompt"),
        verification_system_prompt=prompt_config.get("verification_system_prompt"),
        verification_user_prompt=prompt_config.get("verification_user_prompt"),
        generator_model=generator_model,
        verifier_model=verifier_model,
        cluster_size=cluster_size,
        prompt_type=prompt_preset
    )
    
    # Run featurization
    best_features = run_featurization(
        df, 
        dataset_name, 
        experiment_dir, 
        num_iterations, 
        batch_size,
        prompt_template=prompt_config.get("prompt_template"),
        target_features=target_features
    )
    
    logger.info(f"Adaptive featurization complete. Selected {len(best_features)} features.")
    
    # Return summary of results
    return {
        "experiment_dir": experiment_dir,
        "dataset_name": dataset_name,
        "best_features": best_features,
        "reused": False
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run adaptive featurization on a dataset.")
    parser.add_argument("experiment_name", type=str, help="Name of the experiment")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to analyze")
    parser.add_argument("--output_dir", type=str, default="data", help="Directory to save results")
    parser.add_argument("--num_iterations", type=int, default=10, help="Maximum number of iterations for feature selection")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--prompt_preset", type=str, choices=[p.value for p in PromptPresets], 
                       default=PromptPresets.DEFAULT.value, help="Preset prompt configuration to use")
    parser.add_argument("--custom_generator_system", type=str, help="Custom generator system prompt")
    parser.add_argument("--custom_generator_user", type=str, help="Custom generator user prompt")
    parser.add_argument("--custom_verification_system", type=str, help="Custom verification system prompt")
    parser.add_argument("--custom_verification_user", type=str, help="Custom verification user prompt")
    parser.add_argument("--generator_model", type=str, default=MODEL, help="Model to use for generation")
    parser.add_argument("--verifier_model", type=str, default=MODEL, help="Model to use for verification")
    parser.add_argument("--max_examples_override", type=int, help="Override max_examples in dataset config (controls how many examples to load)")
    parser.add_argument("--target_features", type=int, help="Target number of features to select (overrides num_iterations)")
    parser.add_argument("--cluster_size", type=int, default=CLUSTER_SIZE, 
                        help="Number of similar features to group together during clustering")
    args = parser.parse_args()
    
    # Gather any custom prompts
    custom_prompts = {}
    if args.custom_generator_system:
        custom_prompts["generator_system_prompt"] = args.custom_generator_system
    if args.custom_generator_user:
        custom_prompts["generator_user_prompt"] = args.custom_generator_user  
    if args.custom_verification_system:
        custom_prompts["verification_system_prompt"] = args.custom_verification_system
    if args.custom_verification_user:
        custom_prompts["verification_user_prompt"] = args.custom_verification_user
    
    adaptive_featurization(
        args.experiment_name,
        args.dataset_name,
        args.output_dir,
        args.num_iterations,
        args.batch_size,
        args.prompt_preset,
        custom_prompts if custom_prompts else None,
        args.generator_model,
        args.verifier_model,
        args.target_features,
        args.cluster_size,
        args.max_examples_override
    )
