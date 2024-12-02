import os
from inspect_ai.dataset import Dataset
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import click
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from legalbench.utils import generate_prompts
import jailbreakbench as jbb

@click.command()
@click.option("--dataset_name", type=str, default="truthfulqa/truthful_qa", help="Dataset name")
@click.option("--split", type=str, default="multiple_choice", help="Split name")
@click.option("--hub_name", type=str, default="davisrbr/truthfulqa-embeddings", help="Hub name")
@click.option("--feature", type=str, default="validation", help="Item in dataset dict")
@click.option("--task_name", type=str, default=None, help="LegalBench task name")
@click.option("--task_type", type=click.Choice(['legalbench', 'truthfulqa', 'jailbreakbench']), default='truthfulqa', help="Type of task")
@click.option("--k", type=int, default=100, help="Number of similar/dissimilar examples to find")
def embed_dataset(dataset_name: str, split: str, hub_name: str, feature: str, task_name: str, task_type: str, k: int):
    """
    Embed a dataset and push it to the Hugging Face Hub.

    Depending on the task_type, it embeds either the TruthfulQA dataset or LegalBench dataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load.
        hub_name (str): The name of the Hugging Face Hub repository to push the dataset.
        feature (str): The feature of the dataset to embed.
        task_name (str): The specific LegalBench task name.
        task_type (str): The type of task ('legalbench' or 'truthfulqa').
        k (int): The number of similar/dissimilar examples to find.
    """

    embeddings_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

    if task_type == 'truthfulqa':
        # Load the TruthfulQA dataset
        dataset = load_dataset(dataset_name, split, trust_remote_code=True)
        
        # Compute embeddings and add them to the dataset
        def compute_embedding(example):
            text = example["question"] + "\n" + "\n".join(example["mc1_targets"]["choices"])
            example["embedding"] = embeddings_model.encode(text)
            return example

        dataset[feature] = dataset[feature].map(compute_embedding, batched=False)

    elif task_type == 'legalbench':
        if task_name is None:
            raise ValueError("For LegalBench tasks, please provide the --task_name parameter.")

        # Load the LegalBench dataset
        prompt_template_path = f"legalbench/tasks/{task_name}/base_prompt.txt"

        if not os.path.exists(prompt_template_path):
            raise FileNotFoundError(f"Prompt template not found at {prompt_template_path}")

        with open(prompt_template_path) as in_file:
            prompt_template = in_file.read()

        # Load the dataset
        dataset = load_dataset("nguha/legalbench", name=task_name, split=split)

        # Generate prompts and compute embeddings
        def compute_embedding(record):
            df = pd.DataFrame([record])
            prompts = generate_prompts(prompt_template=prompt_template, data_df=df)
            prompt = prompts[0]

            # Extract the options from the prompt (if any)
            option_pattern = r"Option ([A-Z]): (.*)"
            options_matches = re.findall(option_pattern, prompt)
            choices = [match[1] for match in options_matches]

            # Prepare the text to embed
            text = prompt
            if choices:
                text += "\n" + "\n".join(choices)

            record["embedding"] = embeddings_model.encode(text)
            return record

        dataset = dataset.map(compute_embedding)

    elif task_type == 'jailbreakbench':
        # Load the behaviors dataset
        jb_behaviors_dataset = load_dataset(
            path="JailbreakBench/JBB-Behaviors",
            name="behaviors",
            split="harmful",
            cache_dir="~/data",
        )
        jb_behaviors_df = jb_behaviors_dataset.to_pandas()

        # Initialize lists for collecting data
        data_list = []
        methods = ["DSN", "PAIR", "JBC", "GCG"]
        models = ["llama-2-7b-chat-hf", "vicuna-13b-v1.5", "gpt-3.5-turbo-1106", "gpt-4-0125-preview"]

        # Collect data from JailbreakBench
        for method in methods:
            for model in models:
                try:
                    artifact = jbb.read_artifact(method=method, model_name=model)
                    jailbroken_artifact = [x for x in artifact.jailbreaks if x.jailbroken and x.prompt is not None]
                    
                    for x in jailbroken_artifact:
                        data = {
                            'index': x.index,
                            'goal': x.goal,
                            'behavior': x.behavior,
                            'category': x.category,
                            'prompt': x.prompt,
                            'response': x.response,
                            'number_of_queries': x.number_of_queries,
                            'queries_to_jailbreak': x.queries_to_jailbreak,
                            'jailbroken': x.jailbroken,
                            'method': method,
                            'model_name': model,
                            'prompt_tokens': x.prompt_tokens,
                            'response_tokens': x.response_tokens,
                        }
                        data_list.append(data)
                except Exception as e:
                    print(f"Failed to read artifact for method {method} and model {model}: {e}")

        # Create dataset and compute embeddings
        dataset = Dataset.from_list(data_list)
        
        def compute_embedding(batch):
            embeddings = embeddings_model.encode(batch['goal'], show_progress_bar=False)
            batch['goal_embedding'] = [emb.tolist() for emb in embeddings]
            return batch

        dataset = dataset.map(compute_embedding, batched=True, batch_size=32)

        # Prepare embeddings and goals
        embeddings = np.array(dataset['goal_embedding'])
        goals = dataset['goal']

        def compute_similarities(query_embedding, embeddings):
            embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)
            similarities = np.dot(embeddings_norm, query_embedding_norm)
            return similarities

        # Augment the dataset
        augmented_data = []
        for i in tqdm(range(len(goals)), desc="Augmenting dataset"):
            query = dataset[i]
            query_embedding = embeddings[i]
            similarities = compute_similarities(query_embedding, embeddings)
            
            similarities[i] = -np.inf
            most_similar_indices = np.argsort(-similarities)[:k]
            similarities[i] = np.inf
            least_similar_indices = np.argsort(similarities)[:k]
            
            for idx, sim_idx in enumerate(most_similar_indices):
                sim_idx = int(sim_idx)
                query[f'most_similar_{idx+1}'] = dataset[sim_idx]['goal']
                query[f'most_similarity_{idx+1}'] = float(similarities[sim_idx])
            
            for idx, sim_idx in enumerate(least_similar_indices):
                sim_idx = int(sim_idx)
                query[f'least_similar_{idx+1}'] = dataset[sim_idx]['goal']
                query[f'least_similarity_{idx+1}'] = float(similarities[sim_idx])
            
            augmented_data.append(query)

        # Create and merge datasets
        augmented_dataset = Dataset.from_list(augmented_data)
        augmented_dataset_df = augmented_dataset.to_pandas()
        merged_df = pd.merge(augmented_dataset_df, jb_behaviors_df, left_on='index', right_on='Index', how='left')
        dataset = Dataset.from_pandas(merged_df)

    # Push to hub
    dataset.push_to_hub(hub_name)
    print(f"Dataset pushed to {hub_name}")

if __name__ == "__main__":
    embed_dataset()