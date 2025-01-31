import os
from inspect_ai.dataset import Dataset
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import click
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import jailbreakbench as jbb

@click.command()
@click.option("--dataset_name", type=str, default="truthful_qa", help="Name of the dataset to load")
@click.option("--split", type=str, default="validation", help="Split of the dataset to load")
@click.option("--hub_name", type=str, default="your-username/embedded-dataset", help="HF Hub repo to push the dataset")
@click.option("--feature", type=str, default="train", help="Which feature of the dataset to embed")
@click.option("--task_name", type=str, default=None, help="For specialized tasks, specify the name")
@click.option("--task_type", type=str, default="truthfulqa", help="Type of task (legalbench, truthfulqa, jailbreakbench, politeness, etc.)")
@click.option("--k", type=int, default=100, help="Number of similar/dissimilar examples to find")
def embed_dataset(dataset_name: str, split: str, hub_name: str, feature: str, task_name: str, task_type: str, k: int):
    """
    Embed a dataset and push it to the Hugging Face Hub.

    Depending on the task_type, it embeds:
      - TruthfulQA
      - LegalBench
      - JailbreakBench
      - Politeness
      (or other tasks in the future).

    Args:
        dataset_name (str): The name of the dataset to load.
        split (str): The split of the dataset to load.
        hub_name (str): The name of the Hugging Face Hub repository to push the dataset.
        feature (str): The feature of the dataset to embed.
        task_name (str): The specific task name (if needed).
        task_type (str): The type of task. Options: 'legalbench', 'truthfulqa', 'jailbreakbench', 'politeness', etc.
        k (int): The number of similar/dissimilar examples to find (used for some tasks).
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
            raise ValueError("For LegalBench tasks, please provide the --task_name parameter. I have used a few of the maud* ones previously")

        # Load the LegalBench dataset
        prompt_template_path = f"legalbench/tasks/{task_name}/base_prompt.txt"

        if not os.path.exists(prompt_template_path):
            raise FileNotFoundError(f"Prompt template not found at {prompt_template_path}")

        with open(prompt_template_path) as in_file:
            prompt_template = in_file.read()

        dataset = load_dataset("nguha/legalbench", name=task_name, split=split)

        def compute_embedding(record):
            df = pd.DataFrame([record])
            # Suppose we have a local function to generate prompts
            from solvers.adaptive_utils import generate_prompts  # example usage
            prompts = generate_prompts(prompt_template=prompt_template, data_df=df)
            prompt = prompts[0]

            # Extract potential options
            option_pattern = r"Option ([A-Z]): (.*)"
            options_matches = re.findall(option_pattern, prompt)
            choices = [match[1] for match in options_matches]

            text = prompt
            if choices:
                text += "\n" + "\n".join(choices)

            record["embedding"] = embeddings_model.encode(text)
            return record

        dataset = dataset.map(compute_embedding)

    elif task_type == 'jailbreakbench':
        jb_behaviors_dataset = load_dataset(
            path="JailbreakBench/JBB-Behaviors",
            name="behaviors",
            split="harmful",
            cache_dir="~/data",
        )
        jb_behaviors_df = jb_behaviors_dataset.to_pandas()

        data_list = []
        methods = ["DSN", "PAIR", "JBC", "GCG"]
        models = ["llama-2-7b-chat-hf", "vicuna-13b-v1.5", "gpt-3.5-turbo-1106", "gpt-4-0125-preview"]

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

        dataset = Dataset.from_list(data_list)

        def compute_embedding(batch):
            embeddings = embeddings_model.encode(batch['goal'], show_progress_bar=False)
            batch['goal_embedding'] = [emb.tolist() for emb in embeddings]
            return batch

        dataset = dataset.map(compute_embedding, batched=True, batch_size=32)

        embeddings = np.array(dataset['goal_embedding'])
        goals = dataset['goal']

        def compute_similarities(query_embedding, all_embeddings):
            all_embeddings_norm = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)
            query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)  
            return np.dot(all_embeddings_norm, query_embedding_norm)

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

        augmented_dataset = Dataset.from_list(augmented_data)
        augmented_dataset_df = augmented_dataset.to_pandas()
        merged_df = pd.merge(augmented_dataset_df, jb_behaviors_df, left_on='index', right_on='Index', how='left')
        dataset = Dataset.from_pandas(merged_df)

    elif task_type == 'politeness':
        # Load the multilingual politeness dataset (default configuration)
        dataset = load_dataset(dataset_name, name="default", split=split, trust_remote_code=True)

        # Compute embeddings for each utterance
        def compute_embedding(example):
            utterance = example.get("Utterance", "")
            example["embedding"] = embeddings_model.encode(utterance)
            return example

        dataset = dataset.map(compute_embedding, batched=False)

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # Push to hub (common for all tasks)
    dataset.push_to_hub(hub_name)
    print(f"Dataset pushed to {hub_name}")


if __name__ == "__main__":
    embed_dataset()