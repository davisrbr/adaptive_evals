import os
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import click
import re
import pandas as pd
from legalbench.utils import generate_prompts

@click.command()
@click.option("--dataset_name", type=str, default="truthfulqa/truthful_qa", help="Dataset name")
@click.option("--split", type=str, default="multiple_choice", help="Split name")
@click.option("--hub_name", type=str, default="davisrbr/truthfulqa-embeddings", help="Hub name")
@click.option("--feature", type=str, default="validation", help="Item in dataset dict")
@click.option("--task_name", type=str, default=None, help="LegalBench task name")
@click.option("--task_type", type=click.Choice(['legalbench', 'truthfulqa']), default='truthfulqa', help="Type of task")
def embed_dataset(dataset_name: str, split: str, hub_name: str, feature: str, task_name: str, task_type: str):
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

    else:
        raise ValueError("Invalid task_type. Choose 'truthfulqa' or 'legalbench'.")

    # Add FAISS index to the 'embedding' column
    dataset.add_faiss_index(column='embedding')

    # Push the dataset with the FAISS index to the hub
    dataset.push_to_hub(hub_name)
    print(f"Dataset pushed to {hub_name}")

if __name__ == "__main__":
    embed_dataset()