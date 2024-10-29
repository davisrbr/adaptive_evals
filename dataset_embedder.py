from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import click


@click.command()
@click.option("--dataset_name", type=str, help="Dataset name", default="truthfulqa/truthful_qa")
@click.option("--split", type=str, help="Split name", default="multiple_choice")
@click.option("--hub_name", type=str, help="Hub name", default="davisrbr/truthfulqa-embeddings")
@click.option("--feature", type=str, help="Item in dataset dict", default="validation")
def embed_dataset(dataset_name: str, split: str, hub_name: str, feature: str):
    dataset = load_dataset(dataset_name, split, trust_remote_code=True)
    embeddings_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    dataset[feature] = dataset[feature].add_column(
        "embedding",
        [embeddings_model.encode(example["question"] + "\n" + "\n".join(example["mc1_targets"]["choices"])) for example in dataset[feature]]
    )
    dataset.push_to_hub(hub_name)

if __name__ == "__main__":
    embed_dataset()