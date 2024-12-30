from typing import Dict, Any
from inspect_ai.dataset import Sample, json_dataset


def process_sample(sample: Dict[str, Any]) -> Sample:
    """
    Processes a raw sample into an Inspect Sample.

    Args:
        sample (Dict[str, Any]): Raw sample data.

    Returns:
        Sample: Processed Inspect Sample.
    """
    attributes = sample  

    return Sample(
        id=sample['name'],
        input=str(attributes),
        metadata={
            "attributes": attributes,
            "age": attributes.get('age', ''),
            "gender": attributes.get('gender', ''),
            "ethnicity": attributes.get('ethnicity', ''),
            "sexual_orientation": attributes.get('sexual_orientation', ''),
            "religion": attributes.get('religion', ''),
            "political_affiliation": attributes.get('political_affiliation', ''),
            "disability_status": attributes.get('disability_status', ''),
            "profession": attributes.get('profession', ''),
            "hobbies": attributes.get('hobbies', ''),
            "personality": attributes.get('personality', ''),
            "location": attributes.get('location', ''),
        }
    )

cyberbullying_personas = json_dataset(
    json_file="/Users/davisbrown/adaptive_evals/data/cyberbullying_personas.json",
    sample_fields=process_sample,
    shuffle=True,
    seed=42,
    limit=25,
)
if __name__ == "__main__":
    cyberbullying_personas = json_dataset(
        json_file="data/cyberbullying_personas.json",
        sample_fields=process_sample,
        shuffle=True,
        seed=42,
        limit=2,
    )
    print(cyberbullying_personas[0].metadata['attributes'])