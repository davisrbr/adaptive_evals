from typing import Dict, Any
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from prompting.cyberbullying_prompts import cyberbullying_system_prompt


def process_sample_cyberbullying(sample: Dict[str, Any]) -> Sample:
    """
    Processes a raw sample into an Inspect Sample.

    Args:
        sample (Dict[str, Any]): Raw sample data.

    Returns:
        Sample: Processed Inspect Sample.
    """
    attributes = sample  
    attack_user_message = f"Attributes: {attributes}"
    attack_prompt = cyberbullying_system_prompt()
    input_text = [
        ChatMessageSystem(content=attack_prompt),
        ChatMessageUser(content=attack_user_message)
    ]

    return Sample(
        id=sample['name'],
        input=input_text,
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
            "city": attributes.get('city', ''),
            "state": attributes.get('state', ''),
            "education": attributes.get('education', ''),
            "language": attributes.get('language', ''),
            "income": attributes.get('income', ''),
            "occupation": attributes.get('occupation', ''),
            "urbanRural": attributes.get('urbanRural', ''),
            "politicalAffiliation": attributes.get('politicalAffiliation', ''),
            "scenarios": attributes.get('scenarios', ''),
        }
    )

if __name__ == "__main__":
    cyberbullying_personas = json_dataset(
        json_file="data/synthetic_profiles_complete.json",
        sample_fields=process_sample_cyberbullying,
        shuffle=True,
        seed=42,
        limit=2,
    )
    print(cyberbullying_personas[0].metadata['attributes'])