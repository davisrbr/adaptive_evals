from typing import Dict, Any
from inspect_ai.dataset import Sample, hf_dataset
import numpy as np

from system_prompts import get_attacker_system_prompt
from common import get_init_msg

def process_sample(sample: Dict[str, Any]) -> Sample:
    """
    Processes a raw sample into an Inspect Sample.

    Args:
        sample (Dict[str, Any]): Raw sample data.

    Returns:
        Sample: Processed Inspect Sample.
    """
    goal = sample['Goal']
    target_text = sample['Target']
    attacker_prompt = get_attacker_system_prompt(goal, target_text)

    return Sample(
        id=sample['Index'],
        input=get_init_msg(goal, target_text),
        target=target_text,  
        metadata={
            "Goal": goal,
            "Behavior": sample.get('Behavior', ''),
            "Category": sample.get('Category', ''),
            "Source": sample.get('Source', ''),
            "attacker_prompt": attacker_prompt,
            "Target": target_text
        }
    )

# Load and preprocess the dataset
jb_behaviors_dataset = hf_dataset(
    path="JailbreakBench/JBB-Behaviors",
    name="behaviors",
    split="harmful",
    sample_fields=process_sample,
    cache_dir="~/data",
)
