import os
import random
import re
import logging
from typing import Any, List, Literal, Tuple

from inspect_ai.scorer import CORRECT, INCORRECT, Score, Scorer, Target, accuracy, stderr, scorer
from inspect_ai.solver import solver, Generate, TaskState
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.log import read_eval_log
from inspect_ai.dataset import Sample
from datasets import load_dataset
from legalbench.utils import generate_prompts

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@solver
def adaptive_legal_solver(
    initial_log_path: str,
    task_name: str,
    n_positive_samples: int = 5,
    n_negative_samples: int = 5,
    generator_model_name: str = "openai/gpt-4",
    eval_model_name: str = "openai/gpt-4",
    use_cot: bool = False,
) -> Generate:
    """
    Solver that generates new questions based on the model's performance
    and evaluates the model on them for the LegalBench dataset.

    Args:
        initial_log_path (str): Path to the initial evaluation log.
        task_name (str): The name of the LegalBench task to evaluate.
        n_positive_samples (int): Number of correctly answered samples to use for adaptation.
        n_negative_samples (int): Number of incorrectly answered samples to use for adaptation.
        generator_model_name (str): Name of the model used to generate new questions.
        eval_model_name (str): Name of the model used to evaluate the new questions.
        use_cot (bool): Whether to use chain-of-thought prompting.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Check if we've already generated new samples
        if 'generated_sample' in state.store:
            state.completed = True  # Skip further execution
            return state

        # Initialize generator and evaluator models
        generator_model = get_model(generator_model_name, config=GenerateConfig(max_connections=10000, temperature=0.7))
        eval_model = get_model(eval_model_name, config=GenerateConfig(max_connections=10000, temperature=0))

        # Load the initial evaluation log
        eval_log = read_eval_log(initial_log_path)

        # Access the samples from the eval_log
        sample_logs = eval_log.samples

        # Separate correct and incorrect samples
        correct_samples = [sample for sample in sample_logs if sample.score.value == "C"]
        incorrect_samples = [sample for sample in sample_logs if sample.score.value != "C"]

        num_correct = min(len(correct_samples), n_positive_samples)
        num_incorrect = min(len(incorrect_samples), n_negative_samples)

        # Sample correct and incorrect samples
        sampled_correct = random.sample(correct_samples, num_correct) if num_correct > 0 else []
        sampled_incorrect = random.sample(incorrect_samples, num_incorrect) if num_incorrect > 0 else []

        # Prepare context for the generator model
        context = ""
        for sample_item in sampled_correct + sampled_incorrect:
            status = "Correctly" if sample_item in sampled_correct else "Incorrectly"
            context += f"{status} Answered Example:\n{sample_item.input}\nAnswer: {sample_item.target}\n\n"

        # Generate a new question
        try:
            generation_prompt = f"Based on the examples below, generate a new legal question similar in style.\n\n{context}\nNew Question:"
            generation_response = await generator_model.generate(generation_prompt)
            generated_text = generation_response.completion

            # Parse the generated question
            generated_sample = parse_generated_question(generated_text, task_name)
            if generated_sample is None:
                state.completed = True
                return state

            # Initialize metadata if necessary
            if generated_sample.metadata is None:
                generated_sample.metadata = {}

            # Prepare the prompt for the evaluator model
            evaluation_prompt = generated_sample.input

            # Generate the model's answer
            answer_response = await eval_model.generate(evaluation_prompt)

            # Store the model's answer
            generated_sample.metadata['model_answer'] = answer_response.completion.strip()

            # Score the model's answer
            given_answer = answer_response.completion.strip()
            correct_answer = generated_sample.target

            # Compare given_answer to the correct answer
            if given_answer == correct_answer:
                generated_sample.metadata['score'] = "C"  # Correct
            else:
                generated_sample.metadata['score'] = "I"  # Incorrect

            # Save the generated sample in the state store
            state.store.set('generated_sample', generated_sample)
            state.scores = [generated_sample.metadata['score']]
            state.completed = True
        except Exception as e:
            state.error = f"Error generating or evaluating question: {e}"
            state.completed = True

        return state

    return solve

def parse_generated_question(generated_text: str, task_name: str) -> Sample:
    """
    Parses the generated text to extract the question and answer.
    Returns a Sample object if successful, None otherwise.
    """
    try:
        # For LegalBench tasks, use the prompt template to structure the question
        prompt_template_path = f"legalbench/tasks/{task_name}/base_prompt.txt"
        with open(prompt_template_path) as in_file:
            prompt_template = in_file.read()

        # Assuming the generated text includes the question and the answer separated by a delimiter
        # Adjust parsing logic based on the actual format of generated_text
        split_text = generated_text.strip().split("Answer:")
        if len(split_text) != 2:
            logger.debug("Generated text format is incorrect.")
            return None

        question_text = split_text[0].strip()
        answer_text = split_text[1].strip()

        # Generate the prompt using the template and the question text
        prompt = generate_prompts(prompt_template=prompt_template, data_df=[{'question': question_text}])[0]

        # Assuming choices are ["A", "B", "C"]
        choices = ["A", "B", "C"]

        return Sample(
            input=prompt,
            choices=choices,
            target=answer_text,
            metadata={"task_name": task_name}
        )
    except Exception as e:
        logger.debug(f"Failed to parse generated question: {e}")
        return None

@scorer(metrics=[accuracy(), stderr()])
def adaptive_legal_scorer() -> Scorer:
    """
    Scorer for the adaptive LegalBench task.
    """

    async def score(state: TaskState, target: Target) -> Score:
        try:
            generated_sample = state.store.get('generated_sample')
            value = generated_sample.metadata['score']
            answer = generated_sample.metadata['model_answer']
            explanation = ""
            return Score(
                value=CORRECT if value == "C" else INCORRECT,
                answer=answer,
                target=target,
                explanation=explanation,
            )
        except Exception as e:
            state.error = str(e)
            return Score(
                value=INCORRECT,
                answer="[ERROR]",
                target=target,
                explanation=str(e),
            )

    return score
