import os
import random
import json
import re
import logging
from typing import Any, Optional
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, stderr, scorer, Scorer
from inspect_ai.solver import solver, Generate, TaskState
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.log import read_eval_log
from inspect_ai.dataset import Sample
from legalbench.utils import generate_prompts
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@solver
def adaptive_legal_solver(
    initial_log_path: str,
    task_name: str,
    n_positive_samples: int = 5,
    n_negative_samples: int = 5,
    randomize_sampling: bool = False,
    generator_model_name: str = "openai/gpt-4",
    eval_model_name: str = "openai/gpt-4",
    use_cot_generator: bool = False,
    use_cot_evaluator: bool = False,
    num_attempts: int = 30,
) -> Generate:
    """
    Solver that generates new questions based on the model's performance
    and evaluates the model on them for the LegalBench dataset.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Check if we've already generated new samples
        if 'generated_sample' in state.store:
            state.completed = True  # Skip further execution
            return state

        # Initialize generator and evaluator models
        generator_model = get_model(generator_model_name, config=GenerateConfig(max_connections=10000, temperature=0.5))
        eval_model = get_model(eval_model_name, config=GenerateConfig(max_connections=10000, temperature=0))

        # Load the initial evaluation log
        eval_log = read_eval_log(initial_log_path)

        # Access the samples from the eval_log
        sample_logs = eval_log.samples
        assert len(sample_logs) > 0, "No samples found in the initial evaluation log."

        if not randomize_sampling:
            # Separate correct and incorrect samples
            correct_samples = [sample for sample in sample_logs if sample.score.value == "C"]
            incorrect_samples = [sample for sample in sample_logs if sample.score.value != "C"]

            num_correct = min(len(correct_samples), n_positive_samples)
            num_incorrect = min(len(incorrect_samples), n_negative_samples)

            # Sample correct and incorrect samples
            sampled_correct = random.sample(correct_samples, num_correct) if num_correct > 0 else []
            sampled_incorrect = random.sample(incorrect_samples, num_incorrect) if num_incorrect > 0 else []

        else: # just sample randomly from all samples
            sampled_samples = random.sample(sample_logs, n_positive_samples + n_negative_samples)
            sampled_correct = [sample for sample in sampled_samples if sample.score.value == "C"]
            sampled_incorrect = [sample for sample in sampled_samples if sample.score.value != "C"]

        # Prepare context examples for the generator model
        context_examples = ""
        for sample_item in sampled_correct + sampled_incorrect:
            if not randomize_sampling:
                status = "Correctly" if sample_item in sampled_correct else "Incorrectly"
            else:
                status = "Previously"
            sample_data = sample_item.input
            # format sample_data as json
            sample_data = json.dumps(sample_data)   
            context_examples += f"{status} Answered Example:\n"
            context_examples += f"Sample Data:\n{sample_data}\n"
            context_examples += f"{sample_item.target}\n\n"

        # Load the base prompt for the task
        prompt_template_path = f"legalbench/tasks/{task_name}/base_prompt.txt"
        with open(prompt_template_path, 'r') as f:
            base_prompt = f.read()

        # Identify the placeholder keys in the prompt template
        placeholder_keys = [key.strip() for key in set(re.findall(r'{{(.*?)}}', base_prompt))]
        if not placeholder_keys:
            state.error = "No placeholders found in the base prompt."
            state.completed = True
            return state

        # Generate a new sample data dictionary
        try:
            if use_cot_generator:
                json_format = "{{\"reasoning_for_question\": [REASONING], \"text\": [TEXT], \"reasoning_for_answer\": [REASONING], \"answer\": [ANSWER]}}"
                reasoning_prompt = "'reasoning_for_question' is your step by step reasoning for the question, (for example, 'The model seems to have been confused about the relative importance of the clauses of the statute, and has interpreted them incorrectly; I will make a similar question but have a different clause emphasized in my invented document'), 'reasoning_for_answer' is your step by step reasoning for the answer, here you should explain why the answer to your constructed question is correct (for example, 'Because the question asks about statute A, the correct answer is clause A of statute A')"
            else:
                json_format = "{{\"text\": [TEXT], \"answer\": [ANSWER]}}"
                reasoning_prompt = ""
            # Prepare the generation prompt
            generation_prompt = (
                f"You are to generate a new data sample for the following LegalBench task.\n\n"
                f"Task Name: {task_name}\n\n"
                "Please make your example difficult to answer correctly, considering the examples provided. Note that you should make your example distinct from all of the examples provided.\n\n"
                f"Instructions:\n"
                f"- Generate appropriate values for each of the placeholder keys.\n"
                f"- The values should be suitable for the task.\n"
                f"- Output the result as a JSON object with keys corresponding to the placeholders.\n"
                f"- Ensure that the 'answer' key is included and contains the correct answer.\n"
                f"- Do not include any additional text outside the JSON object.\n\n"
                f"Examples:\n{context_examples}\n"
                f"Now, generate a new data sample. Again, note that you are to make this question extremely difficult to answer correctly. Consider the examples provided, and how they might have caused the model to incorrectly answer the question."
                f"Make your question more like the examples that were answered incorrectly, but make sure that it is distinct from the examples provided.\n\n"
                f"Please format your JSON like {json_format}, where {reasoning_prompt}[TEXT] is the full text of the question, including all details (like documents, etc., but not including the letter of the answer) and [ANSWER] is the letter of the correct answer to the question. Do not prepend or append anything to your JSON, just the brackets and the keys and values. Please be sure to include all {4 if use_cot_generator else 2} keys in the JSON."
            )

            def parse_json(generated_text: str) -> dict:
                # Remove markdown code block if present
                if generated_text.startswith('```json'):
                    # Remove the initial '```json' and following newline if present
                    generated_text = generated_text[7:].lstrip('\n')
                    # Remove the ending '```' if present
                    if generated_text.endswith('```'):
                        generated_text = generated_text[:-3]
                elif generated_text.startswith('json\n'):
                    generated_text = generated_text[5:]

                # Now parse the JSON output
                sample_data = json.loads(generated_text)

                # Ensure all placeholders are filled
                for key in placeholder_keys:
                    if key not in sample_data:
                        raise ValueError(f"Missing key '{key}' in the generated sample data.")
                # Ensure 'answer' key is present
                if 'answer' not in sample_data:
                    raise ValueError("Missing 'answer' key in the generated sample data.")
                return sample_data


            for attempt in range(num_attempts):
                # Generate the new sample data. Try up to num_attempts times if there is an error
                # (typically ~2 attempts is enough to get all correct jsons for a run of 25 epochs with gpt-4o)
                try:
                    generation_response = await generator_model.generate(generation_prompt)
                    generated_text = generation_response.completion.strip()
                    sample_data = parse_json(generated_text)
                    break  # Parsing succeeded, exit the loop
                except ValueError as e:
                    logger.debug(str(e))
                    print(f"Error parsing JSON: {e}, attempt {attempt + 1} of {n}")
            else:
                print("Failed to parse JSON after maximum retry attempts.")
                state.error = "Failed to parse JSON after maximum retry attempts."
                state.completed = True
                return state

            # Generate the prompt using generate_prompts
            data_df = pd.DataFrame([sample_data])
            prompts = generate_prompts(prompt_template=base_prompt, data_df=data_df)
            generated_prompt = prompts[0]
            # prepend the multiple choice template to the prompt
            SINGLE_ANSWER_TEMPLATE = "Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where $LETTER is the letter of the correct answer."
            SINGLE_ANSWER_TEMPLATE_COT = "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where $LETTER is the letter of the correct answer. Think step by step before answering."
            multiple_choice_template = SINGLE_ANSWER_TEMPLATE if not use_cot_evaluator else SINGLE_ANSWER_TEMPLATE_COT
            generated_prompt = multiple_choice_template + "\n\n" + generated_prompt
            correct_answer = sample_data.get('answer', '').strip()
            if not correct_answer:
                state.error = "The 'answer' key is empty in the sample data."
                state.completed = True
                return state

            # Create a Sample object
            generated_sample = Sample(
                input=generated_prompt,
                target=correct_answer,
                metadata={'sample_data': sample_data}
            )

            # Generate the model's answer
            answer_response = await eval_model.generate(generated_prompt)
            model_answer = answer_response.completion.strip()
            # remove ANSWER: from the model's answer
            model_answer = model_answer.split('ANSWER: ')[-1].strip()

            # Store the model's answer
            generated_sample.metadata['model_answer'] = model_answer

            # Score the model's answer
            # Compare given_answer to the correct answer
            if model_answer == correct_answer:
                generated_sample.metadata['score'] = "C"  # Correct
            else:
                generated_sample.metadata['score'] = "I"  # Incorrect

            # Save the generated sample in the state store
            state.store.set('generated_sample', generated_sample)
            state.scores = [generated_sample.metadata['score']]
        except Exception as e:
            state.error = f"Error generating or evaluating the sample: {e}"
            state.completed = True

        return state

    return solve

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
                value=CORRECT, # we don't want to penalize the model for a parsing error
                answer="[ERROR]",
                target=target,
                explanation=str(e),
            )

    return score

@solver
def adaptive_legal_judge_solver(
    initial_log_path: str,
    judge_model_name: Optional[str | list[str]] = None,
    num_attempts: int = 3,
) -> Generate:
    """
    Solver that judges the correctness of generated questions from the adaptive legal solver.

    Args:
        initial_log_path (str): Path to the initial evaluation log containing ground truth questions.
        judge_model_name (str): Name of the model used to judge the generated questions.
        num_attempts (int): Number of attempts to generate a valid judgment.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not judge_model_name:
            logger.debug("No judge model name provided, skipping judgment.")
            state.completed = True
            return state
        # Load the initial evaluation log to get ground truth questions
        initial_eval_log = read_eval_log(initial_log_path)
        ground_truth_samples = initial_eval_log.samples

        # Extract the ground truth questions, shuffle them with a random seed
        ground_truth_questions = [sample.input for sample in ground_truth_samples]
        random.seed(42) # TODO: make random seed passing functional
        random.shuffle(ground_truth_questions)

        # Get the generated sample from the state.store
        generated_sample = state.store.get('generated_sample', None)
        if not generated_sample:
            state.error = "No generated sample found in state."
            state.completed = True
            return state

        generated_question = generated_sample.input

        # Initialize the judge model
        judge_model = get_model(judge_model_name, config=GenerateConfig(max_connections=10000)) # no temperature, assuming using o1-mini or o1-preview :)

        # Prepare the judging prompt using a classifying pattern
        ground_truth_text = "\n".join([f"Question {i+1}: {q}" for i, q in enumerate(ground_truth_questions[:2])])  # Limit to just 2 questions

        judge_prompt = f"""\
You are comparing a generated question to a set of ground truth questions for a legal task. Here is the data:
[BEGIN DATA]
************
[Ground Truth Questions]:
{ground_truth_text}
************
[Generated Question]:
{generated_question}
************
[END DATA]

Determine if the generated question is correct, precise, and free of errors, and follows the same style and format as the ground truth questions. Consider whether it makes sense, contains any factual errors or contradictions that would make it difficult to answer, and is appropriately challenging.

Answer the question by selecting one of the following options:
(A) The generated question is correct, precise, and appropriate. It contains no factual errors or contradictions. Ignore any minor issues with formatting (e.g., two Answer: lines at the end or an extra multiple choice line at the beginning)
(B) The generated question has minor issues but is acceptable. For example, it may be phrased awkwardly or have minor factual errors, but it is still a valid question.
(C) The generated question has significant issues and is incorrect. It contains factual errors or contradictions, or is not a valid question.

Answer the question by calling `select_choice` with your reasoning in a step-by-step manner to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset. Select a single choice by setting the `choice` parameter to a single choice from A, B, or C.

Example:
[Reasoning]:
First, I will compare the generated question to the ground truth questions. The generated question follows the same style and format. It is clear and free of factual errors.
`select_choice(choice="A")`
"""

        # Function to parse the choice from the model output
        def parse_choice(completion: str) -> str:
            match = re.search(r'`select_choice\(choice=["\']([A-C])["\']\)`', completion)
            if match:
                return match.group(1)
            else:
                return None

        # Try generating the judgment
        for _ in range(num_attempts):
            try:
                judge_response = await judge_model.generate(judge_prompt)
                judge_output = judge_response.completion.strip()
                choice = parse_choice(judge_output)
                if choice:
                    # Store the judgment in the sample's metadata
                    generated_sample.metadata['judge_choice'] = choice
                    generated_sample.metadata['judge_reasoning'] = judge_output
                    # Update the state
                    state.store.set('generated_sample', generated_sample)
                    state.completed = True
                    return state
                else:
                    continue  # Retry if parsing failed
            except Exception as e:
                logger.debug(f"Error generating judgment: {e}")
                continue  # Retry on exception

        # If all attempts failed
        state.error = "Failed to get a valid judgment after maximum retry attempts."
        state.completed = True
        return state

    return solve

@scorer(metrics=[accuracy()])
def adaptive_legal_judge_scorer() -> Scorer:
    """
    Scorer that interprets the judgment from the judge model.
    """

    async def score(state: TaskState, target: Target) -> Score:
        try:
            generated_sample = state.store.get('generated_sample')
            judge_choice = generated_sample.metadata.get('judge_choice', None)
            judge_reasoning = generated_sample.metadata.get('judge_reasoning', '')

            if not judge_choice:
                state.error = "No judge_choice found in generated_sample metadata."
                return Score(
                    value=INCORRECT,
                    answer="[NO JUDGE CHOICE]",
                    target=target,
                    explanation="No judge_choice found."
                )

            if judge_choice == "A":
                value = CORRECT
            elif judge_choice == "B":
                value = CORRECT  # You can adjust this if you want to differentiate between A and B
            else:
                value = INCORRECT

            return Score(
                value=value,
                answer=f"Judge Choice: {judge_choice}",
                target=target,
                explanation=judge_reasoning,
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
