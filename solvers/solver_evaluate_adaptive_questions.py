from typing import Optional, Dict, Any, List
import logging
import re
import torch
from inspect_ai.solver import solver, Generate, TaskState
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.dataset import Sample
from data.eval_log_processing import read_eval_log_async
from scorers.scorers_rewording import JUDGE_FILTERED
from scorers.scorers_truthfulqa import INCORRECT, CORRECT
from inspect_ai.solver._multiple_choice import parse_answers
from inspect_ai.solver._multiple_choice import prompt as mc_prompt
from inspect_ai.solver._multiple_choice import pretend_we_didnt_shuffle, valid_template
from inspect_ai.solver._multiple_choice import SINGLE_ANSWER_TEMPLATE, SINGLE_ANSWER_TEMPLATE_COT
from solvers.adaptive_utils import normalize_target

logger = logging.getLogger(__name__)


@solver
def evaluate_adaptive_truthfulqa_questions(
    adaptive_log_path: str,
    new_model_name: str = "openai/gpt-4o-mini",
    multiple_correct: bool = False,
    temperature: float = 0.0,
    use_cot: bool = False,
    custom_template: Optional[str] = None,
) -> Generate:
    """
    Loads the final log from the adaptive_truthfulqa_solver, filters out questions
    that did not pass the judge filter (i.e., judge_choice not in ['A','B']),
    and then evaluates a new model on these filtered questions.

    Args:
        adaptive_log_path (str): Path to the final log from the adaptive_truthfulqa_solver.
        new_model_name (str): Name of the evaluation model to run on the filtered questions.
        multiple_correct (bool): Whether there may be multiple correct answers to the question.
        temperature (float): Temperature for the new model.
        use_cot (bool): If True, include chain-of-thought in prompting (uses SINGLE_ANSWER_TEMPLATE_COT).
        custom_template (Optional[str]): If provided, overrides the default single/multiple answer template.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """
        1. Read the logged samples from the adaptive solver.
        2. Filter out any that were judge-filtered (judge_choice not in ['A','B']).
        3. Evaluate the new model on these retained questions.
        4. Store the results back into the state.
        """
        # 1. Read the log
        eval_log = await read_eval_log_async(adaptive_log_path)
        if not eval_log.samples:
            state.error = "No samples found in the adaptive truthfulqa log."
            state.completed = True
            return state

        # 2. Filter out judge-filtered questions
        # Per adaptive_truthfulqa_scorer_judged, if judge_choice not in ['A','B'], it should be filtered out
        filtered_samples = []
        for sample_item in eval_log.samples:
            # Check the judge_choice (if any)
            judge_choice = sample_item.metadata.get("judge_choice")
            original_score = sample_item.metadata.get("score")
            # If the judge_choice is A or B, it's accepted. If judge_choice is None or something else, it's filtered
            if judge_choice in ["A", "B"] and original_score == INCORRECT:
                filtered_samples.append(sample_item)

        if not filtered_samples:
            logger.debug("No samples passed the judge filter. Nothing to evaluate.")
            state.error = "No samples passed judge filter from the adaptive solver log."
            state.completed = True
            return state

        logger.debug(f"Loaded {len(eval_log.samples)} samples total, {len(filtered_samples)} passed the judge filter.")

        # 3. Evaluate the new model on these questions, storing results in a new key
        new_model = get_model(
            new_model_name,
            config=GenerateConfig(
                temperature=temperature,
                max_connections=10000,
            ),
        )

        # Choose a template if custom_template is not specified
        if custom_template and not valid_template(custom_template):
            logger.debug("Provided custom_template does not contain '{question}' or '{choices}'.")
            state.error = "Invalid custom_template, missing placeholders."
            state.completed = True
            return state

        if not custom_template:
            # Fallback to default single or multiple answer templates
            if multiple_correct:
                template = SINGLE_ANSWER_TEMPLATE_COT if use_cot else SINGLE_ANSWER_TEMPLATE
            else:
                template = SINGLE_ANSWER_TEMPLATE_COT if use_cot else SINGLE_ANSWER_TEMPLATE
        else:
            template = custom_template

        new_evaluation_results: List[Sample] = []
        for sample_item in filtered_samples:
            # Format the prompt
            # sample_item.input => question text
            # sample_item.choices => a list of possible choices
            question_str = sample_item.input.strip()
            choices = sample_item.choices

            # Build the prompt with the new template
            question_prompt = template.format(
                question=question_str,
                choices="\n".join(
                    f"{chr(65+i)}) {choice_text}" for i, choice_text in enumerate(choices)
                ),
                letters=",".join(chr(65 + i) for i in range(len(choices)))
            )

            # Now call the new_model
            response = await new_model.generate(question_prompt)
            model_output = response.completion.strip()

            # 4. Parse the answer from the model's output
            parse_match = parse_answers(model_output)

            # Mark correctness if we have a target
            # The original target is in sample_item.target
            # We can do normalizing
            if hasattr(sample_item, "target"):
                normalized_target = normalize_target(sample_item.target, len(sample_item.choices))
            else:
                normalized_target = []

            if parse_match and parse_match.group(1):
                given_answer_str = parse_match.group(1).upper()
                # can be single or multiple letters, possibly separated by commas
                answer_letters = re.split(r"[,\s]+", given_answer_str)
                answer_indices = [ord(x[0]) - ord("A") for x in answer_letters if x]
            else:
                answer_indices = []

            # Compare with target to decide correctness
            is_correct = set(answer_indices) == set(normalized_target)
            new_score = CORRECT if is_correct else INCORRECT

            # Store new evaluation in metadata
            sample_metadata = dict(sample_item.metadata)  # copy
            sample_metadata["new_model_output"] = model_output
            sample_metadata["new_model_choice_indices"] = answer_indices
            sample_metadata["new_model_choice_str"] = parse_match.group(1) if parse_match else ""
            sample_metadata["new_model_score"] = "C" if is_correct else "I"

            new_item = Sample(
                input=sample_item.input,
                choices=sample_item.choices,
                target=sample_item.target,
                metadata=sample_metadata,
            )
            new_evaluation_results.append(new_item)

        # 5. Store in the state
        state.store.set("re_evaluation_samples", new_evaluation_results)
        state.completed = True
        return state

    return solve 