"""
Adaptive utilities shared between some of the adaptive solvers.
"""

import random
import re
import logging
from inspect_ai.solver import solver
import torch
from typing import List, Tuple, Optional, Dict, Any, Callable
import json

from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.solver import Generate, Solver, TaskState
from data.eval_log_processing import read_eval_log_async
from utils_elicitation.novelty import novelty_scorer

from inspect_ai.solver._multiple_choice import (
    MULTIPLE_ANSWER_TEMPLATE,
    MULTIPLE_ANSWER_TEMPLATE_COT,
    SINGLE_ANSWER_TEMPLATE,
    SINGLE_ANSWER_TEMPLATE_COT,
)
from inspect_ai.solver._multiple_choice import (
    parse_answers,
    set_choices_based_on_generated_response,
    pretend_we_didnt_shuffle,
    valid_template,
    prompt,
)
from inspect_ai.util import resource

logger = logging.getLogger(__name__)


async def parse_log_and_sample(
    initial_log_path: str,
    n_positive_samples: int,
    n_negative_samples: int,
    randomize_sampling: bool = False,
    score_key: str = "C",
) -> Tuple[List[Sample], List[Sample]]:
    """
    Loads the evaluation log, separates correct vs. incorrect samples,
    then does a sample of size n_positive_samples and n_negative_samples,
    optionally randomizing the selection.

    Returns:
        (sampled_correct, sampled_incorrect)
    """
    eval_log = await read_eval_log_async(initial_log_path)
    if not eval_log.samples:
        raise ValueError("No samples found in the initial evaluation log.")

    sample_logs = eval_log.samples
    correct_samples = [s for s in sample_logs if s.score.value == score_key]
    incorrect_samples = [s for s in sample_logs if s.score.value != score_key]

    if not randomize_sampling:
        num_correct = min(len(correct_samples), n_positive_samples)
        num_incorrect = min(len(incorrect_samples), n_negative_samples)
        sampled_correct = random.sample(correct_samples, num_correct) if num_correct > 0 else []
        sampled_incorrect = (
            random.sample(incorrect_samples, num_incorrect) if num_incorrect > 0 else []
        )
    else:
        total_needed = n_positive_samples + n_negative_samples
        chosen = random.sample(sample_logs, min(len(sample_logs), total_needed))
        sampled_correct = [s for s in chosen if s.score.value == score_key]
        sampled_incorrect = [s for s in chosen if s.score.value != score_key]

    return sampled_correct, sampled_incorrect


def build_truthfulqa_context(
    sampled_correct: List[Sample],
    sampled_incorrect: List[Sample],
    randomize_sampling: bool,
    cot_in_context: bool,
) -> str:
    """
    Builds the TruthfulQA context, optionally including chain-of-thought for
    incorrect samples if cot_in_context=True.
    """
    context = ""
    if not randomize_sampling:
        for sample_item in sampled_correct + sampled_incorrect:
            is_correct = sample_item in sampled_correct
            status = "Correctly" if is_correct else "Incorrectly"
            context += f"**{status} Answered Question**: {sample_item.input}\n\n"
            context += f"**Choices**:\n{chr(10).join(sample_item.choices)}\n\n"

            context += "**The correct answer**:\n"
            context += chr(10).join(
                [f"{chr(97 + i)}. {ans}" for i, ans in enumerate(sample_item.target)]
            )
            context += "\n\n"

            if "choice" in sample_item.scores and hasattr(sample_item.scores["choice"], "answer"):
                context += f"**The model's answer**:\n{sample_item.scores['choice'].answer}\n\n"

            if cot_in_context and sample_item.score.value == "I":
                cot_text = sample_item.store.get("chain_of_thought")
                if cot_text:
                    context += f"**The model's incorrect reasoning**:\n {cot_text}\n\n\n"
    else:
        for sample_item in sampled_correct + sampled_incorrect:
            context += "**Previously Answered Question**:\n"
            context += f"**Question**: {sample_item.input}\n\n"
            context += f"**Choices**:\n{chr(10).join(sample_item.choices)}\n\n"
            if isinstance(sample_item.target, list):
                answers_concat = ", ".join(sample_item.target)
            else:
                answers_concat = str(sample_item.target)
            context += f"**Answer**: {answers_concat}\n\n"
    return context


async def check_with_self_and_eval(
    candidate_sample: Sample,
    self_check_model_name: Optional[str],
    eval_model_name: Optional[str],
    score_threshold: int,
    use_eval_model_for_checker: bool,
    self_check_prompt_fn: Callable[[Sample], str],
    parse_self_check_score: Callable[[str], int],
    format_eval_prompt_fn: Callable[[Sample], str],
    parse_eval_answer_fn: Callable[[str, Sample], bool],
    state: TaskState,
    store_chain_of_thought: bool = True,
) -> bool:
    """
    Optionally runs a self-check, then conditionally runs an eval-check:
      1) If self-check is used and fails, reject the question immediately.
      2) If eval-check is used and the model is correct, also reject.
      3) Only accept if the model fails the eval-check (makes the question 'hard').

    Returns:
        True if question is accepted (fails model eval-check), False otherwise.
    """
    # A) Self-check
    if self_check_model_name is not None:
        self_check_model = get_model(
            self_check_model_name, config=GenerateConfig(temperature=0)
        )
        prompt_text = self_check_prompt_fn(candidate_sample)
        response = await self_check_model.generate(prompt_text)
        score = parse_self_check_score(response.completion)

        if score < score_threshold:
            current_scores = state.store.get("self_check_scores", [])
            state.store.set("self_check_scores", current_scores + [score])

            if state.metadata is not None:
                meta_scores = state.metadata.get("self_check_scores", [])
                state.metadata["self_check_scores"] = meta_scores + [score]

            failed_questions = state.store.get("self_check_questions", [])
            state.store.set("self_check_questions", failed_questions + [candidate_sample])
            return False

        else:
            successful_questions = state.store.get("self_check_questions_passed", [])
            self_check_scores_passed = state.store.get("self_check_scores_passed", [])
            state.store.set("self_check_questions_passed", successful_questions + [candidate_sample])
            state.store.set("self_check_scores_passed", self_check_scores_passed + [score])

    if eval_model_name is not None and use_eval_model_for_checker:
        eval_model = get_model(eval_model_name, config=GenerateConfig(temperature=0))
        eval_prompt = format_eval_prompt_fn(candidate_sample)
        eval_resp = await eval_model.generate(eval_prompt)
        model_is_correct = parse_eval_answer_fn(eval_resp.completion, candidate_sample)

        # model is correct -> question is not accepted
        if model_is_correct:
            return False

        # Here the model is incorrect, so we store chain-of-thought if needed
        if store_chain_of_thought:
            candidate_sample.metadata["chain_of_thought"] = eval_resp.completion.strip()

        # Determine final correctness and store
        candidate_sample.metadata["score"] = "C" if model_is_correct else "I"
        candidate_sample.metadata["model_answer"] = eval_resp.completion.strip()

    return True


async def generate_question_with_retries(
    state: TaskState,
    context: str,
    generator_model_name: str,
    max_attempts: int,
    generation_prompt_fn: Callable[[str, int], str],
    parse_question_fn: Callable[[str], Optional[Sample]],
    existing_questions: List[str],
    similarity_threshold: float,
    use_embeddings: bool,
    embedding_model: Any,
    existing_question_embeddings: Optional[torch.Tensor],
    check_question_fn: Callable[[Sample], bool],
    include_previous_reasoning: bool = False,
    previous_reasoning_limit: int = 0,
    extract_reasoning_and_question_fn: Optional[Callable[[str], Tuple[Optional[str], Optional[Sample]]]] = None,
    get_current_examples_fn: Optional[Callable[[int], List[str]]] = None,
    get_current_embeddings_fn: Optional[Callable[[int], Optional[torch.Tensor]]] = None,
) -> Optional[Sample]:
    """
    Repeatedly generates a candidate question and checks it against acceptance criteria:
      • Checks parsing
      • Checks novelty
      • Runs acceptance function (self/eval checks)

    Args:
        get_current_examples_fn: Optional function to get current examples for a specific attempt
        get_current_embeddings_fn: Optional function to get current embeddings for a specific attempt

    Returns:
        The accepted Sample or None if we fail to find one within max_attempts.
    """
    gen_model = get_model(generator_model_name, config=GenerateConfig(temperature=0.5))
    overlap_key = "overlap_count"

    if overlap_key not in state.store:
        state.store.set(overlap_key, 0)

    not_novel_questions = state.store.get("not_novel_questions", [])
    not_novel_scores = state.store.get("not_novel_scores", [])
    novel_questions = state.store.get("novel_questions", [])
    novel_scores = state.store.get("novel_scores", [])

    # Store extracted reasoning
    previous_extracted_reasonings: List[str] = state.store.get("previous_extracted_reasonings", [])
    previous_questions: List[Sample] = state.store.get("previous_questions", [])
    for attempt in range(max_attempts):
        logger.debug(f"Attempt {attempt+1} at generating a novel question...")
        extended_context = context

        # Insert refine instruction right above where we include the actual previous reasoning
        if include_previous_reasoning and previous_reasoning_limit > 0 and previous_extracted_reasonings:
            refine_instructions = (
                "Finally, here is some of your prior reasoning for `reasoning_for_question`. This failed to create a hard enough question, so refine this reasoning in your new `reasoning_for_question`-- correct it given your new observed patterns, and make it more precise, with the goal of creating a new question that is sufficiently hard and novel. Your new question reasoning should be longer and more detailed than the previous reasoning.\n\n"
            )
            example_refine_instructions = (
                "For example, if the previous reasoning was: \n\n"
                "reasoning_for_question: The model often struggles with distinguishing between financial and non-financial considerations in merger agreements. [...] \n\n"
                "You might refine it to: \n\n"
                "reasoning_for_question: The model often struggles with distinguishing between financial and non-financial considerations in merger agreements when discussed in the context of normal company transactions, for example when considering a bank because it conflates the duties of the banks with the duties of the parties in the merger. [...] \n\n"
            )
            extended_context += "\n\n" + refine_instructions + example_refine_instructions

            extended_context += "### Previous Reasoning to analyze the target model's failure modes:\n"
            # Use extracted reasoning/previous question if available, otherwise use full completions
            relevant_reasonings = previous_extracted_reasonings[-previous_reasoning_limit:]
            relevant_questions = previous_questions[-previous_reasoning_limit:]
            for idx, (reasoning_text, question) in enumerate(zip(relevant_reasonings, relevant_questions), start=1):
                extended_context += f"[Previous Question Analysis (not sufficiently difficult) {idx}]:\n{reasoning_text}\n\n"
                extended_context += f"[Previous Question (not sufficiently difficult) {idx}]:\n{question}\n\n"

        # Build prompt using the new function signature with current_attempt
        gen_prompt = generation_prompt_fn(extended_context, attempt)
        gen_resp = await gen_model.generate(gen_prompt)
        generation_text = gen_resp.completion.strip()
        
        # Extract and store reasoning if extraction function is provided
        if extract_reasoning_and_question_fn:
            extracted_reasoning, extracted_question = extract_reasoning_and_question_fn(generation_text)
            if extracted_reasoning:
                previous_extracted_reasonings.append(extracted_reasoning)
                state.store.set("previous_extracted_reasonings", previous_extracted_reasonings)
                previous_questions.append(extracted_question)
                state.store.set("previous_questions", previous_questions)

        candidate = parse_question_fn(generation_text)
        if candidate is None:
            logger.debug("Parsing returned None. Retrying...")
            continue

        # Get current examples and embeddings for this attempt
        current_examples = existing_questions
        current_embeddings = existing_question_embeddings
        
        if get_current_examples_fn:
            current_examples = get_current_examples_fn(attempt)
        
        if get_current_embeddings_fn and use_embeddings:
            current_embeddings = get_current_embeddings_fn(attempt)

        novelty_score = novelty_scorer(
            candidate.input,
            current_examples,
            use_embeddings,
            embedding_model=embedding_model,
            existing_question_embeddings=current_embeddings,
        )
        if novelty_score > similarity_threshold:
            logger.debug("Candidate not novel (score=%.3f). Incrementing overlap_count.", novelty_score)
            new_count = state.store.get(overlap_key, 0) + 1
            state.store.set(overlap_key, new_count)

            not_novel_questions.append(candidate)
            not_novel_scores.append(novelty_score)
            state.store.set("not_novel_questions", not_novel_questions)
            state.store.set("not_novel_scores", not_novel_scores)
            continue

        # it's novel enough
        novel_questions = state.store.get("novel_questions", [])
        novel_scores = state.store.get("novel_scores", [])
        novel_questions.append(candidate)
        novel_scores.append(novelty_score)
        state.store.set("novel_questions", novel_questions)
        state.store.set("novel_scores", novel_scores)

        # check self-check/eval-check acceptance
        accepted = await check_question_fn(candidate)
        if not accepted:
            logger.info("Candidate question was not accepted. Retrying...")
            continue

        candidate.metadata["num_attempts"] = attempt + 1
        candidate.metadata["overlap_count"] = state.store.get(overlap_key, 0)
        return candidate

    # if we get here, we failed to find an acceptable question
    # we will cross our fingers and just choose the least similar question
    self_check_questions_passed = state.store.get("self_check_questions_passed", [])
    if self_check_questions_passed:
        # get the self_check question with the highest score
        self_check_scores_passed = state.store.get("self_check_scores_passed", [])
        best_index = self_check_scores_passed.index(max(self_check_scores_passed))
        return self_check_questions_passed[best_index]
    elif novel_questions:
        least_similar_index = novel_scores.index(min(novel_scores))
        return novel_questions[least_similar_index]
    elif not_novel_questions:
        least_similar_index = not_novel_scores.index(min(not_novel_scores))
        return not_novel_questions[least_similar_index]
    else:
        return None


async def final_eval_check_and_store(
    generated_sample: Sample,
    eval_model_name: str,
    format_eval_prompt_fn: Callable[[Sample], str],
    parse_eval_answer_fn: Callable[[str, Sample], bool],
    store_chain_of_thought: bool = False,
) -> None:
    """
    Final pass of evaluation to store relevant info in metadata:
      • chain_of_thought
      • final "score"
      • "model_answer"
    Only necessary if use_eval_model_for_checker=False, i.e., we have not checked 
    the eval model in the adaptive loop.
    """
    eval_model = get_model(eval_model_name, config=GenerateConfig(temperature=0))
    eval_prompt = format_eval_prompt_fn(generated_sample)
    eval_response = await eval_model.generate(eval_prompt)
    eval_text = eval_response.completion.strip()

    if store_chain_of_thought:
        generated_sample.metadata["chain_of_thought"] = eval_text

    was_correct = parse_eval_answer_fn(eval_text, generated_sample)
    generated_sample.metadata["score"] = "C" if was_correct else "I"
    generated_sample.metadata["model_answer"] = eval_text


def normalize_target(target: Any, num_choices: int) -> List[int]:
    """
    Normalizes the target (string or list) into valid integer indices
    to handle multiple choice letter references (A -> 0, B -> 1, etc.).
    """
    indices: List[int] = []
    if isinstance(target, str):
        target_letters = re.findall(r"[A-Za-z]", target.upper())
        for t in target_letters:
            idx = ord(t) - ord("A")
            if 0 <= idx < num_choices:
                indices.append(idx)
    elif isinstance(target, list):
        for t in target:
            if isinstance(t, str):
                t_letters = re.findall(r"[A-Z]", t.upper())
                for letter in t_letters:
                    idx = ord(letter) - ord("A")
                    if 0 <= idx < num_choices:
                        indices.append(idx)
            elif isinstance(t, int) and 0 <= t < num_choices:
                indices.append(t)
    return indices


def parse_answers_string(answer_response: str) -> re.Match[str] | None:
    """
    Convenience function for extracting answers from the answer response.

    NOTE: This is a lightly modified version of the parse_answers function from _multiple_choice.py

    The generated response must be in the format 'ANSWER: <answers>',
    otherwise we can't extract what the model thinks is "true". We can be a
    bit flexible whether these are "AB" vs "A,B" vs "A B".

    However, if the answer isn't in the expected format the model has
    failed in the task so we'll ultimately just mark it as incorrect
    """
    # First check whether the string strictly ends with the expected answer
    # In this case, we're looking for a single line which contains the expected
    # ANSWER: B,C string with only whitespace after it
    match = re.search(
        r"(?i)^ANSWER\s*:\s*([A-Za-z ,]+)\s*(?:$|\n)",
        answer_response,
        flags=re.MULTILINE,
    )

    # If we couldn't match the strict version, we can try the less strict
    # version for backward compatibility
    if match is None:
        return re.search(
            r"(?i)ANSWER\s*:\s*([A-Za-z ,]+)(?:[^\w]|\n|$)", answer_response
        )
    else:
        return match

def parse_eval_answer(
    eval_text: str, sample: Sample, normalize_fn: Callable[[Any, int], List[int]]
) -> bool:
    """
    Compares the model's normalized answers to the sample's target.
    Returns True if they match exactly, False otherwise.
    """
    if "adjusted_target" in sample.metadata:
        normalized_target_idxs = sample.metadata["adjusted_target"]
    else:
        normalized_target_idxs = normalize_fn(sample.target, len(sample.choices))

    match = parse_answers_string(eval_text)
    if match and match.group(1):
        # Split and filter to only include single-character answers (A, B, C, etc.)
        answer_letters = [letter for letter in re.split(r"[,\s]+", match.group(1).strip().upper()) 
                         if letter and len(letter) == 1]
        sample.metadata["model_answer"] = match.group(1).strip().upper()
    else:
        answer_letters = []
        sample.metadata["model_answer"] = ""

    answer_idxs = [ord(letter) - ord("A") for letter in answer_letters]
    is_correct = set(answer_idxs) == set(normalized_target_idxs)
    sample.metadata["score"] = "C" if is_correct else "I"
    return is_correct


@solver
def multiple_choice_save_cot(
    *,
    template: str | None = None,
    cot: bool = False,
    multiple_correct: bool = False,
    shuffle: bool | random.Random = False,
) -> Solver:
    """Multiple choice question solver.

    Formats a multiple choice question prompt, then calls `generate()`

    ### Usage

    Note that due to the way this solver works, it has some constraints:

        1. The `Sample` must have the `choices` attribute set.
        2. The only built-in compatible scorer is the `choice` scorer.
        3. It calls `generate()` internally, so you don't need to call it again

    ### Shuffling

    If the choices are shuffled, we will unshuffle them in the message history
    after the model has been called, essentially rewriting history. It is
    something to be aware of if writing custom scorers or solvers that interact
    with this scorer.

    Args:
      template (str | None): Template to use for the multiple choice question.
        The defaults vary based on the options and are taken from the `MultipleChoiceTemplate` enum. The template will have questions and possible answers substituted into it before being sent to the model. Consequently it requires three specific template variables:
        - `{question}`: The question to be asked.
        - `{choices}`: The choices available, which will be formatted as a
            list of A) ... B) ... etc. before sending to the model.
        - `{letters}`: (optional) A string of letters representing the choices, e.g.
            "A,B,C". Used to be explicit to the model about the possible answers.
      cot (bool): Default `False`. Whether the solver should perform chain-of-thought
        reasoning before answering. NOTE: this has no effect if you provide a custom template.
      multiple_correct (bool): Default `False`. Whether to allow multiple
        answers to the multiple choice question. For example, "What numbers are
        squares? A) 3, B) 4, C) 9" has multiple correct answers, B and C. Leave
        as `False` if there's exactly one correct answer from the choices
        available. NOTE: this has no effect if you provide a custom template.
      shuffle (bool | Random): Default `False`. Whether to shuffle the choices
        in the multiple.  Passing a `Random` instance will use that for shuffling,
        if `True` a new `Random` instance will be created.
    """
    if template and not valid_template(template):
        raise ValueError(
            "The template must contain '{question}' and '{choices}' placeholders for string substitution."
        )

    if template is None:
        if multiple_correct:
            if cot:
                template = MULTIPLE_ANSWER_TEMPLATE_COT
            else:
                template = MULTIPLE_ANSWER_TEMPLATE
        else:
            if cot:
                template = SINGLE_ANSWER_TEMPLATE_COT
            else:
                template = SINGLE_ANSWER_TEMPLATE

    template = resource(template)

    if shuffle is True:
        shuffle = random.Random()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("The multiple_choice solver requires samples with choices")

        if isinstance(shuffle, random.Random):
            state.choices.shuffle(shuffle)

        # Memoise the current prompt (which is the raw "question" part of the
        # sample). Required in case we unshuffle, because we then alter message
        # history based on the multiple-choice template.
        original_question = state.user_prompt.text

        state.user_prompt.text = prompt(
            question=state.user_prompt.text,
            choices=state.choices,
            template=str(template),
        )

        state = await generate(state)
        # save the entire model response as chain-of-thought
        state.store.set("chain_of_thought", state.output.completion)


        answers = parse_answers(state)
        if answers and answers.group(1):
            # If we've found answers, update the state appropriately
            set_choices_based_on_generated_response(
                state=state, answers=answers.group(1)
            )

            if shuffle:
                pretend_we_didnt_shuffle(
                    state=state,
                    original_question=original_question,
                    template=str(template),
                )

        return state

    return solve