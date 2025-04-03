import logging
import json
import re
import random
import torch
from typing import Any, Dict, List, Literal, Optional, Match

from inspect_ai.solver import solver, Generate, TaskState
from inspect_ai.model import GenerateConfig, get_model
from inspect_ai.dataset import Sample
from data.eval_log_processing import read_eval_log_async
from utils_elicitation.novelty import novelty_scorer

from solvers.adaptive_utils import (
    parse_log_and_sample,
    build_truthfulqa_context,  # We'll reuse this for now; will replace later
    generate_question_with_retries,
    final_eval_check_and_store,
    check_with_self_and_eval,
    parse_answers_string,
)

logger = logging.getLogger(__name__)


def build_politeness_context(
    sampled_correct: List[Sample],
    sampled_incorrect: List[Sample],
    randomize_sampling: bool,
    cot_in_context: bool,
) -> str:
    """
    Builds a context for politeness tasks by showing previously correct and incorrect predictions.
    Updated to include explicit letter mappings in the possible ratings for clarity.
    """
    if not randomize_sampling:
        lines = []
        for sample_item in sampled_correct + sampled_incorrect:
            status = "Correctly" if sample_item in sampled_correct else "Incorrectly"
            lines.append(f"**{status} Rated Utterance**: {sample_item.input}\n")
            # Include an explicit A..E mapping in the possible ratings
            lines.append("Possible Ratings: (A) -2, (B) -1, (C) 0, (D) +1, (E) +2\n")
            lines.append(f"Correct rating letter: {sample_item.target}\n")
            if "choice" in sample_item.scores:
                lines.append(f"Model's rating letter: {sample_item.scores['choice'].answer}\n\n")
            if cot_in_context and sample_item.score.value == "I":
                cot_text = sample_item.store.get("chain_of_thought")
                if cot_text:
                    lines.append(f"**Model's incorrect reasoning**:\n {cot_text}\n\n")
        return "\n".join(lines)
    else:
        # Show a generic summary based on the truthfulqa-style context
        return build_truthfulqa_context(sampled_correct, sampled_incorrect, True, cot_in_context)


def parse_generated_polite_utterance(generated_text: str) -> Optional[Sample]:
    """
    Expects a JSON structure with keys:
      {
         "utterance": "...",
         "politeness_rating": 0 (could be an int from -2 to +2),
      }
    If present, create a Sample with choices = [-2, -1, 0, 1, 2],
    the target = letter corresponding to that rating, e.g. rating = -2 -> "A".
    """
    try:
        json_str_match = re.search(r"\{.*?\}", generated_text, re.DOTALL)
        if not json_str_match:
            logger.debug("No JSON object found in the text.")
            return None
        json_str = json_str_match.group(0)

        data = json.loads(json_str)
        utterance = data.get("utterance", "").strip()
        rating = data.get("politeness_rating", None)

        if not utterance or rating is None:
            logger.debug("Missing required fields in the generated JSON.")
            return None

        # Our main 5 choices
        choices = [str(i) for i in [-2, -1, 0, 1, 2]]
        if str(rating) not in choices:
            logger.debug(f"Rating {rating} not in [-2, -1, 0, 1, 2].")
            return None

        # Convert rating to letter target
        rating_idx = choices.index(str(rating))
        target_letter = chr(ord("A") + rating_idx)

        return Sample(
            input=utterance,
            choices=choices,
            target=target_letter,
            metadata={},
        )
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse JSON: {e}")
        return None


def get_generation_prompt_politeness(context: str) -> str:
    """
    Prompt for generating a new utterance. Currently minimal; can be refined.
    The user can instruct the model to produce an utterance with a certain rating.
    """
    return f"""You are creating a brand new utterance (you may use any language, style, or context). We have some previous examples (correct vs incorrect) below, your goal is to create a new utterance that elicits an incorrect rating from the target model. You should analyze the examples, identify the common mistakes made by the model, and create an example that takes advantage of those mistakes. Your new utterance should be different from the examples below, but should use them for inspiration:
[BEGIN CONTEXT]
{context}
[END CONTEXT]

Please output a JSON object with the following structure:
{{
  "reasoning": "Your reasoning about the model mistakes for creating the new utterance. For example, the model may be confused by German words and culture, or it may have missed a detail about the context of the utterance.",
  "utterance": "A brand-new utterance showcasing politeness or impoliteness in some context",
  "politeness_rating": <integer from -2 to 2>
}}

-2 means very impolite, 2 means very polite. Provide variety.
"""


async def check_politeness_question(candidate_sample: Sample,
                                    self_check_model_name: Optional[str],
                                    eval_model_name: Optional[str],
                                    score_threshold: int,
                                    use_eval_model_for_checker: bool,
                                    state: TaskState
                                    ) -> bool:
    """
    Check the candidate utterance using optional self-check and optional eval-check, just as with truthfulqa logic.
    The model is 'correct' if it matches the rating or not. We *prefer* to find an utterance that forces the model to
    respond incorrectly. So if the model is correct, we reject the question. If the model is incorrect, we accept it.
    """
    def self_check_prompt_fn(sample: Sample) -> str:
        return f"""Check your own rating for the following utterance:
Utterance: {sample.input}
Possible Ratings: {', '.join(sample.choices)}
If your rating matches Politeness_rating, respond with a "score": 5
If you think it's incorrect, respond with a "score": 0
Output JSON: {{ "score": number }}"""

    def parse_self_check_score(text: str) -> int:
        # Look for a JSON with "score": <int>
        match = re.search(r"\"score\"\s*:\s*(\d+)", text)
        return int(match.group(1)) if match else 0

    def format_eval_prompt_fn(sample: Sample) -> str:
        """
        We'll reuse the multiple choice style prompt from the existing approach.
        Politeness rating from A) -2 ... E) +2
        We'll ask the model for the best rating. Then we compare to the candidate's target.
        """
        letters = ["A", "B", "C", "D", "E"]
        rating_str = f"{letters[0]}) -2\n{letters[1]}) -1\n{letters[2]}) 0\n{letters[3]}) 1\n{letters[4]}) 2\n"
        return f"""Consider this utterance for politeness:

Utterance: {sample.input}

The possible ratings are:
{rating_str}

Please pick the best rating. Respond in the format: ANSWER: X
"""

    def parse_eval_answer_fn(eval_text: str, sample: Sample) -> bool:
        # We check if the model rating letter matches sample.target
        # If it does, the model was correct -> we REJECT (False)
        match_obj = parse_politeness_answers(eval_text)
        if not match_obj or not match_obj.group(1):
            return False
        proposed_letter = match_obj.group(1).strip().upper()
        return proposed_letter == sample.target

    return await check_with_self_and_eval(
        candidate_sample,
        self_check_model_name=self_check_model_name,
        eval_model_name=eval_model_name,
        score_threshold=score_threshold,
        use_eval_model_for_checker=use_eval_model_for_checker,
        self_check_prompt_fn=self_check_prompt_fn,
        parse_self_check_score=parse_self_check_score,
        format_eval_prompt_fn=format_eval_prompt_fn,
        parse_eval_answer_fn=lambda text, samp: parse_eval_answer_fn(text, samp),
        state=state,
        store_chain_of_thought=True,
    )


@solver
def adaptive_politeness_solver(
    initial_log_path: str,
    n_positive_samples: int = 8,
    n_negative_samples: int = 8,
    generator_model_name: str = "openai/gpt-4o-mini",
    eval_model_name: str = "openai/gpt-4o-mini",
    self_check_model_name: Optional[str] = None,
    use_embeddings: bool = False,
    embeddings_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    similarity_threshold: float = 0.6,
    score_threshold: int = 4,
    max_attempts: int = 5,
    randomize_sampling: bool = False,
    cot_in_context: bool = False,
    use_cot_generator: bool = False,
    use_cot_evaluator: bool = False,
    original_eval_model_name: Optional[str] = None,
    use_eval_model_for_checker: bool = False,
    # Newly added param to match the approach in solver_adaptive_truthfulqa_refactor
    question_embedding_map: Optional[Dict[str, torch.Tensor]] = None,
) -> Generate:
    """
    Solver that adapts the politeness dataset by generating new utterances
    (with a variety of politeness ratings) that the model is likely to get wrong,
    and then final-evaluates them with the eval model.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if "generated_sample" in state.store:
            state.completed = True
            return state

        # 1) Load initial log and separate correct vs. incorrect from the log
        sampled_correct, sampled_incorrect = await parse_log_and_sample(
            initial_log_path,
            n_positive_samples,
            n_negative_samples,
            randomize_sampling=randomize_sampling,
            score_key="C",
        )

        # 2) Build context for the generator
        context = build_politeness_context(
            sampled_correct, sampled_incorrect, randomize_sampling, cot_in_context
        )

        # 3) Prepare existing utterances for novelty checks
        existing_questions = []
        eval_log = await read_eval_log_async(initial_log_path)
        if eval_log.samples:
            existing_questions = [s.input for s in eval_log.samples]

        existing_question_embeddings = None
        embedding_model = None

        if use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer

                # If we have a precomputed question_embedding_map, fetch embeddings for existing inputs.
                if question_embedding_map is not None:
                    valid_embeddings = [
                        question_embedding_map[q] for q in existing_questions if q in question_embedding_map
                    ]
                    if not valid_embeddings:
                        logger.debug("No embedded utterances found for existing questions. Disabling embedding usage.")
                    else:
                        existing_question_embeddings = torch.stack(valid_embeddings)
                        logger.debug("Using precomputed embeddings for existing utterances.")

                # Fallback: we can still load the embedding model to compute on the fly if needed
                embedding_model = SentenceTransformer(embeddings_model_name)
                logger.debug(f"Embedding model loaded: {embeddings_model_name}")
            except ImportError:
                logger.debug("Could not import sentence_transformers. Disabling embedding usage.")
                embedding_model = None

        # 4) Define a check_question function for novelty + acceptance
        async def check_question_fn(candidate: Sample) -> bool:
            # Check novelty if we have embeddings
            if use_embeddings and embedding_model and existing_question_embeddings is not None:
                # This will handle novelty scoring
                is_novel = await novelty_scorer(
                    candidate.input,
                    embedding_model,
                    existing_question_embeddings,
                    threshold=similarity_threshold,
                )
                if not is_novel:
                    return False
            # Then check if the candidate question is likely to produce an error from the model
            return await check_politeness_question(
                candidate,
                self_check_model_name=self_check_model_name,
                eval_model_name=eval_model_name,
                score_threshold=score_threshold,
                use_eval_model_for_checker=use_eval_model_for_checker,
                state=state,
            )

        # 5) Generate a new utterance (the "question") with retries
        candidate_sample = await generate_question_with_retries(
            state=state,
            context=context,
            generator_model_name=generator_model_name,
            max_attempts=max_attempts,
            generation_prompt_fn=get_generation_prompt_politeness,
            parse_question_fn=parse_generated_polite_utterance,
            existing_questions=existing_questions,
            similarity_threshold=similarity_threshold,
            use_embeddings=use_embeddings,
            embedding_model=embedding_model,
            existing_question_embeddings=existing_question_embeddings,
            check_question_fn=check_question_fn,
        )

        if not candidate_sample:
            state.error = "Failed to find an acceptable new utterance within max_attempts."
            state.completed = True
            return state

        # If we didn't use the eval model for checking, do a final pass now
        if eval_model_name and not use_eval_model_for_checker:
            await final_eval_check_and_store(
                candidate_sample,
                eval_model_name=eval_model_name,
                format_eval_prompt_fn=lambda samp: f"""Utterance: {samp.input}
Possible Ratings: {', '.join(samp.choices)}
Which letter is correct? (Format: ANSWER: X)""",
                parse_eval_answer_fn=lambda txt, s: (
                    parse_politeness_answers(txt).group(1).strip().upper() == s.target
                    if parse_politeness_answers(txt) and parse_politeness_answers(txt).group(1)
                    else False
                ),
                store_chain_of_thought=use_cot_evaluator,
            )

        # Store final results
        state.store.set("generated_sample", candidate_sample)
        # state.scores = {"score": candidate_sample.metadata.get("score", None)}

        if original_eval_model_name is not None:
            state.store.set("original_eval_model_name", original_eval_model_name)

        return state

    return solve

@solver
def adaptive_politeness_judge_solver(
    initial_log_path: str,
    judge_model_name: Optional[str] = None,
    num_attempts: int = 3,
) -> Generate:
    """
    Solver that judges newly generated politeness utterances, reusing
    the judging approach from truthfulqa. For now, uses the same prompt structure.
    We'll refine it later.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not judge_model_name:
            logger.debug("No judge model name provided, skipping judgment.")
            state.completed = True
            return state

        # Load the initial log if we need ground truth data
        initial_eval_log = await read_eval_log_async(initial_log_path)
        ground_truth_samples = initial_eval_log.samples

        # We might shuffle ground truth examples to show to the judge model
        ground_truth_questions = [(sample.input, sample.choices, sample.target) for sample in ground_truth_samples]
        random.shuffle(ground_truth_questions)

        generated_sample = state.store.get("generated_sample", None)
        if not generated_sample:
            state.error = "No generated sample found in state."
            state.completed = True
            return state

        ground_truth_examples = "\n".join([f"Utterance: {g[0]} | Labeled as: ({g[2]})" for g in ground_truth_questions[:4]])
        # Build the judge prompt
        judge_prompt = f"""\
Compare the newly generated politeness utterance to a set of ground-truth examples. Decide if the new utterance is classified correctly in the range between impolite and polite. 
The possible politeness ratings are from -2 to +2, and are mapped to letters A to E: (A) -2, (B) -1, (C) 0, (D) +1, (E) +2

You are to decide if the new utterance is correctly labeled in the range between impolite and polite. Note that the statement should be considered in context-- for example, some critism may be constructive and therefore have a neutral or positive score. Possible decisions:
(A) The generated utterance is labeled correctly.
(B) The generated utterance has a formatting issue but is otherwise acceptable.
(C) The generated utterance is incorrectly labeled.

Write your reasoning step by step, then choose your final answer with:
`select_choice(choice="A")`, `select_choice(choice="B")`, or `select_choice(choice="C")`.

As additional metadata, afterward please also note the language of the utterance. This will be easy to determine from the utterance. Choose with:
`select_language(language="English")`, or `select_language(language="German")`, etc.

[Ground Truth Examples and Ratings]
{ground_truth_examples}

[Generated Utterance]
{generated_sample.input}
[Generated Rating]
{generated_sample.target}
"""
        def parse_judge_choice(completion: str) -> Optional[str]:
            match = re.search(r'`select_choice\(choice=["\']([A-C])["\']\)`', completion)
            return match.group(1) if match else None

        def parse_language(completion: str) -> Optional[str]:
            # Look for language in quotes after select_language
            match = re.search(r'`select_language\(language=["\']([^"\']+)["\']\)`', completion)
            return match.group(1) if match else None

        judge_model = get_model(
            judge_model_name,
            config=GenerateConfig(max_connections=50, temperature=0.1)
        )

        for _ in range(num_attempts):
            try:
                resp = await judge_model.generate(judge_prompt)
                output = resp.completion.strip()
                choice = parse_judge_choice(output)
                language = parse_language(output)
                if choice:
                    generated_sample.metadata["judge_choice"] = choice
                    generated_sample.metadata["judge_reasoning"] = output
                    generated_sample.metadata["language"] = language
                    state.store.set("generated_sample", generated_sample)
                    state.completed = True
                    return state
            except Exception as e:
                logger.debug(f"Error during judge prompt: {e}")
                continue

        state.error = "Failed to get a valid judgment after maximum retry attempts for politeness."
        return state

    return solve

def parse_politeness_answers(text: str) -> Optional[Match]:
    """
    Parse a numeric or letter-based politeness rating (like ANSWER: -2) from the given text,
    then convert to a letter (A..E). If the text has "ANSWER: -2", we convert it to "A", etc.
    Returns a Match-like object whose group(1) is the final letter (A..E).
    """
    # Look for ANSWER: X, where X can be a letter A-E or an integer -2..2
    match_result = re.search(r"ANSWER:\s*([A-E]|-?\d+)", text, re.IGNORECASE)
    if not match_result:
        return None

    ans = match_result.group(1).strip().upper()
    rating_map = {"-2": "A", "-1": "B", "0": "C", "1": "D", "2": "E"}

    # Convert integer rating to letter if needed
    if ans in rating_map:
        ans = rating_map[ans]

    # Return a 'Match-like' object so we can call group(1) on it downstream
    class FakeMatch:
        def __init__(self, value: str):
            self._value = value

        def group(self, idx: Any = 1) -> str:
            return self._value

    return FakeMatch(ans)
