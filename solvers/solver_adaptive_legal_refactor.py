from typing import List, Optional, Tuple

from inspect_ai.dataset import Sample
from inspect_ai.solver import solver
from inspect_ai.solver import Generate, TaskState
from inspect_ai.solver._multiple_choice import SINGLE_ANSWER_TEMPLATE, SINGLE_ANSWER_TEMPLATE_COT
from prompting.adaptive_prompts import get_self_check_judge_prompt, parse_self_check_response
from solvers.adaptive_utils import check_with_self_and_eval, final_eval_check_and_store, generate_question_with_retries, parse_log_and_sample
import json
import re
import logging
import random
import torch

logger = logging.getLogger(__name__)


CUSTOM_SINGLE_ANSWER_TEMPLATE = "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of letters."
CUSTOM_SINGLE_ANSWER_TEMPLATE_COT = "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of letters. Think step by step before answering."


def parse_generated_legal_question(generated_text: str, placeholder_keys: List[str]) -> Optional[Sample]:
    """
    Parses a generated JSON string to extract legal question data.
    Returns a Sample if successful, None otherwise.
    """
    try:
        if generated_text.startswith("```json"):
            generated_text = generated_text[7:].lstrip("\n")
            if generated_text.endswith("```"):
                generated_text = generated_text[:-3]
        elif generated_text.startswith("json\n"):
            generated_text = generated_text[5:]

        data = json.loads(generated_text)
        
        # Validate required keys
        for key in placeholder_keys:
            if key not in data:
                logger.debug(f"Missing key '{key}' in generated data")
                return None
        if "answer" not in data:
            logger.debug("Missing 'answer' key in generated data")
            return None

        # Extract text and answer
        text = data.get('text', '').strip()
        answer = data.get('answer', '').strip()
        
        if not text or not answer:
            logger.debug("Empty text or answer in generated data")
            return None

        # Check for reasoning fields
        if "reasoning_for_question" in data or "reasoning_for_answer" in data:
            # If reasoning fields exist, include them in metadata
            return Sample(
                input=text,
                target=answer,
                metadata={
                    'sample_data': data,
                    'reasoning_for_question': data.get('reasoning_for_question', ''),
                    'reasoning_for_answer': data.get('reasoning_for_answer', '')
                }
            )
        else:
            # Original behavior if no reasoning fields
            return Sample(
                input=text,
                target=answer,
                metadata={'sample_data': data}
            )
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse JSON: {e}")
        return None


def extract_reasoning_and_question_from_json(generation_text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extracts the reasoning_for_question field from JSON text.
    Returns the reasoning and question if found, otherwise returns None.
    """
    try:
        # Clean up markdown code blocks if present
        if generation_text.startswith("```json"):
            generation_text = generation_text[7:].lstrip("\n")
            if generation_text.endswith("```"):
                generation_text = generation_text[:-3]
        elif generation_text.startswith("json\n"):
            generation_text = generation_text[5:]

        # Parse the JSON data
        data = json.loads(generation_text)
        
        # Extract reasoning_for_question if it exists
        reasoning = data.get('reasoning_for_question', '')
        question = data.get('text', '')
        
        # Make sure we're dealing with strings before calling strip()
        if reasoning and isinstance(reasoning, str) and question and isinstance(question, str):
            return reasoning.strip(), question.strip()
        return None, None
    except json.JSONDecodeError:
        return None, None

def build_legal_context(
    sampled_correct: List[Sample],
    sampled_incorrect: List[Sample],
    randomize_sampling: bool,
    cot_in_context: bool,
) -> str:
    """
    Builds the legal context string from sampled examples.
    """
    context_examples = ""
    for idx, sample_item in enumerate(sampled_correct + sampled_incorrect):
        if not randomize_sampling:
            status = "Correctly" if sample_item in sampled_correct else "Incorrectly"
        else:
            status = "Previously"
            
        # show the long initial prompt only for first example, and the other questions for other examples
        # this works for maud_* tasks
        if idx == 0:
            sample_data = sample_item.input
        else:
            sample_data = sample_item.metadata["base_text"]

        sample_data = json.dumps(sample_data)
        
        what_to_do = "(use question/response to understand what the model does well)" if status == "Correctly" else "(use question/response to understand what the model does poorly)"
        context_examples += f"{status} Answered Example {what_to_do}:\n"
        context_examples += f"Sample Data:\n{sample_data}\n"
        
        if cot_in_context:
            context_examples += f"Reasoning trace from the model for this example:\n{sample_item.store.get('chain_of_thought')}\n"
            
        context_examples += f"Correct answer for this example:\n{sample_item.target}\n\n"
    
    return context_examples


def get_legal_generation_prompt(
    task_name: str,
    context: str,
    placeholder_keys: List[str],
    use_cot_generator: bool = False,
    include_previous_reasoning: bool = False,
) -> str:
    """
    Constructs the generation prompt for legal questions.
    """
    if use_cot_generator:
        json_format = '{"reasoning_for_question": [REASONING], "text": [TEXT], "reasoning_for_answer": [REASONING], "answer": [ANSWER]}'
        reasoning_prompt = ("'reasoning_for_question' is your step by step reasoning for the question: **Please make this reasoning as long and detailed as possible, we would like to really find model specific errors**, (for example, "
                          "'The model seems to have been confused about the relative importance of the clauses of the statute, "
                          "and has interpreted them incorrectly; in particular [...] I will make a similar question but have a different clause, [...] because [...],"
                          "emphasized in my invented document'. ), 'reasoning_for_answer' is your step by step reasoning for the "
                          "answer, here you should explain why the answer to your constructed question is correct (for example, "
                          "'Because the question asks about statute A during time period B, the correct answer is clause A during "
                          "time period B instead of statute X during time period Y')")
    else:
        json_format = '{"text": [TEXT], "answer": [ANSWER]}'
        reasoning_prompt = ""

    generation_prompt = (
        "You are to generate a new data sample for the following LegalBench task.\n\n"
        f"Task Name: {task_name}\n\n"
        "Please make your example difficult to answer correctly, considering the examples provided. "
        "Note that you should make your example distinct from all of the examples provided.\n\n"
        f"Instructions:\n"
        f"- Generate appropriate values for each of the placeholder keys.\n"
        f"- The values should be suitable for the task.\n"
        f"- Output the result as a JSON object with keys corresponding to the placeholders.\n"
        f"- Ensure that the 'answer' key is included and contains the correct answer.\n"
        f"- Do not include any additional text outside the JSON object.\n\n"
        f"Consider the examples provided, and how they might have caused the model to incorrectly answer the question. "
        f"Make your question more like the examples that were answered incorrectly, but make sure that it is distinct from the examples provided. "
        f"Your goal is to understand why the failure modes of the model in the incorrectly answered examples, and to use this failure model to construct a new difficult question.\n\n"
        f"The failure modes of the model should be very specific, generalizing across the examples provided in order to inform the construction of a new question. "
        f"Here are some questions that the model has answered, with reasoning traces that you should use to inform your failure mode analysis and question construction. Note that when you generate your question, you will have to include all relevant legal text to answer the question (like the first example provided):\n"
        f"{context}\n"
        f"Now, generate a new data sample. Again, note that you are to make this question extremely difficult to answer correctly. "
        f"Please format your JSON like {json_format}, where {reasoning_prompt}[TEXT] is the full text of the question, including all details. That is, you will have to include all relevant legal text to answer the question (like the first example provided above). You will have to include the full propmt for how to answer the question, including all relevant legal text / potential multiple choice options."
        f"(like documents, etc., but not including the letter of the answer-- do not, under any circumstances, include the letter of the answer in the text field) "
        f"and [ANSWER] is the letter of the correct answer to the question. Do not prepend or append anything to your JSON, just the brackets and the keys and values. "
        f"Please be sure to include all {4 if use_cot_generator else 2} keys in the JSON."
    )

    return generation_prompt


@solver
def adaptive_legal_solver_refactor(
    initial_log_path: str,
    task_name: str,
    n_positive_samples: int = 5,
    n_negative_samples: int = 5,
    randomize_sampling: bool = False,
    generator_model_name: str = "openai/gpt-4",
    eval_model_name: str = "openai/gpt-4",
    self_check_model_name: Optional[str] = None,
    use_embeddings: bool = False,
    embeddings_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    similarity_threshold: float = 0.6,
    score_threshold: int = 4,
    cot_in_context: bool = False,
    use_cot_generator: bool = False,
    use_cot_evaluator: bool = False,
    use_claude: bool = False,
    use_example: bool = True,
    max_attempts: int = 30,
    use_eval_model_for_checker: bool = False,
    original_eval_model_name: Optional[str] = None,
    include_previous_reasoning: bool = False,
    previous_reasoning_limit: int = 0,
    resample_questions: bool = False,
) -> Generate:
    """
    Refactored adaptive solver for legal questions.
    
    Args:
        resample_questions: If True, resample the in-context examples after every question attempt.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if "generated_sample" in state.store:
            state.completed = True
            return state

        # Load base prompt and extract placeholder keys
        if use_claude:
            prompt_template_path = f"/Users/davisbrown/adaptive_evals/legalbench/tasks/{task_name}/claude_prompt.txt"
        elif use_example:
            prompt_template_path = f"/Users/davisbrown/adaptive_evals/legalbench/tasks/{task_name}/base_prompt.txt"
        else:
            prompt_template_path = f"/Users/davisbrown/adaptive_evals/legalbench/tasks/{task_name}/base_prompt_wo_example.txt"

        with open(prompt_template_path, "r") as f:
            base_prompt = f.read()

        placeholder_keys = [key.strip() for key in set(re.findall(r"{{(.*?)}}", base_prompt))]
        if not placeholder_keys:
            state.error = "No placeholders found in the base prompt."
            state.completed = True
            return state

        # Load all samples for potential resampling
        all_correct_samples, all_incorrect_samples = await parse_log_and_sample(
            initial_log_path,
            n_positive_samples,
            n_negative_samples,
            randomize_sampling,
            score_key="C",
        )
        
        # Store all samples in state for resampling
        state.store.set("all_correct_samples", all_correct_samples)
        state.store.set("all_incorrect_samples", all_incorrect_samples)
        
        # Initial sampling
        sampled_correct = random.choices(all_correct_samples, k=min(n_positive_samples, len(all_correct_samples))) if all_correct_samples else []
        sampled_incorrect = random.choices(all_incorrect_samples, k=min(n_negative_samples, len(all_incorrect_samples))) if all_incorrect_samples else []

        # Initial context
        context = build_legal_context(
            sampled_correct,
            sampled_incorrect,
            randomize_sampling,
            cot_in_context,
        )

        existing_questions = [s.input for s in sampled_correct + sampled_incorrect]

        # Setup embedding model if needed
        embedding_model = None
        existing_question_embeddings = None
        if use_embeddings:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer(embeddings_model_name)
            if existing_questions:
                existing_question_embeddings = embedding_model.encode(
                    existing_questions,
                    convert_to_tensor=True,
                )

        async def accept_question(candidate: Sample) -> bool:
            """Self-check + eval-check acceptance function."""
            def _self_check_prompt_fn(s: Sample) -> str:
                return get_self_check_judge_prompt(
                    s.input,
                    s.choices,
                    [s.target],
                )

            def _parse_self_check_score(resp: str) -> int:
                result = parse_self_check_response(resp)
                return result.get("score", 0)

            def _format_eval_prompt_fn(s: Sample) -> str:
                multiple_choice_template = CUSTOM_SINGLE_ANSWER_TEMPLATE_COT if use_cot_evaluator else CUSTOM_SINGLE_ANSWER_TEMPLATE
                return multiple_choice_template + "\n\n" + s.input

            # Here we mimic solver_adaptive_legal.py: simply check if final model output matches candidate.target
            def _parse_eval_answer_fn(eval_text: str, samp: Sample) -> bool:
                correct_answer = str(samp.target).strip()
                model_answer = eval_text.strip()
                if "ANSWER:" in model_answer:
                    model_answer = model_answer.split("ANSWER:")[-1].strip()
                return model_answer == correct_answer

            return await check_with_self_and_eval(
                candidate_sample=candidate,
                self_check_model_name=self_check_model_name,
                eval_model_name=eval_model_name,
                score_threshold=score_threshold,
                use_eval_model_for_checker=use_eval_model_for_checker,
                self_check_prompt_fn=_self_check_prompt_fn,
                parse_self_check_score=_parse_self_check_score,
                format_eval_prompt_fn=_format_eval_prompt_fn,
                parse_eval_answer_fn=_parse_eval_answer_fn,
                state=state,
                store_chain_of_thought=True,
            )

        def _generation_prompt_fn(ctx: str, current_attempt: int) -> str:
            """
            Builds the final prompt used to generate new questions.
            Resamples in-context examples if resample_questions is True.
            """
            # Use the original context for the first attempt
            if not resample_questions or current_attempt == 0:
                return get_legal_generation_prompt(
                    task_name=task_name,
                    context=ctx,
                    placeholder_keys=placeholder_keys,
                    use_cot_generator=use_cot_generator,
                )
            
            # For subsequent attempts with resampling enabled, create a new context
            all_correct = state.store.get("all_correct_samples", [])
            all_incorrect = state.store.get("all_incorrect_samples", [])
            
            # Resample with replacement
            resampled_correct = random.choices(
                all_correct, 
                k=min(n_positive_samples, len(all_correct))
            ) if all_correct else []
            
            resampled_incorrect = random.choices(
                all_incorrect,
                k=min(n_negative_samples, len(all_incorrect))
            ) if all_incorrect else []
            
            # Build new context with resampled examples
            new_context = build_legal_context(
                resampled_correct,
                resampled_incorrect,
                randomize_sampling,
                cot_in_context,
            )
            
            # Store the new examples for novelty checking
            new_examples = [s.input for s in resampled_correct + resampled_incorrect]
            state.store.set(f"examples_attempt_{current_attempt}", new_examples)
            
            # Update embeddings if using them
            if use_embeddings and embedding_model is not None and new_examples:
                new_embeddings = embedding_model.encode(
                    new_examples,
                    convert_to_tensor=True,
                )
                state.store.set(f"embeddings_attempt_{current_attempt}", new_embeddings)
            
            return get_legal_generation_prompt(
                task_name=task_name,
                context=new_context,
                placeholder_keys=placeholder_keys,
                use_cot_generator=use_cot_generator,
            )

        def _parse_question_fn(text: str) -> Optional[Sample]:
            return parse_generated_legal_question(text, placeholder_keys)

        # Custom function to get current examples for novelty checking
        def _get_current_examples(attempt: int) -> List[str]:
            if not resample_questions or attempt == 0:
                return existing_questions
            return state.store.get(f"examples_attempt_{attempt}", existing_questions)
        
        # Custom function to get current embeddings for novelty checking
        def _get_current_embeddings(attempt: int) -> Optional[torch.Tensor]:
            if not resample_questions or attempt == 0:
                return existing_question_embeddings
            return state.store.get(f"embeddings_attempt_{attempt}", existing_question_embeddings)

        # Modify generate_question_with_retries to use our custom functions
        generated_sample = await generate_question_with_retries(
            state=state,
            context=context,
            generator_model_name=generator_model_name,
            max_attempts=max_attempts,
            generation_prompt_fn=_generation_prompt_fn,
            parse_question_fn=_parse_question_fn,
            existing_questions=existing_questions,  # Initial value, will be updated by _get_current_examples
            similarity_threshold=similarity_threshold,
            use_embeddings=use_embeddings,
            embedding_model=embedding_model,
            existing_question_embeddings=existing_question_embeddings,  # Initial value, will be updated by _get_current_embeddings
            check_question_fn=accept_question,
            include_previous_reasoning=include_previous_reasoning,
            previous_reasoning_limit=previous_reasoning_limit,
            extract_reasoning_and_question_fn=extract_reasoning_and_question_from_json,
            get_current_examples_fn=_get_current_examples if resample_questions else None,
            get_current_embeddings_fn=_get_current_embeddings if resample_questions and use_embeddings else None,
        )

        if not generated_sample:
            state.error = f"Failed to generate valid sample after {max_attempts} attempts"
            state.completed = True
            return state

        def _format_eval_prompt_fn(s: Sample) -> str:
            multiple_choice_template = CUSTOM_SINGLE_ANSWER_TEMPLATE_COT if use_cot_evaluator else CUSTOM_SINGLE_ANSWER_TEMPLATE
            return multiple_choice_template + "\n\n" + s.input

        def _parse_eval_answer_fn(eval_text: str, s: Sample) -> bool:
            correct_answer = str(s.target).strip()
            model_answer = eval_text.strip()
            if "ANSWER:" in model_answer:
                model_answer = model_answer.split("ANSWER:")[-1].strip()
            return model_answer == correct_answer

        if not use_eval_model_for_checker or generated_sample.metadata.get("score") is None:
            await final_eval_check_and_store(
                generated_sample,
                eval_model_name,
                _format_eval_prompt_fn,
                _parse_eval_answer_fn,
                store_chain_of_thought=True,
            )

        state.store.set("generated_sample", generated_sample)

        if original_eval_model_name:
            state.store.set("original_eval_model_name", original_eval_model_name)
        else:
            state.store.set("original_eval_model_name", eval_model_name)

        state.scores = [generated_sample.metadata["score"]]
        return state

    return solve