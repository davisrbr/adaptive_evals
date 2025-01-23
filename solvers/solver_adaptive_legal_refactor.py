from typing import List, Optional

from inspect_ai.dataset import Sample
from inspect_ai.solver import solver
from inspect_ai.solver import Generate, TaskState
from inspect_ai.solver._multiple_choice import SINGLE_ANSWER_TEMPLATE, SINGLE_ANSWER_TEMPLATE_COT
from prompting.adaptive_prompts import get_self_check_judge_prompt, parse_self_check_response
from solvers.adaptive_utils import check_with_self_and_eval, normalize_target, parse_eval_answer, parse_log_and_sample
import json
import re
import logging

logger = logging.getLogger(__name__)

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

        return Sample(
            input=text,
            target=answer,
            metadata={'sample_data': data}
        )
    except json.JSONDecodeError as e:
        logger.debug(f"Failed to parse JSON: {e}")
        return None

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
    for sample_item in sampled_correct + sampled_incorrect:
        if not randomize_sampling:
            status = "Correctly" if sample_item in sampled_correct else "Incorrectly"
        else:
            status = "Previously"
            
        sample_data = sample_item.input
        sample_data = json.dumps(sample_data)
        
        context_examples += f"{status} Answered Example:\n"
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
) -> str:
    """
    Constructs the generation prompt for legal questions.
    """
    if use_cot_generator:
        json_format = '{"reasoning_for_question": [REASONING], "text": [TEXT], "reasoning_for_answer": [REASONING], "answer": [ANSWER]}'
        reasoning_prompt = ("'reasoning_for_question' is your step by step reasoning for the question, (for example, "
                          "'The model seems to have been confused about the relative importance of the clauses of the statute, "
                          "and has interpreted them incorrectly; I will make a similar question but have a different clause "
                          "emphasized in my invented document'), 'reasoning_for_answer' is your step by step reasoning for the "
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
        f"{context}\n"
        f"Now, generate a new data sample. Again, note that you are to make this question extremely difficult to answer correctly. "
        f"Consider the examples provided, and how they might have caused the model to incorrectly answer the question. "
        f"Make your question more like the examples that were answered incorrectly, but make sure that it is distinct from the examples provided. "
        f"Your goal is to understand why the failure modes of the model in the incorrectly answered examples, and to use this failure model to construct a new difficult question.\n\n"
        f"Please format your JSON like {json_format}, where {reasoning_prompt}[TEXT] is the full text of the question, including all details "
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
) -> Generate:
    """
    Refactored adaptive solver for legal questions.
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
            
        with open(prompt_template_path, 'r') as f:
            base_prompt = f.read()

        placeholder_keys = [key.strip() for key in set(re.findall(r"{{(.*?)}}", base_prompt))]
        if not placeholder_keys:
            state.error = "No placeholders found in the base prompt."
            state.completed = True
            return state

        # Load samples & build context
        sampled_correct, sampled_incorrect = await parse_log_and_sample(
            initial_log_path,
            n_positive_samples,
            n_negative_samples,
            randomize_sampling,
            score_key="C",
        )
        
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
                multiple_choice_template = SINGLE_ANSWER_TEMPLATE_COT if use_cot_evaluator else SINGLE_ANSWER_TEMPLATE
                return multiple_choice_template + "\n\n" + s.input

            def _parse_eval_answer_fn(eval_text: str, samp: Sample) -> bool:
                return parse_eval_answer(eval_text, samp, normalize_target)

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

        def _generation_prompt_fn(ctx: str) -> str:
            return get_legal_generation_prompt(
                task_name=task_name,
                context=ctx,
                placeholder_keys=placeholder_keys,
                use_cot_generator=use_cot_generator,
            )

        def _parse_question_fn(text: str) -> Optional[Sample]:
            return parse_generated_legal_question(text, placeholder_keys)

        generated_sample = await generate_question_with_retries(
            state=state,
            context=context,
            generator_model_name=generator_model_name,
            max_attempts=max_attempts,
            generation_prompt_fn=_generation_prompt_fn,
            parse_question_fn=_parse_question_fn,
            existing_questions=existing_questions,
            similarity_threshold=similarity_threshold,
            use_embeddings=use_embeddings,
            embedding_model=embedding_model,
            existing_question_embeddings=existing_question_embeddings,
            check_question_fn=accept_question,
        )

        if not generated_sample:
            state.error = f"Failed to generate valid sample after {max_attempts} attempts"
            state.completed = True
            return state

        def _format_eval_prompt_fn(s: Sample) -> str:
            multiple_choice_template = SINGLE_ANSWER_TEMPLATE_COT if use_cot_evaluator else SINGLE_ANSWER_TEMPLATE
            return multiple_choice_template + "\n\n" + s.input

        def _parse_eval_answer_fn(eval_text: str, s: Sample) -> bool:
            return parse_eval_answer(eval_text, s, normalize_target)

        # Final evaluation if needed
        if not use_eval_model_for_checker or generated_sample.metadata.get("score", None) is None:
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