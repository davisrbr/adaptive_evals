import logging
import re
from inspect_ai.solver import Solver, solver, Generate, TaskState
from solvers.adaptive_utils import normalize_target
from solvers.solver_adaptive_truthfulqa import parse_answers_string

logger = logging.getLogger(__name__)

@solver
def evaluate_adaptive_truthfulqa_questions() -> Solver:
    """
    Evaluates questions using the model specified in the task configuration.
    
    Args:
        multiple_correct: Whether multiple answers can be correct
        temperature: Temperature for model generation
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # Generate response using the configured model
        state = await generate(state)
        
        # Parse answer from model output
        model_output = state.output.completion.strip()
        parse_match = parse_answers_string(model_output)
        
        if parse_match and parse_match.group(1):
            given_answer_str = parse_match.group(1).upper()
            answer_letters = re.split(r"[,\s]+", given_answer_str)
            answer_indices = [ord(x[0]) - ord("A") for x in answer_letters if x]
        else:
            try:
                given_answer = model_output.split("ANSWER:")[1].strip().upper()
                answer_letters = re.split(r",\s*", given_answer)
                answer_indices = [ord(letter[0]) - ord("A") for letter in answer_letters if letter]
            except IndexError:
                answer_indices = []

        # Score response
        normalized_target = normalize_target(state.target, len(state.choices))
        is_correct = set(answer_indices) == set(normalized_target)
        
        # Store results in state metadata
        state.metadata.update({
            "model_output": model_output,
            "choice_indices": answer_indices,
            "choice_str": parse_match.group(1) if parse_match else "",
            "score": "C" if is_correct else "I"
        })
        
        return state

    return solve 