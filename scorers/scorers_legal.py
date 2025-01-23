from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    Scorer,
    stderr,
)
from inspect_ai.solver import TaskState

# Import the newly introduced judge-filtered constant and "judged" metrics.
# Adjust the import path if your project structure or naming differs.
from scorers.scorers_rewording import accuracy_judged, stderr_judged, JUDGE_FILTERED


@scorer(metrics=[accuracy(), stderr()])
def adaptive_legal_scorer() -> Scorer:
    """
    Scorer for the adaptive LegalBench task's multiple-choice correctness.
    """
    async def score(state: TaskState, target: Target) -> Score:
        try:
            generated_sample = state.store.get("generated_sample")
            raw_value = generated_sample.metadata["score"]
            answer = generated_sample.metadata.get("model_answer", "")

            return Score(
                value=CORRECT if raw_value == "C" else INCORRECT,
                answer=answer,
                explanation="Multiple-choice correctness from adaptive_legal_scorer.",
            )
        except Exception as e:
            state.error = str(e)
            return Score(
                value=INCORRECT,
                answer="[ERROR]",
                explanation=f"Error in adaptive_legal_scorer: {e}",
            )
    return score


@scorer(metrics=[accuracy()])
def adaptive_legal_judge_scorer() -> Scorer:
    """
    Scorer that interprets the judgment from the judge model.
    """
    async def score(state: TaskState, target: Target) -> Score:
        try:
            generated_sample = state.store.get("generated_sample")
            judge_choice = generated_sample.metadata.get("judge_choice", None)
            judge_reasoning = generated_sample.metadata.get("judge_reasoning", "")

            if not judge_choice:
                state.error = "No judge_choice found in generated_sample metadata."
                return Score(
                    value=INCORRECT,
                    answer="[NO JUDGE CHOICE]",
                    target=target,
                    explanation="No judge_choice found.",
                )

            if judge_choice in ["A", "B"]:
                value = CORRECT
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


@scorer(metrics=[accuracy_judged(), stderr_judged()])
def adaptive_legal_scorer_judged() -> Scorer:
    """
    Variation of the adaptive_legal_scorer that also checks the judge's verdict.
    If the solver's correctness is "C" but the judge's choice is NOT A/B, we mark JUDGE_FILTERED.
    Else, we fall back to CORRECT or INCORRECT as usual.
    """
    async def score(state: TaskState, target: Target) -> Score:
        try:
            generated_sample = state.store.get("generated_sample")
            raw_value = generated_sample.metadata.get("score", "I")
            answer = generated_sample.metadata.get("model_answer", "")
            judge_choice = generated_sample.metadata.get("judge_choice", "")

            # If solver claims correct, check judge's verdict
            if raw_value == "C":
                # A / B => still correct; other => judge says there's a problem
                if judge_choice in ["A", "B"]:
                    final_value = CORRECT
                else:
                    final_value = JUDGE_FILTERED
            else:
                # Solver is incorrect => final is incorrect
                final_value = INCORRECT

            explanation = (
                "Multiple-choice correctness from adaptive_legal_scorer_judged.\n"
                f"Judge verdict => {judge_choice or '[NONE]'}"
            )
            return Score(
                value=final_value,
                answer=answer,
                explanation=explanation,
            )
        except Exception as e:
            state.error = str(e)
            return Score(
                value=INCORRECT,
                answer="[ERROR]",
                explanation=f"Error in adaptive_legal_scorer_judged: {e}",
            )
    return score
