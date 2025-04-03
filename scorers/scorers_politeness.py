from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    Scorer,
)
from inspect_ai.solver import TaskState
from scorers.scorers_rewording import (
    JUDGE_FILTERED,
    accuracy_judged,
    stderr_judged,
)


@scorer(metrics=[accuracy()])
def adaptive_politeness_scorer() -> Scorer:
    """
    Baseline scorer for an adaptive Politeness evaluation.
    Checks the generated_sample's metadata["score"] to see if it is "C" or "I".
    """
    async def score(state: TaskState, target: Target) -> Score:
        try:
            generated_sample = state.store.get("generated_sample", None)
            if not generated_sample:
                state.error = "No generated_sample found in state.store."
                return Score(
                    value=INCORRECT,
                    answer="[NO SAMPLE]",
                    explanation="No generated_sample found in state.store.",
                )

            raw_value = generated_sample.metadata.get("score", None)
            answer = generated_sample.metadata.get("model_answer", "")

            if raw_value is None:
                explanation = "No 'score' field in generated_sample metadata."
                state.error = explanation
                return Score(
                    value=INCORRECT,
                    answer="[NO SCORE]",
                    explanation=explanation,
                    metadata={"generated_sample": generated_sample, "scorer_id": "adaptive_politeness_scorer"},
                )

            return Score(
                value=CORRECT if raw_value == "C" else INCORRECT,
                answer=answer,
                explanation="Multiple-choice correctness from adaptive_politeness_scorer.",
                metadata={
                    "generated_sample": generated_sample,
                    "scorer_id": "adaptive_politeness_scorer"
                },
            )
        except Exception as e:
            state.error = str(e)
            return Score(
                value=INCORRECT,
                answer="[ERROR]",
                explanation=f"Error in adaptive_politeness_scorer: {e}",
            )
    return score


@scorer(metrics=[accuracy()])
def adaptive_politeness_judge_scorer() -> Scorer:
    """
    Scorer that interprets the judgment from a separate judge model for politeness.
    Uses judge_choice in [A, B, C], where:
        A => correct and acceptable
        B => minor issues but acceptable
        C => invalid or incorrectly formed
    Here, A/B => CORRECT, C => INCORRECT.
    """
    async def score(state: TaskState, target: Target) -> Score:
        try:
            generated_sample = state.store.get("generated_sample")
            if not generated_sample:
                state.error = "No generated_sample found in state.store."
                return Score(
                    value=INCORRECT,
                    answer="[NO SAMPLE]",
                    explanation="No generated_sample found in state.store.",
                )

            judge_choice = generated_sample.metadata.get("judge_choice", None)
            judge_reasoning = generated_sample.metadata.get("judge_reasoning", "")

            if not judge_choice:
                state.error = "No judge_choice found in generated_sample metadata."
                return Score(
                    value=INCORRECT,
                    answer="[NO JUDGE CHOICE]",
                    explanation="No judge_choice found.",
                    metadata={
                        "generated_sample": generated_sample,
                        "scorer_id": "adaptive_politeness_judge_scorer",
                    },
                )

            if judge_choice in ["A", "B"]:
                value = CORRECT
            else:
                value = INCORRECT

            return Score(
                value=value,
                answer=f"Judge Choice: {judge_choice}",
                explanation=judge_reasoning,
                metadata={
                    "generated_sample": generated_sample,
                    "scorer_id": "adaptive_politeness_judge_scorer"
                },
            )
        except Exception as e:
            state.error = str(e)
            return Score(
                value=INCORRECT,
                answer="[ERROR]",
                explanation=str(e),
            )
    return score


@scorer(metrics=[accuracy_judged(), stderr_judged()])
def adaptive_politeness_scorer_judged() -> Scorer:
    """
    Variation of the adaptive_politeness_scorer that also checks the judge's verdict.
    If solver's correctness is 'INCORRECT' and the judge accepts it (A or B),
    we continue to treat it as 'INCORRECT'.
    If solver's correctness is 'INCORRECT' but the judge actually doesn't accept it,
    we mark it as JUDGE_FILTERED. Otherwise, we propagate CORRECT or INCORRECT as usual.
    """
    async def score(state: TaskState, target: Target) -> Score:
        try:
            generated_sample = state.store.get("generated_sample")
            if not generated_sample:
                state.error = "No generated_sample found in state.store."
                return Score(
                    value=JUDGE_FILTERED,
                    answer="[NO SAMPLE]",
                    explanation="No generated_sample found in state.store.",
                )

            raw_value = generated_sample.metadata.get("score", CORRECT)
            answer = generated_sample.metadata.get("model_answer", "")
            judge_choice = generated_sample.metadata.get("judge_choice", "")

            # If the judge accepts the sample, we use the original scoring
            if judge_choice in ["A", "B"]:
                final_value = raw_value
            else:
                final_value = JUDGE_FILTERED

            explanation = (
                "Multiple-choice correctness from adaptive_politeness_scorer_judged.\n"
                f"Judge verdict => {judge_choice or '[NONE]'}"
            )
            return Score(
                value=final_value,
                answer=answer,
                explanation=explanation,
                metadata={
                    "scorer_id": "adaptive_politeness_scorer_judged",
                    "generated_sample": generated_sample
                },
            )
        except Exception as e:
            state.error = str(e)
            return Score(
                value=JUDGE_FILTERED,  # fallback if something goes wrong
                answer="[ERROR]",
                explanation=f"Error in adaptive_politeness_scorer_judged: {e}",
            )
    return score
