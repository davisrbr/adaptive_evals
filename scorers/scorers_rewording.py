from inspect_ai.scorer import CORRECT, INCORRECT, PARTIAL, Metric, Score, Target, accuracy, metric, scorer, Scorer, stderr, value_to_float
from inspect_ai.scorer._choice import _choices_are_shuffled, _shuffled_explanation, _score_target, unshuffle_choices
from inspect_ai.solver import TaskState


# @scorer
# def choice_judged() -> Scorer:
#     """
#     A variant of the choice scorer that only marks a question as correct
#     if the multiple-choice selection is correct AND a `rewording_[dataset]_judge_solver`
#     solver has also marked it as correct in the TaskState metadata.

#     This scorer expects that:
#       • The multiple_choice solver marks the correct choice(s) in state.choices.
#       • The `rewording_[dataset]_judge_solver` solver sets a boolean
#         'rewording_[dataset]_judge_correct' in state.metadata indicating its judgment.

#     Returns:
#         A Scorer callable that computes a list of floats (1.0 if correct, 0.0 if not).
#     """
#     def _score(states: List[TaskState]) -> List[float]:
#         results = []
#         for state in states:
#             # Determine correctness from the multiple_choice solver
#             multiple_choice_correct = any(choice.is_target and choice.correct for choice in state.choices)

#             # Determine correctness as judged by rewording_[dataset]_judge_solver
#             judge_correct = bool(state.store.get("reworded_judge_choice", None) == "A")

#             # Final correctness only if both the multiple_choice solver and judge agree
#             results.append(1.0 if (multiple_choice_correct and judge_correct) else 0.0)
#         return results

#     return _score

JUDGE_FILTERED = "F"  # new constant for questions filtered by the judge

@metric("accuracy_judged")
def accuracy_judged() -> Metric:
    """
    Compute the accuracy of answers excluding any that are JUDGE_FILTERED.
    Scores with value == JUDGE_FILTERED are skipped.
    """
    def compute(scores: list[Score]) -> float:
        filtered_scores = [s for s in scores if s.value != JUDGE_FILTERED]
        # if there are no scores after filtering, return 1.0
        if not filtered_scores:
            return 1.0
        to_float = value_to_float()
        float_vals = [to_float(s.value) for s in filtered_scores]
        return sum(float_vals) / len(filtered_scores)
    return compute

@metric("count_incorrect_after_judged")
def count_incorrect_after_judged() -> Metric:
    """
    Compute the total number of samples that are answered incorrectly, excluding those that are JUDGE_FILTERED.
    """
    def compute(scores: list[Score]) -> float:
        filtered_scores = [s for s in scores if s.value == INCORRECT]
        # if there are no scores after filtering, return 1.0
        if not filtered_scores:
            return 0
        return len(filtered_scores)
    return compute


@metric("stderr_judged")
def stderr_judged() -> Metric:
    """
    Compute the standard error of the mean for answers (interpreted as floats) 
    excluding any that are JUDGE_FILTERED.
    """
    def compute(scores: list[Score]) -> float:
        # Filter out all JUDGE_FILTERED
        filtered_scores = [s for s in scores if s.value != JUDGE_FILTERED]
        if len(filtered_scores) <= 1:
            return 0.0
        to_float = value_to_float()
        float_vals = [to_float(s.value) for s in filtered_scores]
        mean_val = sum(float_vals) / len(float_vals)
        variance = sum((v - mean_val) ** 2 for v in float_vals) / (len(float_vals) - 1)
        return (variance / len(float_vals)) ** 0.5
    return compute

@scorer(metrics=[accuracy(), stderr()])
def choice() -> Scorer:
    """
    Original multiple-choice scorer. Marks a question CORRECT if model-selected
    answers match the target letters.
    """

    async def score(state: TaskState, target: Target) -> Score:
        choices = state.choices

        if _choices_are_shuffled(choices):
            explanation = _shuffled_explanation(choices)
            choices = unshuffle_choices(choices)
        else:
            explanation = state.output.completion

        target_positions, answers = _score_target(target, choices)
        generated_selected_choices = [
            i for i, choice in enumerate(choices) if choice.correct is True
        ]

        target_matches_choices = generated_selected_choices == sorted(target_positions)

        return Score(
            value=CORRECT if target_matches_choices else INCORRECT,
            answer=", ".join(answers),
            explanation=explanation,
        )

    return score

@scorer(metrics=[accuracy(), stderr()])
def judge_scoring() -> Scorer:
    """
    A scorer that only marks a question as CORRECT if the rewording_legal_judge_solver
    also marked the rewording as correct.
    """
    async def score(state: TaskState, target: Target) -> Score:
        judge_output = state.store.get("reworded_judge_choice", None)
        judge_reasoning = state.store.get("reworded_judge_reasoning", None)
        return Score(
            value=CORRECT if judge_output == "A" else PARTIAL if judge_output == "B" else INCORRECT,
            explanation=judge_reasoning,
        )
    return score

@scorer(metrics=[accuracy_judged(), stderr_judged()])
def choice_judged() -> Scorer:
    """
    Variation of the multiple-choice scorer that only marks a question as CORRECT
    if:
      1. The model-selected answers match the target letters, and
      2. The rewording_legal_judge_solver also marked the rewording as correct.

    This assumes the solver handling the judgment has placed a boolean flag
    (e.g., "rewording_judged_correct") in the state's metadata if the
    rewording_legal_judge_solver passed.
    """

    async def score(state: TaskState, target: Target) -> Score:
        choices = state.choices

        # If the model's multiple-choice answers were shuffled, unshuffle
        if _choices_are_shuffled(choices):
            explanation = _shuffled_explanation(choices)
            choices = unshuffle_choices(choices)
        else:
            explanation = state.output.completion

        # Regular choice-scoring logic
        target_positions, answers = _score_target(target, choices)
        generated_selected_choices = [
            i for i, choice in enumerate(choices) if choice.correct is True
        ]
        target_matches_choices = generated_selected_choices == sorted(target_positions)

        # Check if the rewording_legal_judge_solver flagged the rewording as correct
        # The solver can store a boolean like "rewording_judged_correct" in metadata.
        judged_correct = state.store.get("reworded_judge_choice", None) == "A"
        judged_reasoning = state.store.get("reworded_judge_reasoning", None)

        if target_matches_choices and judged_correct:
            score_value = CORRECT
        elif target_matches_choices and not judged_correct:
            score_value = JUDGE_FILTERED
        else:
            score_value = INCORRECT

        return Score(
            value=score_value,
            answer=", ".join(answers),
            explanation=(
                explanation
                + "\n\nRewording Judge Verdict: "
                + ("PASSED" if judged_correct else "FAILED")
                + "\n\n"
                + judged_reasoning
            ),
        )

    return score