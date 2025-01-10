from random import Random
from inspect_ai.solver import solver, Solver, TaskState, Generate
from inspect_ai.solver._multiple_choice import MULTIPLE_ANSWER_TEMPLATE, MULTIPLE_ANSWER_TEMPLATE_COT, SINGLE_ANSWER_TEMPLATE, SINGLE_ANSWER_TEMPLATE_COT
from inspect_ai.solver._multiple_choice import parse_answers, set_choices_based_on_generated_response, pretend_we_didnt_shuffle, valid_template, prompt
from inspect_ai.util import resource


@solver
def multiple_choice_save_cot(
    *,
    template: str | None = None,
    cot: bool = False,
    multiple_correct: bool = False,
    shuffle: bool | Random = False,
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
        shuffle = Random()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        if not state.choices:
            raise ValueError("The multiple_choice solver requires samples with choices")

        if isinstance(shuffle, Random):
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