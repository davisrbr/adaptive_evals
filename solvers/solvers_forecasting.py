from inspect_ai import solver
from inspect_ai.model import GenerateConfig, Model, ChatMessageUser
from inspect_ai.solver import (
    solver,
    TaskState,
    Generate, 
)


ZERO_SHOT_PROMPT = (
    """You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's date: {date_begin}
Question close date: {date_end}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. Do not output anything else.
Answer: {{ Insert answer here }}""",
    ("QUESTION", "BACKGROUND", "RESOLUTION_CRITERIA", "DATES"),
)


@solver
def zero_shot_forecasting_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            # Format the prompt using metadata
            prompt = ZERO_SHOT_PROMPT[0].format(
                question=state.input_text,
                background=state.metadata["background"],
                resolution_criteria=state.metadata["resolution_criteria"],
                date_begin=state.metadata["date_begin"],
                date_end=state.metadata["date_close"]
            )
            
            # Add the formatted prompt as a user message
            state.messages.append(ChatMessageUser(content=prompt))
            
            # Generate response using the model
            state = await generate(state)

            return state
            
        except Exception as e:
            # If there's an error, mark the state as completed
            state.completed = True
            # Store error info in metadata for debugging
            state.metadata['solver_error'] = str(e)
            return state
        
    return solve


