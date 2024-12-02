from inspect_ai.scorer import scorer, Score, Target, mean
from inspect_ai.solver import TaskState
import re

#Utils
def extract_probability(text: str) -> float:
    """Extract probability value from text enclosed in asterisks"""
    # Look for numbers between asterisks
    pattern = r"\*([\d\.]+)\*"
    matches = re.findall(pattern, text)
    
    if matches:
        try:
            prob = float(matches[-1])  # Take last match
            if 0 <= prob <= 1:
                return prob
        except ValueError:
            pass
            
    return 0.5  # Default fallback


#Brier Score
@scorer(metrics=[mean()])
def brier_score():
    async def score(state: TaskState, target: Target) -> Score:
        try:
            # Check if we have model output
            if state.output is None or not hasattr(state.output, 'completion'):
                return Score(
                    value=1.0,  # Worst possible Brier score
                    answer="[NO OUTPUT]",
                    target=target,
                    explanation="Model did not generate any output"
                )
            
            # Extract prediction from model output
            output_text = state.output.completion
            if "*" in output_text:
                prediction = extract_probability(output_text)
            else:
                prediction = float(output_text)

            
            # Store the extracted prediction in metadata
            state.metadata['extracted_prediction'] = prediction
            
            # Get target value
            target_value = float(target.text)
            
            # Validate prediction and target
            if not (0 <= prediction <= 1) or not (0 <= target_value <= 1):
                return Score(
                    value=1.0,
                    answer=str(prediction),
                    target=target,
                    explanation=f"Prediction ({prediction}) or target ({target_value}) outside [0,1] range"
                )
            
            # Calculate Brier score
            brier = (prediction - target_value) ** 2
            
            return Score(
                value=brier,
                answer=str(prediction),
                target=target,
                explanation=f"Prediction: {prediction}, Target: {target_value}, Brier Score: {brier:.4f}"
            )
            
        except Exception as e:
            return Score(
                value=1.0,  # Worst possible Brier score
                answer=str(state.output.completion if state.output else "[NO OUTPUT]"),
                target=target,
                explanation=f"Error in scoring: {str(e)}"
            )
    
    return score