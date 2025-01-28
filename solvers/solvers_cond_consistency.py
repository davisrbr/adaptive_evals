from inspect_ai.solver import solver, TaskState, Generate 
import logging
import re
from utils_consistency.forecaster import BASIC_COT_FORECASTER
from inspect_ai.model import ChatMessageUser
from typing import Dict, Any
import asyncio
import numpy as np
import asyncio 
import numpy as np

# TODO: Add extremization, generates initial results for all questions
@solver
def cond_consistency_solver(N_forecasts: int = 5):
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            async def generate_forecast(question: Dict[str, str]) -> Dict[str, Any]:
                """Generate multiple forecasts and aggregate results"""
                # Generate multiple forecasts in parallel
                forecast_states = [
                    TaskState(
                        input=question['title'],
                        sample_id=state.sample_id,
                        epoch=state.epoch,
                        messages=[ChatMessageUser(content=BASIC_COT_FORECASTER.format(
                            question=f"{question['title']}\n\n{question['body']}"
                        ))],
                        model=state.model,
                        metadata=state.metadata
                    ) for _ in range(N_forecasts)
                ]

                # Run all forecasts in parallel
                responses = await asyncio.gather(*[generate(s) for s in forecast_states])
                
                probabilities = []
                reasonings = []
                
                for response_state in responses:
                    if response_state.output and hasattr(response_state.output, 'completion'):
                        content = response_state.output.completion
                        matches = re.findall(r'\*([0-9]*\.?[0-9]+)\*', content)
                        try:
                            prob = float(matches[-1]) if matches else 0.5
                            probabilities.append(prob)
                            reasonings.append(content)
                        except:
                            pass  # Handle any parsing errors
                
                # Handle case where no valid forecasts were generated
                if not probabilities:
                    return {
                        'average_probability': 0.5,
                        'median_reasoning': "No valid forecasts generated",
                        'all_probabilities': [],
                        'all_reasonings': []
                    }

                # Find median forecast
                sorted_indices = np.argsort(probabilities)
                median_idx = sorted_indices[len(probabilities)//2]
                
                return {
                    'average_probability': np.mean(probabilities),
                    'median_reasoning': reasonings[median_idx],
                    'all_probabilities': probabilities,
                    'all_reasonings': reasonings
                }

            # Get forecasts with reasoning for all three questions
            p_result = await generate_forecast({
                'title': state.metadata['P_title'],
                'body': state.metadata['P_body']
            })
            
            q_given_p_result = await generate_forecast({
                'title': state.metadata['Q_given_P_title'],
                'body': state.metadata['Q_given_P_body']
            })
            
            p_and_q_result = await generate_forecast({
                'title': state.metadata['P_and_Q_title'],
                'body': state.metadata['P_and_Q_body']
            })

            # Store forecasts and reasoning in metadata for scorer
            state.metadata['forecasts'] = {
                'P': p_result['average_probability'],
                'Q_given_P': q_given_p_result['average_probability'], 
                'P_and_Q': p_and_q_result['average_probability']
            }
            
            state.metadata['reasoning'] = {
                'P': p_result['median_reasoning'],
                'Q_given_P': q_given_p_result['median_reasoning'],
                'P_and_Q': p_and_q_result['median_reasoning']
            }

            # Store all forecasts and reasoning
            state.metadata['all_forecasts'] = {
                'P': p_result['all_probabilities'],
                'Q_given_P': q_given_p_result['all_probabilities'],
                'P_and_Q': p_and_q_result['all_probabilities']
            }
            
            state.metadata['all_reasoning'] = {
                'P': p_result['all_reasonings'],
                'Q_given_P': q_given_p_result['all_reasonings'],
                'P_and_Q': p_and_q_result['all_reasonings']
            }

            logging.info(f"Forecasts generated:")
            logging.info(f"P: {p_result['average_probability']:.3f}")
            logging.info(f"Q|P: {q_given_p_result['average_probability']:.3f}")
            logging.info(f"P∧Q: {p_and_q_result['average_probability']:.3f}")

            return state

        except Exception as e:
            logging.error(f"Error in cond_consistency_solver: {str(e)}")
            raise

    return solve
    
