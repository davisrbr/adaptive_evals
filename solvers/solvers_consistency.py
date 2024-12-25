from inspect_ai.solver import solver, TaskState, Generate 
from inspect_ai.model import ChatMessageUser, ModelOutput, GenerateConfig, get_model
from inspect_ai.log import read_eval_log
from typing import Dict, Any
from inspect_ai.dataset import Sample
import asyncio
from enum import Enum
import logging
import random
import re
from utils_consistency.consistency_question_generator import (
    question_generate, 
    llm_generate,
    question_generation_prompt
)
from utils_consistency.forecaster import BINARY_SCRATCH_PAD_PROMPT_NEW_6
import json
from utils_consistency.adaptive_consistency_generator import (
    AdversarialQuestionGenerator, 
    GenerationConfig,
    main_adversarial_generation
)
from datetime import datetime
import pandas as pd
import uuid
from typing import List

class ConsistencyType(Enum):
    NOT = "not"
    AND = "and"
    OR = "or"
    CONSEQUENCE = "consequence"
    PARAPHRASE = "paraphrase"

NUMBER_OF_RELATED_QUESTIONS = 1
NUM_ADAPTIVE_QUESTIONS_TO_GENERATE = 10

@solver
def consistency_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            original_data = state.metadata['original_data']
            
            # Format base question (P)
            base_question = {
                'id': state.metadata.get('id', 'original'),
                'title': original_data['title'],
                'body': original_data['body'],
                'resolution_date': original_data['resolution_date'],
                'created_date': original_data.get('created_date', ''),
                'question_type': 'binary',
                'data_source': state.metadata.get('data_source', 'original')
            }

            async def generate_forecast(question: Dict[str, Any]) -> Dict[str, Any]:
                """Generate a forecast for a single question using scratchpad"""
                try:
                    # Create a copy of the state for this forecast
                    forecast_state = TaskState(
                        input=question['title'],
                        sample_id=state.sample_id,
                        epoch=state.epoch,
                        messages=[],
                        model=state.model,
                        metadata=state.metadata
                    )

                    # Format prompt using scratchpad template
                    prompt = BINARY_SCRATCH_PAD_PROMPT_NEW_6.format(
                        question=question['title'],
                        background=question['body']
                    )
                    
                    # Add the formatted prompt as a user message
                    forecast_state.messages.append(ChatMessageUser(content=prompt))
                    
                    # Generate response using the model
                    response_state = await generate(forecast_state)
                    
                    # Extract content and probability
                    if response_state.output and hasattr(response_state.output, 'completion'):
                        content = response_state.output.completion
                        # Use the probability extraction from forecaster.py
                        pattern = r'\*([0-9]*\.?[0-9]+)\*'
                        match = re.search(pattern, content)
                        probability = float(match.group(1)) if match else 0.5
                    else:
                        content = "No output generated"
                        probability = 0.5

                    return {
                        'reasoning': content,
                        'probability': probability
                    }
                except Exception as e:
                    logging.error(f"Forecast generation error for {question['title']}: {str(e)}")
                    return {'reasoning': str(e), 'probability': 0.5}

            # Generate related questions
            related_questions = await asyncio.to_thread(
                question_generate,
                input_question=f"{original_data['title']}\n\n{original_data['body']}",
                prompt_template=question_generation_prompt,
                n_generated_relevant_questions=NUMBER_OF_RELATED_QUESTIONS
            )
            
            if not related_questions:
                raise ValueError("Failed to generate related questions")
            
            formatted_related = [{
                'id': q['id'],
                'title': q['title'],
                'body': q['body'],
                'resolution_date': original_data['resolution_date'],
                'created_date': original_data.get('created_date', ''),
                'question_type': 'binary',
                'data_source': 'generated'
            } for q in related_questions]

            async def check_consistency(ct: ConsistencyType):
                try:
                    logging.info(f"Starting consistency check for type: {ct.value}")
                    
                    if ct in {ConsistencyType.AND, ConsistencyType.OR}:
                        related_q = random.choice(formatted_related)
                        transformed = await asyncio.to_thread(
                            llm_generate,
                            operator=ct.value,
                            questions=[base_question, related_q]
                        )
                        
                        # Generate forecasts with reasoning
                        p_forecast = await generate_forecast(base_question)
                        q_forecast = await generate_forecast(related_q)
                        r_forecast = await generate_forecast(transformed)
                        
                        logging.info(f"Forecasts for {ct.value}:")
                        logging.info(f"P: {p_forecast['probability']}")
                        logging.info(f"Q: {q_forecast['probability']}")
                        logging.info(f"R: {r_forecast['probability']}")
                        
                        return {
                            'P': {**base_question, 'forecast': p_forecast['probability'], 'forecast_reasoning': p_forecast['reasoning']},
                            'Q': {**related_q, 'forecast': q_forecast['probability'], 'forecast_reasoning': q_forecast['reasoning']},
                            'R': {**transformed, 'forecast': r_forecast['probability'], 'forecast_reasoning': r_forecast['reasoning']}
                        }
                    else:
                        transformed = await asyncio.to_thread(
                            llm_generate,
                            operator=ct.value,
                            questions=[base_question]
                        )
                        
                        # Generate forecasts with reasoning
                        p_forecast = await generate_forecast(base_question)
                        q_forecast = await generate_forecast(transformed)
                        
                        logging.info(f"Forecasts for {ct.value}:")
                        logging.info(f"P: {p_forecast['probability']}")
                        logging.info(f"Q: {q_forecast['probability']}")
                        
                        return {
                            'P': {**base_question, 'forecast': p_forecast['probability'], 'forecast_reasoning': p_forecast['reasoning']},
                            'Q': {**transformed, 'forecast': q_forecast['probability'], 'forecast_reasoning': q_forecast['reasoning']}
                        }
                except Exception as e:
                    logging.error(f"Error in {ct.value} check: {str(e)}")
                    return None

            # Create and run tasks
            tasks = []
            for ct in ConsistencyType:
                tasks.append(asyncio.create_task(check_consistency(ct)))
            
            results = await asyncio.gather(*tasks)
            
            # Store results in metadata
            state.metadata['consistency_checks'] = {
                ct.value: result 
                for ct, result in zip(ConsistencyType, results) 
                if result is not None
            }
            
            # Set main output
            state.output = ModelOutput.from_content(
                state.model.name,
                str(state.metadata['consistency_checks'])
            )
            
            return state
            
        except Exception as e:
            state.completed = True
            state.metadata['solver_error'] = str(e)
            logging.error(f"Error in consistency solver: {str(e)}")
            return state
            
    return solve


@solver
def adaptive_consistency_solver(
    initial_log_path: str,
    consistency_types: List[str] = ['not', 'paraphrase', 'consequence', 'and', 'or'], 
    generator_model_name: str = "openai/gpt-4o"
) -> Generate:
    """
    Solver that generates adversarial consistency questions based on previous performance
    by extracting examples from eval logs.
    """
    generator_model = get_model(generator_model_name, config=GenerateConfig(temperature=0.5))
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        logging.info("=== Starting Adaptive Solver ===")
        logging.info(f"Initial state metadata: {state.metadata}")
        
        if 'consistency_checks' in state.metadata:
            state.completed = True
            return state
            
        try:
            
            # Load evaluation log and log details
            eval_log = read_eval_log(initial_log_path)
            logging.info(f"Loaded eval log with {len(eval_log.samples)} samples")
            
            # Check structure of first sample
            if eval_log.samples:
                sample = eval_log.samples[0]
                logging.info("\n=== Sample Structure ===")
                logging.info(f"Sample scores: {sample.scores}")
                logging.info(f"Sample metadata: {sample.metadata}")
                if 'consistency_checks' in sample.metadata:
                    logging.info(f"Sample consistency_checks: {sample.metadata['consistency_checks']}")
            
            # Create prompt with challenging examples from logs
            generation_prompt = """You are an expert in analyzing forecasting models and generating challenging questions.
Given the example questions and their scores below, generate new challenging question pairs/triplets.

Examples of questions that tested consistency poorly (had high inconsistency scores):\n"""

            # Get examples from logs for each consistency type
            for ct in consistency_types:
                generation_prompt += f"\n=== {ct.upper()} Examples ===\n"
                logging.info(f"\nProcessing consistency type: {ct}")
                
                # Get samples with this consistency type and sort by score
                scored_samples = [
                    sample for sample in eval_log.samples 
                    if ct in sample.metadata.get('consistency_checks', {})
                ]
                logging.info(f"Found {len(scored_samples)} samples for {ct}")
                
                if scored_samples:
                    # Sort by score if present in metadata
                    scored_samples.sort(key=lambda x: x.metadata['consistency_checks'][ct].get('score', 1.0), reverse=True)
                    
                    # Take 3 worst examples
                    for sample in scored_samples[:3]:
                        checks = sample.metadata['consistency_checks'][ct]
                        generation_prompt += f"\nP: {checks['P']['title']}\n"
                        generation_prompt += f"Q: {checks['Q']['title']}\n"
                        if 'R' in checks:  # For AND/OR
                            generation_prompt += f"R: {checks['R']['title']}\n"
                        generation_prompt += "---\n"

            # Add generation instructions using same format as AdversarialQuestionGenerator
            generation_prompt += f"\nGenerate {NUM_ADAPTIVE_QUESTIONS_TO_GENERATE} new challenging questions in this exact JSON format:"
            generation_prompt += """
{
    "question_pairs": [
        {
            "original": {
                "title": "Question P title",
                "body": "Full question P body with resolution criteria",
                "challenge_factors": ["List specific factors that make this pair/triplet challenging"]
            },
            "second": {
                "title": "Question Q title",
                "body": "Full question Q body with resolution criteria"
            },
            "combined": {
                "title": "Question R title (only for AND/OR)",
                "body": "Full question R body (only for AND/OR)"
            },
            "consistency": {
                "type": "not|consequence|paraphrase|and|or",
                "challenge_rationale": "Explain why this type was chosen and why it's challenging"
            }
        }
    ]
}"""
            logging.info("\n=== Generation Prompt ===")
            logging.info(generation_prompt)

            # Get generated questions
            response = await generator_model.generate(generation_prompt)
            response_text = response.completion
            logging.info("\n=== LLM Response ===")
            logging.info(response_text)
            
            # REPLACE THIS ENTIRE BLOCK with the new parsing code:
            try:
                # Extract questions JSON from the response
                if "```json" in response_text:
                    # If the response includes markdown formatting
                    json_start = response_text.find("{")
                    json_end = response_text.rfind("}") + 1
                    if json_start != -1 and json_end != -1:
                        response_text = response_text[json_start:json_end]
                
                # Parse JSON
                generated_questions = json.loads(response_text)['question_pairs']
                logging.info(f"\nSuccessfully parsed {len(generated_questions)} questions")
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse LLM response: {e}")
                logging.error(f"Response text: {response_text}")
                raise

            # Initialize consistency_checks dict
            if 'consistency_checks' not in state.metadata:
                state.metadata['consistency_checks'] = {}

            async def generate_forecast(q: Dict[str, Any]) -> Dict[str, Any]:
                forecast_state = TaskState(
                    input=q['title'],
                    sample_id=state.sample_id,
                    epoch=state.epoch,
                    messages=[],
                    model=state.model,
                    metadata=state.metadata
                )

                prompt = BINARY_SCRATCH_PAD_PROMPT_NEW_6.format(
                    question=q['title'],
                    background=q['body']
                )
                
                forecast_state.messages.append(ChatMessageUser(content=prompt))
                response_state = await generate(forecast_state)
                
                if response_state.output and hasattr(response_state.output, 'completion'):
                    content = response_state.output.completion
                    pattern = r'\*([0-9]*\.?[0-9]+)\*'
                    match = re.search(pattern, content)
                    probability = float(match.group(1)) if match else 0.5
                else:
                    content = "No output generated"
                    probability = 0.5

                return {
                    'reasoning': content,
                    'probability': probability
                }
            # Replace the current storage section with:

            # Initialize consistency_checks dict with ALL types
            state.metadata['consistency_checks'] = {
                ct: None for ct in consistency_types  # Initialize all types to None
            }


            # Process each generated question
            for i, question in enumerate(generated_questions):
                logging.info(f"\n=== Processing Question {i+1} ===")
                ct = question['consistency']['type']
                logging.info(f"Consistency type: {ct}")
                
                # Run forecasts on generated P and Q
                p_forecast = await generate_forecast(question['original'])
                q_forecast = await generate_forecast(question['second'])
                
                logging.info(f"P forecast: {p_forecast}")
                logging.info(f"Q forecast: {q_forecast}")

                # Create base question object (P)
                base_question = {
                    'id': str(uuid.uuid4()),  # or however you want to generate IDs
                    'title': question['original']['title'],
                    'body': question['original']['body'],
                    'resolution_date': question.get('resolution_date', '2024-06-29'),
                    'created_date': datetime.now().strftime('%Y-%m-%d'),
                    'question_type': 'binary',
                    'data_source': 'generated',
                    'forecast': p_forecast['probability'],
                    'forecast_reasoning': p_forecast['reasoning']
                }

                # Create Q question object
                transformed_question = {
                    'title': question['second']['title'],
                    'body': question['second']['body'],
                    'resolution_date': question.get('resolution_date', '2024-06-29'),
                    'question_type': 'binary',
                    'data_source': 'llm_generated',
                    'created_date': datetime.now().strftime('%Y-%m-%d'),
                    'id': f'transformed-{ct}-{base_question["id"]}',
                    'forecast': q_forecast['probability'],
                    'forecast_reasoning': q_forecast['reasoning']
                }

                check_result = {
                    'P': base_question,
                    'Q': transformed_question
                }

                # Add R forecast for AND/OR
                if ct in ['and', 'or'] and 'combined' in question:
                    r_forecast = await generate_forecast(question['combined'])
                    logging.info(f"R forecast: {r_forecast}")
                    combined_question = {
                        'title': question['combined']['title'],
                        'body': question['combined']['body'],
                        'resolution_date': question.get('resolution_date', '2024-06-29'),
                        'question_type': 'binary',
                        'data_source': 'llm_generated',
                        'created_date': datetime.now().strftime('%Y-%m-%d'),
                        'id': f'combined-{ct}-{base_question["id"]}',
                        'forecast': r_forecast['probability'],
                        'forecast_reasoning': r_forecast['reasoning']
                    }
                    check_result['R'] = combined_question

                # Store results
                state.metadata['consistency_checks'][ct] = check_result

                logging.info(f"Stored check result for {ct}")

            logging.info("\n=== Final State ===")
            logging.info(f"Final metadata: {state.metadata}")
            
            state.output = ModelOutput.from_content(
                state.model.name,
                str(state.metadata['consistency_checks'])
            )
            
        except Exception as e:
            state.error = f"Error in adaptive consistency solver: {str(e)}"
            state.completed = True
            logging.error(f"Error in solver: {str(e)}", exc_info=True)
            
        return state
        
    return solve