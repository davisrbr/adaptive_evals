from inspect_ai.solver import TaskState
from inspect_ai.scorer import scorer, Score, Target, mean, stderr, Scorer, accuracy, CORRECT, INCORRECT
from solvers.solvers_consistency import ConsistencyType
import logging
from utils_consistency.metrics import (
    negation_frequentist_metric,
    or_frequentist_metric,
    and_frequentist_metric, 
    consequence_frequentist_metric, 
    paraphrase_frequentist_metric, 
    but_frequentist_metric, 
    cond_frequentist_metric,
    andor_frequentist_metric, 
    expevidence_frequentist_metric
)
import numpy as np
from typing import List

@scorer(
    metrics={
        **{f"{ct.value}_score": [mean(), stderr()] for ct in ConsistencyType},
        "overall_score": [mean(), stderr()]  # Add overall score metric
    }
)
def consistency_scorer(consistency_types: List[str] = None) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        checks = state.metadata.get('consistency_checks', {})
        # Initialize scores with defaults for all consistency types
        scores = {f"{ct.value}_score": None for ct in ConsistencyType}
        scores["overall_score"] = 0.0
        # scores = {}
        beta_min = 1e-3  # regularization term

        all_type_scores = []  # Store all scores to calculate overall mean
        
        for ct_value, check_data in checks.items():

            key = f"{ct_value}_score"
            logging.info(f"\n=== Scoring {ct_value} consistency ===")
            
            if not check_data:
                logging.warning(f"No check data for {ct_value}")
                scores[key] = None
                continue
            
            # Convert single check to list for uniform processing
            check_data_list = check_data if isinstance(check_data, list) else [check_data]
            type_scores = []
            
            for check in check_data_list:
                # Get forecasts
                p_forecast = check['P'].get('forecast', 0.0)
                q_forecast = check['Q'].get('forecast', 0.0)
                
                # Log the forecasts
                logging.info(f"P forecast: {p_forecast:.3f} - {check['P']['title']}")
                logging.info(f"Q forecast: {q_forecast:.3f} - {check['Q']['title']}")
                
                try:
                    # Score based on consistency type using frequentist metrics
                    if ct_value in consistency_types:
                        if ct_value == 'not':
                            score = negation_frequentist_metric(p_forecast, q_forecast, beta_min)
                            logging.info(f"NOT check: negation_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f})")
                            
                        elif ct_value == 'or':
                            r_forecast = check['R'].get('forecast', 0.0)
                            logging.info(f"R forecast: {r_forecast:.3f} - {check['R']['title']}")
                            score = or_frequentist_metric(p_forecast, q_forecast, r_forecast, beta_min)
                            logging.info(f"OR check: or_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f}, R={r_forecast:.3f})")
                            
                        elif ct_value == 'and':
                            r_forecast = check['R'].get('forecast', 0.0)
                            logging.info(f"R forecast: {r_forecast:.3f} - {check['R']['title']}")
                            score = and_frequentist_metric(p_forecast, q_forecast, r_forecast, beta_min)
                            logging.info(f"AND check: and_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f}, R={r_forecast:.3f})")
                        
                        elif ct_value == 'consequence':
                            score = consequence_frequentist_metric(p_forecast, q_forecast, beta_min)
                            logging.info(f"CONSEQUENCE check: consequence_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f})")
                        
                        elif ct_value == 'paraphrase':
                            score = paraphrase_frequentist_metric(p_forecast, q_forecast, beta_min)
                            logging.info(f"PARAPHRASE check: paraphrase_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f})")

                        elif ct_value == 'but':
                            r_forecast = check['R'].get('forecast', 0.0)
                            logging.info(f"R forecast: {r_forecast:.3f} - {check['R']['title']}")
                            score = but_frequentist_metric(p_forecast, q_forecast, r_forecast, beta_min)
                            logging.info(f"BUT check: but_frequentist_metric(P={p_forecast:.3f}, not P and Q={q_forecast:.3f}, P or Q={r_forecast:.3f})")

                        elif ct_value == 'cond':
                            r_forecast = check['R'].get('forecast', 0.0)
                            logging.info(f"R forecast: {r_forecast:.3f} - {check['R']['title']}")
                            score = cond_frequentist_metric(p_forecast, q_forecast, r_forecast, beta_min)
                            logging.info(f"COND check: but_frequentist_metric(P={p_forecast:.3f}, Q|P={q_forecast:.3f}, P and Q={r_forecast:.3f})")

                        elif ct_value == 'andor':
                            r_forecast = check['R'].get('forecast', 0.0)
                            s_forecast = check['S'].get('forecast', 0.0)
                            logging.info(f"R forecast: {r_forecast:.3f} - {check['R']['title']}")
                            logging.info(f"S forecast: {s_forecast:.3f} - {check['S']['title']}")
                            score = andor_frequentist_metric(p_forecast, q_forecast, r_forecast, s_forecast, beta_min)
                            logging.info(f"AndOr check: andor_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f}, P and Q={r_forecast:.3f}, P or Q={s_forecast:.3f})")

                        elif ct_value == 'expevidence':
                            r_forecast = check['R'].get('forecast', 0.0)
                            s_forecast = check['S'].get('forecast', 0.0)
                            logging.info(f"R forecast: {r_forecast:.3f} - {check['R']['title']}")
                            logging.info(f"S forecast: {s_forecast:.3f} - {check['S']['title']}")
                            score = expevidence_frequentist_metric(p_forecast, q_forecast, r_forecast, s_forecast, beta_min)
                            logging.info(f"ExpEvidence check: expevidence_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f}, P | Q={r_forecast:.3f}, P | -Q={s_forecast:.3f})")
                        
                        
                        check['score'] = float(score)
                        type_scores.append(score)
                        all_type_scores.append(score)  # Add to overall scores list
                        logging.info(f"Frequentist metric score: {score:.3f}\n")
                        
                except Exception as e:
                    logging.error(f"Error calculating {ct_value} metric: {str(e)}")
                    check['score'] = None
                    type_scores.append(0.0)
            
            # Average the scores for this type
            if type_scores:
                mean_score = sum(type_scores) / len(type_scores)
                stderr_score = np.std(type_scores, ddof=1) / np.sqrt(len(type_scores)) if len(type_scores) > 1 else 0
                scores[key] = mean_score
                logging.info(f"{key}: mean={mean_score:.3f}, stderr={stderr_score:.3f}")
            else:
                scores[key] = None

        # Calculate overall score as mean of all individual scores
        if all_type_scores:
            overall_mean = sum(all_type_scores) / len(consistency_types)
            overall_stderr = np.std(all_type_scores, ddof=1) / np.sqrt(len(consistency_types)) if len(consistency_types) > 1 else 0
            scores["overall_score"] = overall_mean
            logging.info(f"\n=== Overall Score: mean={overall_mean:.3f}, stderr={overall_stderr:.3f} ===")
        else:
            scores["overall_score"] = None
            logging.info("\n=== Overall Score: mean=0.000, stderr=0.000 ===")

        return Score(
            value=scores,
            answer=str(checks)
        )
    return score

@scorer(metrics=[accuracy()])
def adaptive_consistency_judge_scorer() -> Scorer:
    """
    Scorer that interprets the judgment from the judge model.
    """

    async def score(state: TaskState, target: Target) -> Score:
        try:
            generated_sample = state.store.get('generated_sample')
            judge_choice = generated_sample.metadata.get('judge_choice', None)
            judge_reasoning = generated_sample.metadata.get('judge_reasoning', '')

            if not judge_choice:
                state.error = "No judge_choice found in generated_sample metadata."
                return Score(
                    value=INCORRECT,
                    answer="[NO JUDGE CHOICE]",
                    target=target,
                    explanation="No judge_choice found."
                )

            if judge_choice == "A":
                value = CORRECT
            elif judge_choice == "B":
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

@scorer(metrics=[mean(), stderr()])
def temp_adaptive_consistency_judge_scorer() -> Scorer:
    """
    Scorer that processes the judgments from the adaptive consistency judge solver.
    Returns the accuracy of the judged samples.
    """
    
    async def score(state: TaskState, target: Target) -> Score:
        try:
            # Get accuracy from state metadata
            accuracy = state.metadata.get('accuracy', 0.0)
            judgments = state.metadata.get('judgments', [])
            
            # Calculate accuracy string
            judgment_counts = {
                'A': sum(1 for j in judgments if j['judgment'] == 'A'),
                'B': sum(1 for j in judgments if j['judgment'] == 'B'),
                'C': sum(1 for j in judgments if j['judgment'] == 'C')
            }
            
            judgment_summary = (
                f"Total samples: {len(judgments)}\n"
                f"Excellent (A): {judgment_counts['A']}\n"
                f"Acceptable (B): {judgment_counts['B']}\n"
                f"Unsuitable (C): {judgment_counts['C']}\n"
                f"Accuracy: {accuracy:.3f}"
            )

            return Score(
                value=accuracy,
                answer=judgment_summary,
                explanation=f"Processed {len(judgments)} judgments. Accuracy is ratio of acceptable (A/B) to total judgments."
            )
            
        except Exception as e:
            state.error = str(e)
            return Score(
                value=0.0,
                answer="[ERROR]",
                explanation=str(e)
            )

    return score