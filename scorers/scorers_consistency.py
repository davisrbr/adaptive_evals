from inspect_ai.solver import TaskState
from inspect_ai.scorer import scorer, Score, Target, mean, stderr
from solvers.solvers_consistency import ConsistencyType
import logging
from utils_consistency.metrics import (
    negation_frequentist_metric,
    or_frequentist_metric,
    and_frequentist_metric, 
    consequence_frequentist_metric, 
    paraphrase_frequentist_metric
)

@scorer(
    metrics={
        f"{ct.value}_score": [mean(), stderr()]
        for ct in ConsistencyType
    }
)
def consistency_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        checks = state.metadata.get('consistency_checks', {})
        scores = {}
        beta_min = 1e-3  # regularization term
        
        for ct_value, check_data in checks.items():
            key = f"{ct_value}_score"
            logging.info(f"\n=== Scoring {ct_value} consistency ===")
            
            if not check_data:
                logging.warning(f"No check data for {ct_value}")
                scores[key] = 0.0
                continue
                
            # Get forecasts
            p_forecast = check_data['P'].get('forecast', 0.0)
            q_forecast = check_data['Q'].get('forecast', 0.0)
            
            # Log the forecasts
            logging.info(f"P forecast: {p_forecast:.3f} - {check_data['P']['title']}")
            logging.info(f"Q forecast: {q_forecast:.3f} - {check_data['Q']['title']}")
            
            try:
                # Score based on consistency type using frequentist metrics
                if ct_value == 'not':
                    scores[key] = negation_frequentist_metric(p_forecast, q_forecast, beta_min)
                    logging.info(f"NOT check: negation_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f})")
                    
                elif ct_value == 'or':
                    r_forecast = check_data['R'].get('forecast', 0.0)
                    logging.info(f"R forecast: {r_forecast:.3f} - {check_data['R']['title']}")
                    scores[key] = or_frequentist_metric(p_forecast, q_forecast, r_forecast, beta_min)
                    logging.info(f"OR check: or_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f}, R={r_forecast:.3f})")
                    
                elif ct_value == 'and':
                    r_forecast = check_data['R'].get('forecast', 0.0)
                    logging.info(f"R forecast: {r_forecast:.3f} - {check_data['R']['title']}")
                    scores[key] = and_frequentist_metric(p_forecast, q_forecast, r_forecast, beta_min)
                    logging.info(f"AND check: and_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f}, R={r_forecast:.3f})")
                
                elif ct_value == 'consequence':
                    scores[key] = consequence_frequentist_metric(p_forecast, q_forecast, beta_min)
                    logging.info(f"CONSEQUENCE check: consequence_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f})")
                
                elif ct_value == 'paraphrase':
                    scores[key] = paraphrase_frequentist_metric(p_forecast, q_forecast, beta_min)
                    logging.info(f"PARAPHRASE check: paraphrase_frequentist_metric(P={p_forecast:.3f}, Q={q_forecast:.3f})")
                
                logging.info(f"Frequentist metric score: {scores[key]:.3f}\n")
                
            except Exception as e:
                logging.error(f"Error calculating {ct_value} metric: {str(e)}")
                scores[key] = 0.0

        return Score(
            value=scores,
            answer=str(checks)
        )
    return score