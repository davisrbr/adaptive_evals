from pathlib import Path
from inspect_ai.log import read_eval_log
import json


def generate_in_context_examples(log_path: str, model_dir: str) -> None:
    """
    Generate in-context examples from evaluation logs and save to a file.
    Sorts question triples by consistency score and saves top 20.
    
    Args:
        log_path: Path to the evaluation log file
    """
    # Read evaluation log
    eval_log = read_eval_log(log_path)
    
    # Create list to store examples with their scores
    examples = []
    
    # Process each sample
    for sample in eval_log.samples:
        # Get consistency score
        consistency_score = sample.scores['cond_consistency_scorer'].value
        
        # Create example dictionary with all relevant information
        example = {
            'consistency_score': consistency_score,
            'questions': {
                'P': {
                    'title': sample.metadata['P_title'],
                    'body': sample.metadata['P_body'],
                    'forecast': sample.metadata['forecasts']['P'],
                    'reasoning': sample.metadata['reasoning']['P']
                },
                'Q_given_P': {
                    'title': sample.metadata['Q_given_P_title'],
                    'body': sample.metadata['Q_given_P_body'],
                    'forecast': sample.metadata['forecasts']['Q_given_P'],
                    'reasoning': sample.metadata['reasoning']['Q_given_P']
                },
                'P_and_Q': {
                    'title': sample.metadata['P_and_Q_title'],
                    'body': sample.metadata['P_and_Q_body'],
                    'forecast': sample.metadata['forecasts']['P_and_Q'],
                    'reasoning': sample.metadata['reasoning']['P_and_Q']
                }
            }
        }
        examples.append(example)
    
    # Sort examples by consistency score (highest to lowest)
    examples.sort(key=lambda x: x['consistency_score'], reverse=True)
    
    # Take top 20 examples
    top_examples = examples[:20]
    
    # Create output directory if it doesn't exist
    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'top_inconsistent_examples.json'
    with open(output_path, 'w') as f:
        json.dump(top_examples, f, indent=2)
    
def main():
    log_path = "/Users/bprithvi/Desktop/Research Projects/Brachio/adaptive_evals/tasks/cond_100/gpt-4o-mini/2025-01-24T11-52-36-05-00_cond-initial-consistency_V2878JZ8DW8RYAfPvC4yZP.eval"
    model_dir = "/Users/bprithvi/Desktop/Research Projects/Brachio/adaptive_evals/tasks/cond_100/gpt-4o-mini"
    
    generate_in_context_examples(log_path=log_path, model_dir=model_dir)

if __name__ == "__main__":
    main()