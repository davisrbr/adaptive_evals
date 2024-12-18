#utils_consistency/metrics.py
import numpy as np
from typing import Dict
import numpy as np
import pandas as pd
import logging 

#Negation Arbitrage Metric
def negation_arbitrage_metric(fp: float, f_neg_p: float) -> float:
    """
    Calculate the value of the RHS of the given formula using NumPy.

    Args:
        fp (float): The value of F(P).
        f_neg_p (float): The value of F(¬P).

    Returns:
        float: The computed value of the RHS.
    """
    if fp < 0 or fp > 1 or f_neg_p < 0 or f_neg_p > 1:
        raise ValueError("Both F(P) and F(¬P) must be between 0 and 1.")
    
    term1 = np.sqrt(fp * (1 - f_neg_p))
    term2 = np.sqrt((1 - fp) * f_neg_p)
    result = -2 * np.log(term1 + term2)

    # Correct small negative floating-point values
    if np.isclose(result, 0.0):
        result = 0.0
    
    return result

def negation_frequentist_metric(fp: float, f_neg_p: float, beta_min: float = 1e-3) -> float:
    """
    Calculate the negation consistency violation metric.

    Args:
        fp (float): The value of F(P), the forecasted probability of P.
        f_neg_p (float): The value of F(¬P), the forecasted probability of not P.
        beta_min (float): Small regularization term to avoid division by zero (default: 1e-3).

    Returns:
        float: The negation consistency violation metric.
    """
    if fp < 0 or fp > 1 or f_neg_p < 0 or f_neg_p > 1:
        raise ValueError("F(P) and F(¬P) must be probabilities between 0 and 1.")

    # Numerator: |F(P) + F(¬P) - 1|
    numerator = abs(fp + f_neg_p - 1)

    # Denominator: sqrt((1 - F(P))F(P) + (1 - F(¬P))F(¬P)) + beta_min
    term1 = (1 - fp) * fp
    term2 = (1 - f_neg_p) * f_neg_p
    denominator = np.sqrt(term1 + term2) + beta_min

    # Compute the negation consistency violation metric
    result = numerator / denominator

    return result


def consequence_frequentist_metric(fp: float, fq: float, beta_min: float = 1e-3) -> float:
    """
    Calculate the consequence consistency violation metric.
    Always returns a non-negative value.
    """
    if fp < 0 or fp > 1 or fq < 0 or fq > 1:
        raise ValueError("F(P) and F(Q) must be probabilities between 0 and 1.")

    # Iverson bracket [F(P) > F(Q)]
    iverson = 1 if fp > fq else 0
    
    # Calculate absolute difference |F(P) - F(Q)|
    numerator = abs(fp - fq)
    
    # Calculate denominator
    term1 = fp * (1 - fp)
    term2 = fq * (1 - fq)
    denominator = np.sqrt(term1 + term2 + beta_min) 
    
    # Compute the consequence consistency violation metric
    # Use np.abs to ensure non-negative result
    result = np.abs(iverson * (numerator / denominator))
    
    # Handle potential negative zero
    return result if result != 0 else 0.0

def paraphrase_frequentist_metric(fp: float, fq: float, beta_min: float = 1e-3) -> float:
    """
    Calculate the paraphrase consistency violation metric.
    Always returns a non-negative value.
    """
    if fp < 0 or fp > 1 or fq < 0 or fq > 1:
        raise ValueError("F(P) and F(Q) must be probabilities between 0 and 1.")
    
    # Calculate absolute difference |F(P) - F(Q)|
    numerator = abs(fp - fq)
    
    # Calculate denominator
    term1 = fp * (1 - fp)
    term2 = fq * (1 - fq)
    denominator = np.sqrt(term1 + term2 + beta_min)
    
    # Compute the paraphrase consistency violation metric
    # Use np.abs to ensure non-negative result
    result = np.abs(numerator / denominator)
    
    # Handle potential negative zero
    return result if result != 0 else 0.0


# Example usage
# fp = 0.5      # Example value for F(P)
# fq = 0.6 # Example value for F(Q)
# beta_min = 1e-3

# v_negation = negation_frequentist_metric(fp, fq, beta_min)
# print(f"The negation frequentist consistency metric is: {v_negation}")

# v_consequence = consequence_frequentist_metric(fp, fq, beta_min)
# print(f"The consequence frequentist consistency metric is: {v_consequence}")

# v_paraphrase = paraphrase_frequentist_metric(fp, fq, beta_min)
# print(f"The paraphrase consistency violation metric is: {v_paraphrase}")

def calculate_consistency_metrics(df: pd.DataFrame, beta_min: float = 1e-3) -> pd.DataFrame:
    """
    Calculate appropriate consistency metrics based on consistency_type.
    
    Args:
        df: DataFrame with forecast probabilities and consistency types
        beta_min: Regularization term
        
    Returns:
        DataFrame with added consistency_score column
    """
    result_df = df.copy()
    
    # Initialize consistency score column
    result_df['consistency_score'] = np.nan
    
    # Calculate metrics based on consistency type
    for idx, row in df.iterrows():
        fp = row['original_forecast_probability']
        fq = row['consistency_forecast_probability']
        
        # Skip if probabilities are missing
        if pd.isna(fp) or pd.isna(fq):
            continue
            
        try:
            if row['consistency_type'] == 'not':
                score = negation_frequentist_metric(fp, fq, beta_min)
            elif row['consistency_type'] == 'consequence':
                score = consequence_frequentist_metric(fp, fq, beta_min)
            elif row['consistency_type'] == 'paraphrase':
                score = paraphrase_frequentist_metric(fp, fq, beta_min)
            else:
                logging.warning(f"Unknown consistency type: {row['consistency_type']}")
                continue
                
            result_df.at[idx, 'consistency_score'] = score
            
        except Exception as e:
            logging.error(f"Error calculating metric for row {idx}: {str(e)}")
            continue
    
    return result_df

def analyze_consistency_scores(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Analyze consistency scores by type.
    
    Args:
        df: DataFrame with consistency scores
        
    Returns:
        Dictionary of statistics by consistency type
    """
    stats = {}
    
    for consistency_type in df['consistency_type'].unique():
        type_df = df[df['consistency_type'] == consistency_type]
        scores = type_df['consistency_score'].dropna()
        
        stats[consistency_type] = {
            'mean': scores.mean(),
            'median': scores.median(),
            'std': scores.std(),
            'min': scores.min(),
            'max': scores.max(),
            'count': len(scores),
            'null_count': type_df['consistency_score'].isna().sum()
        }
    
    return stats

# Creating Consistency Scores for the Above CSV, i.e, taking the combined CSV with forecasts and calculating the consistency scores
# if __name__ == "__main__":
#     # Load results
#     df = pd.read_csv('forecast_results_combined_20241217_222800.csv')
    
#     # Calculate consistency metrics
#     result_df = calculate_consistency_metrics(df)
    
#     # Analyze results
#     stats = analyze_consistency_scores(result_df)
    
#     # Print summary
#     print("\nConsistency Score Analysis:")
#     for consistency_type, metrics in stats.items():
#         print(f"\n{consistency_type.upper()} Metrics:")
#         print(f"Mean: {metrics['mean']:.4f}")
#         print(f"Median: {metrics['median']:.4f}")
#         print(f"Std Dev: {metrics['std']:.4f}")
#         print(f"Range: [{metrics['min']:.4f}, {metrics['max']:.4f}]")
#         print(f"Sample Size: {metrics['count']}")
#         print(f"Missing Values: {metrics['null_count']}")
    
#     # Save results
#     output_file = f"consistency_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
#     result_df.to_csv(output_file, index=False)
#     print(f"\nSaved results to: {output_file}")