#utils_consistency/metrics.py
import numpy as np
from typing import Dict
import numpy as np
import pandas as pd
import logging 
from datetime import datetime

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

def and_frequentist_metric(fp: float, fq: float, f_p_and_q: float, beta_min: float = 1e-3) -> float:
    """
    Calculate the AND consistency violation metric.
    
    Args:
        fp (float): F(P), probability of P
        fq (float): F(Q), probability of Q
        f_p_and_q (float): F(P∧Q), probability of P AND Q
        beta_min (float): Small regularization term
    
    Returns:
        float: The AND consistency violation metric
    """
    # Input validation
    for p in [fp, fq, f_p_and_q]:
        if not 0 <= p <= 1:
            raise ValueError("All probabilities must be between 0 and 1")
            
    # Check strict inequalities
    lhs_bound = max(fp + fq - 1, 0)
    rhs_bound = min(fp, fq)
    
    if lhs_bound < f_p_and_q < rhs_bound:
        return 0.0
        
    # Calculate LHS violation
    if f_p_and_q <= lhs_bound and (fp + fq - 1 > 0):
        v_and_lhs = abs(fp + fq - 1 - f_p_and_q) / (
            np.sqrt(fp * (1-fp) + fq * (1-fq) + f_p_and_q * (1-f_p_and_q)) + beta_min
        )
    else:
        v_and_lhs = 0.0
        
    # Calculate RHS violation
    min_prob = min(fp, fq)
    if f_p_and_q >= min_prob:
        v_and_rhs = abs(f_p_and_q - min_prob) / (
            np.sqrt(f_p_and_q * (1-f_p_and_q) + min_prob * (1-min_prob)) + beta_min
        )
    else:
        v_and_rhs = 0.0
        
    return max(v_and_lhs, v_and_rhs)

def or_frequentist_metric(fp: float, fq: float, f_p_or_q: float, beta_min: float = 1e-3) -> float:
    """
    Calculate the OR consistency violation metric.
    
    Args:
        fp (float): F(P), probability of P
        fq (float): F(Q), probability of Q
        f_p_or_q (float): F(P∨Q), probability of P OR Q
        beta_min (float): Small regularization term
    
    Returns:
        float: The OR consistency violation metric
    """
    # Input validation
    for p in [fp, fq, f_p_or_q]:
        if not 0 <= p <= 1:
            raise ValueError("All probabilities must be between 0 and 1")
            
    # Check strict inequalities
    lhs_bound = max(fp, fq)
    rhs_bound = min(1, fp + fq)
    
    if lhs_bound < f_p_or_q < rhs_bound:
        return 0.0
        
    # Calculate LHS violation
    max_prob = max(fp, fq)
    if f_p_or_q <= max_prob:
        v_or_lhs = abs(max_prob - f_p_or_q) / (
            np.sqrt(max_prob * (1-max_prob) + f_p_or_q * (1-f_p_or_q)) + beta_min
        )
    else:
        v_or_lhs = 0.0
        
    # Calculate RHS violation
    if f_p_or_q >= rhs_bound and (fp + fq < 1):
        v_or_rhs = abs(f_p_or_q - fp - fq) / (
            np.sqrt(f_p_or_q * (1-f_p_or_q) + fp * (1-fp) + fq * (1-fq)) + beta_min
        )
    else:
        v_or_rhs = 0.0
        
    return max(v_or_lhs, v_or_rhs)
# [Keep all existing imports and the metric calculation functions above]

def but_frequentist_metric(fp: float, f_not_p_and_q: float, f_p_or_q: float, beta_min: float = 1e-3) -> float:
    """
    Calculate the BUT consistency violation metric.
    
    Args:
        fp (float): F(P), probability of P
        f_not_p_and_q (float): F(¬P∧Q), probability of NOT P AND Q
        f_p_or_q (float): F(P∨Q), probability of P OR Q
        beta_min (float): Small regularization term to avoid division by zero
    
    Returns:
        float: The BUT consistency violation metric
    """
    # Input validation
    for p in [fp, f_not_p_and_q, f_p_or_q]:
        if not 0 <= p <= 1:
            raise ValueError("All probabilities must be between 0 and 1")
    
    # Calculate numerator: |F(P∨Q) - F(P) - F(¬P∧Q)|
    numerator = abs(f_p_or_q - fp - f_not_p_and_q)
    
    # Calculate denominator: sqrt(F(P∨Q)(1-F(P∨Q)) + F(P)(1-F(P)) + F(¬P∧Q)(1-F(¬P∧Q))) + β_min
    term1 = f_p_or_q * (1 - f_p_or_q)
    term2 = fp * (1 - fp)
    term3 = f_not_p_and_q * (1 - f_not_p_and_q)
    denominator = np.sqrt(term1 + term2 + term3  + beta_min)
    
    # Compute the BUT consistency violation metric
    result = numerator / denominator
    
    return result

def cond_frequentist_metric(fp: float, f_q_given_p: float, f_p_and_q: float, beta_min: float = 1e-3) -> float:
    """
    Calculate the COND (conditional) consistency violation metric.
    
    Args:
        fp (float): F(P), probability of P
        f_q_given_p (float): F(Q|P), conditional probability of Q given P
        f_p_and_q (float): F(P∧Q), probability of P AND Q
        beta_min (float): Small regularization term to avoid division by zero
    
    Returns:
        float: The COND consistency violation metric
    """
    # Input validation
    for p in [fp, f_q_given_p, f_p_and_q]:
        if not 0 <= p <= 1:
            raise ValueError("All probabilities must be between 0 and 1")
    
    # Using the notation from the paper:
    # a' = F(P)
    # b' = F(Q|P)
    # c' = F(P∧Q)
    a = fp
    b = f_q_given_p
    c = f_p_and_q
    
    # Calculate numerator: |a'b' - c'| = |F(P)F(Q|P) - F(P∧Q)|
    numerator = abs(a * b - c)
    
    # Calculate denominator: sqrt(a'b'(a'(1-b') + b'(1-a')) + c'(1-c')) + β_min
    term1 = a * b * (a * (1 - b) + b * (1 - a))
    term2 = c * (1 - c)
    denominator = np.sqrt(term1 + term2 + beta_min) 
    
    # Compute the COND consistency violation metric
    result = numerator / denominator
    
    return result

def andor_frequentist_metric(fp: float, fq: float, f_p_and_q: float, f_p_or_q: float, beta_min: float = 1e-3) -> float:
    """
    Calculate the ANDOR consistency violation metric.
    
    Args:
        fp (float): F(P), probability of P
        fq (float): F(Q), probability of Q
        f_p_and_q (float): F(P∧Q), probability of P AND Q
        f_p_or_q (float): F(P∨Q), probability of P OR Q
        beta_min (float): Small regularization term to avoid division by zero
    
    Returns:
        float: The ANDOR consistency violation metric
    """
    # Input validation
    for p in [fp, fq, f_p_and_q, f_p_or_q]:
        if not 0 <= p <= 1:
            raise ValueError("All probabilities must be between 0 and 1")
    
    # Calculate numerator: |F(P) + F(Q) - F(P∨Q) - F(P∧Q)|
    numerator = abs(fp + fq - f_p_or_q - f_p_and_q)
    
    # Calculate denominator: sqrt(F(P)(1-F(P)) + F(Q)(1-F(Q)) + F(P∨Q)(1-F(P∨Q)) + F(P∧Q)(1-F(P∧Q))) + β_min
    term1 = fp * (1 - fp)
    term2 = fq * (1 - fq)
    term3 = f_p_or_q * (1 - f_p_or_q)
    term4 = f_p_and_q * (1 - f_p_and_q)
    denominator = np.sqrt(term1 + term2 + term3 + term4 + beta_min) 
    
    # Compute the ANDOR consistency violation metric
    result = numerator / denominator
    
    return result

def expevidence_frequentist_metric(
    fp: float,           # a = F(P)
    fq: float,           # d = F(Q)
    f_p_given_q: float,  # b = F(P|Q)
    f_p_given_not_q: float,  # c = F(P|¬Q)
    beta_min: float = 1e-3,
    sigma : float = 0.05,
) -> float:
    """
    Calculate the EXPEVIDENCE (Expected Evidence) consistency violation metric.
    
    Args:
        fp (float): F(P), probability of P
        fq (float): F(Q), probability of Q
        f_p_given_q (float): F(P|Q), conditional probability of P given Q
        f_p_given_not_q (float): F(P|¬Q), conditional probability of P given not Q
        beta_min (float): Small regularization term to avoid division by zero
        sigma (float): based on n, default n = 400 => sigma = 0.05
    
    Returns:
        float: The EXPEVIDENCE consistency violation metric
    """
    # Input validation
    for p in [fp, fq, f_p_given_q, f_p_given_not_q]:
        if not 0 <= p <= 1:
            raise ValueError("All probabilities must be between 0 and 1")
    
    # Using the notation from the paper:
    # a = F(P)
    # d = F(Q)
    # b = F(P|Q)
    # c = F(P|¬Q)
    a = fp
    d = fq
    b = f_p_given_q
    c = f_p_given_not_q
    
    # Calculate numerator: |bd + c(1-d) - a|
    numerator = abs(b * d + c * (1 - d) - a)
    
    # Calculate denominator: σ√(a(1-a) + d²b(1-b) + (1-d)²c(1-c) + (b-c)²d(1-d)) + β_min
    term1 = a * (1 - a)
    term2 = d * d * b * (1 - b)
    term3 = (1 - d) * (1 - d) * c * (1 - c)
    term4 = (b - c) * (b - c) * d * (1 - d)
    denominator = sigma * np.sqrt(term1 + term2 + term3 + term4 + beta_min) 
    
    # Compute the EXPEVIDENCE consistency violation metric
    result = numerator / denominator
    
    return result

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
    result_df['consistency_score'] = np.nan
    
    for idx, row in df.iterrows():
        # Skip if required probabilities are missing
        if pd.isna(row['question_P_forecast_probability']):
            continue
            
        try:
            consistency_type = row['consistency_type'].lower()
            fp = row['question_P_forecast_probability']
            
            if consistency_type in ['not', 'consequence', 'paraphrase']:
                if pd.isna(row['question_Q_forecast_probability']):
                    continue
                fq = row['question_Q_forecast_probability']
                
                if consistency_type == 'not':
                    score = negation_frequentist_metric(fp, fq, beta_min)
                elif consistency_type == 'consequence':
                    score = consequence_frequentist_metric(fp, fq, beta_min)
                elif consistency_type == 'paraphrase':
                    score = paraphrase_frequentist_metric(fp, fq, beta_min)
                    
            elif consistency_type in ['and', 'or']:
                if pd.isna(row['question_Q_forecast_probability']) or pd.isna(row['question_R_forecast_probability']):
                    continue
                fq = row['question_Q_forecast_probability']
                fr = row['question_R_forecast_probability']
                
                if consistency_type == 'and':
                    score = and_frequentist_metric(fp, fq, fr, beta_min)
                else:  # or
                    score = or_frequentist_metric(fp, fq, fr, beta_min)
            else:
                logging.warning(f"Unknown consistency type: {consistency_type}")
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
        
        if len(scores) > 0:
            stats[consistency_type] = {
                'mean': scores.mean(),
                'median': scores.median(),
                'std': scores.std(),
                'min': scores.min(),
                'max': scores.max(),
                'count': len(scores),
                'null_count': type_df['consistency_score'].isna().sum()
            }
        else:
            stats[consistency_type] = {
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0,
                'null_count': type_df['consistency_score'].isna().sum()
            }
    
    return stats

# Test code
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load results
    input_file = "forecast_results_20241219_003530.csv"
    df = pd.read_csv(input_file)
    
    # Calculate consistency metrics
    result_df = calculate_consistency_metrics(df)
    
    # Analyze results
    stats = analyze_consistency_scores(result_df)
    
    # Print summary
    print("\nConsistency Score Analysis:")
    for consistency_type, metrics in stats.items():
        print(f"\n{consistency_type.upper()} Metrics:")
        print(f"Mean: {metrics['mean']:.4f}")
        print(f"Median: {metrics['median']:.4f}")
        print(f"Std Dev: {metrics['std']:.4f}")
        print(f"Range: [{metrics['min']:.4f}, {metrics['max']:.4f}]")
        print(f"Sample Size: {metrics['count']}")
        print(f"Missing Values: {metrics['null_count']}")
    
    # Save results
    output_file = f"consistency_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    result_df.to_csv(output_file, index=False)
    print(f"\nSaved results to: {output_file}")