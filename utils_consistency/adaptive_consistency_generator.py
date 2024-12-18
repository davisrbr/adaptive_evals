#utils_consistency/adaptive_consistency_generator.py
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from openai import OpenAI
import pandas as pd
import logging 
from dataclasses import dataclass
import os
import json

@dataclass
class GenerationConfig:
    n_questions: int
    samples_per_bucket: Dict[str, int]
    samples_per_consistency: Dict[str, int]
    model: str = "gpt-4o"
    temperature: float = 0.7
    checkpoint_dir: str = "adaptive_consistency_checkpoints"

class AdversarialQuestionGenerator:
    def __init__(
        self,
        config: GenerationConfig,
        api_key: Optional[str] = None
    ):
        self.config = config
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        logging.basicConfig(
            filename=f'{self.config.checkpoint_dir}/generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def bucket_questions(self, df: pd.DataFrame) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Two-level bucketing: first by consistency type, then by performance."""
        consistency_buckets = {}
        
        for consistency_type in df['consistency_type'].unique():
            type_df = df[df['consistency_type'] == consistency_type]
            
            # Sort by consistency score
            df_sorted = type_df.sort_values('consistency_score', ascending=True)
            
            # Create performance buckets within this consistency type
            performance_buckets = {}
            for bucket_name, (start_pct, end_pct) in zip(
                ['best', 'medium', 'poor', 'worst'],
                [(0, 25), (25, 50), (50, 75), (75, 100)]
            ):
                start_idx = int(len(df_sorted) * start_pct / 100)
                end_idx = int(len(df_sorted) * end_pct / 100)
                bucket_df = df_sorted.iloc[start_idx:end_idx]
                
                n_samples = min(
                    self.config.samples_per_bucket[bucket_name],
                    self.config.samples_per_consistency[consistency_type]
                )
                
                if len(bucket_df) > n_samples:
                    bucket_df = bucket_df.sample(n_samples, random_state=42)
                
                performance_buckets[bucket_name] = bucket_df
            
            consistency_buckets[consistency_type] = performance_buckets
            
        return consistency_buckets

    def create_prompt(self, consistency_buckets: Dict[str, Dict[str, pd.DataFrame]]) -> str:
        """Create the generation prompt with examples from all consistency types."""
        prompt = self._create_prompt_header()
        prompt += self._add_examples(consistency_buckets)
        prompt += self._create_prompt_instructions()
        return prompt

    def _create_prompt_header(self) -> str:
        return """You are an expert in analyzing forecasting models and generating challenging questions. You'll analyze question pairs with their consistency scores and then generate new challenging questions.

DEFINITIONS OF CONSISTENCY TYPES:
1. Logical Negation (NOT): The negation must be the exact logical negation of the original question.
2. Consequence: For the original question to resolve positively, the consequence question should have resolved positively. For the original question P, and consequence question Q, the Probability of Q should be greater than the Probability of P. Examples of good P/Q pairs for Consequence, so the question P should imply that Q is a pre-requisite:
      P: "Will computers in Italy be able to send and receive emails in 2050?"
      Q: "Will computers in Italy be connected to the internet"
      (Q is a necessary component for P, because computers can't send emails without the internet)

      P: "Will human life expectancy in all countries exceed 95 years by 2030?"
      Q: "Will human life expectancy in the United States exceed 90 years by 2030?"
      (Achieving > 95 years expectancy in all countries requires > 90 years expectancy in one country - the United States)
3. Paraphrase: A restatement (Q) of the original question (P) that should preserve exactly the same meaning. So, probability of P = probability of Q


PART 1: EXAMPLE QUESTIONS BY CONSISTENCY TYPE AND PERFORMANCE
"""

    def _add_examples(self, consistency_buckets: Dict[str, Dict[str, pd.DataFrame]]) -> str:
        examples = ""
        for consistency_type, performance_buckets in consistency_buckets.items():
            examples += f"\n=== {consistency_type.upper()} CONSISTENCY ===\n"
            
            for bucket_name, bucket_df in performance_buckets.items():
                examples += f"\n{bucket_name.upper()} PERFORMING {consistency_type.upper()} QUESTIONS:\n"
                for _, row in bucket_df.iterrows():
                    examples += self._format_example(row, consistency_type)
        return examples

    def _format_example(self, row: pd.Series, consistency_type: str) -> str:
        return f"""
Original Question: {row['original_title']}
Consistency Question ({consistency_type}): {row['consistency_title']}
Original Body: {row['original_body']}
Consistency Body: {row['consistency_body']}
Consistency Score: {row['consistency_score']:.4f}
Original Probability: {row['original_forecast_probability']:.4f}
Consistency Probability: {row['consistency_forecast_probability']:.4f}
---\n"""

    def _create_prompt_instructions(self) -> str:
        json_format = """{
        "question_pairs": [
            {
                "original": {
                    "title": "Question title",
                    "body": "Full question body with resolution criteria",
                    "challenge_factors": ["List specific factors that make this pair challenging"]
                },
                "consistency": {
                    "type": "not|consequence|paraphrase",  # Specify which type
                    "title": "Consistency question title",
                    "body": "Full consistency question body with resolution criteria",
                    "challenge_rationale": "Explain why this consistency type was chosen and why it's challenging"
                }
            }
        ]
    }"""

        return f"""
        PART 2: YOUR TASK
        1. First, analyze the patterns in these questions, focusing on:
        - Which consistency type tends to be most challenging and why?
        - What makes questions challenging within each consistency type - is it due to forecasting difficulty for the original question or due to the consistency question of the pair of questions being difficult?
        - What patterns distinguish good vs poor performance in across each consistency type?
        - What factors (topic, timeframe, region, complexity, etc.) contribute to the challenges?
        - What patterns do you notice in the worst-performing questions?
        - Are there common factors that make questions difficult across all types?

        2. Then, generate {self.config.n_questions} challenging question pairs, choosing appropriate consistency types:
        - Choose which consistency type to use based on where you identify the most potential for challenging questions
        - Create questions that specifically target the patterns you identified as most challenging
        - Ensure proper logical relationships based on the chosen consistency type
        - Make questions diverse in topics and challenge types

        Format your response exactly as follows:

        <Begin Analysis>
        [Your analysis of patterns across consistency types and performance levels]
        <End Analysis>

        <Begin Questions>
        [Generate {self.config.n_questions} question pairs in this exact JSON format]
        {json_format}
        <End Questions>"""

    def generate(self, prompt: str) -> str:
        """Generate response using GPT-4."""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in analyzing forecasting models and generating challenging questions."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=self.config.temperature
        )
        return response.choices[0].message.content

    def save_generation(self, prompt: str, response: str, timestamp: str) -> None:
        """Save generation artifacts."""
        # Save prompt and response
        with open(f'{self.config.checkpoint_dir}/generation_{timestamp}.txt', 'w') as f:
            f.write("=== PROMPT ===\n\n")
            f.write(prompt)
            f.write("\n\n=== RESPONSE ===\n\n")
            f.write(response)
        logging.info(f"Saved generation output to generation_{timestamp}.txt")

    def parse_generation_response(self, response: str, timestamp: str) -> Tuple[str, List[Dict]]:
        """Parse analysis and questions from response."""
        try:
            # Save raw response first
            raw_response_path = f'{self.config.checkpoint_dir}/raw_response_{timestamp}.txt'
            with open(raw_response_path, 'w') as f:
                f.write(response)
            logging.info(f"Saved raw response to {raw_response_path}")
            
            # Extract analysis
            analysis_start = response.find("<Begin Analysis>") + len("<Begin Analysis>")
            analysis_end = response.find("<End Analysis>")
            analysis = response[analysis_start:analysis_end].strip()
            
            # Extract questions JSON
            questions_start = response.find("<Begin Questions>") + len("<Begin Questions>")
            questions_end = response.find("<End Questions>")
            questions_json = response[questions_start:questions_end].strip()
            
            # Clean JSON string - remove markdown formatting
            questions_json = questions_json.replace('```json', '').replace('```', '').strip()
            
            # Parse JSON
            questions_data = json.loads(questions_json)
            
            return analysis, questions_data["question_pairs"]
            
        except Exception as e:
            logging.error(f"Error parsing generation response: {str(e)}")
            logging.error(f"Raw response: {response}")
            raise



    def save_to_csv(self, questions: List[Dict], timestamp: str) -> None:
        """Save generated questions to CSV."""
        rows = []
        for q in questions:
            row = {
                'original_title': q['original']['title'],
                'original_body': q['original']['body'],
                'original_challenge_factors': ','.join(q['original']['challenge_factors']),
                'consistency_type': q['consistency']['type'],
                'consistency_title': q['consistency']['title'],
                'consistency_body': q['consistency']['body'],
                'consistency_challenge_rationale': q['consistency']['challenge_rationale']
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = f'{self.config.checkpoint_dir}/generated_questions_{timestamp}.csv'
        df.to_csv(csv_path, index=False)
        logging.info(f"Saved generated questions to {csv_path}")
        return df

# Then update the main function accordingly:
def main_adversarial_generation(
    input_csv: str,
    config: GenerationConfig
) -> Tuple[str, pd.DataFrame]:
    """Main function to generate adversarial questions."""
    try:
        # Load data
        df = pd.read_csv(input_csv)
        logging.info(f"Loaded {len(df)} rows from {input_csv}")
        
        # Initialize generator
        generator = AdversarialQuestionGenerator(config)
        
        # Create buckets
        consistency_buckets = generator.bucket_questions(df)
        
        # Create prompt
        prompt = generator.create_prompt(consistency_buckets)
        
        # Generate questions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        response = generator.generate(prompt)
        
        # Save all outputs
        generator.save_generation(prompt, response, timestamp)
        
        # Parse response
        analysis, questions = generator.parse_generation_response(response, timestamp)
        
        # Save results to CSV
        generated_df = generator.save_to_csv(questions, timestamp)
        
        print("Generation complete! Check the outputs in:", config.checkpoint_dir)
        print(f"Raw response saved to: {config.checkpoint_dir}/raw_response_{timestamp}.txt")
        print(f"Generation log saved to: {config.checkpoint_dir}/generation_{timestamp}.log")
        print(f"Generated questions saved to: {config.checkpoint_dir}/generated_questions_{timestamp}.csv")
        
        return analysis, generated_df
        
    except Exception as e:
        logging.error(f"Error in adversarial generation: {str(e)}")
        raise

# Generating Adversarial Questions based on NOT, Paraphrase, and Consequence forecasts along with frequentist consistency scores
# if __name__ == "__main__":
#     config = GenerationConfig(
#         n_questions=10,
#         samples_per_bucket={
#             'worst': 5,
#             'poor': 5,
#             'medium': 2,
#             'best': 2
#         },
#         samples_per_consistency={
#             'not': 15,
#             'consequence': 15,
#             'paraphrase': 15
#         },
#         model="gpt-4o",
#         temperature=0.7,
#         checkpoint_dir="adversarial_checkpoints"
#     )
    
#     analysis, generated_df = main_adversarial_generation(
#         input_csv='consistency_scores_20241217_225616.csv',
#         config=config
#     )
    
#     print("\nGenerated Questions Summary:")
#     print("Total questions:", len(generated_df))
#     print("\nDistribution of consistency types:")
#     print(generated_df['consistency_type'].value_counts())