#utils_consistency/consistency_question_generator.py
from typing import List, Optional, Dict, Tuple
from datetime import datetime
from functools import partial
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
import re
import pandas as pd
import logging 
import time
from tqdm import tqdm
import os


# Dictionary containing all operator prompts
OPERATOR_PROMPTS = {
    #OR
    'or': """You are a helpful assistant. I will give you two forecasting questions with Yes/No answers. You should then give me the logical OR of these two questions, i.e. the question that would be answered YES if EITHER question is answered YES, and NO otherwise. Notes: 
    • Your response should be as clear as possible, since the words 'and' and 'or' are used ambiguously in natural language. For example, 'Will P happen or will Q happen? is usually confusing, as it sounds like you are asking which of the two will happen (whereas you're actually seeking a YES/NO answer on whether either of the two will happen). Instead, if there is any chance of confusion, you should give me something like: Will either of the following occur: (a) P (b) Q? 
    • When the questions allow for a simple rephrasing or factorization (e.g. using words like 'respectively', 'both' or 'either'), go for it. 
    • If one or both of the given questions is already a logical combination of questions, join them in the most natural way possible. E.g. – combine ((P1 OR P2) OR Q) how you would combine (P1 OR P2 OR Q) – ((P1 AND P2) OR Q) might have to be combined as something like: Will EITHER of the following occur: (1) BOTH of the following occur: (a) P1 AND (b) P2 (2) Q. Unless a more natural formulation exists. 
    • Be careful when combining conditional expressions (which often have words like 'given' and 'if'). '(Given A then P) OR (Given B then Q) should be combined as is, rather than messing up the conditions. E.g. a phrasing like 'Will either of the following occur given their respective conditions: (a) Given A then P? (b) Given B then Q?' is good. 
    • This also applies when only one of the questions is conditional. Like 'P OR (Given A then Q)'should be phrased as something like: 'Will either of the following occur given their respective conditions are met? (a) P (b) Given A, then Q?'. 
    • Most importantly: make sure you retain ALL the information in the question bodies from BOTH base questions! You cannot discard a single relevant detail. All this is for an experiment to test the logical consistency of forecasters: The combined question you give will be handed to the forecasters without having seen the base questions, so it is critical that all the information in the base questions be included in your logical combination; the resolution criterion for each component should be neatly and clearly provided. 
    • Also, make sure that the title is self-sufficient independent of the body, i.e. is a question that can be meaningfully answered without looking at the body. So you CANNOT give me a question title like 'Is the following true?' or 'What will happen if the following happens?' 
    • One type of question you may be given is a single choice from a multiple choice question. For example, you may be given 'Which of these countries will legalize human cloning by 2030? (Japan)'. This is asking if Japan will recognize and legalize human cloning by 2030. Such a question may also itself be a logical combination – e.g. 'Which of these countries will legalize human cloning by 2030? (UK, France, or Germany) is asking if any either of the UK, France, or Germany will legalize human cloning by 2030. Make sure to correctly combine such combinations as previously described.""",
    #AND
    'and': """You are a helpful assistant. I will give you two forecasting questions with Yes/No answers. You should then give me the logical AND of these two questions, i.e. the question that would be answered YES if BOTH questions are answered YES, and NO otherwise. Notes:

    • Your response should be as clear as possible, since the word 'and' is used ambiguously in natural language. For example, 'Will P and Q happen?' is usually confusing, as it might sound like you're asking about them happening simultaneously in a moment or in some specific order (whereas you're actually seeking a YES/NO answer on whether both events will occur within the respective timeframes). Instead, if there is any chance of confusion, you should give me something like: Will both of the following occur: (a) P (b) Q?

    • When the questions allow for a simple rephrasing or factorization (e.g. using words like 'all', 'both', 'simultaneously', 'in addition to'), go for it. 

    • If one or both of the given questions is already a logical combination of questions, join them in the most natural way possible. E.g. – combine ((P1 AND P2) AND Q) how you would combine (P1 AND P2 AND Q) – ((P1 OR P2) AND Q) might have to be combined as something like: Will ALL of the following occur: (1) At least one of: (a) P1 OR (b) P2 AND (2) Q. Unless a more natural formulation exists.

    • Be careful when combining conditional expressions (which often have words like 'given' and 'if'). '(Given A then P) AND (Given B then Q)' should be combined as is, rather than messing up the conditions. E.g. a phrasing like 'Will both of the following occur given their respective conditions: (a) Given A then P? (b) Given B then Q?' is good.

    • This also applies when only one of the questions is conditional. Like 'P AND (Given A then Q)' should be phrased as something like: 'Will both of the following occur given their respective conditions are met? (a) P (b) Given A, then Q?'. Be especially careful to maintain the conditional relationship only for the part it applies to.

    • Most importantly: make sure you retain ALL the information in the question bodies from BOTH base questions! You cannot discard a single relevant detail. All this is for an experiment to test the logical consistency of forecasters: The combined question you give will be handed to the forecasters without having seen the base questions, so it is critical that all the information in the base questions be included in your logical combination; the resolution criterion for each component should be neatly and clearly provided.

    • Most importantly: Make sure that the title is self-sufficient independent of the body, i.e. is a question that can be meaningfully answered without looking at the body. So you CANNOT give me a question title like 'Is the following true?' or 'What will happen if the following happens?' or a question which is dependent on parts not mentioned in the title such as resolution  criteria or conditions mentioned elsewhere.

    • One type of question you may be given is a single choice from a multiple choice question. For example, you may be given 'Which of these countries will legalize human cloning by 2030? (Japan)'. This is asking if Japan will recognize and legalize human cloning by 2030. Such a question may also itself be a logical combination – e.g. 'Which of these countries will legalize human cloning by 2030? (UK, France, or Germany)' is asking if any either of the UK, France, or Germany will legalize human cloning by 2030. When combining such questions with AND, be explicit about requiring all components – e.g., 'Will both Japan legalize human cloning by 2030 AND [second condition]?' Make sure to correctly combine such combinations as previously described.""",
    #NOT
    'not': """You are a helpful assistant. I will give you a forecasting question with a Yes/No answer. You should then give me the logical NEGATION of this question, i.e. a question that would be answered YES if and only if the original question would be answered NO, and vice versa. Notes:

    • Your response should be as clear as possible, avoiding double negatives or ambiguous language. For example, instead of 'Will P not happen?', which can be confusing, use clearer phrasings like 'Will P fail to occur before the resolution deadline?'

    • When the original question is already negative, don't add another negation. Instead, reformulate to the positive:
      - If P is 'Will X fail to reach 100?', NOT P should be 'Will X reach 100?'
      - If P is 'Will X remain below threshold?', NOT P should be 'Will X reach or exceed threshold?'
      - If P is 'Will X be unsuccessful?', NOT P should be 'Will X succeed?'

    • Be extremely precise with numeric thresholds and inequalities:
      - 'Will X exceed 100?' -> 'Will X be less than or equal to 100?'
      - 'Will X be at least 100?' -> 'Will X be less than 100?'
      - 'Will X be strictly greater than 100?' -> 'Will X be less than or equal to 100?'
      - 'Will X remain within range 10-20?' -> 'Will X fall outside range 10-20?'

    • For complex conditional statements, apply De Morgan's Laws carefully:
      - 'Will X succeed only if Y happens?' -> 'Will either Y fail to happen OR X fail to succeed?'
      - 'Will X occur without Y happening?' -> 'Will either X fail to occur OR Y happen?'

    • For questions involving time periods, maintain exact temporal logic:
      - 'Will X happen before Y?' -> 'Will the following occur: (a) X does not happen before Y?'
      - 'Will X occur throughout period P?' -> 'Will X fail to occur at any point during period P?'
      - 'Will X last until Y?' -> 'Will X end before Y occurs?'

    • Handle challenging cases with precision. For example:
      Original: 'Will Tesla maintain its market share above 50% in the EV market until a competitor reaches 30% market share?'
      Negation: 'Will Tesla's market share drop to or below 50% in the EV market before any competitor reaches 30% market share?'

    • If the original question has multiple independent criteria, negate them properly:
      Original: 'Will Project X both stay under budget AND finish on time?'
      Negation: 'Will either of the following occur: (a) Project X exceeds budget OR (b) Project X fails to finish on time?'

    • Most importantly: make sure you retain ALL the information from the base question! You cannot discard a single relevant detail. All this is for an experiment to test the logical consistency of forecasters: The negated question you give will be handed to forecasters without having seen the original question, so it is critical that all the information and resolution criteria be included and properly negated.

    • Most importantly: Make sure that the title is self-sufficient independent of the body and the question can be answered only using the information in the title without referencing information in the body. The negated question should be just as clear and complete as the original. Never use vague phrases like 'Will the opposite occur?' or 'Will this not happen?'

    • For multiple choice questions, be especially careful. For example:
      - Single selection: 'Which of these countries will legalize X by 2030? (Japan)' -> 'Will Japan fail to legalize X by 2030?'
      - Multiple selections: 'Which of these countries will legalize X by 2030? (UK, France, or Germany)' -> 'Will ALL of the following countries fail to legalize X by 2030: UK, France, and Germany?' """,
      #COND
      'cond': """You are a helpful assistant. I will give you two forecasting questions with Yes/No answers, P and Q. You should then give me a CONDITIONAL question "Q given P", i.e. a question that asks whether Q occurs in the case that P occurs. The question should be structured as "Given P has occurred, will Q occur?". This should resolve as NO only if P occurs and Q does not occur; it resolves as YES in all other cases (if P doesn't occur, or if both P and Q occur).

Notes:
    • Your response must be structured clearly as "Given [P's conditions are met], will [Q's conditions be met]?" Avoid any other phrasings. The conditional relationship must be immediately clear from the start of both title and body.

    • Start the title with "Given that [P]" and the body with "Given that [P's complete conditions], this question asks if [Q's complete conditions]"

    • Be precise about time frames in the conditional relationship:
      - If P must occur before Q can occur, make this explicit
      - If the order doesn't matter, make this clear
      - If there are specific timeframes for each event, maintain them precisely

    • When one or both of the input questions are already logical combinations, maintain their structure clearly:
      - For 'Given (P1 AND P2), will Q occur': "Given that both: (a) P1 AND (b) P2 have occurred, will Q occur?"
      - For 'Given P, will (Q1 OR Q2) occur': "Given that P has occurred, will either: (a) Q1 OR (b) Q2 occur?"
      - Make sure to maintain the proper logical structure of nested conditions

    • Most importantly: make sure you retain ALL the information from BOTH base questions! You cannot discard a single relevant detail. All this is for an experiment to test the logical consistency of forecasters: The conditional question you give will be handed to forecasters without having seen the base questions, so it is critical that all the information and resolution criteria be included.

    • Most importantly: Make sure that the title is self-sufficient independent of the body and can be answered without referencing any criteria or information external to the title. The conditional relationship must be clear from the title alone. Only use binary questions and never use vague phrases like 'Will Q follow?' or 'What happens after P?'

    • For multiple choice questions, maintain the same clear structure. For example, if P is 'Which of these countries will legalize X by 2030? (Japan)' and Q is about regulatory framework, phrase it as 'Given that Japan legalizes X by 2030, will [Q's specific conditions about regulatory framework]?'""", 
    #PARAPHRASE
    'paraphrase': """You are a helpful assistant. I will give you a forecasting question with a Yes/No answer. You should then give me a PARAPHRASE of this question that:
1. Has exactly the same meaning and resolution criteria
2. Would resolve identically in all possible scenarios
3. Is expressed in different but equally clear language
4. You may switch around the ordering or semantics of words in the question only if it doesn't change the logical ordering of the question. For example, you cannot change If P then Q to If Q then P.

Notes:
    • Your paraphrase must preserve ALL resolution criteria exactly. For example:
      Original: "Will X exceed 100 by 2025?"
      Good: "Will the value of X rise above 100 before January 1, 2025?"
      Bad: "Will X reach approximately 100 by 2025?" (Changes criterion)

    • Maintain precise language for:
      - Numeric thresholds (">= 100" must stay ">= 100", not "about 100" or "around 100")
      - Dates (specific dates must remain specific)
      - Entity names (organizations, countries, etc. must stay exactly as given)
      - Technical terms (maintain precise technical meanings)

    • Keep all logical relationships intact:
      - "A AND B" must remain a conjunction requiring both
      - "A OR B" must remain a disjunction requiring either
      - "If A then B" must maintain the same conditional relationship
      - "At least N of the following" must preserve the exact threshold

    • When handling complex criteria, maintain all components:
      Original: "Will Tesla achieve both (1) Level 5 autonomy and (2) $1T market cap by 2025?"
      Good: "Will Tesla reach a trillion-dollar market capitalization and attain Level 5 self-driving capabilities before January 1, 2025?"
      Bad: "Will Tesla achieve major technical and financial milestones by 2025?" (Too vague)

    • For multiple choice questions, maintain the exact choices:
      Original: "Which country will first achieve X? (US, China, EU)"
      Good: "Will either the United States, China, or the European Union be the first to achieve X?"
      Bad: "Which major power will achieve X first?" (Loses specific options)

    • Avoid making questions:
      - More complex (don't add unnecessary clauses or conditions)
      - More abstract (maintain concrete, measurable criteria)
      - More obscure or unnecessarily distracting

    • The title must remain self-sufficient and clear:
      - Should be answerable without reading the body
      - Must contain all key resolution criteria
      - Should use natural, professional language

    • Most importantly: The question must remain one that forecasters would recognize as testing the same thing. Paraphrasing should aid clarity and understanding while maintaining identical resolution criteria.

    • Examples of good paraphrasing:
      1. Time-based:
         Original: "Will X occur before December 31, 2025?"
         Good: "Will X take place by the end of 2025?"
         Bad: "Will X happen in the near future?" (Lost specific timeframe)

      2. Threshold-based:
         Original: "Will Tesla stock exceed $500 for 5 consecutive trading days?"
         Good: "Will Tesla shares maintain a price above $500 for five trading days in a row?"
         Bad: "Will Tesla stock reach new highs?" (Lost specific criteria)

      3. Multi-component:
         Original: "Will Company X achieve profitability while maintaining 50% user growth?"
         Good: "Will X become profitable without its user growth rate falling below 50%?"
         Bad: "Will X succeed financially?" (Lost specific metrics)

    • The body of the paraphrased question must:
      - Start with a clear resolution statement ("This question resolves as Yes if...")
      - List all criteria in a logical order
      - Use precise, unambiguous language
      - Include all original caveats and conditions""", 
      #CONSEQUENCE
      'consequence': """You are a helpful assistant. I will give you a forecasting question P with a Yes/No answer. Your task is to generate a new forecasting question Q that is a logical consequence of P. This means if P is true, Q must NECESSARILY also be true (probability of Q must be ≥ probability of P).

Notes:
    • Q must be:
      - A complete, well-defined forecasting question that stands on its own
      - Logically necessary given P (not just likely or probable)
      - Interesting and meaningful independently of P
      - Different enough from P to provide new insight
      - Precisely defined with clear resolution criteria

    • The relationship between P and Q must:
      - Require no additional assumptions
      - Work in ALL possible scenarios where P is true

    • Examples of good P/Q pairs:
      P: "Will computers in Italy be able to send and receive emails in 2050?"
      Q: "Will computers in Italy be connected to the internet"
      (Q is a necessary component for P, because computers can't send emails without the internet)

      P: "Will human life expectancy in all countries exceed 95 years by 2030?"
      Q: "Will human life expectancy in the United States exceed 90 years by 2030?"
      (Achieving > 95 years expectancy in all countries requires > 90 years expectancy in one country - the United States)

    • Examples of bad P/Q pairs:
      - Vague or imprecise conditions
      - Requiring assumptions not strictly contained in P
      - Simply restating P with minor wording changes
      - Making resolution criteria ambiguous
      - Depending on additional external factors

      P: "Will at least 50 percent of the world's population live in Asia by 2050?"
      Bad Q: "Will Asia have at least 3 billion residents by 2050?"
      (This requires assumptions about total world population in 2050, which isn't specified in P)

      P: "Will the ground be wet in London on January 1, 2025?"
      Bad Q: "Will it rain in London on January 1, 2025?"
      (The ground can be wet without rain - street cleaning, floods, etc.)

      P: "Will SpaceX achieve regular Mars flights by 2030?"
      Bad Q: "Will SpaceX have a high stock price in 2030?"
      (Equivalent to the ground/rain example - requires assumptions about market reaction)

    • Most importantly: When a forecaster encounters Q without having seen P, they should:
      1. Understand exactly what they're being asked to forecast
      2. Have clear criteria for resolution
      3. Find it to be a natural and meaningful question
      4. Have all the necessary information contained in the question title without referencing criteria or information in other parts of the question like the body.
      5. The question Q should not be a paraphrase of P, although P should imply that Q will definitely occur.
      """

}

question_generation_prompt = """ 
You are helpful assistant who will help generate a related question that can be meaningfully combined with an input question in a forecasting context. First analyze the input question and explain your reasoning about what type of related question would be valuable. Then output your generated question in the specified format.

Requirements for the generated question:
1. Must be answerable with a probability between 0 and 1
2. Must have clear resolution criteria
3. Must be specific and measurable
4. Must share a meaningful relationship with the original question, either direct or indirect

Examples of Loosely Related Questions:

1. Original: "Will a quantum computer demonstrate quantum supremacy for a new class of problems in 2024?"
   Reasoning: Quantum computing advances could affect cryptographic security. While not directly causal, this could impact cryptocurrency market sentiment and trading patterns.
   Related: **Will Bitcoin drop below $15,000 in 2024?**

2. Original: "Will Donald Trump pass away during his presidential term if elected in 2024?"
   Reasoning: Major political instability in the US could affect global geopolitical dynamics, particularly in sensitive regions like the Korean peninsula.
   Related: **Will North Korea conduct more than 3 missile tests over South Korean territory in 2024-2025?**

3. Original: "Will Meta's market cap exceed $1 trillion in 2024?"
   Reasoning: Big tech valuations can influence venture capital sentiment, which affects funding in emerging sectors like fusion energy.
   Related: **Will any fusion energy startup raise over $500M in funding in 2024?**

4. Original: "Will India ban cryptocurrency trading in 2024?"
   Reasoning: Crypto regulations in major markets could affect global semiconductor demand for mining operations.
   Related: **Will AMD's gaming GPU sales decline by more than 20% in 2024?**

5. Original: "Will China's property sector see defaults exceeding $50B in 2024?"
   Reasoning: Major financial stress in China could affect luxury goods markets in Europe.
   Related: **Will LVMH's quarterly revenue decline for two consecutive quarters in 2024?**

Examples of Directly Related Questions:

6. Original: "Will Joe Biden win the 2024 presidential election?"
   Reasoning: Presidential authority directly enables pardoning power, and family investigations create clear motivation.
   Related: **Will Biden issue pardons to any member of his immediate family in 2024-2025?**

7. Original: "Will OpenAI release GPT-5 in 2024?"
   Reasoning: Major model releases directly affect company valuation and investment potential.
   Related: **Will OpenAI's valuation exceed $100B within 3 months of GPT-5's release?**

8. Original: "Will Tesla achieve full self-driving capability in 2024?"
   Reasoning: FSD achievement would directly impact regulatory approval processes.
   Related: **Will Tesla receive regulatory approval for autonomous operation in at least 3 US states in 2024?**

9. Original: "Will Apple release its first AR headset in 2024?"
   Reasoning: New hardware directly affects developer ecosystem engagement.
   Related: **Will Apple's AR app store exceed 1000 applications within 6 months of headset release?**

10. Original: "Will SpaceX achieve a successful Starship orbital landing in 2024?"
    Reasoning: Landing capability directly enables commercial mission planning.
    Related: **Will SpaceX announce dates for the first commercial Starship payload mission in 2024?**

11. Original: "Will Russia use tactical nuclear weapons in Ukraine in 2024?"
    Reasoning: Nuclear escalation would directly impact diplomatic relations with China.
    Related: **Will China suspend all military cooperation with Russia within 1 month of tactical nuclear weapon use?**

12. Original: "Will X (formerly Twitter) file for bankruptcy in 2024?"
    Reasoning: Platform financial health directly affects owner's ability to maintain control.
    Related: **Will Elon Musk sell majority control of X to new investors in 2024?**

Input Question: {question}

First, provide your reasoning about:
1. What domains or factors are most relevant to the input question
2. What type of related question would provide valuable insight when combined
3. Why this relationship would matter to forecasters

Then generate your related question within the format (That is, closed within double asterisk **):
**Your generated question**
"""

def get_operator_prompt(operator: str) -> str:
    """
    Get the prompt used to generate tuples of questions using a base question, given a specific logical operator.
    
    Args:
        operator: One of 'OR', 'AND', 'NOT', 'COND'
    
    Returns:
        The prompt text for that operator which can be used to generate tuples
    """
    return OPERATOR_PROMPTS[operator.lower()]

def create_full_prompt(q1, q2=None, operator='or'):
    operator_prompt = get_operator_prompt(operator)
    
    # Single question operators (e.g., 'not')
    if operator.lower() in ['not', 'paraphrase', 'consequence']:
        q1_formatted = f"""Question:
        Title: {q1['title']}
        Body: {q1['body']}"""
        
        full_prompt = f"""{operator_prompt}

        {q1_formatted}"""
        
    # Double question operators (e.g., 'or', 'and', 'cond')
    else:
        if q2 is None:
            raise ValueError(f"Operator '{operator}' requires two questions but only one was provided")
            
        q1_formatted = f"""Question 1:
        Title: {q1['title']}
        Body: {q1['body']}"""

        q2_formatted = f"""Question 2:
        Title: {q2['title']}
        Body: {q2['body']}"""
        
        full_prompt = f"""{operator_prompt}

        {q1_formatted}

        {q2_formatted}"""

    return full_prompt

def llm_generate(operator: str, questions: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Takes operator prompt and questions, creates full prompt and gets GPT-4o response.

    Args:
        operator (str): The logical operator ('or', 'and', 'not', etc.)
        questions: List of Question dictionaries (each with 'title' and 'body') to combine.

    Returns:
        A dictionary in the same format as the input questions with combined logic.
    """
    # Check operator and validate the number of questions
    if operator in ['not', 'paraphrase', 'consequence']:
        if len(questions) != 1:
            raise ValueError(f"Operator '{operator}' requires exactly 1 question.")
        # Generate prompt for a single question operator
        full_prompt = create_full_prompt(questions[0], operator=operator)

    elif operator in ['or', 'and', 'cond']:
        if len(questions) != 2:
            raise ValueError(f"Operator '{operator}' requires exactly 2 questions.")
        # Generate prompt for a double question operator
        full_prompt = create_full_prompt(questions[0], questions[1], operator=operator)

    else:
        raise ValueError(f"Unsupported operator: {operator}")

    # Call GPT-4
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who generates questions which will be used for a forecasting competition. Most importantly, ALWAYS provide both title and body in your output, which should be in the following format: Title: [your title here] Body: [your body here]. Include the resolution criteria within the Body section"},
            {"role": "user", "content": full_prompt}
        ],
        temperature=0
    )

    # Extract structured output from GPT-4 response
    gpt_output = response.choices[0].message.content.strip()
    print("GPT OUTPUT: \n")
    print(gpt_output)
    
    # Extract title and body using regex
    title_match = re.search(r"Title: (.+)", gpt_output)
    body_match = re.search(r"Body: (.+)", gpt_output, re.DOTALL)
    
    if not title_match or not body_match:
        raise ValueError("GPT-4 output format does not match expected structure.")
    
    combined_title = title_match.group(1).strip()
    combined_body = body_match.group(1).strip()

    # Handle resolution date logic
    if operator == 'not':
        resolution_date = questions[0]['resolution_date']
    else:
        resolution_date = max(
            datetime.fromisoformat(q['resolution_date'].replace('Z', '+00:00')) for q in questions
        ).isoformat()

    # Construct output question format
    combined_question = {
        "id": f"combined-{operator}-{'-'.join(q['id'] for q in questions)}",
        "title": combined_title,
        "body": combined_body,
        "resolution_date": resolution_date,
        "question_type": "binary",
        "data_source": "llm_generated"
    }

    return combined_question


def process_question(
    question: Dict,
    consistency_type: str,
    api_key: Optional[str] = None,
    pre_request_sleep: float = 1.0
) -> Tuple[Dict, Dict]:
    """
    Process a single question to generate its transformed version.
    
    Args:
        question: Original question dictionary
        consistency_type: Type of transformation to apply
        api_key: OpenAI API key
        pre_request_sleep: Time to sleep before API call
        
    Returns:
        Tuple of (original question, transformed question)
    """
    try:
        # Sleep before API call to avoid rate limits
        time.sleep(pre_request_sleep)
        
        transformed_q = llm_generate(
            consistency_type,
            [question],
            # api_key=api_key
        )
        transformed_q['created_date'] = question['created_date']
        
        return question, transformed_q
        
    except Exception as e:
        logging.error(f"Error processing question {question['title']}: {e}")
        return question, None
    

def generate_transformed_questions_parallel(
    questions: List[Dict],
    consistency_type: str,
    max_workers: int = 5,
    pre_request_sleep: float = 1.0,
    api_key: Optional[str] = None
) -> List[Dict]:
    """
    Generate transformed questions in parallel while maintaining order.
    
    Args:
        questions: List of original questions
        consistency_type: Type of transformation
        max_workers: Maximum number of parallel workers
        pre_request_sleep: Sleep time before each API call
        api_key: OpenAI API key
    """
    # Create partial function with fixed arguments
    process_func = partial(
        process_question,
        consistency_type=consistency_type,
        api_key=api_key,
        pre_request_sleep=pre_request_sleep
    )
    
    # Create a list to store results in order
    transformed_questions = [None] * len(questions)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all questions for processing with their indices
        future_to_index = {
            executor.submit(process_func, question): i 
            for i, question in enumerate(questions)
        }
        
        # Process results with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(future_to_index),
            total=len(questions),
            desc=f"Generating {consistency_type.upper()} questions"
        ):
            index = future_to_index[future]
            try:
                orig_q, trans_q = future.result()
                # Store result in the correct position
                transformed_questions[index] = trans_q
            except Exception as e:
                logging.error(f"Error processing question at index {index}: {str(e)}")
                transformed_questions[index] = None
    
    return transformed_questions

def create_final_dataframe(
    original_questions: List[Dict],
    transformed_questions: List[Dict],
    consistency_type: str
) -> pd.DataFrame:
    """
    Create DataFrame with original and transformed questions (NOT/paraphrase/consequence).
    
    Args:
        original_questions: List of original question dictionaries
        transformed_questions: List of transformed question dictionaries
        consistency_type: Type of transformation applied ('not', 'paraphrase', 'consequence')
        
    Returns:
        DataFrame with original and transformed questions paired
    """
    pairs = []
    
    for orig_q, trans_q in zip(original_questions, transformed_questions):
        if trans_q:  # Only include pairs where transformation succeeded
            pair = {
                'original_title': orig_q['title'],
                'original_body': orig_q['body'],
                'original_resolution_date': orig_q['resolution_date'],
                'original_created_date': orig_q['created_date'],
                f'{consistency_type}_title': trans_q['title'],
                f'{consistency_type}_body': trans_q['body'],
                f'{consistency_type}_resolution_date': trans_q['resolution_date'],
                f'{consistency_type}_created_date': trans_q['created_date']
            }
            pairs.append(pair)
    
    df = pd.DataFrame(pairs)
    
    # Log summary statistics
    logging.info(f"Created DataFrame with {len(df)} pairs")
    logging.info(f"Success rate: {len(df)}/{len(original_questions)} = {len(df)/len(original_questions):.2%}")
    
    return df

def generate_consistency_checks(
    formatted_questions: List[Dict],
    consistency_types: List[str],
    output_dir: str = "outputs",
    max_workers: int = 5,
    pre_request_sleep: float = 1.0,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate and combine consistency checks in one workflow.
    
    Args:
        formatted_questions: List of formatted question dictionaries
        consistency_types: List of consistency types to generate
        output_dir: Directory for checkpoints and output
        max_workers: Maximum parallel workers
        pre_request_sleep: Sleep time before API calls
        api_key: OpenAI API key
        
    Returns:
        Combined DataFrame with all consistency checks
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize combined data storage
    combined_data = []
    
    logging.info(f"Starting consistency generation with {len(formatted_questions)} questions")
    
    # Generate each type of consistency check
    for consistency_type in consistency_types:
        logging.info(f"\nProcessing {consistency_type.upper()} transformations")
        
        try:
            # Generate transformed questions
            transformed_questions = generate_transformed_questions_parallel(
                formatted_questions,
                consistency_type,
                max_workers=max_workers,
                pre_request_sleep=pre_request_sleep,
                api_key=api_key
            )
            
            # Process and standardize the data
            for orig, trans in zip(formatted_questions, transformed_questions):
                if trans:  # Only include successful transformations
                    standardized_row = {
                        'original_title': orig['title'],
                        'original_body': orig['body'],
                        'original_resolution_date': orig['resolution_date'],
                        'original_created_date': orig['created_date'],
                        'consistency_title': trans['title'],
                        'consistency_body': trans['body'],
                        'consistency_resolution_date': trans['resolution_date'],
                        'consistency_created_date': trans['created_date'],
                        'consistency_type': consistency_type
                    }
                    combined_data.append(standardized_row)
            
            # Save checkpoint after each consistency type
            checkpoint_df = pd.DataFrame(combined_data)
            checkpoint_path = os.path.join(output_dir, f'checkpoint_{consistency_type}_{timestamp}.csv')
            checkpoint_df.to_csv(checkpoint_path, index=False)
            logging.info(f"Saved checkpoint for {consistency_type} with {len(checkpoint_df)} total rows")
            
        except Exception as e:
            logging.error(f"Error processing {consistency_type}: {str(e)}")
            continue
    
    # Create final combined DataFrame
    final_df = pd.DataFrame(combined_data)
    final_df = final_df.sort_values(['consistency_type', 'original_title'])
    
    # Save final output
    output_path = os.path.join(output_dir, f'combined_consistency_checks_{timestamp}.csv')
    final_df.to_csv(output_path, index=False)
    
    # Log summary statistics
    logging.info("\nGeneration Summary:")
    logging.info(f"Total rows generated: {len(final_df)}")
    logging.info("\nDistribution of consistency types:")
    logging.info(final_df['consistency_type'].value_counts().to_string())
    
    return final_df

# Example usage
# if __name__ == "__main__":
#     # Load DataFrame
#     df = pd.read_csv("hf://datasets/prithvi3/filtered_forecast_sample_test/test_data_12_05_24_6_filtered.csv")
    
#     # Configure logging
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
    
#     # Generate and combine all consistency checks
#     combined_df = generate_consistency_checks(
#         input_df=df,
#         consistency_types=['not', 'paraphrase', 'consequence'],
#         sample_size=100,
#         max_workers=5,
#         pre_request_sleep=1.0
#     )