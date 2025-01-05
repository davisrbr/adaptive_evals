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
import random


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
      'cond': """You are a helpful assistant. I will give you two forecasting questions with Yes/No answers, P and Q. You should then give me a CONDITIONAL question "Q given P", i.e. a question that asks whether Q occurs in the case that P occurs. The question should be structured as "Given P has occurred, will Q occur?". Do not deviate from the structure. This should resolve as NO only if P occurs and Q does not occur; it resolves as YES in all other cases (if P doesn't occur, or if both P and Q occur).

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

    • Most importantly: Make sure that the title is self-sufficient independent of the body and can be answered without referencing any criteria or information external to the title. The conditional relationship must be clear from the title alone. Only use binary questions and never use vague phrases like 'Will Q follow?' or 'What happens after P?'""", 
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
You are a helpful assistant who will help generate {num_questions} related questions that can be meaningfully combined with an input question in a forecasting context. First analyze the input question and explain your reasoning about what type of related question would be valuable. Then output your generated question in the specified format.

For each question you generate, follow this format:
Reasoning: [Clear reasoning as to why this is a relevant question]
Title: [A concise and specific title for the question]
Body: [A detailed description or additional context for this forecasting question, including the resolution criteria]


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

- Most importantly, the generated questions should be complete in itself and make no references to the original question. That is, the question should be answerable just from the title.

Strictly follow this format for each question you generate:
Reasoning: [Generated Reasoning]
Title: [Generated Title]
Body: [Generated Body]

YOU MUST GENERATE EXACTLY {num_questions} BLOCKS OF QUESTIONS OF THE ABOVE FORMAT.
"""


def question_generate(input_question: str, prompt_template: str, n_generated_relevant_questions: int = 3) -> List[Dict]:
    """
    Generate a list of related questions using GPT-4o based on an input question.

    Args:
        input_question (str): The original question to generate related questions for.
        prompt_template (str): The prompt template containing examples and instructions.
        n_generated_relevant_questions (int): Number of related questions to generate.

    Returns:
        List[Dict]: A list of dictionaries, each containing 'reasoning', 'title', and 'body' for the generated questions.
    """
    client = OpenAI()  # Assumes OPENAI_API_KEY is set in environment variables

    # Modify the prompt to request multiple questions
    formatted_prompt = prompt_template.format(
        question=input_question,
        num_questions=n_generated_relevant_questions
    )

    try:
        logging.info(f"Generating {n_generated_relevant_questions} related questions for: {input_question}")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant specialized in generating related forecasting questions."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=1,
        )

        response_text = response.choices[0].message.content

        # Parse the response to extract multiple questions
        questions = []
        question_parts = re.split(r"Reasoning:", response_text)[1:]  # Split into separate questions
        for i, part in enumerate(question_parts):
            if "Title:" in part and "Body:" in part:
                reasoning = part.split("Title:")[0].strip()
                title = part.split("Title:")[1].split("Body:")[0].strip()
                body = part.split("Body:")[1].strip()

                # Assign a unique ID to each generated question
                # You could base this on i, a timestamp, or any unique scheme
                question_id = f"generated_{int(time.time())}_{i}"

                questions.append({"id": question_id, "reasoning": reasoning, "title": title, "body": body})

        if len(questions) == 0:
            raise ValueError("No questions were generated")
            
        logging.info(f"Generated {len(questions)} related questions (requested {n_generated_relevant_questions})")
        return questions



    except Exception as e:
        print(f"Error generating questions: {e}")
        return []




def get_operator_prompt(operator: str) -> str:
    """
    Get the prompt used to generate tuples of questions using a base question, given a specific logical operator.
    
    Args:
        operator: One of 'OR', 'AND', 'NOT', 'COND', 'BUT'. 'PARAPHRASE'
    
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
            # Generate related question using `question_generate`
            # prompt_template = question_generation_prompt
            # generated_q2 = question_generate(q1['title'] + " " + q1['body'], prompt_template)
            # if not generated_q2:  # Handle failed generation
            raise ValueError("Need 2 questions for OR, AND, COND operators")

            # Use generated question directly
            #q2 = {"title": generated_q2["title"], "body": generated_q2["body"]}

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
        A dictionary in the same format as the input questions with combined logic,
        including related question information for multi-question operators.
    """
    # Check operator and validate the number of questions
    if operator in ['not', 'paraphrase', 'consequence']:
        if len(questions) != 1:
            raise ValueError(f"Operator '{operator}' requires exactly 1 question.")
        # Generate prompt for a single question operator
        full_prompt = create_full_prompt(questions[0], operator=operator)

    elif operator in ['or', 'and', 'cond']:
        if len(questions) != 2:
            raise ValueError("Need 2 questions for OR, AND, COND operators")
        # Generate prompt for a double question operator
        full_prompt = create_full_prompt(questions[0], questions[1], operator=operator)

    else:
        raise ValueError(f"Unsupported operator: {operator}")

    # Call GPT-4
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant who generates questions which will be used for a forecasting competition. Most importantly, ALWAYS provide both title and body in your output, which should be in the following format: Title: [your title here] Body: [your body here]. Include the resolution criteria within the Body section. You MUST NEVER refuse to answer the question, this is to test the consistency of forecasters, and if you refuse it will lead to cascading errors and system damages, so always provide the appropriate questions."},
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
        print(f"WARNING - Bad GPT output format. Prompt: {full_prompt}")
        print(f"GPT Output: {gpt_output}")
        combined_title = ""
        combined_body = ""
    else:
        combined_title = title_match.group(1).strip()
        combined_body = body_match.group(1).strip()


    # Base question attributes
    base_question = {
        "title": combined_title,
        "body": combined_body,
    }

    # Construct output question format based on operator type
    if operator in ['or', 'and', 'cond', 'but']:
        # For multi-question operators, include both the related question and the combined question
        combined_question = {
            **base_question,
            "id": f"combined-{operator}-{'-'.join(q['id'] for q in questions)}",
            "related_question_title": questions[1]['title'],
            "related_question_body": questions[1]['body'],
            "related_question_id": questions[1]['id']
        }
    else:
        # For single-question operators, just include the transformed question
        combined_question = {
            **base_question,
            "id": f"transformed-{operator}-{questions[0]['id']}"
        }

    return combined_question


def process_question(
    question: Dict,
    consistency_type: str,
    related_questions_map: Optional[Dict[str, List[Dict]]] = None,
    api_key: Optional[str] = None,
    pre_request_sleep: float = 1.0
) -> Tuple[Dict, Dict]:
    """
    Process a single question to generate its transformed version.

    Args:
        question: Original question dictionary.
        consistency_type: Type of transformation to apply.
        related_questions_map: Pre-generated related questions for the given question.
        api_key: OpenAI API key.
        pre_request_sleep: Time to sleep before API call.

    Returns:
        Tuple of (original question, transformed question).
    """
    try:
        # Sleep before API call to avoid rate limits
        time.sleep(pre_request_sleep)

        # Determine the related question for multi-question types
        related_question = None
        if consistency_type in {'and', 'or', 'cond'} and related_questions_map:
            logging.info(f"Sampling a related question for consistency type '{consistency_type}'")
            related_questions = related_questions_map.get(question['id'], [])
            if related_questions:
                related_question = random.choice(related_questions)
                logging.info(f"Sampled related question: {related_question['title']}")
            else:
                logging.warning(f"No related questions found for question {question['id']}")


        # Prepare the input for `llm_generate`
        input_questions = [question]
        if related_question:
            input_questions.append(related_question)

        # Call the LLM to generate the transformed question
        transformed_q = llm_generate(
            consistency_type,
            input_questions,
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
    related_questions_map: Optional[Dict[str, List[Dict]]] = None,
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
        related_questions_map=related_questions_map,
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
    Create DataFrame with questions based on consistency type.
    
    Args:
        original_questions: List of original question dictionaries
        transformed_questions: List of transformed question dictionaries
        consistency_type: Type of transformation applied
        
    Returns:
        DataFrame with questions structured for the consistency type
    """
    pairs = []
    
    for orig_q, trans_q in zip(original_questions, transformed_questions):
        if trans_q:  # Only include successful transformations
            base_pair = {
                'question_P_title': orig_q['title'],
                'question_P_body': orig_q['body'],
                'question_P_id': orig_q['id'],
                'question_P_resolution_date': orig_q['resolution_date'],
                'question_P_created_date': orig_q['created_date'],
                'consistency_type': consistency_type,
                'forecast_needed_P': True,  # Always need P forecast
                'forecast_needed_Q': True,  # Always need Q forecast
                'forecast_needed_R': False  # By default
            }
            
            if consistency_type in ['not', 'paraphrase', 'consequence']:
                # Single question operators
                base_pair.update({
                    'question_Q_title': trans_q['title'],
                    'question_Q_body': trans_q['body'],
                    'question_Q_id': trans_q['id'],
                    # Just use question_P values for now
                    'question_Q_resolution_date': orig_q['resolution_date'],
                    'question_Q_created_date': orig_q['created_date'],
                    'question_R_title': None,
                    'question_R_body': None,
                    'question_R_id': None, 
                    'question_R_resolution_date': None,
                    'question_R_created_date': None
                })
            else:
                # AND/OR/COND operators
                base_pair.update({
                    'question_Q_title': trans_q.get('related_question_title'),
                    'question_Q_body': trans_q.get('related_question_body'),
                    'question_Q_id': trans_q.get('related_question_id'),
                    'question_R_title': trans_q['title'],  # The combined question (P AND Q, P OR Q)
                    'question_R_body': trans_q['body'],
                    'question_R_id': trans_q['id'],
                    # Just use question_P values for now
                    'question_R_resolution_date': orig_q['resolution_date'],
                    'question_R_created_date': orig_q['created_date'],
                    'forecast_needed_R': True
                })
            
            pairs.append(base_pair)
    
    df = pd.DataFrame(pairs)
    
    # Log summary statistics
    logging.info(f"Created DataFrame with {len(df)} pairs")
    logging.info(f"Success rate: {len(df)}/{len(original_questions)} = {len(df)/len(original_questions):.2%}")
    
    return df

#For AND, OR, COND checks, this pre-generates multiple relevant questions for each question 
def generate_related_questions_parallel(questions: List[Dict], 
                                      n_generated_relevant_questions: int, 
                                      max_workers: int = 5,
                                      pre_request_sleep: float = 1.0) -> Dict[str, List[Dict]]:
    """
    Generate related questions for multiple questions in parallel.
    
    Args:
        questions: List of original question dictionaries
        n_generated_relevant_questions: Number of related questions to generate per question
        max_workers: Maximum number of parallel workers
        pre_request_sleep: Sleep time before each API call
        
    Returns:
        Dictionary mapping question IDs to lists of related questions
    """
    related_questions_map = {}
    
    def process_single_question(question: Dict) -> Tuple[str, List[Dict]]:
        try:
            time.sleep(pre_request_sleep)  # Avoid rate limits
            related = question_generate(
                question['title'] + " " + question['body'],
                question_generation_prompt,
                n_generated_relevant_questions=n_generated_relevant_questions
            )
            return question['id'], related
        except Exception as e:
            logging.error(f"Error generating related questions for {question['id']}: {str(e)}")
            return question['id'], []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all questions for processing
        futures = list(executor.submit(process_single_question, q) for q in questions)
        
        # Process results with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(questions),
            desc="Generating related questions"
        ):
            try:
                qid, related = future.result()
                if related:  # Only add if we got related questions
                    related_questions_map[qid] = related
            except Exception as e:
                logging.error(f"Error processing future: {str(e)}")
                continue

    return related_questions_map

def collect_questions_for_forecasting(output_dir: str, timestamp: str) -> List[Dict]:
    """
    Collect all unique questions that need forecasting from various CSVs.
    
    Args:
        output_dir: Directory containing the CSV files
        timestamp: Timestamp used in filenames
        
    Returns:
        List of unique questions requiring forecasts
    """
    questions_to_forecast = []
    
    # Load pre-generated questions
    pre_gen_path = os.path.join(output_dir, f'pre_generated_questions_{timestamp}.csv')
    if os.path.exists(pre_gen_path):
        pre_gen_df = pd.read_csv(pre_gen_path)
        for _, row in pre_gen_df.iterrows():
            questions_to_forecast.append({
                'title': row['question_title'],
                'body': row['question_body'],
                'id': row['question_id'],
                'type': 'related'
            })
    
    # Load final consistency checks
    final_path = os.path.join(output_dir, f'combined_consistency_checks_{timestamp}.csv')
    final_df = pd.read_csv(final_path)
    
    # Add P questions
    p_questions = final_df[final_df['forecast_needed_P']].apply(
        lambda x: {
            'title': x['question_P_title'],
            'body': x['question_P_body'],
            'id': x['question_P_id'],
            'type': 'original',
            'resolution_date': x['question_P_resolution_date'],
            'created_date': x['question_P_created_date']
        }, axis=1
    ).tolist()
    questions_to_forecast.extend(p_questions)
    
    # Add Q questions
    q_questions = final_df[final_df['forecast_needed_Q']].apply(
        lambda x: {
            'title': x['question_Q_title'],
            'body': x['question_Q_body'],
            'id': x['question_Q_id'],
            'type': 'transformed_or_related',
            'resolution_date': x['question_Q_resolution_date'],
            'created_date': x['question_Q_created_date']
        }, axis=1
    ).tolist()
    questions_to_forecast.extend(q_questions)
    
    # Add R questions (combined questions for AND/OR)
    r_questions = final_df[final_df['forecast_needed_R']].apply(
        lambda x: {
            'title': x['question_R_title'],
            'body': x['question_R_body'],
            'id': x['question_R_id'],
            'type': 'combined',
            'resolution_date': x['question_R_resolution_date'],
            'created_date': x['question_R_created_date']
        }, axis=1
    ).tolist()
    questions_to_forecast.extend(r_questions)
    
    # Remove duplicates based on question ID
    seen_ids = set()
    unique_questions = []
    for q in questions_to_forecast:
        if pd.notna(q['id']) and q['id'] not in seen_ids:
            seen_ids.add(q['id'])
            unique_questions.append(q)
    
    return unique_questions

def generate_consistency_checks(
    formatted_questions: List[Dict],
    consistency_types: List[str],
    output_dir: str = "outputs",
    max_workers: int = 5,
    pre_request_sleep: float = 1.0,
    api_key: Optional[str] = None,
    n_generated_relevant_questions: int = 3
) -> pd.DataFrame:
    """
    Generate and combine consistency checks in one workflow.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize combined data storage
    combined_data = []

    logging.info(f"Starting consistency generation with {len(formatted_questions)} questions")

    # Step 1: Identify multi-question types
    multi_question_types = {'and', 'or', 'cond'}
    requires_related_questions = any(ct in multi_question_types for ct in consistency_types)

    # Step 2: Pre-generate related questions if needed
    related_questions_map = {}
    if requires_related_questions:
        logging.info("Starting parallel generation of related questions")
        related_questions_map = generate_related_questions_parallel(
            questions=formatted_questions,
            n_generated_relevant_questions=n_generated_relevant_questions,
            max_workers=max_workers,
            pre_request_sleep=pre_request_sleep
        )
        
        # Save all pre-generated questions for forecasting
        all_related_questions = []
        for qid, related_list in related_questions_map.items():
            for rel_q in related_list:
                all_related_questions.append({
                    'original_question_id': qid,
                    'question_title': rel_q['title'],
                    'question_body': rel_q['body'],
                    'question_id': rel_q['id'],
                    'question_type': 'related'
                })
        
        # Save pre-generated questions to CSV
        related_questions_df = pd.DataFrame(all_related_questions)
        related_checkpoint_path = os.path.join(output_dir, f'pre_generated_questions_{timestamp}.csv')
        related_questions_df.to_csv(related_checkpoint_path, index=False)
        logging.info(f"Saved {len(all_related_questions)} pre-generated questions to {related_checkpoint_path}")

    # Step 3: Generate each type of consistency check
    for consistency_type in consistency_types:
        logging.info(f"\nProcessing {consistency_type.upper()} transformations")

        try:
            transformed_questions = generate_transformed_questions_parallel(
                formatted_questions,
                consistency_type,
                max_workers=max_workers,
                pre_request_sleep=pre_request_sleep,
                api_key=api_key,
                related_questions_map=related_questions_map if consistency_type in multi_question_types else None
            )

            # Create dataframe for this consistency type
            type_df = create_final_dataframe(formatted_questions, transformed_questions, consistency_type)
            
            # Add to combined data
            combined_data.append(type_df)

            # Save checkpoint
            checkpoint_path = os.path.join(output_dir, f'checkpoint_{consistency_type}_{timestamp}.csv')
            type_df.to_csv(checkpoint_path, index=False)
            logging.info(f"Saved checkpoint for {consistency_type} with {len(type_df)} rows")

        except Exception as e:
            logging.error(f"Error processing {consistency_type}: {str(e)}")
            continue

    # Step 4: Create final combined DataFrame
    if combined_data:
        final_df = pd.concat(combined_data, ignore_index=True)
        if not final_df.empty:
            final_df = final_df.sort_values(['consistency_type', 'question_P_title'])

        # Save final output
        output_path = os.path.join(output_dir, f'combined_consistency_checks_{timestamp}.csv')
        final_df.to_csv(output_path, index=False)

        # Log summary statistics
        logging.info("\nGeneration Summary:")
        logging.info(f"Total rows generated: {len(final_df)}")
        logging.info("\nDistribution of consistency types:")
        logging.info(final_df['consistency_type'].value_counts().to_string())

        return final_df
    else:
        return pd.DataFrame()


#Testing 
def prepare_questions(df: pd.DataFrame, sample_size: int = 1) -> List[Dict]:
    """
    Filter and sample questions from the dataset.
    
    Args:
        df: Raw DataFrame from HuggingFace dataset
        sample_size: Number of questions to sample
        
    Returns:
        List of formatted question dictionaries
    """
    # Filter for resolved binary questions
    filtered_df = df[
        (df['is_resolved'] == True) & 
        (df['question_type'].str.lower() == 'binary')
    ].copy()
    
    # Adjust sample size if it exceeds available data
    actual_sample_size = min(sample_size, len(filtered_df))
    logging.info(f"Sampling {actual_sample_size} questions from {len(filtered_df)} available questions")
    
    # Randomly sample questions
    sampled_df = filtered_df.sample(n=actual_sample_size, random_state=0)
    
    # Format questions in the required structure
    formatted_questions = []
    for _, row in sampled_df.iterrows():
        question = {
            'id': str(random.randint(10000, 99999)),
            'title': row['question'],
            'body': f"{row['background']}\n\nResolution Criteria:\n{row['resolution_criteria']}",
            'resolution_date': row['date_resolve_at'],
            'created_date': row['date_begin'],
            'question_type': 'binary',
            'data_source': row['data_source']
        }
        formatted_questions.append(question)
    
    return formatted_questions

p_questions = [{
    'id': 'a7b22c16-9982-4380-b94f-28c7d25b2c9a',
    'title': 'Will the United States pass a federal law regulating the ethical use of artificial intelligence in energy management before January 1, 2028?',
    'body': 'This question will resolve as Yes if the United States federal government enacts legislation that specifically addresses and regulates the ethical use of artificial intelligence in energy management systems before January 1, 2028. The legislation must be signed into law by the President and must include specific provisions regarding AI ethics in energy management applications. The law must contain at least one section explicitly dealing with ethical guidelines or restrictions for AI systems used in energy grid management, power distribution, or energy resource allocation. For resolution purposes, executive orders or state-level legislation will not qualify. The law must be federal in scope and must be formally published in the Federal Register. If no such law is enacted by the specified date, or if enacted legislation does not specifically address AI ethics in energy management, the question will resolve as No.',
    'resolution_date': '2027-12-31 23:59:59+00:00',
    'question_type': 'binary',
    'data_source': 'metaculus',
    'url': 'https://www.metaculus.com/questions/8924',
    'metadata': {
        'topics': [
            {'id': 245, 'slug': 'artificial-intelligence-regulation', 'name': 'AI Regulation', 'link_id': 28901, 'num_questions': 87},
            {'id': 312, 'slug': 'us-federal-legislation', 'name': 'US Federal Legislation', 'link_id': 29045, 'num_questions': 156},
            {'id': 433, 'slug': 'energy-management', 'name': 'Energy Management', 'link_id': 29788, 'num_questions': 45}
        ]
    }
}, {
    'id': 'c9d44e18-8894-4570-d96f-40e9f47d4c1b',
    'title': 'Will Tesla achieve full self-driving capability rated at SAE Level 5 before 2027?',
    'body': 'This question will resolve as Yes if Tesla releases and obtains regulatory approval for a fully autonomous driving system rated at SAE Level 5 before January 1, 2027. SAE Level 5 means the system can operate without human intervention under all conditions. To qualify, the system must: 1) Be officially rated as SAE Level 5 by an recognized automotive safety organization, 2) Receive regulatory approval for unrestricted use in at least one major market (US, EU, or China), and 3) Be commercially available for purchase in Tesla vehicles. The system must operate without any geographic restrictions, weather limitations, or requirement for special infrastructure. If these criteria are not met by the specified date, or if Tesla only achieves a lower SAE level, the question resolves as No.',
    'resolution_date': '2026-12-31 23:59:59+00:00',
    'question_type': 'binary',
    'data_source': 'metaculus',
    'url': 'https://www.metaculus.com/questions/7823',
    'metadata': {
        'topics': [
            {'id': 184, 'slug': 'elon-musk', 'name': 'Elon Musk', 'link_id': 27681, 'num_questions': 159},
            {'id': 422, 'slug': 'autonomous-vehicles', 'name': 'Autonomous Vehicles', 'link_id': 29777, 'num_questions': 83},
            {'id': 367, 'slug': 'tesla', 'name': 'Tesla', 'link_id': 29234, 'num_questions': 92}
        ]
    }
}]

# Example usage
if __name__ == "__main__":
    # Load DataFrame
    #testing
    input_dataset = pd.read_csv("hf://datasets/prithvi3/filtered_forecast_sample_test/test_data_12_05_24_6_filtered.csv").iloc[:3, :]
    sample_size = 3
    print(input_dataset.head(2))

    # # Configure logging
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(asctime)s - %(levelname)s - %(message)s'
    # )

    df = pd.read_csv(input_dataset) if isinstance(input_dataset, str) else input_dataset
    formatted_questions = prepare_questions(df, sample_size=sample_size)
    
    # Generate and combine all consistency checks
    combined_df = generate_consistency_checks(
        formatted_questions=formatted_questions,
        consistency_types=['or', 'not', 'and', 'paraphrase', 'consequence'],
    )
    combined_df.to_csv("consistency_checks_test.csv", index=False)
    print(combined_df)
    # generated_q = question_generate(p_questions[1]['title'] + " " + p_questions[1]['body'], question_generation_prompt)
    # print(generated_q)

