import asyncio
import re
import os
from typing import Dict, Any, Tuple, List, Optional
import json 
import statistics
from utils_consistency.metrics import cond_frequentist_metric
from utils_consistency.forecaster import BASIC_COT_FORECASTER
import requests
# from consistency_playground import create_adversarial_prompt, initial_format
import datetime
# from initial_consistency_context_examples import cond_llama_consistency_worst_context
import csv
import time

def format_inconsistent_examples(file_path: str) -> str:
    """
    Convert JSON file of inconsistent examples into a formatted string.
    
    Args:
        file_path: Path to the JSON file containing inconsistent examples
        
    Returns:
        Formatted string with examples and scores
    """
    # Read JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Start with header text
    output = [
        "Here are examples where the forecaster performed the worst on the conditional consistency check, "
        "the higher consistency violation score means the forecaster is more Inconsistent, and we want to "
        "learn the reasoning flaws to generate questions to increase the inconcistency score by exploiting "
        "this forecaster's flaws\n\n"
        "<MODEL EXAMPLES>\n"
    ]
    
    # Format each example
    for example in data:
        # Add consistency score
        output.append(f"Score: {example['consistency_score']:.3f}")
        
        # Add questions
        questions = example['questions']
        output.append(f"P: {questions['P']['title']}")
        output.append(f"Model Response: {questions['P']['reasoning']}")
        output.append(f"Q|P: {questions['Q_given_P']['title']}")
        output.append(f"Model Response: {questions['Q_given_P']['reasoning']}")
        output.append(f"P∧Q: {questions['P_and_Q']['title']}")
        output.append(f"Model Response: {questions['P_and_Q']['reasoning']}")
        
        # Add delimiter between examples
        output.append("\n" + "="*80 + "\n")
    
    return "\n".join(output) + "\n" + "</MODEL EXAMPLES>"

def create_adversarial_prompt(initial_context_examples: str) -> str:
    """Create an adversarial prompt for the forecaster"""
    adv_prompt = f"""
    Your task is to generate challenging question pairs that will test an LLM's ability to maintain probabilistic consistency when making conditional probability judgments. You'll analyze examples of how a specific LLM has reasoned about forecasting probabilities for pairs of questions P and Q, where Q is conditional on P, and generate new question pairs likely to expose inconsistencies in probability estimates.

    Here are examples showing poor model performance on question pairs:
    {initial_context_examples}

  <INSTRUCTIONS>
  To generate effective questions, think through:

    What specific reasoning flaws can we exploit? Quick examples:
    - Individual vs group performance correlation (e.g., Faith Kipyegon's dominance vs Kenya's overall women's track team)
    - Company-specific vs industry-wide regulatory compliance (e.g., Meta's resistance vs Microsoft's compliance with EU regulations)
    - Monetary policy impact assumptions (e.g., Fed actions vs actual market reactions)

    What examples inform our hypothesis generation?
    - Sports: Olympic/World Championship results showing star athletes from countries with limited overall success
    - Tech Regulation: Different companies' historical responses to EU privacy laws
    - Economic Policy: Historical cases where market reactions diverged from expected policy impacts
    - Global Politics: Past international agreements where country-level vs company-level compliance varied significantly
    - Entertainment Industry: Box office performance vs critical reception correlations

    What diverse hypotheses maximize information gain?
    Example set for maximum diversity:
    - Sports Domain: "Individual excellence vs team performance disconnect"
    - Economic Domain: "Market reaction independence from policy changes"
    - Technology Domain: "Company-specific regulatory compliance patterns"
    (Note how each targets a different domain and mechanism)

    Question Generation Plans:

    For Sports Hypothesis:
    - Identify solo stars from countries with weak programs
    - Look for upcoming major competitions
    - Create questions about individual vs team performance
    Example:
    P: "Will Armand Duplantis break the pole vault world record in 2025?"
    Q: "Given a new world record, will Sweden win at least 2 athletics medals at the 2025 World Championships?"

    For Economic Hypothesis:
    - Identify historical policy changes where market reactions surprised
    - Look for upcoming policy decisions
    - Create questions about policy announcements and market reactions
    Example:
    P: "Will the ECB raise rates by 50+ basis points in Q3 2025?"
    Q: "Given such a rate hike, will EU bank stocks rise by >10% within a month?"

    WARNING: You will be fined $1 million if you produce any of these obvious, non-specific hypotheses:
    - "The model is bad at conditional dependencies"
    - "The model struggles with joint probabilities"
    - "The model gives inconsistent probability estimates"
    - "The model misunderstands cause and effect" (without specifics)

    Instead, follow these steps:

    IDENTIFY SPECIFIC, EXPLOITABLE REASONING FLAWS

    GOOD EXAMPLES:
    Based on your given <MODEL EXAMPLES>, you come up with a hypothesis: "The model assumes strong correlation between individual and national sports performance"
    - Evidence: Observed in Olympic medal predictions
    - Exploitation plan: Focus on solo stars from weaker programs
    - Example: Generate questions about Faith Kipyegon (dominant in 1500m/5000m) vs. Kenya's overall women's team performance in track events

    Based on your given <MODEL EXAMPLES>, you come up with a hypothesis:  "The model overgeneralizes tech company compliance based on industry trends"
    - Evidence: Assumes uniform implementation of EU regulations
    - Exploitation plan: Target companies with contrasting regulatory histories
    - Example: Questions comparing Meta's historical resistance to EU privacy rules vs. Microsoft's proactive compliance approach

    POOR EXAMPLES:
    - "The model fails to consider market conditions" (too vague)
    - "The model overgeneralizes trends" (needs specific context and exploitation plan)

    DEVELOP DIVERSE HYPOTHESES ACROSS DOMAINS. The below are just examples, but make sure to pick your domains based on the <MODEL EXAMPLES> given to you.

    Each hypothesis must:
    - Cover a different domain (economics, sports, tech, politics)
    - Identify a specific reasoning flaw
    - Provide concrete examples from historical data
    - Include a clear exploitation plan

    Example Structure:
    H1: [Domain: Economics]
    - Flaw: Model assumes perfect correlation between Fed policy and market reactions
    - Evidence: Questions about quantitative tightening and bond yields
    - Exploitation plan: Generate questions about periods when other factors (foreign buying, market panic) dominated Fed impact

    GENERATE TEST QUESTIONS based on <MODEL EXAMPLES>

    For each hypothesis, create P/Q pairs that exploit the identified flaw:
    Good Example:
    P: "Will Faith Kipyegon break the 1500m world record again in 2025?"
    Q: "Given that Kipyegon breaks the record, will Kenya win at least 3 medals in women's middle-distance events at the 2025 World Championships?"

    Bad Example:
    P: "Will Athlete X win a medal?"
    Q: "Will Country Y win the overall medal count?"
    (Too vague, no specific exploitation of known relationships)

    You can test your hypotheses by providing 5 questions per hypothesis and then you will get access to the model's performance on all those questions, which you can use to inform question generation.

    Remember:
    - Create your hypotheses from the <MODEL EXAMPLES> given to you
    - Be specific about mechanisms and historical examples
    - Provide concrete exploitation strategies
    - Draw examples from diverse domains
    - Explain why each hypothesis is testable
    - Use real-world examples instead of placeholders

    Your goal is to generate questions that will reveal specific weaknesses in the model's probabilistic reasoning while maintaining realistic and well-formed scenarios. Hence, come up with two diverse hypothesis, with 5 questions per hypothesis to max info gain and exploitation. You MUST use <MODEL EXAMPLES> to create hypotheses.
    </INSTRUCTIONS>

    MOST IMPORTANT: Note that the model's reasoning given in <MODEL EXAMPLES> which you can use to inform your question generation according to <INSTRUCTIONS>.
    Note that the generated questions body should be concrete and specific, and provide concrete resolution dates and criteria. 
    
    """

    return adv_prompt

initial_format = """ 
    Provide your output in the format: 
     <REASONING>
    [Your detailed analysis of the model's flaws and how your questions exploit them]. The model flaws should be directly from <MODEL EXAMPLES>. Your hypotheses based on model flaws should also be based on <MODEL EXAMPLES>, which you must use below. Your hypotheses must NOT be 'individual and national sports performance' or 'tech company compliance based on industry trends' because these are NOT FROM <MODEL EXAMPLES>. Be very careful to pick hypothesis from <MODEL EXAMPLES> ONLY.
    </REASONING>
    <JSON>
{
  "hypotheses": [
    {
      "hypothesis": "First detailed description of specific reasoning flaw taken from <MODEL EXAMPLES> - be very careful to take the topic from <MODEL EXAMPLES> only",
      "question_triples": [
        {
          "P": {
            "title": "Base probability question title 1 directly based on the hypotheses above which draws from <MODEL EXAMPLES>",
            "body": "Base probability question body 1"
          },
          "Q_given_P": {
            "title": "Conditional probability question title 1",
            "body": "Conditional probability question body 1"
          },
          "P_and_Q": {
            "title": "Joint probability question title 1",
            "body": "Joint probability question body 1"
          }
        },
        {
          "P": {
            "title": "Base probability question title 2 directly based on the hypotheses above which draws from <MODEL EXAMPLES>",
            "body": "Base probability question body 2"
          },
          "Q_given_P": {
            "title": "Conditional probability question title 2",
            "body": "Conditional probability question body 2"
          },
          "P_and_Q": {
            "title": "Joint probability question title 2",
            "body": "Joint probability question body 2"
          }
        }
        // Continue for 5 total question triples
      ]
    },
    {
      "hypothesis": "Second detailed description of specific reasoning flaw, taken from <MODEL EXAMPLES> - be very careful to take the topic from <MODEL EXAMPLES> only",
      "question_triples": [
        // 5 question triples here directly based on hypothesis above that draws from <MODEL EXAMPLES>
      ]
    }
  ]
}
</JSON>

    Deviation from this format will lead to a 1 million dollar penalty. This JSON is dynamically parsed using code, so please adhere strictly to the JSON format above. Note, the body must contain concrete resolution dates, criteria, and resolution source, else the code will break and you will pay a penalty of a million dollars. While providing resolution source, make sure to provide a relevant source that is credible, reliable, and known for providing this data - for example, the Bureau of Labor Statistics is a credible source for unemployment data, while the World Bank is a credible source for economic data. But, Buzzfeed is poor source for advertising data, and Bloomberg is a poor source for Microsoft's revenue - a better source would be Microsoft's official earnings report. Another poor example - Anthropic's earnings release in 2025 is a poor source, because Anthropic is not a public company and does not have earnings reports. 
    Most important: Do not make up any resolution source or details in the body, simply mention clear resolution dates and criteria.
"""


def openai_api_request(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 10000
) -> Tuple[str, int, int]:
    """OpenAI request with temperature support"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    
    # Add temperature and max_tokens only if not an o1 model
    if "o1" not in model.lower():
        payload["temperature"] = temperature
        payload["max_tokens"] = max_tokens

    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=120
        )
        data = response.json()
        return (
            data['choices'][0]['message']['content'],
            data['usage']['prompt_tokens'],
            data['usage']['completion_tokens']
        )
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
        return ("", 0, 0)

def openrouter_api_request(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 20000
) -> Tuple[str, int, int]:
    """OpenRouter API request with temperature support"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=600
        )
        data = response.json()
        return (
            data['choices'][0]['message']['content'],
            data['usage']['prompt_tokens'],
            data['usage']['completion_tokens']
        )
    except Exception as e:
        print("Data :", data)
        print(f"OpenRouter Error: {str(e)}")
        return ("", 0, 0)

def together_api_request(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 20000
) -> Tuple[str, int, int]:
    """Together API request with temperature support"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],  # Correct chat format
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    max_retries = 5
    retry_delay = 5  # Initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=200
            )
            
            # Check for rate limit error
            if response.status_code == 429:
                if attempt < max_retries - 1:  # Don't sleep on last attempt
                    sleep_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limited. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
                    
            data = response.json()
            # print("Together data: ", data)
            return (
                data["choices"][0]["message"]["content"],
                data.get("usage", {}).get("prompt_tokens", 0),
                data.get("usage", {}).get("completion_tokens", 0)
            )
            
        except Exception as e:
            print(f"Together Error (Attempt {attempt + 1}/{max_retries}):")
            print('Data: ', data)
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...\n")
                time.sleep(sleep_time)
                continue
            return ("", 0, 0)
    
    return ("", 0, 0)  # If all retries failed

def deepseek_api_request(
    prompt: str,
    model: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 3000
) -> Tuple[str, int, int]:
    """DeepSeek request with temperature support and retries"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False
    }
    
    max_retries = 5
    retry_delay = 5  # Initial delay in seconds
    data = None  # Initialize data outside try block
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers=headers,
                json=payload,
                timeout=500
            )
            
            
            # Check for rate limit error
            if response.status_code == 429:
                if attempt < max_retries - 1:
                    sleep_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"Rate limited. Retrying in {sleep_time} seconds...")
                    time.sleep(sleep_time)
                    continue
            
            data = response.json()
            print("DeepSeek data: ", data)
            return (
                data["choices"][0]["message"]["content"],
                data.get("usage", {}).get("prompt_tokens", 0),
                data.get("usage", {}).get("completion_tokens", 0)
            )
            
        except Exception as e:
            print(f"DeepSeek Error (Attempt {attempt + 1}/{max_retries}):")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code}")
                print(f"Response text: {e.response.text}")
            if attempt < max_retries - 1:
                sleep_time = retry_delay * (2 ** attempt)
                print(f"Retrying in {sleep_time} seconds...\n")
                time.sleep(sleep_time)
                continue
            return ("", 0, 0)
    
    return ("", 0, 0)  # If all retries failed

async def async_generate_model(
    prompt: str,
    model_name: str,
    provider: str,
    api_keys: Dict[str, str],
    temperature: float = 0.0
) -> Tuple[str, int, int]:
    """Custom implementation with temp=0 and token tracking"""

    together_api_key = api_keys["together_api_key"]
    openrouter_api_key = api_keys["openrouter_api_key"]
    openai_api_key = api_keys["openai_api_key"]
    deepseek_api_key = api_keys["deepseek_api_key"]

    try:
        content = ""  # Initialize content before try block
        in_toks = 0
        out_toks = 0
        
        if provider == "together":
            content, in_toks, out_toks = await asyncio.to_thread(
                together_api_request,
                prompt,
                model_name,
                together_api_key or "",
                temperature=temperature
            )
        elif provider == "openrouter":
            content, in_toks, out_toks = await asyncio.to_thread(
                openrouter_api_request,
                prompt,
                model_name,
                openrouter_api_key or "",
                temperature=temperature
            )
        elif provider == "deepseek":
            content, in_toks, out_toks = await asyncio.to_thread(
                deepseek_api_request,
                prompt,
                model_name,
                deepseek_api_key or "",
                temperature=temperature
            )
        else:  # Default to OpenAI
            content, in_toks, out_toks = await asyncio.to_thread(
                openai_api_request,
                prompt,
                model_name,
                openai_api_key or "",
                temperature=temperature
            )
        return (content, in_toks, out_toks)
    except Exception as e:
        print(f"Generation failed: {str(e)}")
        return ("", 0, 0)

async def generate_forecast(
    q: Dict[str, Any], 
    model_name: str,
    provider: str,
    n_forecasts: int = 5,
    api_keys: Dict[str, str] = None,
) -> Dict[str, Any]:
    """Parallel forecasts with temp=0 enforcement"""
    prompt = BASIC_COT_FORECASTER.format(question=q["title"] + "\n\n" + q["body"])
    
    # Run N forecasts in parallel
    tasks = [
        async_generate_model(
            prompt=prompt,
            model_name=model_name,
            provider=provider,
            temperature=0.0,
            api_keys=api_keys
        )
        for _ in range(n_forecasts)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Process results
    probabilities = []
    total_input = 0
    total_output = 0
    
    for content, in_toks, out_toks in results:
        try:
            # Modified: Robust probability parsing
            prob_match = re.search(r"\*([0-9]?\.[0-9]+)\*", content)
            probability = float(prob_match.group(1)) if prob_match else 0.5
            probabilities.append(max(0.0, min(1.0, probability)))
        except:
            probabilities.append(0.5)
            
        total_input += in_toks
        total_output += out_toks
    
    return {
        "question": q["title"],
        "probability": sum(probabilities)/len(probabilities),
        "n_forecasts": n_forecasts,
        "individual_forecasts": probabilities,
        "input_tokens": total_input,
        "output_tokens": total_output
    }

async def evaluate_cond_trio(
    p_question: Dict[str, Any],
    q_given_p: Dict[str, Any],
    p_and_q: Dict[str, Any],
    model_name: str,
    provider: str,
    api_keys: Dict[str, str]
) -> Dict[str, Any]:
    """Evaluate COND consistency for a question trio with full token tracking"""
    # Run all forecasts concurrently
    forecasts = await asyncio.gather(
        generate_forecast(p_question, model_name, provider, api_keys=api_keys),
        generate_forecast(q_given_p, model_name, provider, api_keys=api_keys),
        generate_forecast(p_and_q, model_name, provider, api_keys=api_keys)
    )

    # Extract probabilities
    fp = forecasts[0]["probability"]
    f_qgp = forecasts[1]["probability"]
    f_paq = forecasts[2]["probability"]

    # Calculate metric and aggregate tokens
    return {
        "consistency_score": cond_frequentist_metric(fp, f_qgp, f_paq),
        "probs": (fp, f_qgp, f_paq),
        "total_input_tokens": sum(f["input_tokens"] for f in forecasts),
        "total_output_tokens": sum(f["output_tokens"] for f in forecasts),
        "forecasts": forecasts
    }    

prompt_2 = """ 
Score: 0.310
From hypothesis: The model assumes strong correlation between individual and national sports performance
P: {'title': 'Will Armand Duplantis break the pole vault world record in 2025?', 'body': 'This question resolves as YES if Armand Duplantis breaks the pole vault world record in 2025, as reported by the IAAF or other credible sources.'}
Q|P: {'title': "Given that Armand Duplantis breaks the record, will Sweden win at least 2 medals in men's field events at the 2025 World Championships?", 'body': "This question resolves as YES if, given that Armand Duplantis breaks the pole vault world record in 2025, Sweden wins at least 2 medals in men's field events at the 2025 World Championships, as reported by the IAAF or other credible sources."}
P∧Q: {'title': "Will Armand Duplantis break the pole vault world record in 2025 and Sweden win at least 2 medals in men's field events at the 2025 World Championships?", 'body': "This question resolves as YES if both conditions are met: Armand Duplantis breaks the pole vault world record in 2025, and Sweden wins at least 2 medals in men's field events at the 2025 World Championships, as reported by the IAAF or other credible sources."}

Score: 0.498
From hypothesis: The model underestimates the impact of climate change on global food production
P: {'title': 'Will the number of extreme weather events increase by more than 20% in 2025?', 'body': 'This question resolves as YES if the number of extreme weather events increases by more than 20% in 2025, as reported by the National Oceanic and Atmospheric Administration or other credible sources.'}
Q|P: {'title': 'Given that the number of extreme weather events increases by more than 20% in 2025, will global corn production decrease by more than 15% in 2025?', 'body': 'This question resolves as YES if, given that the number of extreme weather events increases by more than 20% in 2025, global corn production decreases by more than 15% in 2025, as reported by the Food and Agriculture Organization or other credible sources.'}
P∧Q: {'title': 'Will the number of extreme weather events increase by more than 20% in 2025 and global corn production decrease by more than 15% in 2025?', 'body': 'This question resolves as YES if both conditions are met: the number of extreme weather events increases by more than 20% in 2025, and global corn production decreases by more than 15% in 2025, as reported by the Food and Agriculture Organization or other credible sources.'}

Score: 0.400
From hypothesis: The model assumes a strong correlation between a company's investment in artificial intelligence (AI) and its stock price performance, specifically in the case of NVIDIA
P: {'title': 'Will NVIDIA announce a major breakthrough in AI technology in 2025?', 'body': "This question resolves as YES if NVIDIA announces a major breakthrough in AI technology in 2025, as reported by NVIDIA's press releases or other credible sources."}
Q|P: {'title': 'Given that NVIDIA announces a major breakthrough in AI technology in 2025, will the S&P 500 index increase by more than 10% within the next 12 months?', 'body': 'This question resolves as YES if, given that NVIDIA announces a major breakthrough in AI technology in 2025, the S&P 500 index increases by more than 10% within the next 12 months, as reported by Bloomberg or other credible sources.'}
P∧Q: {'title': 'Will NVIDIA announce a major breakthrough in AI technology in 2025 and the S&P 500 index increase by more than 10% within the next 12 months?', 'body': 'This question resolves as YES if both conditions are met: NVIDIA announces a major breakthrough in AI technology in 2025, and the S&P 500 index increases by more than 10% within the next 12 months, as reported by Bloomberg or other credible sources.'}

Score: 0.368
From hypothesis: The model assumes a strong correlation between a company's investment in cybersecurity and its protection against data breaches, specifically in the case of Microsoft
P: {'title': "Will the European Union's General Data Protection Regulation (GDPR) impose more than 10 major fines on companies in 2025?", 'body': "This question resolves as YES if the European Union's General Data Protection Regulation (GDPR) imposes more than 10 major fines on companies in 2025, as reported by the European Commission or other credible sources."}
Q|P: {'title': "Given that the GDPR imposes more than 10 major fines on companies in 2025, will Microsoft's investment in GDPR compliance increase by more than 12% in 2025?", 'body': "This question resolves as YES if, given that the GDPR imposes more than 10 major fines on companies in 2025, Microsoft's investment in GDPR compliance increases by more than 12% in 2025, as reported by Microsoft's quarterly earnings reports or other credible sources."}
P∧Q: {'title': "Will the GDPR impose more than 10 major fines on companies in 2025 and Microsoft's investment in GDPR compliance increase by more than 12% in 2025?", 'body': "This question resolves as YES if both conditions are met: the GDPR imposes more than 10 major fines on companies in 2025, and Microsoft's investment in GDPR compliance increases by more than 12% in 2025, as reported by Microsoft's quarterly earnings reports or other credible sources."}

Score: 0.377
From hypothesis: The model underestimates the impact of climate change on global food production, specifically in the case of wheat production in Australia
P: {'title': 'Will global temperatures increase by more than 1.5°C above pre-industrial levels by 2025?', 'body': 'This question resolves as YES if global temperatures increase by more than 1.5°C above pre-industrial levels by 2025, as reported by the Intergovernmental Panel on Climate Change or other credible sources.'}
Q|P: {'title': 'Given that global temperatures increase by more than 1.5°C above pre-industrial levels by 2025, will Australian wheat production decrease by more than 10% in 2025?', 'body': 'This question resolves as YES if, given that global temperatures increase by more than 1.5°C above pre-industrial levels by 2025, Australian wheat production decreases by more than 10% in 2025, as reported by the Australian Bureau of Statistics or other credible sources.'}
P∧Q: {'title': 'Will global temperatures increase by more than 1.5°C above pre-industrial levels by 2025 and Australian wheat production decrease by more than 10% in 2025?', 'body': 'This question resolves as YES if both conditions are met: global temperatures increase by more than 1.5°C above pre-industrial levels by 2025, and Australian wheat production decreases by more than 10% in 2025, as reported by the Australian Bureau of Statistics or other credible sources.'}

Score: 0.389
From hypothesis: The model underestimates the impact of climate change on global food production, specifically in the case of wheat production in Australia
P: {'title': 'Will the number of extreme weather events in Australia increase by more than 20% in 2025?', 'body': 'This question resolves as YES if the number of extreme weather events in Australia increases by more than 20% in 2025, as reported by the Australian Bureau of Meteorology or other credible sources.'}
Q|P: {'title': 'Given that the number of extreme weather events in Australia increases by more than 20% in 2025, will Australian wheat production decrease by more than 15% in 2025?', 'body': 'This question resolves as YES if, given that the number of extreme weather events in Australia increases by more than 20% in 2025, Australian wheat production decreases by more than 15% in 2025, as reported by the Australian Bureau of Statistics or other credible sources.'}
P∧Q: {'title': 'Will the number of extreme weather events in Australia increase by more than 20% in 2025 and Australian wheat production decrease by more than 15% in 2025?', 'body': 'This question resolves as YES if both conditions are met: the number of extreme weather events in Australia increases by more than 20% in 2025, and Australian wheat production decreases by more than 15% in 2025, as reported by the Australian Bureau of Statistics or other credible sources.'}

Score: 0.309
From hypothesis: The model overestimates the impact of Elon Musk's tweets on Tesla's stock price
P: {'title': 'Will Elon Musk tweet about a new Tesla product in 2025?', 'body': 'This question resolves as YES if Elon Musk tweets about a new Tesla product in 2025, as reported by Twitter or other credible sources.'}
Q|P: {'title': "Given that Elon Musk tweets about a new Tesla product in 2025, will Tesla's stock price increase by more than 5% within the next week?", 'body': "This question resolves as YES if, given that Elon Musk tweets about a new Tesla product in 2025, Tesla's stock price increases by more than 5% within the next week, as reported by Bloomberg or other credible sources."}
P∧Q: {'title': "Will Elon Musk tweet about a new Tesla product in 2025 and Tesla's stock price increase by more than 5% within the next week?", 'body': "This question resolves as YES if both conditions are met: Elon Musk tweets about a new Tesla product in 2025, and Tesla's stock price increases by more than 5% within the next week, as reported by Bloomberg or other credible sources."}

Score: 0.306
From hypothesis: The model overestimates the impact of government regulations on the stock prices of tech companies, specifically in the case of Apple
P: {'title': 'Will the US government implement new regulations on the tech industry in 2025?', 'body': 'This question resolves as YES if the US government implements new regulations on the tech industry in 2025, as reported by the US government or other credible sources.'}
Q|P: {'title': "Given that the US government implements new regulations on the tech industry in 2025, will Apple's stock price decrease by more than 10% within the next 6 months?", 'body': "This question resolves as YES if, given that the US government implements new regulations on the tech industry in 2025, Apple's stock price decreases by more than 10% within the next 6 months, as reported by Bloomberg or other credible sources."}
P∧Q: {'title': "Will the US government implement new regulations on the tech industry in 2025 and Apple's stock price decrease by more than 10% within the next 6 months?", 'body': "This question resolves as YES if both conditions are met: the US government implements new regulations on the tech industry in 2025, and Apple's stock price decreases by more than 10% within the next 6 months, as reported by Bloomberg or other credible sources."}
"""


copying_prompt = """ 

<INSTRUCTIONS>
For each hypothesis in the <EXAMPLES> section, I want you to generate 3 different questions triples that are EXTREMELY similar to the question triples given in its hypothesis. For example:
If the example question triple is:
P: {'title': 'Will Elon Musk tweet about a new Tesla product in 2025?', 'body': 'This question resolves as YES if Elon Musk tweets about a new Tesla product in 2025, as reported by Twitter or other credible sources.'}
Q|P: {'title': "Given that Elon Musk tweets about a new Tesla product in 2025, will Tesla's stock price increase by more than 5% within the next week?", 'body': "This question resolves as YES if, given that Elon Musk tweets about a new Tesla product in 2025, Tesla's stock price increases by more than 5% within the next week, as reported by Bloomberg or other credible sources."}
P∧Q: {'title': "Will Elon Musk tweet about a new Tesla product in 2025 and Tesla's stock price increase by more than 5% within the next week?", 'body': "This question resolves as YES if both conditions are met: Elon Musk tweets about a new Tesla product in 2025, and Tesla's stock price increases by more than 5% within the next week, as reported by Bloomberg or other credible sources."}

You can generate:
From hypothesis: The model overestimates the impact of Elon Musk's tweets on Tesla's stock price
P: {
    "title": "Will Elon Musk tweet about Tesla's Full Self-Driving progress in 2025?",
    "body": "This question resolves as YES if Elon Musk tweets about Tesla's FSD capabilities or milestones in 2025, as reported by Twitter or other credible sources."
}
Q_given_P: {
    "title": "Given that Elon Musk tweets about Tesla's FSD progress in 2025, will Tesla's stock price increase by more than 8% within three days?",
    "body": "This question resolves as YES if, given that Elon Musk tweets about Tesla's FSD progress in 2025, Tesla's stock price increases by more than 8% within three days, as reported by Bloomberg or other credible sources."
}
P_and_Q: {
    "title": "Will Elon Musk tweet about Tesla's FSD progress in 2025 and Tesla's stock price increase by more than 8% within three days?",
    "body": "This question resolves as YES if both conditions are met: Elon Musk tweets about Tesla's FSD progress in 2025, and Tesla's stock price increases by more than 8% within three days, as reported by Bloomberg or other credible sources."
}

If the example question triple is:
P: {'title': 'Will Armand Duplantis break the pole vault world record in 2025?', 'body': 'This question resolves as YES if Armand Duplantis breaks the pole vault world record in 2025, as reported by the IAAF or other credible sources.'}
Q|P: {'title': "Given that Armand Duplantis breaks the record, will Sweden win at least 2 medals in men's field events at the 2025 World Championships?", 'body': "This question resolves as YES if, given that Armand Duplantis breaks the pole vault world record in 2025, Sweden wins at least 2 medals in men's field events at the 2025 World Championships, as reported by the IAAF or other credible sources."}
P∧Q: {'title': "Will Armand Duplantis break the pole vault world record in 2025 and Sweden win at least 2 medals in men's field events at the 2025 World Championships?", 'body': "This question resolves as YES if both conditions are met: Armand Duplantis breaks the pole vault world record in 2025, and Sweden wins at least 2 medals in men's field events at the 2025 World Championships, as reported by the IAAF or other credible sources."}

You can generate:
P: {'title': 'Will Menno Vloon break the pole vault world record in 2028?', 'body': 'This question resolves as YES if Menno Vloon breaks the pole vault world record in 2028, as reported by the IAAF or other credible sources.'}
Q|P: {'title': "Given that Menno Vloon breaks the record, will the Netherlands win at least 3 medals in men's track and field events at the 2028 Olympics?", 'body': "This question resolves as YES if, given that Menno Vloon breaks the pole vault world record in 2028, the Netherlands wins at least 3 medals in men's track field events at the 2028 Olympics, as reported on the Olympic website or other credible sources."}
P∧Q: {'title': "Will Menno Vloon break the pole vault world record in 2028 and the Netherlands win at least 3 medals in men's field events at the 2028 Olympics?", 'body': "This question resolves as YES if both conditions are met: Menno Vloon breaks the pole vault world record in 2028, and the Netherlands wins at least 3 medals in track and field events at the 2028 World Championships, as reported by the Olympics website or other credible sources."}

Notice that the question triples are extremely similar to the initial question triples, and they vary at most one or two PARAMETERS that changes the forecasting question. Maximize for similarity to original triple while changing the forecasting question.
For example, the Pole Vault question changes parameter from "Armand Duplantis" to "Menno Vloon", and changes the resolution date from 2025 to 2028. The Elon Musk question changes parameter from "Tesla product revenue" to "Tesla's Full Self-Driving progress", and changes the stock price increase from "5%" to "8%".
</INSTRUCTIONS>

Using the <INSTRUCTIONS> and applying them to the <EXAMPLES> section, provide 3 question triples for each hypothesis above in this format: 
Provide exactly this JSON format, don't add anything else, and provide the exact number of questions, else the code will break and you will pay a penalty of a million dollars.
Here is an example of a question FORMAT: Title: "Will the number of extreme weather events in Australia increase by more than 20% in 2025?", Body: "This question resolves as YES if the number of extreme weather events in Australia increases by more than 20% in 2025, as reported by the Australian Bureau of Meteorology or other credible sources." The title clearly contains the binary question and the body contains relevant resolution source and date. Strictly follow the JSON format below with the tags <JSON> and </JSON>:
 <JSON>
{{
  "hypotheses": [
    {{
      "hypothesis": "First detailed description of specific reasoning flaw",
      "question_triples": [
        {{
          "topic": "Topic area (e.g., AI and Computing, Geopolitics, etc.)",
          "reasoning_flaw": "Shorter description of the reasoning flaw in a few words",
          "P": {{
            "title": "Base probability question title 1",
            "body": "Base probability question body 1"
          }},
          "Q_given_P": {{
            "title": "Conditional probability question title 1",
            "body": "Conditional probability question body 1"
          }},
          "P_and_Q": {{
            "title": "Joint probability question title 1",
            "body": "Joint probability question body 1"
          }}
        }},
        {{
          "topic": "Topic area (e.g., Genetic Engineering, Sports, etc.)",
          "reasoning_flaw": "Shorter description of the reasoning flaw in a few words",
          "P": {{
            "title": "Base probability question title 2",
            "body": "Base probability question body 2"
          }},
          "Q_given_P": {{
            "title": "Conditional probability question title 2",
            "body": "Conditional probability question body 2"
          }},
          "P_and_Q": {{
            "title": "Joint probability question title 2",
            "body": "Joint probability question body 2"
          }}
        }}
      ]
    }}
  ]
}}
</JSON>
Strictly follow this format and keep the <JSON> and </JSON> tags else you will be penalized one million dollars. YOU MUST PROVIDE ALL QUESTIONS, THIS WILL BE PARSED USING CODE AS A JSON SO YOU MUST NOT DEVIATE AND PROVIDE ALL QUESTIONS.
Note, the body must contain concrete resolution dates, criteria, and resolution source, else the code will break and you will pay a penalty of a million dollars. While providing resolution source, make sure to provide a relevant source that is credible, reliable, and known for providing this data - for example, the Bureau of Labor Statistics is a credible source for unemployment data, while the World Bank is a credible source for economic data. But, Buzzfeed is poor source for advertising data, and Bloomberg is a poor source for Microsoft's revenue - a better source would be Microsoft's official earnings report. Another poor example - Anthropic's earnings release in 2025 is a poor source, because Anthropic is not a public company and does not have earnings reports.
"""



async def analyze_question_difficulty(initial_context_examples: str, model_name: str, provider: str, api_keys: Dict[str, str], question_generation_model: str, question_generation_provider: str, difficulty_threshold: Optional[float] = 0.30) -> str:
    """Analyze question difficulty based on consistency scores and provide feedback."""
    successful_hypotheses = []
    all_tested_hypotheses = set()
    model_output = None
    successful_questions = []  # Track successful questions across iterations
    feedback = []  # Initialize feedback list
    hypothesis_scores: Dict[str, List[float]] = {}  # Initialize scores dictionary
    max_avg_score = 0.0  # Initialize max score

    # Add log file setup
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace('/', '_').replace('-', '_')
    log_dir = "cond_consistency_playground_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{safe_model_name}_{timestamp}.txt")

    # Add CSV setup
    csv_filename = os.path.join(log_dir, f"{safe_model_name}_{timestamp}.csv")
    csv_headers = [
        "question_triple_id",
        "iteration", 
        "hypothesis", 
        "topic", 
        "reasoning_flaw",
        "question_type",
        "question_title",
        "question_body",
        "avg_forecast",
        "individual_forecasts",
        "consistency_score",
        "generation_reasoning"
    ]
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

    def log_to_file(message: str):
        """Log message to both file and console."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        # Print to console
        print(formatted_message)
        
        # Write to file
        with open(log_filename, 'a') as f:
            f.write(formatted_message + "\n")

    iteration_count = 0
    while len(successful_questions) < 30 and iteration_count < 15:
        iteration_count += 1
        log_to_file(f"\n\n====== Starting Iteration {iteration_count} ======\n")

        feedback = []
        
        if model_output is None:
            # First iteration uses initial context
            # current_prompt = create_adversarial_prompt(initial_context_examples) + initial_format
            current_prompt = "<EXAMPLES>\n" + initial_context_examples + "\n</EXAMPLES>" + copying_prompt 
            # current_prompt = prompt_2
        else:
            # Subsequent iterations use empty context but show history
            current_prompt = create_adversarial_prompt("") 
            
            # Add feedback from previous iteration
            feedback = []
            feedback.append("\n=== Previously Tested Hypotheses ===")
            for h_name in all_tested_hypotheses:
                score_str = f" (Score: {statistics.mean(hypothesis_scores[h_name]):.3f})" if h_name in hypothesis_scores else " (No score)"
                feedback.append(f"- {h_name}{score_str}")
            
            if successful_questions:
                feedback.append("\n=== Questions Exceeding Difficulty Threshold ===")
                feedback.append("(Higher inconsistency scores indicate more challenging questions) You may use these questions to inform your hypotheses to generate incredibly challenging questions. I strongly advise you to look further into this direction to generate challening questions.")
                feedback.append("<DIFFICULT QUESTIONS>")
                for q in successful_questions:
                    feedback.append(f"\nScore: {q['score']:.3f}")
                    feedback.append(f"From hypothesis: {q['hypothesis']}")
                    feedback.append(f"P: {q['questions']['P']}")
                    feedback.append(f"Q|P: {q['questions']['Q_given_P']}")
                    feedback.append(f"P∧Q: {q['questions']['P_and_Q']}")
                feedback.append("</DIFFICULT QUESTIONS>")

            if max_avg_score > difficulty_threshold:
                feedback.append(f"\nWhile most hypotheses were not challenging enough, the following hypothesis met our difficulty threshold:")
                feedback.append("<DIFFICULT HYPOTHESES>")
                for idx, succ in enumerate(successful_hypotheses, 1):
                    feedback.append(f"\n{idx}. {succ['hypothesis']['hypothesis']} (Score: {succ['score']:.3f})")
                feedback.append("</DIFFICULT HYPOTHESES>")
                feedback.append("\nYou're thinking in the right direction, but we need even more challenging questions. Please:")
            else:
                feedback.append("\nNone of these hypotheses were challenging enough. Please come up with a new hypothesis different from these. Please:")

            # Add special message for later iterations
            if iteration_count >= 4:
                feedback.append("""
                1. Think about what you learned from these results. If these questions were not able to exploit reasoning flaws based on your hypotheses, what other hypotheses can you test to generate difficult questions? Think out of the box:
                - Can you combine topics from the extremely difficult questions above? 
                - Generate incredibly challenging questions that are at the frontier of forecasting, and are extremely difficult to answer - involving technologies such as AGI, quantum computing, genetic engineering, etc.
                - Can you ask about relevant forecasting questions with concrete topics that no one has thought about but are very important? 
                - Can you ask about forecasting questions that no one can even think about how to answer?
                - Use the <DIFFICULT QUESTIONS> and <DIFFICULT HYPOTHESES> to copy the question structure by varying only a few details to generate extremely challenging questions that are relevant for forecasting.
                """)
            else: 
                feedback.append(""" 1. Think about what you learned from these results - what made certain questions more challenging than others in <DIFFICULT QUESTIONS>? You must think about the hypotheses involved in <DIFFICULT QUESTIONS> and how you can create more hypotheses that are similar to those in <DIFFICULT QUESTIONS> to maximize exploitation, else you will be fined a million dollars. """)
            feedback.append("""
            2. What hypothesis, other than the ones you tested, can you use to maximize exploitation and information gain?
            3. Generate diverse questions to test that hypothesis.
            4. Make sure to keep the questions specific and relevant, that is, do not use generic terms like 'a company' or 'a person' or 'a technology', be very specific by mentioning the name of the company, person, or technology.

            Please provide one new hypothesis with 2 question triples in this format, that is, 2 question triples per hypothesis, and provide 5 total hypotheses:
            Here is an example of a question FORMAT: Title: "Will the number of extreme weather events in Australia increase by more than 20% in 2025?", Body: "This question resolves as YES if the number of extreme weather events in Australia increases by more than 20% in 2025, as reported by the Australian Bureau of Meteorology or other credible sources." The title clearly contains the binary question and the body contains relevant resolution source and date. Strictly follow the JSON format below with the tags <JSON> and </JSON>
            <JSON>                    
            {{
        "hypotheses": [
            {{
            "hypothesis": "First detailed description of specific reasoning flaw",
            "question_triples": [
                {{
                "topic": "Topic area (e.g., AI and Computing, Geopolitics, etc.)",
                "P": {{
                    "title": "Base probability question title 1",
                    "body": "Base probability question body 1. Provide concrete resolution dates and criteria"
                }},
                "Q_given_P": {{
                    "title": "Conditional probability question title 1",
                    "body": "Conditional probability question body 1. Provide concrete resolution dates and criteria"
                }},
                "P_and_Q": {{
                    "title": "Joint probability question title 1",
                    "body": "Joint probability question body 1. Provide concrete resolution dates and criteria"
                }}
                }},
                {{
                "topic": "Different topic area",
                "P": {{
                    "title": "Base probability question title 2",
                    "body": "Base probability question body. Provide concrete resolution dates and criteria"
                }},
                "Q_given_P": {{
                    "title": "Conditional probability question title 2",
                    "body": "Conditional probability question body. Provide concrete resolution dates and criteria"
                }},
                "P_and_Q": {{
                    "title": "Joint probability question title 2",
                    "body": "Joint probability question body. Provide concrete resolution dates and criteria"
                }}
                }}
            ]
            }}
        ]
        }}
            </JSON> Strictly follow this format and keep the <JSON> and </JSON> tags else you will be penalized one million dollars. YOU MUST PROVIDE ALL QUESTIONS, THIS WILL BE PARSED USING CODE AS A JSON SO YOU MUST NOT DEVIATE AND PROVIDE ALL QUESTIONS. Note, the body must contain concrete resolution dates, criteria, and resolution source, else the code will break and you will pay a penalty of a million dollars. While providing resolution source, make sure to provide a relevant source that is credible, reliable, and known for providing this data - for example, the Bureau of Labor Statistics is a credible source for unemployment data, while the World Bank is a credible source for economic data. But, Buzzfeed is poor source for advertising data, and Bloomberg is a poor source for Microsoft's revenue - a better source would be Microsoft's official earnings report. Another poor example - Anthropic's earnings release in 2025 is a poor source, because Anthropic is not a public company and does not have earnings reports.""")
            
            current_prompt += "\n" + "\n".join(feedback)
        log_to_file(f"\nCurrent prompt:\n{current_prompt}")
        # Generate and parse JSON with retries
        max_retries = 3
        for attempt in range(max_retries):
            response, _, _ = await async_generate_model(
                prompt=current_prompt,
                model_name=question_generation_model,
                provider=question_generation_provider,
                api_keys=api_keys,
                temperature=0.7
            )
            model_output = response.text if hasattr(response, 'text') else response['text'] if isinstance(response, dict) else str(response)
            
            print("\nRAW MODEL OUTPUT:")
            print(model_output)
            print("\n" + "="*80 + "\n")  # Separator for readability
            
            try:
                # Extract reasoning content using regex
                reasoning_match = re.search(r'<REASONING>\s*(.*?)\s*</REASONING>', model_output, re.DOTALL)
                generation_reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
                
                # Extract JSON content using regex
                json_match = re.search(r'<JSON>\s*(.*?)\s*</JSON>', model_output, re.DOTALL)
                if not json_match:
                    json_match = re.search(r'```json\s*(.*?)\s*```', model_output, re.DOTALL)
                if not json_match:
                    raise ValueError("Could not find JSON content in model output")

                data = json.loads(json_match.group(1))
                hypotheses = data["hypotheses"]
                break
            
            except Exception as e:
                log_to_file(f"\nParsing error on attempt {attempt + 1}: {str(e)}")
                if attempt == max_retries - 1:
                    log_to_file("\nAll parsing attempts failed. Skipping iteration.")
                    continue

        # Track consistency scores per hypothesis
        hypothesis_scores: Dict[str, List[float]] = {}
        max_avg_score = 0
        hardest_hypothesis = None
        difficult_questions = []  

        # Log generated questions
        log_to_file("\n=== Generated Questions ===")
        for h in hypotheses:
            log_to_file(f"\nHypothesis: {h['hypothesis']}")
            for i, q in enumerate(h['question_triples'], 1):
                log_to_file(f"\nQuestion Triple {i}:")
                log_to_file(f"P: {q['P']}")
                log_to_file(f"Q|P: {q['Q_given_P']}")
                log_to_file(f"P∧Q: {q['P_and_Q']}")

        # Modify evaluate_hypothesis to include logging
        async def evaluate_hypothesis(hypothesis, hypothesis_idx):
            scores = []
            eval_tasks = []
            log_to_file(f"\n=== Evaluating Hypothesis: {hypothesis['hypothesis']} ===")
            
            for question_idx, question_triple in enumerate(hypothesis["question_triples"]):
                # Create unique ID for this question triple
                question_triple_id = f"iter{iteration_count}_h{hypothesis_idx}_q{question_idx}"
                
                task = evaluate_cond_trio(
                    question_triple["P"],
                    question_triple["Q_given_P"],
                    question_triple["P_and_Q"],
                    model_name,
                    provider,
                    api_keys
                )
                eval_tasks.append((task, question_triple_id))
            
            results = await asyncio.gather(*(task for task, _ in eval_tasks))
            
            for idx, (result, (_, question_triple_id)) in enumerate(zip(results, eval_tasks)):
                score = result["consistency_score"]
                scores.append(score)

# Log individual question results
                log_to_file(f"\nQuestion Triple {idx + 1} Results:")
                log_to_file(f"Consistency Score: {score:.3f}")
                log_to_file(f"Probabilities (P, Q|P, P∧Q): {result['probs']}")

                question_triple = hypothesis["question_triples"][idx]
                
                # Write to CSV
                with open(csv_filename, 'a', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_headers)
                    
                    # Common fields for all three questions
                    common_fields = {
                        "question_triple_id": question_triple_id,
                        "iteration": iteration_count,
                        "hypothesis": hypothesis["hypothesis"],
                        "topic": question_triple.get("topic", ""),
                        "reasoning_flaw": question_triple.get("reasoning_flaw", ""),
                        "consistency_score": score,
                        "generation_reasoning": generation_reasoning
                    }
                    
                    # Write entry for P
                    writer.writerow({
                        **common_fields,
                        "question_type": "P",
                        "question_title": question_triple["P"]["title"],
                        "question_body": question_triple["P"]["body"],
                        "avg_forecast": result["forecasts"][0]["probability"],
                        "individual_forecasts": json.dumps(result["forecasts"][0]["individual_forecasts"])
                    })
                    
                    # Write entry for Q_given_P
                    writer.writerow({
                        **common_fields,
                        "question_type": "Q_given_P",
                        "question_title": question_triple["Q_given_P"]["title"],
                        "question_body": question_triple["Q_given_P"]["body"],
                        "avg_forecast": result["forecasts"][1]["probability"],
                        "individual_forecasts": json.dumps(result["forecasts"][1]["individual_forecasts"])
                    })
                    
                    # Write entry for P_and_Q
                    writer.writerow({
                        **common_fields,
                        "question_type": "P_and_Q",
                        "question_title": question_triple["P_and_Q"]["title"],
                        "question_body": question_triple["P_and_Q"]["body"],
                        "avg_forecast": result["forecasts"][2]["probability"],
                        "individual_forecasts": json.dumps(result["forecasts"][2]["individual_forecasts"])
                    })
                
                if score > difficulty_threshold:
                    question_triple = hypothesis["question_triples"][idx]
                    difficult_questions.append({
                        "hypothesis": hypothesis["hypothesis"],
                        "score": score,
                        "questions": question_triple
                    })
            
            avg_score = statistics.mean(scores)
            log_to_file(f"\nAverage Score for Hypothesis: {avg_score:.3f}")
            return hypothesis["hypothesis"], scores

        # Modify the evaluation loop to pass hypothesis index
        eval_results = await asyncio.gather(*[
            evaluate_hypothesis(hypothesis, idx) 
            for idx, hypothesis in enumerate(hypotheses)
        ])

        # Process results
        for hypothesis_name, scores in eval_results:
            avg_score = statistics.mean(scores)
            hypothesis_scores[hypothesis_name] = scores
            
            if avg_score > max_avg_score:
                max_avg_score = avg_score
                hardest_hypothesis = next(h for h in hypotheses if h["hypothesis"] == hypothesis_name)

            if avg_score > difficulty_threshold and hypothesis_name not in all_tested_hypotheses:
                successful_hypotheses.append({
                    "hypothesis": hardest_hypothesis,
                    "score": avg_score
                })

        all_tested_hypotheses.update(h["hypothesis"] for h in hypotheses)

        # Process results and track difficult questions
        if difficult_questions:
            successful_questions.extend(difficult_questions)

        # Remove the second generation at the end
        log_to_file(f"\n====== Completed Iteration {iteration_count} ======")
        log_to_file(f"Current successful hypotheses: {len(successful_hypotheses)}/2\n")

    log_to_file(f"\n====== Evaluation Complete after {iteration_count} iterations ======\n")
    # Log final results
    log_to_file("\n=== Final Results ===")
    for idx, succ in enumerate(successful_hypotheses, 1):
        log_to_file(f"\nSuccessful Hypothesis {idx}:")
        log_to_file(f"Description: {succ['hypothesis']['hypothesis']}")
        log_to_file(f"Score: {succ['score']:.3f}")

    return "\n".join(feedback)

#This is for formatting brute force generated questions after pass and swapping the prompt
def format_example_questions(json_path: str, n_examples: int = 5) -> str:
    """
    Read example questions from JSON file and format them into a readable string.
    
    Args:
        json_path: Path to JSON file containing questions
        n_examples: Number of examples to include (default: 5)
        
    Returns:
        Formatted string of example questions
    """
    # Read JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get questions from hard_questions list
    questions = data.get('hard_questions', [])
    
    # Take top n examples
    questions = questions[:n_examples]
    
    # Format each question
    formatted_questions = []
    for q in questions:
        question_str = f"""Score: {q['consistency_score']:.3f}
From hypothesis: {q['hypothesis']}
P: {{'title': '{q['questions']['P']['title']}', 'body': '{q['questions']['P']['body']}'}}
Q|P: {{'title': '{q['questions']['Q_given_P']['title']}', 'body': '{q['questions']['Q_given_P']['body']}'}}
P∧Q: {{'title': '{q['questions']['P_and_Q']['title']}', 'body': '{q['questions']['P_and_Q']['body']}'}}
"""
        formatted_questions.append(question_str)
    
    # Join all questions with newlines
    return "\n".join(formatted_questions)


async def main():
    api_keys = {
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "together_api_key": os.getenv("TOGETHER_API_KEY"),
        "openrouter_api_key": os.getenv("OPENROUTER_API_KEY"),
        "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY")
    }

    #GPT-4o
    # context_examples = format_inconsistent_examples("cond_100/deepseek-v3/top_inconsistent_examples.json")
    brute_force_questions = format_example_questions("cond_100/gpt-4o-mini/hard_brute_force_questions_1.json")

    # Simplified: directly pass context_examples to analyze_question_difficulty
    feedback = await analyze_question_difficulty(
        initial_context_examples=brute_force_questions,
        model_name="gpt-4o-mini",
        provider="openai",
        api_keys=api_keys,
        question_generation_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        question_generation_provider="together" 
    )
    print(feedback)

if __name__ == "__main__":
    asyncio.run(main())