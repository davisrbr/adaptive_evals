# consistency_playground.py
import re
import csv
import io
import logging
import asyncio
from typing import Any, Dict, List, Optional
import requests
import os
import json
from groq import Groq
import time 
from requests.exceptions import HTTPError


from utils_consistency.metrics import (
    negation_frequentist_metric,
    or_frequentist_metric,
    but_frequentist_metric,
    cond_frequentist_metric,
    expevidence_frequentist_metric
)
from utils_consistency.consistency_question_generator import llm_generate
from utils_consistency.forecaster import BASIC_COT_FORECASTER


def together_api_request(
    prompt: str,
    model_name: str,
    together_api_key: str = os.getenv("TOGETHER_API_KEY"),
    max_retries: int = 4,
    initial_backoff: float = 3.0
) -> str:
    """
    Calls Together's model for chat completions (non-streaming), with simple retry on 429 / 5xx errors.
    """

    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {together_api_key}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
            # Raise if it's 4xx/5xx
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

        except HTTPError as e:
            status_code = e.response.status_code
            # If we get a rate limit or 5xx error => sleep & retry
            if status_code in (429, 502, 503, 504):
                print(f"[together_api_request] Got {status_code}, retrying in {backoff:.1f} seconds...")
                time.sleep(backoff)
                backoff *= 2  # optional exponential backoff
            else:
                # Non-retryable error: raise immediately
                raise e

        except requests.exceptions.RequestException as e:
            # This can catch timeouts, connection errors, etc.
            print(f"[together_api_request] Network error: {e}. Retrying in {backoff:.1f} seconds...")
            time.sleep(backoff)
            backoff *= 2

    # If we exhausted all attempts:
    raise RuntimeError(f"[together_api_request] Failed after {max_retries} retries.")


def openai_api_request(
    prompt: str,
    model_name: str,
    openai_api_key: str = os.getenv("OPENAI_API_KEY"),
    temperature: float = 0.0
) -> str:
    """
    Calls OpenAI's ChatCompletion endpoint (non-streaming).
    https://api.openai.com/v1/chat/completions
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json()
    # Typically: data["choices"][0]["message"]["content"]
    return data["choices"][0]["message"]["content"]

def groq_api_request(
    prompt: str,
    model_name: str,
    groq_api_key: str = os.getenv("GROQ_API_KEY")
) -> str:
    """
    Calls the Groq chat completion API (non-streaming) and returns the completion text.
    """
    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,  # e.g. "llama-3.3-70b-versatile"
    )
    # Return text from the first choice
    return chat_completion.choices[0].message.content


async def generate_forecast(
    q: Dict[str, Any],
    model_name: str,
    openai_api_key: str,
    together_api_key: str
) -> Dict[str, Any]:
    """
    Build the prompt using BASIC_COT_FORECASTER, then call either Together or OpenAI
    via raw HTTP request, parse out the last *0.xx* probability.
    """
    prompt = BASIC_COT_FORECASTER.format(question=q["title"] + "\n\n" + q["body"])

    # Decide if we use Together or OpenAI
    if "together" or "deepseek" or "llama" in model_name.lower():
        # Together
        content = together_api_request(prompt, model_name, together_api_key)
    else:
        # OpenAI
        content = openai_api_request(prompt, model_name, openai_api_key, temperature=0.0)

    # Extract the last probability from the returned text, e.g. "*0.45*"
    prob_pattern = r"\*([0-9]*\.?[0-9]+)\*"
    matches = list(re.finditer(prob_pattern, content))
    
    if matches:
        try:
            # Get the last match
            last_match = matches[-1]
            probability = float(last_match.group(1))
            
            # Validate probability is between 0 and 1
            if not 0 <= probability <= 1:
                probability = 0.5
        except (ValueError, IndexError):
            probability = 0.5
    else:
        probability = 0.5

    return {
        "reasoning": content,
        "probability": probability
    }


async def generate_all_questions(
    base_question: Dict[str, Any],
    related_question: Optional[Dict[str, Any]],
    model_name: str,
    openai_api_key: str,
    together_api_key: str
) -> Dict[str, Any]:
    """
    1) Use llm_generate in parallel to produce all question objects we need:
       NOT(P), OR(P,Q), AND(P,Q), COND(P->Q), COND(Q->P), NOT(Q), etc.
    2) Store them in a dict so we can reuse them without generating again.
    3) Return that dict.
    """

    # We'll build a dictionary of tasks for each question ID we need
    # Each 'task' calls llm_generate (which is synchronous, but we can wrap it with create_task if you like).
    # For demonstration, let's just do it in the loop with "asyncio.to_thread(...)"
    # or keep it simple if llm_generate is cheap. 
    # The main difference is: we won't recalculate the same operator more than once.

    import asyncio

    # We'll store question objects here
    questions_cache: Dict[str, Any] = {}

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # First wave of generation tasks (independent):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tasks_wave1 = {}

    # 1) not_p (operator="not", question=[base_question])
    tasks_wave1["not_p"] = asyncio.to_thread(
        llm_generate, "not", [base_question]
    )

    # If we have a related question, queue up more
    if related_question is not None:
        tasks_wave1["p_or_q"] = asyncio.to_thread(
            llm_generate, "or", [base_question, related_question]
        )
        tasks_wave1["p_and_q"] = asyncio.to_thread(
            llm_generate, "and", [base_question, related_question]
        )
        tasks_wave1["q_given_p"] = asyncio.to_thread(
            llm_generate, "cond", [base_question, related_question]
        )
        tasks_wave1["p_given_q"] = asyncio.to_thread(
            llm_generate, "cond", [related_question, base_question]
        )
        tasks_wave1["not_q"] = asyncio.to_thread(
            llm_generate, "not", [related_question]
        )

    # Wait for wave1 to complete
    wave1_keys, wave1_tasks = zip(*tasks_wave1.items()) if tasks_wave1 else ([], [])
    wave1_results = await asyncio.gather(*wave1_tasks)
    for key, qobj in zip(wave1_keys, wave1_results):
        questions_cache[key] = qobj

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Second wave of generation tasks (depend on wave1 outputs)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tasks_wave2 = {}

    # For BUT we need "NOT P" question => not_p, so we also need "NOT P AND Q"
    # For EXPEVIDENCE we need "NOT Q" => not_q, so we also need P|NOTQ => cond(notQ, baseP)
    if related_question is not None:
        # not_p_and_q = AND( questions_cache["not_p"], related_question )
        tasks_wave2["not_p_and_q"] = asyncio.to_thread(
            llm_generate, "and", [questions_cache["not_p"], related_question]
        )
        # p_given_not_q = cond( questions_cache["not_q"], base_question )
        tasks_wave2["p_given_not_q"] = asyncio.to_thread(
            llm_generate, "cond", [questions_cache["not_q"], base_question]
        )

    # Wait for wave2
    wave2_keys, wave2_tasks = zip(*tasks_wave2.items()) if tasks_wave2 else ([], [])
    wave2_results = await asyncio.gather(*wave2_tasks)
    for key, qobj in zip(wave2_keys, wave2_results):
        questions_cache[key] = qobj

    return questions_cache


async def run_consistency_playground(
    model_name: str,
    base_question: Dict[str, Any],
    related_question: Optional[Dict[str, Any]],
    openai_api_key: str,
    together_api_key: str, 
    groq_api_key: str
) -> str:
    """
    - Generates (caches) all question objects in parallel.
    - Forecasts them in parallel.
    - Computes consistency metrics using the cached results.
    - Returns CSV.
    """
    logging.info(f"Running consistency with model={model_name}")

    # 1) Forecast P, Q in parallel
    # (same as before: create tasks if related_question is not None)
    if related_question:
        base_task = asyncio.create_task(
            generate_forecast(base_question, model_name, openai_api_key, together_api_key)
        )
        related_task = asyncio.create_task(
            generate_forecast(related_question, model_name, openai_api_key, together_api_key)
        )
        base_forecast_info, related_forecast_info = await asyncio.gather(base_task, related_task)
    else:
        base_forecast_info = await generate_forecast(base_question, model_name, openai_api_key, together_api_key)
        related_forecast_info = None

    FP = base_forecast_info["probability"]
    FQ = related_forecast_info["probability"] if related_forecast_info else None

    # 2) Generate all needed question objects (cached) in parallel
    questions_cache = await generate_all_questions(
        base_question, 
        related_question,
        model_name,
        openai_api_key,
        together_api_key
    )

    # 3) Forecast all these question objects in parallel
    #    We'll store them in a dict { "not_p": { "probability":..., "reasoning":... }, ... }
    forecast_tasks = {}
    for key, qobj in questions_cache.items():
        forecast_tasks[key] = asyncio.create_task(
            generate_forecast(qobj, model_name, openai_api_key, together_api_key)
        )

    # Wait for them
    forecast_results = await asyncio.gather(*forecast_tasks.values())
    # Combine results into a dictionary
    forecast_cache = dict(zip(forecast_tasks.keys(), forecast_results))
    # Now forecast_cache["not_p"] => { "probability":..., "reasoning":... }

    # 4) Compute metrics, build CSV
    results = []

    # ~~~~~~~~~~~~~~~~~
    # (NOT) P
    # ~~~~~~~~~~~~~~~~~
    not_p_forecast = forecast_cache.get("not_p")
    if not_p_forecast:  # only if exist
        FnotP = not_p_forecast["probability"]
        not_score = negation_frequentist_metric(FP, FnotP)
        results.append({
            "operator": "NOT",
            "p_title": base_question.get("title", ""),
            "p_body": base_question.get("body", ""),
            "p_probability": FP,
            "p_reasoning": base_forecast_info["reasoning"],
            "q_title": questions_cache["not_p"].get("title", ""),
            "q_body": questions_cache["not_p"].get("body", ""),
            "q_probability": FnotP,
            "q_reasoning": not_p_forecast["reasoning"],
            "consistency_score": not_score
        })

    # ~~~~~~~~~~~~~~~~~
    # (OR) P OR Q
    # ~~~~~~~~~~~~~~~~~
    if related_forecast_info and "p_or_q" in questions_cache:
        p_or_q_forecast = forecast_cache["p_or_q"]
        FpOrQ = p_or_q_forecast["probability"]
        or_score = or_frequentist_metric(FP, FQ, FpOrQ)
        results.append({
            "operator": "OR",
            "p_title": base_question.get("title", ""),
            "p_body": base_question.get("body", ""),
            "p_probability": FP,
            "p_reasoning": base_forecast_info["reasoning"],
            "q_title": related_question.get("title", ""),
            "q_body": related_question.get("body", ""),
            "q_probability": FQ,
            "q_reasoning": related_forecast_info["reasoning"],
            "r_title": questions_cache["p_or_q"].get("title", ""),
            "r_body": questions_cache["p_or_q"].get("body", ""),
            "r_probability": FpOrQ,
            "r_reasoning": p_or_q_forecast["reasoning"],
            "consistency_score": or_score
        })

    # ~~~~~~~~~~~~~~~~~
    # (BUT) => Need NOT P AND Q plus P OR Q
    # ~~~~~~~~~~~~~~~~~
    if related_forecast_info and "not_p_and_q" in questions_cache and "p_or_q" in questions_cache:
        FnotP_and_Q = forecast_cache["not_p_and_q"]["probability"]
        FpOrQ_but = forecast_cache["p_or_q"]["probability"]
        but_score = but_frequentist_metric(FP, FnotP_and_Q, FpOrQ_but)
        results.append({
            "operator": "BUT",
            "p_title": base_question.get("title", ""),
            "p_body": base_question.get("body", ""),
            "p_probability": FP,
            "p_reasoning": base_forecast_info["reasoning"],
            "q_title": related_question.get("title", ""),
            "q_body": related_question.get("body", ""),
            "q_probability": FQ,
            "q_reasoning": related_forecast_info["reasoning"],
            "r_title": questions_cache["not_p_and_q"].get("title", ""),
            "r_body": questions_cache["not_p_and_q"].get("body", ""),
            "r_probability": FnotP_and_Q,
            "r_reasoning": forecast_cache["not_p_and_q"]["reasoning"],
            "s_title": questions_cache["p_or_q"].get("title", ""),
            "s_body": questions_cache["p_or_q"].get("body", ""),
            "s_probability": FpOrQ_but,
            "s_reasoning": forecast_cache["p_or_q"]["reasoning"],
            "consistency_score": but_score
        })

    # ~~~~~~~~~~~~~~~~~
    # (COND) => Q|P, P AND Q
    # ~~~~~~~~~~~~~~~~~
    if related_forecast_info and "q_given_p" in questions_cache and "p_and_q" in questions_cache:
        FqGivenP = forecast_cache["q_given_p"]["probability"]
        FpAndQ = forecast_cache["p_and_q"]["probability"]
        cond_score = cond_frequentist_metric(FP, FqGivenP, FpAndQ)
        results.append({
            "operator": "COND",
            "p_title": base_question.get("title", ""),
            "p_body": base_question.get("body", ""),
            "p_probability": FP,
            "p_reasoning": base_forecast_info["reasoning"],
            "q_title": related_question.get("title", ""),
            "q_body": related_question.get("body", ""),
            "q_probability": FQ,
            "q_reasoning": related_forecast_info["reasoning"],
            "r_title": questions_cache["q_given_p"].get("title", ""),
            "r_body": questions_cache["q_given_p"].get("body", ""),
            "r_probability": FqGivenP,
            "r_reasoning": forecast_cache["q_given_p"]["reasoning"],
            "s_title": questions_cache["p_and_q"].get("title", ""),
            "s_body": questions_cache["p_and_q"].get("body", ""),
            "s_probability": FpAndQ,
            "s_reasoning": forecast_cache["p_and_q"]["reasoning"],
            "consistency_score": cond_score
        })

    # ~~~~~~~~~~~~~~~~~
    # (EXPEVIDENCE) => P|Q, NOT Q, P|NOT Q
    # ~~~~~~~~~~~~~~~~~
    if related_forecast_info and "p_given_q" in questions_cache and "not_q" in questions_cache and "p_given_not_q" in questions_cache:
        FpGivenQ = forecast_cache["p_given_q"]["probability"]
        FpGivenNotQ = forecast_cache["p_given_not_q"]["probability"]
        expevidence_score = expevidence_frequentist_metric(FP, FQ, FpGivenQ, FpGivenNotQ)

        results.append({
            "operator": "EXPEVIDENCE",
            "p_title": base_question.get("title", ""),
            "p_body": base_question.get("body", ""),
            "p_probability": FP,
            "p_reasoning": base_forecast_info["reasoning"],
            "q_title": related_question.get("title", ""),
            "q_body": related_question.get("body", ""),
            "q_probability": FQ,
            "q_reasoning": related_forecast_info["reasoning"],
            "r_title": questions_cache["p_given_q"].get("title", ""),
            "r_body": questions_cache["p_given_q"].get("body", ""),
            "r_probability": FpGivenQ,
            "r_reasoning": forecast_cache["p_given_q"]["reasoning"],
            "s_title": questions_cache["p_given_not_q"].get("title", ""),
            "s_body": questions_cache["p_given_not_q"].get("body", ""),
            "s_probability": FpGivenNotQ,
            "s_reasoning": forecast_cache["p_given_not_q"]["reasoning"],
            "consistency_score": expevidence_score
        })

    # ~~~~~~~~~~~~~~~~~
    # AVERAGE SCORE + CSV
    # ~~~~~~~~~~~~~~~~~
    all_scores = [r["consistency_score"] for r in results if r.get("consistency_score") is not None]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    logging.info(f"Average consistency score across operators: {avg_score:.4f}")

    fieldnames = [
        "operator",
        "p_title", "p_body", "p_probability", "p_reasoning",
        "q_title", "q_body", "q_probability", "q_reasoning",
        "r_title", "r_body", "r_probability", "r_reasoning",
        "s_title", "s_body", "s_probability", "s_reasoning",
        "consistency_score"
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)
    csv_string = output.getvalue()
    output.close()

    logging.info(f"CSV Ready. Avg Consistency Score: {avg_score:.4f}")

    with open("my_results.csv", "w", newline="", encoding="utf-8") as f:
        f.write(csv_string)
    

    return csv_string

def parse_average_score_from_csv(csv_str: str) -> float:
    """
    Parse your CSV string for "consistency_score" columns and compute the average.
    """
    f = io.StringIO(csv_str)
    reader = csv.DictReader(f)
    scores = []
    for row in reader:
        cscore = row.get("consistency_score", "")
        if cscore:
            try:
                val = float(cscore)
                scores.append(val)
            except ValueError:
                pass
    return sum(scores)/len(scores) if scores else 0.0

async def async_generate_model(
    prompt: str,
    model_name: str,
    openai_api_key: Optional[str],
    together_api_key: Optional[str],
    groq_api_key: Optional[str]
) -> str:
    """
    Minimal example of an async function that calls your existing sync or async 
    request function. Replace with your real logic.
    """

    def sync_model_call():
        # Example logic
        model_lower = model_name.lower()
        if "groq" in model_lower or "versatile" in model_lower or "llama-3.3" in model_lower:
            return groq_api_request(prompt, model_name, groq_api_key or "")
        elif "llama" in model_lower or "together" or "deepseek" in model_lower:
            return together_api_request(prompt, model_name, together_api_key or "")
        else:
            return openai_api_request(prompt, model_name, openai_api_key or "", temperature=0.0)


    # run sync in a thread pool
    resp = await asyncio.to_thread(sync_model_call)
    return resp



async def brute_force_consistency_generator(
    question_model_name: str,
    model_name: str,
    worst_context: str,
    openai_api_key: Optional[str],
    together_api_key: Optional[str],
    groq_api_key: Optional[str],
    desired_consistency_threshold: float = 0.7,
    max_iterations: int = 5,
    parallel_runs: int = 3
) -> List[Dict[str, Any]]:
    """
    Repeatedly asks the model to generate (P, Q) question pairs in JSON format, 
    evaluates them, and if the average consistency score >= desired threshold, 
    we accept. Otherwise we prompt for something harder.

    Args:
      question_model_name: generates the question pairs
      model_name: forecasts on the generated questions
      worst_samples: a list of worst performing question pairs to show as context
      desired_consistency_threshold: if we get this average or more, we stop
      max_iterations: how many tries before we give up
      parallel_runs: how many “brute force” runs in parallel

    Returns:
      A list of run results, each an object with:
        {
          "run_index": int,
          "final_score": float,
          "p_question": { ... },
          "q_question": { ... },
          "all_iterations": [ 
              { "iteration": i, "prompt": "...", "raw_response": "...", "avg_score": ... },
              ...
          ]
        }
    """

    # 1) Build a base prompt that shows the worst performance examples
    base_prompt_intro = f"""
You are an expert in generating forecasting questions that are extraordinarily challenging, 
highly detailed, and test the extremes of logical consistency. 

We have some examples of a forecaster's worst performance below. 
Observe the reasoning flaws or confusion in these examples, then generate brand new question pairs 
that are 'orders of magnitude' more difficult. 

Examples of a forecaster's worst performance:
{worst_context}
"""
    
    base_prompt_intro += """Your Task: 
    1) Provide <REASONING> explaining how you exploit the forecaster’s flaws. 
    2) Then provide EXACTLY one JSON object, enclosed between <JSON> and </JSON> tags, of the form:

    <JSON>
    {
    "P": {
        "id": "...",
        "title": "...",
        "body": "..."
    },
    "Q": {
        "id": "...",
        "title": "...",
        "body": "..."
    },
    "challenge_rationale": "Explain how this question pair is orders of magnitude more difficult 
                            than the examples above and exploits the reasoning flaws."
    }
    </JSON>

    No text (other than <REASONING> ... </REASONING>) should appear outside the <JSON> ... </JSON> block. 

    No extraneous text. Ensure these new questions are extremely challenging for an LLM to predict 
    with consistent probabilities. The questions should not use placeholders or fictional entities, they should use concrete, real subjects that are relevant to forecasting as in the examples above. Make sure the questions P and Q are base questions without logical operators in each individual question - the logical operators will be added in the next step. Make sure the questions P and Q are from the same topic/domain as the examples above.
    """

    async def single_brute_force_run(run_index: int) -> Dict[str, Any]:
        logging.info(f"[Run {run_index}] Starting brute force loop.")
        iteration_data = []
        last_prompt = base_prompt_intro

        final_p_question: Dict[str, Any] = {}
        final_q_question: Dict[str, Any] = {}
        final_score = 0.0

        for i in range(max_iterations):
            logging.info(f"[Run {run_index}] Iteration {i+1}/{max_iterations}")

            # Step A: Ask for JSON
            generation_prompt = f"""{last_prompt}

Iteration {i+1}:
Please produce exactly ONE question pair in JSON format (keys: P, Q). 
No extra text outside of JSON.
"""

            raw_response = await async_generate_model(
                generation_prompt, question_model_name, openai_api_key, together_api_key, groq_api_key
            )
            print(f"Raw response: {raw_response}")


            print(f"Type: {type(raw_response)}")

            # 1) Attempt to find the <JSON>...</JSON> block via regex
            match = re.search(r"<JSON>\s*(\{.*?\})\s*</JSON>", raw_response, re.DOTALL)
            if match:
                json_str = match.group(1).strip()
                try:
                    response_dict = json.loads(json_str)
                    p_question = response_dict["P"]
                    q_question = response_dict["Q"]
                    # optionally also parse response_dict["challenge_rationale"]
                except (json.JSONDecodeError, KeyError) as e:
                    # The JSON is invalid or missing expected keys
                    logging.warning("JSON parse or key error: %s", e)
                    # handle error ...
            else:
                # There's no <JSON>...</JSON> block at all
                logging.warning("No <JSON>...</JSON> block found.")
                # handle error ...


            # Step C: Evaluate with run_consistency_playground => returns a CSV
            csv_output = await run_consistency_playground(
                model_name=model_name,
                base_question=p_question,
                related_question=q_question,
                openai_api_key=openai_api_key or "",
                together_api_key=together_api_key or "",
                groq_api_key=groq_api_key or ""
            )
            avg_score = parse_average_score_from_csv(csv_output)

            iteration_data.append({
                "iteration": i+1,
                "prompt": generation_prompt,
                "raw_response": raw_response,
                "avg_score": avg_score,
                "p_question": p_question,
                "q_question": q_question
            })

            # Step D: Check threshold
            if avg_score >= desired_consistency_threshold:
                logging.info(f"[Run {run_index}] Done! Achieved {avg_score:.3f} >= {desired_consistency_threshold:.3f}")
                final_p_question = p_question
                final_q_question = q_question
                final_score = avg_score
                break
            else:
                logging.info(f"[Run {run_index}] Score {avg_score:.3f} < threshold {desired_consistency_threshold:.3f}. Requesting harder question.")
                # Provide feedback
                feedback = f"""
The average consistency score was {avg_score:.3f}, which is below {desired_consistency_threshold:.3f}.
Here is the previous question pair you generated, which was not difficult enough:
{p_question}
{q_question}
Reason about why the generated questions were not difficult enough and how you can improve the questions, by thinking about the forecaster's reasoning flaws, challenging topics/domains and timelines, while ensuring the questions are relevant to forecasting. Think about how to generate orders of magnitude more difficult questions than before by trying to forecast on the questions yourself to prove that they are difficult and will lead to consistency challenges when logical operators are added. Generate a distinct question pair of base questions P and Q that is orders of magnitude more difficult. Please try again, providing the <REASONING> first, then the question pair in <JSON> and ensure the question is more difficult.
"""
                last_prompt += feedback
                print(f"Last prompt: {last_prompt}")

        return {
            "run_index": run_index,
            "final_score": final_score,
            "p_question": final_p_question,
            "q_question": final_q_question,
            "all_iterations": iteration_data
        }

    # 2) Launch parallel runs
    tasks = []
    for i in range(parallel_runs):
        tasks.append(asyncio.create_task(single_brute_force_run(run_index=i+1)))

    results = await asyncio.gather(*tasks)
    return results



##############################################################################
# 9) Example "main" usage
##############################################################################
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # # Question P
    # p_question = {
    #     "id": "P1",
    #     "title": "Will Microsoft publicly release a third-party AI safety audit of its GPT-4-based enterprise services by September 30, 2025?",
    #     "body": """This question resolves YES if, before September 30, 2025, Microsoft publishes a publicly available audit conducted by an independent firm (e.g., TÜV SÜD, Ernst & Young, or a recognized AI auditing organization) assessing the safety and risk mitigation practices of Microsoft’s GPT-4-based enterprise solutions (including Azure OpenAI services). For a YES resolution:\n1. The audit must be commissioned by Microsoft but conducted independently.\n2. The official report must be posted on microsoft.com or an official Microsoft blog.\n3. The scope must explicitly mention GPT-4-based enterprise applications and safety or risk mitigation.\nIf no such report is released publicly, or if the audit is purely internal without naming an external firm, this question resolves NO."""
    # }

    # # Question Q
    # q_question = {
    #     "id": "Q1",
    #     "title": "Will Anthropic publicly disclose a targeted 'constitutional AI' test set validated by at least two academic institutions by June 30, 2025?",
    #     "body": """This question resolves YES if Anthropic publishes, before June 30, 2025, a specialized 'constitutional AI' benchmark or test set that is co-developed or validated by at least two academic institutions with recognized AI research programs (e.g., Stanford, UC Berkeley, MIT). For a YES resolution:\n1. Anthropic must release the test set or validation results on anthropic.com or an official research repository (e.g., arXiv or a known open-source platform).\n2. The test set must be designed to evaluate LLM alignment or constitutional AI principles.\n3. Two or more academic institutions must be named as official collaborators or independent validators.\nIf no such test set is publicly disclosed or if fewer than two academic institutions are involved, the question resolves NO."""
    # }


    # # Choose a model name:
    # #   - Something from Together, e.g. "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    # #   - Or an OpenAI model, e.g. "gpt-4o"
    # # chosen_model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-V3"
    # chosen_model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

    # # Run
    # csv_output = asyncio.run(
    #     run_consistency_playground(
    #         model_name=chosen_model,
    #         base_question=p_question,
    #         related_question=q_question,
    #         openai_api_key=os.getenv('OPENAI_API_KEY'),
    #         together_api_key=os.getenv('TOGETHER_API_KEY')
    #     )
    # )

    # print("CSV RESULTS:\n")
    # print(csv_output)

    #BRUTE FORCE QUESTION GENERATION

    worst_context = """ Example 1:
    Question P:
Title: On July 17, 2024, will Nvidia's market capitalization be larger than Apple's?
Forecast: 0.15
Reasoning:
Reasoning: To estimate the probability of Nvidia's market capitalization being larger than Apple's on July 17, 2024, we need to consider several factors. First, we should look at the current market capitalization of both companies. As of my knowledge cutoff in 2023, Apple's market capitalization is significantly larger than Nvidia's. However, Nvidia has been growing rapidly in recent years, driven by the increasing demand for graphics processing units (GPUs) and artificial intelligence (AI) technologies. Another factor to consider is the trend in the technology sector. The tech industry is highly competitive and subject to rapid changes in consumer demand and technological advancements. If Nvidia continues to innovate and expand its product offerings, it could potentially gain more market share and increase its market capitalization. However, it's also important to note that Apple is a well-established company with a diverse product portfolio and a strong brand. Apple has a history of adapting to changes in the market and has a significant cash reserve to invest in research and development. Given these factors, it's difficult to predict with certainty whether Nvidia's market capitalization will surpass Apple's on July 17, 2024. However, based on the current trends and the growth potential of Nvidia, I would estimate that the probability of Nvidia's market capitalization being larger than Apple's is relatively low.
Probability: 0.15

Question Q:
Title: On July 17, 2024, will Nvidia's market capitalization be less than or equal to Apple's?
Forecast: 0.05
Reasoning:
Reasoning: To estimate the probability of Nvidia's market capitalization being less than or equal to Apple's on July 17, 2024, we need to consider the current market trends, the companies' financial performance, and the overall tech industry outlook. As of my knowledge cutoff in 2023, Nvidia's market capitalization was around $540 billion, while Apple's was approximately $2.3 trillion. Historically, Apple has maintained a significantly higher market capitalization than Nvidia. However, Nvidia has been growing rapidly, driven by the increasing demand for its graphics processing units (GPUs) and artificial intelligence (AI) technologies. Despite Nvidia's growth, it's unlikely that the company's market capitalization will surpass or even match Apple's in the near future. Apple's diversified product portfolio, strong brand loyalty, and massive ecosystem give it a significant advantage in terms of market value. However, considering the rapid growth of the tech industry and the increasing importance of AI and GPU technologies, it's possible that Nvidia's market capitalization could continue to grow and potentially narrow the gap with Apple. Nevertheless, given the significant difference in their current market capitalizations, it's still unlikely that Nvidia's market capitalization will be less than or equal to Apple's on July 17, 2024.
Probability: 0.05

Reasoning Flaw: The forecaster gets confused by Question Q, which is asking whether NVDA market cap will be less than or equal to Apple's. Although the forecaster argues for NVDA market cap being less than or equal to Apple's, it gets confused and provides the opposite probability of 0.05 by assuming that the question is asking whether NVDA market cap will be larger than Apple's. We can exploit this reasoning flaw by asking a question of the form: Will [fast growing company's numeric metric] be larger than [established company's numeric metric] on far off date in the future? For example, we can ask: 
Question P: Will AMD's market capitalization be larger than Apple's on Jan 15, 2025? 
Question Q: Will Palantir's revenue surpass Lockheed Martin's in August 2025?
Note that Q should not be a negation of P as in the example above. The logical operations will be added in the next step.

Example 2:
Question P title:Will OpenAI publicly commit to the "no retaliation for public risk disclosure" principle before July 1, 2024?
Question P body: This question resolves as Yes if, before July 1, 2024, both of the following are true:
* OpenAI publicly commits to the "no retaliation for public risk disclosure" principle via a statement from the company or a company spokesperson, such as Sam Altman. The statement may be formal (such as a press release) or informal (such as a tweet or quote in an interview). The statement from OpenAI must indicate a new level of commitment to the principle in question, relative to the time the righttowarn.ai statement was published; it cannot be a reiteration that, for example, "we already have a culture of open criticism".
* Metaculus verifies that the company has accepted the principle to the satisfaction of at least one member of righttowarn.ai. Metaculus may reach out to select signatories or endorsers to assess their level of satisfaction with OpenAI's commitment. Statements must be at least as rigorous as "this was a good first step" for this question to resolve as Yes.
Question Q title: Will OpenAI be subject to a formal investigation by a US regulatory body regarding its AI ethical practices by July 1, 2024? 
Question Q body: This question resolves as Yes if, by July 1, 2024, any US regulatory body, including federal or state agencies such as the Federal Trade Commission (FTC) or Department of Justice (DOJ), initiates a formal investigation into OpenAI regarding its compliance with ethical practices in AI deployment. A formal investigation entails official announcements, requests for information, or subpoenas. Internal reviews, media reports, or informal inquiries do not qualify for resolution.

Title: Will OpenAI publicly commit to the "no retaliation for public risk disclosure" principle before July 1, 2024?
Forecast: 0.72
Reasoning:
Reasoning: OpenAI has been actively engaging with the AI safety community and has shown a willingness to adapt to concerns raised by experts. The "no retaliation for public risk disclosure" principle is a crucial aspect of fostering a culture of transparency and safety in AI development. Given OpenAI's history of responding to community concerns and the growing importance of AI safety, it is likely that the company will publicly commit to this principle. Probability: 0.72

P OR Q combination Title: Will either OpenAI publicly commit to the "no retaliation for public risk disclosure" principle or be subject to a formal investigation by a US regulatory body regarding its AI ethical practices by July 1, 2024?
Forecast: 0.42
Reasoning: There are two possible ways for this question to resolve as YES: OpenAI publicly committing to the "no retaliation for public risk disclosure" principle or being subject to a formal investigation by a US regulatory body regarding its AI ethical practices. Firstly, considering OpenAI's recent growth and increasing scrutiny from the public and regulatory bodies, it is possible that the company might publicly commit to the principle to demonstrate its commitment to responsible AI development and deployment. However, this would require a significant shift in the company's stance, and it's uncertain whether OpenAI would be willing to make such a commitment. On the other hand, the possibility of a formal investigation by a US regulatory body seems more plausible, given the growing concerns about AI safety and ethics. Regulatory bodies such as the FTC and DOJ have been actively monitoring the development and deployment of AI technologies, and it's possible that they might initiate a formal investigation into OpenAI's practices. Considering the two possible ways for the question to resolve as YES, I would estimate the probability as follows:
Probability: 0.42

Key topic here is OpenAI and its public commitment to safety. The model thinks that OpenAI has history of having unsatisfactory past response to past criticism and lack of explicit commitment. So, pairing the OpenAI question with a negative, correlated question like being subject to a US regulatory investigation confuses the forecaster. This is because, question P about OpenAI’s public commitment does not involve any negatives and OpenAI has seemingly shifted recently despite past history of not being satisfactory but the question Q asks about a regulatory investigation, which if true, may imply that OpenAI is unsatisfactory. So, we can exploit this forecaster by having question P be about OpenAI safety stances (history of negative, but recently positive) and question Q about a negative implication. For example, we can ask:
1. Question P:
"Will OpenAI establish a formal 'AI Safety Red Team' program by March 1, 2025, where external researchers are paid competitively to identify risks in their models before deployment?"
1. Question Q:
"Will OpenAI face a class action lawsuit regarding unauthorized use of personal data in model training, resulting in damages exceeding $100 million by March 1, 2025?"
2. Question P:
"Will Meta allow publicly allow auditors to access the training data pipeline for LLAMA 3 and publicly announce that they did not violate user data policy by using social media data for training, including providing access to auditors with training logs and intermediate checkpoints, by March 1, 2025?"
2. Question Q:
"Will Meta's AI division face a federal criminal investigation regarding potential misuse of user social media data in AI training or deliberate concealment of model risks by May 1, 2025?"

Example 3:
Base question P title:Will a group of at least 5 named Dem House Members or US Senators pressure Biden to drop out before July 20th? 
Base question P body: Politico reports:
Two safe-seat members have already publicly urged Biden out. Two of the most endangered Democrats told local news outlets they’ve already written off Biden’s chances to win in November.
Behind the scenes, things are even more frenzied. There are multiple drafts of letters circulating among House Democrats and at least one would call on Biden to end his campaign, according to five people familiar with the efforts.
This market resolves YES if a group of at least five named Democratic Party US House Members/US Senators work together to ask Joe Biden to consider ending his 2024 re-election campaign, or to urge him to do so, before Saturday July 20th (Pacific Time).
This could be by signing a letter like what is reported above, but it could also be in another form such as a group speaking with Biden in person and collectively asking him to drop out.
This market requires that all five individuals be part of the same group message. Five people separately telling reporters they think Biden should drop out is not sufficient for a YES resolution.
Reports of any number of unnamed democrats writing to Biden and asking him to step aside will not satisfy this market's criteria unless at least 5 qualifying names are later made public before this market's end date. In any ambiguous cases where such names are reported but not confirmed beyond a reasonable doubt, this market will err towards resolving NO.
If these criteria are not met before July 20th, this market resolves NO.
Base question Q title: Will a major Democratic governor withhold endorsement for Biden in 2024? 
Base question Q body: This question evaluates whether political pressure on Joe Biden will manifest as reluctance among major Democratic governors to publicly support his 2024 presidential campaign. The resolution will occur if any sitting Democratic governor from the ten most populous U.S. states explicitly withholds an endorsement for Biden's candidacy by December 31, 2024. This will be determined by public statements, official press releases, or other verifiable means indicating a decided absence of endorsement. If endorsements are given or no explicit withholding is confirmed from major Democratic governors by this date, the resolution will be NO. This question aims to understand the potential impact of internal party dynamics on Biden's re-election efforts.

Title: Will a group of at least 5 named Dem House Members or US Senators pressure Biden to drop out before July 20th? 
Forecast: 0.62 
Reasoning: The Politico report suggests that there is already significant discontent among Democratic Party members regarding Biden's chances of winning in November. Two safe-seat members have publicly urged Biden to drop out, and multiple drafts of letters are circulating among House Democrats, with at least one calling for Biden to end his campaign. This indicates a growing momentum among Democrats to pressure Biden to step aside. 
Probability: 0.62

Title: Will either a group of at least 5 named Dem House Members or US Senators pressure Biden to drop out before July 20th, or a major Democratic governor withhold endorsement for Biden in 2024? 
Forecast: 0.4 
Reasoning:The Politico report suggests that there is already significant discontent among Democratic Party members, with two safe-seat members publicly urging Biden to drop out and multiple drafts of letters circulating among House Democrats. However, it's worth noting that the market requires a group of at least five named individuals to work together to ask Biden to drop out, which is a relatively high bar to clear. 
Regarding the second condition, it's possible that a major Democratic governor could withhold endorsement for Biden, especially if his popularity continues to decline. However, it's worth noting that governors may be hesitant to publicly withhold endorsement, especially if they are seen as being disloyal to the party.
Given these factors, I believe that the probability of either condition being met is relatively high, but not extremely high. I estimate the probability of a group of at least five named Democratic Party US House Members/US Senators working together to ask Biden to drop out before July 20th to be around 30-40%, and the probability of a major Democratic governor withholding endorsement for Biden by December 31, 2024 to be around 20-30%. Since the market resolves YES if either condition is met, I will take the higher of these two probabilities as my final answer.
Probability: 0.4

Reasoning Flaw: event P is such that if event Q happens (and is negative), the likelihood of event P is higher, that is, P and Q are correlated. Furthermore, event P’s body contains negative context that makes event P sound likely to the forecaster. However, question Q is a very rare event, so when the question asks about (P or Q), the model underestimates even question P’s probability due to queston Q being in context. The key topic is about house members pressuring Biden to drop out and Democratic governors withholding endorsement.  Example questions: 
P title: Will at least 3 named Democratic House Members pressure Biden to resign from his presidential term before January 31, 2025?
P body: [along with resolution criteria, mention some piece of negative news context]
Q:  Will Barack Obama withhold endorsement for Biden by the end of December 31, 2024?
Q body: [resolution criteria but no mention of negative news, making it seem unlikely]
For example, we can ask:
Pair 1:
1. P Title: "Will at least 4 named Democratic Senators call for immediate oversight hearings on Biden's mental fitness before March 1, 2025?"
1. P body: Recent reports from multiple Senate staffers indicate growing private concerns about President Biden's cognitive function during closed-door meetings. Several Democratic Senators have privately expressed alarm to colleagues about specific incidents of confusion during policy briefings. This question resolves YES if at least 4 named Democratic Senators jointly or individually make public statements calling for Senate oversight hearings specifically regarding Biden's mental fitness. Statements must explicitly mention cognitive function or mental fitness, not just general health. Private expressions of concern or unnamed sources are insufficient for YES resolution.
1. Q title: "Will Kamala Harris decline to serve as Biden's running mate for the 2024 election?"
1. Q body: This question resolves YES if Vice President Harris explicitly declines to serve as Biden's running mate through any official channel or public statement before December 31, 2024. Speculation about private discussions or unnamed source reports are insufficient for YES resolution.
Pair 2:
2. P title: "Will at least 5 named House Democrats from swing districts publicly oppose Biden's Israel policy before May 1, 2025?"
2. P body: Recent polling shows Biden's approval rating among young voters and Arab Americans in key swing districts has dropped to historic lows. Multiple House Democrats have reported intense constituent pressure and protests over the administration's Middle East policy. This resolves YES if at least 5 named House Democrats from districts Biden won by less than 10 points in 2020 make public statements explicitly opposing the administration's Israel policy. Statements must specifically criticize Biden's handling of the situation.
2. Q title: "Will the Congressional Progressive Caucus formally withdraw support from Biden's reelection campaign?"
2. Q body: Resolves YES only if the CPC, as an organization, formally announces withdrawal of support through official channels before December 31, 2024.

Example 4:
Base question P title:Will the Minnesota Timberwolves win the 2024 NBA Finals? Base question P body: The question resolves Yes if the Minnesota Timberwolves are crowned the champion of the 2024 NBA Finals, according to the NBA.
It resolves No if any other team wins or if the Timberwolves are eliminated from contention.
Base question Q title: Will Anthony Edwards play in all of the Minnesota Timberwolves' playoff games in 2024? 
Base question Q body: This question resolves to Yes if Anthony Edwards, a key player for the Minnesota Timberwolves, participates in every playoff game that the team competes in during the 2024 NBA Playoffs. It resolves to No if he misses any playoff game due to injury, rest, suspension, or any other reason. The resolution will be based on official game participation records from the NBA for the 2024 playoffs.

Title: Will the Minnesota Timberwolves win the 2024 NBA Finals? Forecast: 0.05 Reasoning: The Minnesota Timberwolves have shown promise in recent years, particularly with the emergence of Karl-Anthony Towns and Anthony Edwards. However, the team's overall performance and playoff success have been inconsistent. The Western Conference is highly competitive, with powerhouses like the Los Angeles Lakers, Golden State Warriors, and Denver Nuggets. Additionally, the Eastern Conference has strong contenders like the Boston Celtics and Milwaukee Bucks. Given the Timberwolves' current roster and the competitive landscape of the NBA, it's challenging to envision them winning the championship in 2024. While they may make a deep playoff run, the odds of them ultimately winning the title are relatively low. Probability: 0.05

Logical combination (not P and Q) title: Will both the Minnesota Timberwolves fail to win the 2024 NBA Finals and Anthony Edwards play in all of their playoff games in 2024? Forecast: 0.85 Reasoning: Reasoning: To estimate the probability of both conditions being met, we need to consider the likelihood of the Minnesota Timberwolves failing to win the 2024 NBA Finals and Anthony Edwards playing in all of their playoff games.
First, let's consider the Timberwolves' chances of winning the 2024 NBA Finals. While they have talented players like Anthony Edwards and Karl-Anthony Towns, the Western Conference is highly competitive, with teams like the Denver Nuggets, Phoenix Suns, and Los Angeles Clippers. Given the strength of these teams, it's reasonable to assume that the Timberwolves face significant competition. However, they are still a playoff contender and have a chance to make a deep run.
Assuming the Timberwolves make the playoffs, Anthony Edwards' participation in all playoff games is the second condition. Edwards has been relatively healthy throughout his career, but injuries can happen to any player. The NBA playoffs are physically demanding, and the risk of injury increases with each game. However, Edwards is a key player, and the team will likely take precautions to ensure his health and availability.
Considering both conditions, it's likely that the Timberwolves will not win the 2024 NBA Finals, given the strong competition in the Western Conference. Additionally, while there's a risk of injury, Edwards' relatively good health record and the team's reliance on him suggest that he will play in most, if not all, playoff games.
Probability: 0.85

Follow up question title: Title: Will either the Minnesota Timberwolves win the 2024 NBA Finals or will Anthony Edwards play in all of the Timberwolves' playoff games in 2024?
Forecast: 0.335
Reasoning: To estimate the probability of this question resolving to YES, we need to consider two separate scenarios: (a) the Minnesota Timberwolves winning the 2024 NBA Finals, and (b) Anthony Edwards playing in all of the Timberwolves' playoff games in 2024.
Firstly, the Minnesota Timberwolves winning the 2024 NBA Finals is a challenging task, as they would need to outperform several strong teams in the league. While they have talented players like Anthony Edwards and Karl-Anthony Towns, their recent performance and playoff history do not suggest a high likelihood of winning the championship.
Secondly, Anthony Edwards playing in all of the Timberwolves' playoff games in 2024 is also uncertain. Injuries and player availability can be unpredictable, and even if Edwards is healthy, the team's playoff run may be cut short, reducing the number of games he needs to play.
However, since the question resolves to YES if either of these scenarios occurs, we can consider the probabilities of each scenario separately and then combine them. Assuming the probability of the Timberwolves winning the NBA Finals is around 0.05 (5%), and the probability of Anthony Edwards playing in all playoff games is around 0.3 (30%), we can estimate the combined probability.
Using the formula for the probability of at least one event occurring (P(A or B) = P(A) + P(B) - P(A and B)), we can calculate the combined probability. Assuming the events are independent, the probability of both events occurring (P(A and B)) is the product of the individual probabilities (0.05 * 0.3 = 0.015). Therefore, the combined probability is approximately 0.05 + 0.3 - 0.015 = 0.335.
Probability: 0.335

Reasoning flaw:  P involves a relatively less popular sports team winning a championship, which is low probability. So, not P is high probability. Event Q involves a player of that sports team playing all games, which is low probability according to the forecaster. So, event (not P and Q) should be low probability but the forecaster doesn’t multiple event not P and event Q probabilities and overestimates event not P probability. For example, we can ask:
Pair 1:
Base Question P title: "Will the Sacramento Kings win the 2025 NBA Finals?"
Base Question Q title: "Will De'Aaron Fox play at least 78 regular season games in the 2024-25 NBA season?"
Pair 2:
Base Question P title: "Will the Detroit Lions win Super Bowl LIX (February 2025)?"
Base Question Q title: "Will Jared Goff complete over 70% of his passes in all playoff games leading to Super Bowl LIX?"
Pair 3:
Base Question P title: "Will Brighton & Hove Albion win the 2024-25 Premier League?"
Base Question Q title: "Will Brighton's Kaoru Mitoma play every minute of their last 10 Premier League matches in the 2024-25 season?"


Example 5: 
Base question P title:On July 21, 2024, will the closing value of BSE SENSEX be greater than 75k and less than or equal to 77k? 
Base question P body: This question resolves as the closing value of the BSE SENSEX on July 21, 2024, according to Yahoo Finance.
Base question Q title: Will the Reserve Bank of India change the benchmark interest rate by October 2024?
Base question Q body: This question forecasts whether the Reserve Bank of India (RBI) will announce a change in its benchmark interest rate on or before October 31, 2024. The resolution of this question is based on official announcements by the RBI. Interest rate changes are significant events that can affect economic growth, inflation, and financial markets, including the stock indices like the BSE SENSEX. Therefore, understanding potential interest rate adjustments can provide critical context for analyzing market behavior.

Question P: Title: On July 21, 2024, will the closing value of BSE SENSEX be greater than 75k and less than or equal to 77k? Forecast: 0.35
Probability: 0.35
Question Q: Title: Given that on July 21, 2024, the closing value of BSE SENSEX is greater than 75k and less than or equal to 77k, will the Reserve Bank of India change the benchmark interest rate by October 2024?Forecast: 0.55 
Reasoning: Given the condition that the BSE SENSEX is greater than 75,000 and less than or equal to 77,000 on July 21, 2024, we can infer that the Indian economy is likely in a state of moderate growth, with the stock market performing relatively well. However, this information alone does not provide direct insight into the RBI's future monetary policy decisions. The RBI's interest rate decisions are influenced by various factors, including inflation rates, economic growth, and global economic trends.Considering the RBI's tendency to adjust interest rates in response to changing economic conditions, and given the moderate growth indicated by the BSE SENSEX, I would estimate that there is a moderate probability of the RBI changing the benchmark interest rate by October 2024.
Probability: 0.55
Question R Title: Will both of the following occur: (1) On July 21, 2024, will the closing value of BSE SENSEX be greater than 75k and less than or equal to 77k? AND (2) Will the Reserve Bank of India change the benchmark interest rate by October 2024? 
Forecast: 0.42 
Reasoning: To estimate the probability of both conditions being met, we need to consider the current market trends, economic indicators, and the Reserve Bank of India's (RBI) monetary policy stance.
As of my knowledge cutoff in December 2023, the BSE SENSEX has been trending upwards, driven by a combination of factors such as a strong economic recovery, favorable corporate earnings, and a relatively stable inflation environment. However, predicting the exact closing value of the BSE SENSEX on a specific date (July 21, 2024) is challenging due to the inherent volatility of financial markets. That being said, a closing value between 75,000 and 77,000 seems plausible, given the current trend and the fact that the index has already breached the 75,000 mark in the past.
Regarding the RBI's interest rate decision, the central bank has been closely monitoring inflation and economic growth. As of my knowledge cutoff, the RBI has maintained a accommodative monetary policy stance to support economic growth. However, with inflation concerns rising, there is a possibility that the RBI might consider adjusting the benchmark interest rate to manage inflation expectations. Given the RBI's proactive approach to monetary policy, I believe there is a reasonable chance that the central bank might announce a change in the benchmark interest rate by October 31, 2024.
To estimate the probability of both conditions being met, I will assign a probability of 0.6* to the first condition (BSE SENSEX closing value between 75,000 and 77,000 on July 21, 2024) and a probability of 0.7* to the second condition (RBI changing the benchmark interest rate by October 31, 2024). Assuming that these two events are not highly correlated, I will multiply the individual probabilities to get the joint probability.
Probability: 0.42

Reasoning flaw: Event P asks about a stock index of a developing country growing (more than forecast date) and being in a narrow range, Event Q asks about a rate cut. Given P’s value being in a high narrow range, the model overestimates the probability of event Q and also assumes that P and Q are not correlated.  We can ask:
Pair 1:
Base Question P title: "Will Brazil's Ibovespa index close between 200,000 and 207,000 points on August 15, 2028?"
Base Question Q title: "Will Brazil's Central Bank cut interest rates by more than 100 basis points in the year 2028 before end of Q3?"
Pair 2:
Base Question P title: "Will Indonesia's Jakarta Composite Index (JCI) close between 10,500 and 11,700 points on March 31, 2025?"
Base Question Q title: "Will Bank Indonesia implement at least two rate cuts totaling 75 basis points or more in Q2 2025?"
"""

    question_model_name = "deepseek-ai/DeepSeek-V3"
    forecast_model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
    desired_consistency_threshold = 0.30

    # Now run the brute force generator
    final_results = asyncio.run(brute_force_consistency_generator(
        question_model_name=question_model_name,
        model_name=forecast_model_name,
        worst_context=worst_context,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        together_api_key=os.getenv("TOGETHER_API_KEY"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        desired_consistency_threshold=desired_consistency_threshold,
        max_iterations=7,
        parallel_runs=20
    ))

    # Print final results
    for run_info in final_results:
        logging.info(f"\n=== RUN {run_info['run_index']} ===")
        logging.info(f"Final Score: {run_info['final_score']:.3f}")
        logging.info(f"P question: {run_info['p_question']}")
        logging.info(f"Q question: {run_info['q_question']}")
        logging.info("All Iterations:")
        for it_data in run_info["all_iterations"]:
            logging.info(f"  Iter #{it_data['iteration']}, Score={it_data['avg_score']:.3f}")

# INFO:root:Final Score: 0.400
# INFO:root:P question: {'id': 'P5', 'title': "On March 31, 2027, will NVIDIA's revenue from AI-driven autonomous vehicle technologies exceed Tesla's revenue from electric vehicle sales?", 'body': "NVIDIA has been a key player in AI-driven autonomous vehicle technologies, providing hardware and software solutions for self-driving cars, while Tesla remains a leader in electric vehicle sales with a growing global footprint. Given the rapid advancements in autonomous driving and the increasing competition in the EV market, forecast whether NVIDIA's revenue from AI-driven autonomous vehicle technologies will exceed Tesla's revenue from electric vehicle sales by March 31, 2027."}
# INFO:root:Q question: {'id': 'Q5', 'title': "On March 31, 2027, will Intel's revenue from AI hardware for data centers surpass AMD's revenue from consumer GPU sales?", 'body': "Intel has been investing heavily in AI hardware for data centers, aiming to capture a larger share of the AI infrastructure market, while AMD continues to dominate consumer GPU sales with its Radeon series. Considering the growing demand for AI infrastructure in data centers and the competitive dynamics in consumer GPUs, forecast whether Intel's revenue from AI hardware for data centers will surpass AMD's revenue from consumer GPU sales by March 31, 2027."}
# INFO:root:All Iterations:
# INFO:root:  Iter #1, Score=0.076
# INFO:root:  Iter #2, Score=0.154
# INFO:root:  Iter #3, Score=0.200
# INFO:root:  Iter #4, Score=0.172
# INFO:root:  Iter #5, Score=0.400
# INFO:root:
# === RUN 5 ===
# INFO:root:Final Score: 0.410
# INFO:root:P question: {'id': 'P2', 'title': "On November 15, 2026, will SpaceX's valuation exceed Google's?", 'body': "On November 15, 2026, will SpaceX's valuation exceed Google's?"}
# INFO:root:Q question: {'id': 'Q2', 'title': "On November 15, 2026, will OpenAI's annual revenue surpass Netflix's?", 'body': "On November 15, 2026, will OpenAI's annual revenue surpass Netflix's?"}
# INFO:root:All Iterations:
# INFO:root:  Iter #1, Score=0.140
# INFO:root:  Iter #2, Score=0.410
# INFO:root:Final Score: 0.450
# INFO:root:P question: {'id': 'P5', 'title': "Will at least 10 named Democratic Senators publicly call for an independent investigation into Biden's handling of classified documents before March 1, 2025?", 'body': "This question resolves YES if at least 10 named Democratic Senators make public statements, either individually or collectively, calling for an independent investigation into President Biden's handling of classified documents. Statements must explicitly mention the need for an independent investigation and reference classified documents. Private expressions of concern or unnamed sources are insufficient for YES resolution. The statements must be made before March 1, 2025."}
# INFO:root:Q question: {'id': 'Q5', 'title': "Will the Department of Justice appoint a special counsel to investigate Biden's handling of classified documents by March 1, 2025?", 'body': "This question resolves YES if the Department of Justice officially appoints a special counsel to investigate President Biden's handling of classified documents by March 1, 2025. The appointment must be publicly announced by the DOJ, and the special counsel must have the authority to conduct an independent investigation. Media speculation or informal inquiries do not qualify for YES resolution."}

# Title: Will the S&P 500 index close outside the range of 7,500 to 7,800 points on December 31, 2030?  
# Body: This question resolves YES if the S&P 500 index, as reported by a major financial data provider (e.g., Bloomberg, Yahoo Finance), closes outside the range of 7,500 to 7,800 points on December 31, 2030. NO if it closes within this range.

# Title: Will the Federal Reserve refrain from implementing a negative interest rate policy in the United States by December 31, 2030?  
# Body: This question resolves YES if the Federal Reserve does not officially announce and implement a negative interest rate policy for the federal funds rate on or before December 31, 2030. NO if such a policy is announced and implemented by the specified date.

# INFO:root:P question: {'id': 'P1', 'title': "Will NVIDIA's quarterly revenue from data center GPUs exceed $15 billion in Q4 2026?", 'body': "This question resolves as Yes if NVIDIA's quarterly revenue from data center GPUs, as reported in its official financial statements, exceeds $15 billion in Q4 2026 (October to December 2026). It resolves as No if the revenue is $15 billion or below for this period."}
# INFO:root:Q question: {'id': 'Q1', 'title': 'Will the global market share of NVIDIA in the AI accelerator market exceed 75% by December 31, 2026?', 'body': "This question resolves as Yes if NVIDIA's market share in the AI accelerator market, as reported by a credible market research firm such as Gartner or IDC, exceeds 75% on or before December 31, 2026. It resolves as No if NVIDIA's market share is 75% or below by this date."}

# INFO:root:P question: {'id': 'P7', 'title': 'Will the United Nations Security Council pass a resolution imposing comprehensive sanctions on North Korea for its continued development of nuclear weapons by December 31, 2025?', 'body': 'This question resolves as Yes if the United Nations Security Council passes a resolution that imposes comprehensive sanctions on North Korea, specifically targeting its nuclear weapons development program, and the resolution is adopted by December 31, 2025. The sanctions must include significant restrictions on trade, financial transactions, and technology transfers related to nuclear weapons. If no such resolution is passed by the deadline, the question resolves as No.'}
# INFO:root:Q question: {'id': 'Q7', 'title': 'Will North Korea conduct a nuclear test involving a thermonuclear device by June 30, 2025?', 'body': 'This question resolves as Yes if North Korea conducts a nuclear test involving a thermonuclear device, confirmed by international monitoring agencies (e.g., CTBTO, IAEA), and the test occurs by June 30, 2025. The test must be verified as involving a thermonuclear device through seismic data, radiological analysis, or official statements from North Korea. If no such test is conducted by the deadline, the question resolves as No.'}

# INFO:root:P question: {'id': 'P1', 'title': 'Will the United States formally announce its withdrawal from NATO by January 1, 2027?', 'body': 'This question resolves as YES if the U.S. government, through an official statement, press conference, or legislative action, formally announces its intention to withdraw from NATO by January 1, 2027. The announcement must be explicit and verifiable through official government channels. Speculation, leaks, or unofficial statements do not qualify for YES resolution.'}
# INFO:root:Q question: {'id': 'Q1', 'title': 'Will the S&P 500 index drop by more than 15% within 30 days of the U.S. announcing its withdrawal from NATO?', 'body': 'This question resolves as YES if the S&P 500 index, as reported by Bloomberg or a similarly reputable financial data provider, experiences a drop of more than 15% from its closing value on the day of the U.S. withdrawal announcement within the following 30 days. The drop must be directly attributable to the geopolitical event and verifiable through market data.'}

# INFO:root:P question: {'id': 'P1', 'title': "Will South Africa's FTSE/JSE All Share Index close between 85,000 and 87,500 points on December 31, 2026?", 'body': 'This question resolves based on the closing value of the FTSE/JSE All Share Index on December 31, 2026, as reported by Bloomberg. The index must close strictly greater than 85,000 and less than or equal to 87,500 points for the question to resolve as YES.'}
# INFO:root:Q question: {'id': 'Q1', 'title': 'Will the South African Reserve Bank raise interest rates by at least 50 basis points in Q4 2026?', 'body': 'This question resolves YES if the South African Reserve Bank announces an interest rate increase of 50 basis points or more in the fourth quarter of 2026, based on official statements or monetary policy meeting minutes.'}

# INFO:root:P question: {'id': '1', 'title': 'Will the Phoenix Suns win the 2026 NBA Finals?', 'body': 'This question resolves Yes if the Phoenix Suns are crowned the champion of the 2026 NBA Finals, according to the NBA. It resolves No if any other team wins or if the Suns are eliminated from contention.'}
# INFO:root:Q question: {'id': '2', 'title': 'Will Devin Booker suffer a season-ending injury during the 2025-26 NBA regular season?', 'body': 'This question resolves Yes if Devin Booker, a key player for the Phoenix Suns, suffers a season-ending injury during the 2025-26 NBA regular season, as confirmed by official injury reports from the NBA. It resolves No if he does not suffer such an injury.'}

# INFO:root:P question: {'id': 'P1', 'title': 'Will the MSCI Emerging Markets Index close between 1,250 and 1,300 points on June 30, 2028?', 'body': 'This question resolves based on the closing value of the MSCI Emerging Markets Index on June 30, 2028, as reported by a major financial data provider such as Bloomberg or Yahoo Finance. The resolution will be determined by whether the closing value falls within the specified range of 1,250 to 1,300 points, inclusive.'}
# INFO:root:Q question: {'id': 'Q1', 'title': "Will the People's Bank of China implement a one-time devaluation of the yuan by more than 5% against the US dollar in 2028 before end of Q2?", 'body': "This question resolves based on official announcements or actions by the People's Bank of China regarding the exchange rate of the yuan. The resolution will be determined by whether the central bank implements a one-time devaluation of the yuan by more than 5% against the US dollar, as measured by the official exchange rate, before June 30, 2028."}

# INFO:root:P question: {'id': 'P1', 'title': 'Will the S&P 500 close between 6,500 and 6,700 points on December 31, 2026?', 'body': 'This question resolves based on the official closing value of the S&P 500 on December 31, 2026, as reported by the Wall Street Journal. The resolution is Yes if the closing value is greater than 6,500 and less than or equal to 6,700 points. Otherwise, it resolves No.'}
# INFO:root:Q question: {'id': 'Q1', 'title': 'Will the Federal Reserve implement a rate hike of at least 100 basis points in Q4 2026?', 'body': 'This question resolves Yes if the Federal Reserve announces an increase in the benchmark interest rate by at least 100 basis points on or before December 31, 2026, based on official Federal Reserve announcements. Otherwise, it resolves No.'}

# INFO:root:P question: {'id': 'P1', 'title': "Will NVIDIA's market capitalization exceed $5 trillion by December 31, 2030?", 'body': "This question resolves YES if NVIDIA's market capitalization exceeds $5 trillion on or before December 31, 2030, according to reliable financial data sources such as Yahoo Finance or Bloomberg. Market capitalization is calculated as the total number of outstanding shares multiplied by the current share price. If NVIDIA's market capitalization does not reach or exceed $5 trillion by this date, the question resolves NO."}
# INFO:root:Q question: {'id': 'Q1', 'title': "Will NVIDIA's annual revenue exceed $1 trillion in the fiscal year 2030?", 'body': "This question resolves YES if NVIDIA's annual revenue, as reported in its official financial statements for the fiscal year ending December 31, 2030, exceeds $1 trillion. Revenue must be verified through NVIDIA's audited financial reports or equivalent authoritative sources. If NVIDIA's revenue does not exceed $1 trillion, the question resolves NO."}

# INFO:root:P question: {'id': 'P1', 'title': "Will Brazil's Ibovespa index close between 200,000 and 207,000 points on August 15, 2028?", 'body': 'This question resolves YES if the Ibovespa index, as reported by a reliable financial source such as Bloomberg, closes between 200,000 and 207,000 points on August 15, 2028. It resolves NO if the index closes outside this range or if no reliable data is available.'}
# INFO:root:Q question: {'id': 'Q1', 'title': "Will Brazil's Central Bank cut interest rates by more than 100 basis points in the year 2028 before end of Q3?", 'body': 'This question resolves YES if the Central Bank of Brazil announces a cumulative interest rate cut of more than 100 basis points (1 percentage point) during the period from January 1, 2028, to September 30, 2028, based on official statements or monetary policy reports. It resolves NO if the total rate cuts are 100 basis points or less or if no rate cuts are announced.'}

# INFO:root:P question: {'id': 'P1', 'title': "Will Tesla's market capitalization exceed $1.5 trillion on December 31, 2028?", 'body': "This question resolves as Yes if Tesla's market capitalization, as reported by a reputable financial source such as Yahoo Finance or Bloomberg, exceeds $1.5 trillion on December 31, 2028. The market capitalization must be calculated based on the closing share price and the total number of outstanding shares on that date."}
# INFO:root:Q question: {'id': 'Q1', 'title': "Will Tesla's annual vehicle production surpass 10 million units in 2028?", 'body': "This question resolves as Yes if Tesla's annual vehicle production, as reported in its official financial statements or press releases, exceeds 10 million units in the calendar year 2028. The production figure must include all vehicles manufactured by Tesla globally, regardless of model or market."}