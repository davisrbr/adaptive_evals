from inspect_ai import solver
from inspect_ai.model import GenerateConfig, Model, ChatMessageUser
from inspect_ai.solver import (
    solver,
    TaskState,
    Generate, 
)
from inspect_ai.util import subtask
from inspect_ai.model import ModelOutput

# Forecasting helper modules are imported lazily in advanced paths so
# zero-shot forecasting can run without optional retrieval dependencies.

ZERO_SHOT_PROMPT = (
    """You are an expert superforecaster, familiar with the work of Tetlock and others. Make a prediction of the probability that the question will be resolved as true. You MUST give a probability estimate between 0 and 1 UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, pick the base rate, but return a number between 0 and 1.

Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's date: {date_begin}
Question close date: {date_end}

Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal. Do not output anything else.
Answer: {{ Insert answer here }}""",
    ("QUESTION", "BACKGROUND", "RESOLUTION_CRITERIA", "DATES"),
)

SEARCH_QUERY_PROMPT_0 = (
    """I will provide you with a forecasting question and the background information for the question. I will then ask you to generate short search queries (up to {max_words} words each) that I'll use to find articles on Google News to help answer the question.

Question:
{question}

Question Background:
{background}

Today's date: {date_begin}
Question close date: {date_end}

You must generate this exact amount of queries: {num_keywords}

Start off by writing down sub-questions. Then use your sub-questions to help steer the search queries you produce.

Your response should take the following structure:
Thoughts:
{{ Insert your thinking here. }}
Search Queries:
{{ Insert the queries here. Use semicolons to separate the queries. }}""",
    (
        "QUESTION",
        "BACKGROUND",
        "DATES",
        "NUM_KEYWORDS",
        "MAX_WORDS",
    ),
)

SEARCH_QUERY_PROMPT_1 = (
    """I will provide you with a forecasting question and the background information for the question.

Question:
{question}

Question Background:
{background}

Today's date: {date_begin}
Question close date: {date_end}

Task:
- Generate brief search queries (up to {max_words} words each) to gather information on Google that could influence the forecast.

You must generate this exact amount of queries: {num_keywords}

Your response should take the following structure:
Thoughts:
{{ Insert your thinking here. }}
Search Queries:
{{ Insert the queries here. Use semicolons to separate the queries. }}""",
    (
        "QUESTION",
        "BACKGROUND",
        "DATES",
        "NUM_KEYWORDS",
        "MAX_WORDS",
    ),
)

SUMMARIZATION_PROMPT = (
    """I want to make the following article shorter (condense it to no more than 100 words).

Article:
---
{article}
---

When doing this task for me, please do not remove any details that would be helpful for making considerations about the following forecasting question.

Forecasting Question: {question}
Question Background: {background}""",
    ("QUESTION", "BACKGROUND"),
)

RANKING_PROMPT = (
    """Please consider the following forecasting question and its background information.
After that, I will give you a news article and ask you to rate its relevance with respect to the forecasting question.

Question:
{question}

Question Background:
{background}

Question Resolution Criteria:
{resolution_criteria}

Article:
{article}

Please rate the relevance of the article to the question, at the scale of 1-6
1 -- irrelevant
2 -- slightly relevant
3 -- somewhat relevant
4 -- relevant
5 -- highly relevant
6 -- most relevant

Guidelines:
- You don't need to access any external sources. Just consider the information provided.
- Focus on the content of the article, not the title.
- If the text content is an error message about JavaScript, paywall, cookies or other technical issues, output a score of 1.

Your response should look like the following:
Thoughts: {{ insert your thinking }}
Rating: {{ insert your rating }}""",
    ("QUESTION", "BACKGROUND", "RESOLUTION_CRITERIA", "ARTICLE"),
)


BINARY_SCRATCH_PAD_PROMPT_1 = (
    """Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's date: {date_begin}
Question close date: {date_end}

We have retrieved the following information for this question:
{retrieved_info}


Instructions:
1. Write down any additional relevant information that is not included above. This should be specific facts that you already know the answer to, rather than information that needs to be looked up.
{{ Insert additional information }}

2. Provide at least 3 reasons why the answer might be no.
{{ Insert your thoughts }}

3. Provide at least 3 reasons why the answer might be yes.
{{ Insert your thoughts }}

4. Rate the strength of each of the reasons given in the last two responses. Think like a superforecaster (e.g. Nate Silver).
{{ Insert your rating of the strength of each reason }}

5. Aggregate your considerations.
{{ Insert your aggregated considerations }}

6. Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
{{ Insert your answer }}""",
    ("QUESTION", "BACKGROUND", "RESOLUTION_CRITERIA", "DATES", "RETRIEVED_INFO"),
)

BINARY_SCRATCH_PAD_PROMPT_2 = (
    """Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's date: {date_begin}
Question close date: {date_end}

We have retrieved the following information for this question:
{retrieved_info}


Think step by step: {{ Insert your step by step consideration }}
Aggregating considerations: {{ Aggregate your considerations }}
Answer: {{ Output your answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal }}""",
    ("QUESTION", "BACKGROUND", "RESOLUTION_CRITERIA", "DATES", "RETRIEVED_INFO"),
)

BINARY_SCRATCH_PAD_PROMPT_NEW_2 = (
    """Question: {question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's date: {date_begin}
Question close date: {date_end}

We have retrieved the following information for this question:
{retrieved_info}


Instructions:
1. Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.
{{ Insert rephrased and expanded question.}}

2. Using your knowledge of the world and topic, as well as the information provided, provide a few reasons why the answer might be no. Rate the strength of each reason.
{{ Insert your thoughts }}

3. Using your knowledge of the world and topic, as well as the information provided, provide a few reasons why the answer might be yes. Rate the strength of each reason.
{{ Insert your thoughts }}

4. Aggregate your considerations. Think like a superforecaster (e.g. Nate Silver).
{{ Insert your aggregated considerations }}

5. Output an initial probability (prediction) given steps 1-4.
{{ Insert initial probability. }}

6. Evaluate whether your calculated probability is excessively confident or not confident enough. Also, consider anything else that might affect the forecast that you did not before consider (e.g. base rate of the event).
{{ Insert your thoughts }}

7. Output your final prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
{{ Insert your answer }}""",
    ("QUESTION", "BACKGROUND", "RESOLUTION_CRITERIA", "DATES", "RETRIEVED_INFO"),
)

BINARY_SCRATCH_PAD_PROMPT_NEW_3 = (
    """Question: {question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's date: {date_begin}
Question close date: {date_end}

We have retrieved the following information for this question:
{retrieved_info}


Instructions:
1. Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.
{{ Insert rephrased and expanded question.}}

2. Provide a few reasons why the answer might be no. Rate the strength of each reason.
{{ Insert your thoughts }}

3. Provide a few reasons why the answer might be yes. Rate the strength of each reason.
{{ Insert your thoughts }}

4. Aggregate your considerations. Think like a superforecaster (e.g. Nate Silver).
{{ Insert your aggregated considerations }}

5. Output an initial probability (prediction) given steps 1-4.
{{ Insert initial probability. }}

6. Feel free to adjust your probability now. Here is a non-exhaustive list of some things you'll want to check:
- Is your calculated probability excessively confident or not confident enough?
- Is there anything else that might affect the forecast that you did not before consider (e.g. base rate of the event)?
- Use your intuition and feel for the question.
{{ Insert your thoughts }}

7. Output your final answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
{{ Insert your answer }}""",
    ("QUESTION", "BACKGROUND", "RESOLUTION_CRITERIA", "DATES", "RETRIEVED_INFO"),
)

BINARY_SCRATCH_PAD_PROMPT_NEW_6 = (
    """Question:
{question}

Question Background:
{background}

Resolution Criteria:
{resolution_criteria}

Today's date: {date_begin}
Question close date: {date_end}

We have retrieved the following information for this question:
{retrieved_info}


Instructions:
1. Given the above question, rephrase and expand it to help you do better answering. Maintain all information in the original question.
{{ Insert rephrased and expanded question.}}

2. Develop a decision tree outlining possible paths to both 'Yes' and 'No' outcomes.
{{ Insert decision tree outline }}

3. Analyze the probability of each branch of the decision tree based on current information.
{{ Insert branch probability analysis }}

4. Discuss any potential game-changers or wildcard events. Use your knowledge of the topic as well as the information provided.
{{ Insert discussion on wildcards }}

5. Output an initial probability (prediction) given steps 1-4.
{{ Insert initial probability. }}

6. Feel free to adjust your probability now. Here is a non-exhaustive list of some things you'll want to check:
- Is your calculated probability is excessively confident or not confident enough.
- Is there anything else that might affect the forecast that you did not before consider.
- Use your intuition and feel for the question.
{{ Insert your thoughts }}

7. Output your final answer (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
{{ Insert your answer }}""",
    ("QUESTION", "BACKGROUND", "RESOLUTION_CRITERIA", "DATES", "RETRIEVED_INFO"),
)

ALIGNMENT_PROMPT = (
    """Question:
{question}

Background:
{background}

Resolution Criteria:
{resolution_criteria}

Model’s Thinking:
{reasoning}

Task:
Evaluate the alignment between the model's thinking and its prediction. If someone were given the reasoning alone (without the prediction), would they likely arrive at the same prediction?

Alignment Ratings:
1 — Very Not Aligned
2 — Not Aligned
3 — Slightly Not Aligned
4 — Slightly Aligned
5 — Aligned
6 — Very Aligned

Please use these ratings to indicate the degree of alignment between the model's reasoning and its prediction.

Note: If the response indicates that this question is old or it's already been resolved, give it an alignment rating of 1.

I want your answer to follow this format:

Thinking: {{ insert your thinking here }}
Rating: {{ insert your alignment rating here (a number between 1 and 6) }}""",
    ("QUESTION", "BACKGROUND", "RESOLUTION_CRITERIA", "REASONING"),
)

ENSEMBLE_PROMPT_0 = (
    """I need your assistance with making a forecast. Here is the question and its metadata.
Question: {question}

Background: {background}

Resolution criteria: {resolution_criteria}

Today's date: {date_begin}
Question close date: {date_end}

I have retrieved the following information about this question.
Retrieved Info:
{retrieved_info}

In addition, I have generated a collection of other responses and reasonings from other forecasters:
{base_reasonings}

Your goal is to aggregate the information and make a final prediction.

Instructions:
1. Provide reasons why the answer might be no.
{{ Insert your thoughts here }}

2. Provide reasons why the answer might be yes.
{{ Insert your thoughts here }}

3. Aggregate your considerations.
{{ Insert your aggregated considerations here }}

4. Output your prediction (a number between 0 and 1) with an asterisk at the beginning and end of the decimal.
{{ Insert the probability here }}""",
    (
        "QUESTION",
        "BACKGROUND",
        "RESOLUTION_CRITERIA",
        "RETRIEVED_INFO",
        "DATES",
        "BASE_REASONINGS",
    ),
)


RETRIEVAL_CONFIG = {
    "NUM_SEARCH_QUERY_KEYWORDS": 3, #3 queries * 2 News APIs * 2 prompts
    "MAX_WORDS_NEWSCATCHER": 5,
    "MAX_WORDS_GNEWS": 8,
    "SEARCH_QUERY_MODEL_NAME": "gpt-4o-2024-08-06",
    "SEARCH_QUERY_TEMPERATURE": 0.0,
    "SEARCH_QUERY_PROMPT_TEMPLATES": [
        SEARCH_QUERY_PROMPT_0,
        SEARCH_QUERY_PROMPT_1 ,
    ],
    # Top 10
    "NUM_ARTICLES_PER_QUERY": 5,
    #4o-mini
    "SUMMARIZATION_MODEL_NAME": "gpt-4o-mini-2024-07-18",
    "SUMMARIZATION_TEMPERATURE": 0.2,
    "SUMMARIZATION_PROMPT_TEMPLATE": SUMMARIZATION_PROMPT,
    "NUM_SUMMARIES_THRESHOLD": 10,
    "PRE_FILTER_WITH_EMBEDDING": True,
    "PRE_FILTER_WITH_EMBEDDING_THRESHOLD": 0.32,
    #4o-mini
    "RANKING_MODEL_NAME": "gpt-4o-mini-2024-07-18",
    "RANKING_TEMPERATURE": 0.0,
    "RANKING_PROMPT_TEMPLATE": RANKING_PROMPT,
    "RANKING_RELEVANCE_THRESHOLD": 4,
    "RANKING_COSINE_SIMILARITY_THRESHOLD": 0.5,
    "SORT_BY": "date",
    "RANKING_METHOD": "llm-rating",
    "RANKING_METHOD_LLM": "title_250_tokens",
    "NUM_SUMMARIES_THRESHOLD": 20,
    "EXTRACT_BACKGROUND_URLS": True,
}

REASONING_CONFIG = {
    "BASE_REASONING_MODEL_NAMES": ["gpt-4o-2024-08-06"], #Suppose 1st model is finetuned and is provided as an input by the user
    "BASE_REASONING_TEMPERATURE": 0.5,
    "BASE_REASONING_PROMPT_TEMPLATES": [
        [
            ZERO_SHOT_PROMPT,
            ZERO_SHOT_PROMPT,
            ZERO_SHOT_PROMPT
        ],
        [
            BINARY_SCRATCH_PAD_PROMPT_NEW_2,
            BINARY_SCRATCH_PAD_PROMPT_NEW_6,
            BINARY_SCRATCH_PAD_PROMPT_1
        ],
    ],
    "ALIGNMENT_MODEL_NAME": "gpt-4o-mini-2024-07-18",
    "ALIGNMENT_TEMPERATURE": 0,
    "ALIGNMENT_PROMPT": ALIGNMENT_PROMPT,
    "AGGREGATION_METHOD": "meta",
    "AGGREGATION_PROMPT_TEMPLATE": ENSEMBLE_PROMPT_0,
    "AGGREGATION_TEMPERATURE": 0.2,
    "AGGREGATION_MODEL_NAME": "gpt-4o-2024-08-06",
    "AGGREGATION_WEIGTHTS": None,
}


@solver
def zero_shot_forecasting_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            # Format the prompt using metadata
            prompt = ZERO_SHOT_PROMPT[0].format(
                question=state.input_text,
                background=state.metadata["background"],
                resolution_criteria=state.metadata["resolution_criteria"],
                date_begin=state.metadata["date_begin"],
                date_end=state.metadata["date_close"]
            )
            
            # Add the formatted prompt as a user message
            state.messages.append(ChatMessageUser(content=prompt))
            
            # Generate response using the model
            state = await generate(state)

            return state
            
        except Exception as e:
            # If there's an error, mark the state as completed
            state.completed = True
            # Store error info in metadata for debugging
            state.metadata['solver_error'] = str(e)
            return state
        
    return solve


# Subtask to retrieve and rank articles, then summarize them to use for reasoning with base models
@subtask
async def generate_summaries(
    question: str,
    background_info: str,
    resolution_criteria: str,
    retrieval_dates: list,
    urls_in_background: list = [],
    config: dict = RETRIEVAL_CONFIG,
) -> str:
    """
    A subtask to retrieve, summarize, and rank articles.
    
    Parameters:
    - question (str): The main question to guide the article retrieval and summarization.
    - background_info (str): Background context for better search relevance.
    - resolution_criteria (str): Criteria to determine what constitutes a successful resolution.
    - retrieval_dates (list): List of dates to constrain the search.
    - urls_in_background (list): Optional list of pre-existing URLs for additional context.
    - config (dict): Configuration dictionary for retrieval and summarization.
    
    Returns:
    - str: Concatenated summaries of the top-ranked articles for reasoning.
    """
    from utils_forecasting import ranking, summarize

    # Retrieve and rank articles
    (
        ranked_articles,
        all_articles,
        search_queries_list_gnews,
        search_queries_list_nc,
    ) = await ranking.retrieve_summarize_and_rank_articles(
        question,
        background_info,
        resolution_criteria,
        retrieval_dates,
        urls=urls_in_background,
        config=config,
        return_intermediates=True,
    )

    # Generate summaries from the ranked articles
    all_summaries = summarize.concat_summaries(
        ranked_articles[: config["NUM_SUMMARIES_THRESHOLD"]]
    )
    
    return all_summaries


@solver
def advanced_forecasting_solver():
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        try:
            from utils_forecasting import ensemble

            # Extract user-provided model name and normalize it
            user_model_name = state.model.name
            #normalized_user_model_name = normalize_model_name(user_model_name)
            normalized_user_model_name = user_model_name

            # Prepend user-provided model to backend models
            all_models = [normalized_user_model_name] + REASONING_CONFIG["BASE_REASONING_MODEL_NAMES"]

            # Perform summarization and ensemble reasoning
            all_summaries = await generate_summaries(
                question=state.input_text,
                background_info=state.metadata["background"],
                resolution_criteria=state.metadata["resolution_criteria"],
                #retrieval_date
                #retrieval_dates=[state.metadata["date_begin"], state.metadata["date_close"]], 
                retrieval_dates=[state.metadata["date_begin"], state.metadata["retrieval_date"]], 
                urls_in_background=state.metadata.get("extracted_urls", []),
            )

            ensemble_dict = await ensemble.meta_reason(
                question=state.input_text,
                background_info=state.metadata["background"],
                resolution_criteria=state.metadata["resolution_criteria"],
                # today_to_close_date = [retrieval_dates[1], question_dates[1]], change later
                today_to_close_date_range=[state.metadata["retrieval_date"], state.metadata["date_close"]],
                retrieved_info=all_summaries,
                reasoning_prompt_templates=REASONING_CONFIG["BASE_REASONING_PROMPT_TEMPLATES"],
                base_model_names=all_models,
                base_temperature=REASONING_CONFIG["BASE_REASONING_TEMPERATURE"],
                aggregation_method=REASONING_CONFIG["AGGREGATION_METHOD"],
                answer_type="probability",
                weights=REASONING_CONFIG["AGGREGATION_WEIGTHTS"],
                meta_model_name=REASONING_CONFIG["AGGREGATION_MODEL_NAME"],
                meta_prompt_template=REASONING_CONFIG["AGGREGATION_PROMPT_TEMPLATE"],
                meta_temperature=REASONING_CONFIG["AGGREGATION_TEMPERATURE"],
            )

            # Use meta_prediction as output
            state.output = ModelOutput.from_content(
                state.model.name, str(ensemble_dict["meta_prediction"])
            )

            return state

        except Exception as e:
            state.completed = True
            state.metadata["solver_error"] = str(e)
            return state
    
    return solve
