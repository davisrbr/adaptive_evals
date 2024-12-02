OAI_SOURCE = "OAI"
ANTHROPIC_SOURCE = "ANTHROPIC"
TOGETHER_AI_SOURCE = "TOGETHER"
GOOGLE_SOURCE = "GOOGLE"
HUGGINGFACE_SOURCE = "HUGGINGFACE"

CHARS_PER_TOKEN = 4

MODEL_NAME_TO_SOURCE = {
    "claude-2.1": ANTHROPIC_SOURCE,
    "claude-2": ANTHROPIC_SOURCE,
    "claude-3-opus-20240229": ANTHROPIC_SOURCE,
    "claude-3-sonnet-20240229": ANTHROPIC_SOURCE,
    "gpt-4": OAI_SOURCE,
    "gpt-3.5-turbo-1106": OAI_SOURCE,
    "gpt-3.5-turbo-16k": OAI_SOURCE,
    "gpt-3.5-turbo": OAI_SOURCE,
    "gpt-4-1106-preview": OAI_SOURCE,
    "gpt-4o-2024-08-06": OAI_SOURCE,
    "gpt-4o": OAI_SOURCE,
    "gpt-4o-mini": OAI_SOURCE,
    "gpt-4o-mini-2024-07-18": OAI_SOURCE,
    "gemini-pro": GOOGLE_SOURCE,
    "togethercomputer/llama-2-7b-chat": TOGETHER_AI_SOURCE,
    "togethercomputer/llama-2-13b-chat": TOGETHER_AI_SOURCE,
    "togethercomputer/llama-2-70b-chat": TOGETHER_AI_SOURCE,
    "togethercomputer/LLaMA-2-7B-32K": TOGETHER_AI_SOURCE,
    "togethercomputer/StripedHyena-Hessian-7B": TOGETHER_AI_SOURCE,
    "mistralai/Mistral-7B-Instruct-v0.2": TOGETHER_AI_SOURCE,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": TOGETHER_AI_SOURCE,
    "zero-one-ai/Yi-34B-Chat": TOGETHER_AI_SOURCE,
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": TOGETHER_AI_SOURCE,
    "NousResearch/Nous-Hermes-2-Yi-34B": TOGETHER_AI_SOURCE,
}

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


DEFAULT_RETRIEVAL_CONFIG = {
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

END_WORDS_TO_PROBS_6 = {
    "No": 0.05,
    "Very Unlikely": 0.15,
    "Unlikely": 0.35,
    "Likely": 0.55,
    "Very Likely": 0.75,
    "Yes": 0.95,
}

END_WORDS_TO_PROBS_10 = {
    "No": 0.05,
    "Extremely Unlikely": 0.15,
    "Very Unlikely": 0.25,
    "Unlikely": 0.35,
    "Slightly Unlikely": 0.45,
    "Slightly Likely": 0.55,
    "Likely": 0.65,
    "Very Likely": 0.75,
    "Extremely Likely": 0.85,
    "Yes": 0.95,
}

TOKENS_TO_PROBS_DICT = {
    "six_options": END_WORDS_TO_PROBS_6,
    "ten_options": END_WORDS_TO_PROBS_10,
}

MODEL_TOKEN_LIMITS = {
    "claude-2.1": 200000,
    "claude-2": 100000,
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "gpt-4": 8000,
    "gpt-3.5-turbo-1106": 16000,
    "gpt-3.5-turbo-16k": 16000,
    "gpt-3.5-turbo": 8000,
    "gpt-4-1106-preview": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4o-mini-2024-07-18": 128000,
    "gemini-pro": 30720,
    "togethercomputer/llama-2-7b-chat": 4096,
    "togethercomputer/llama-2-13b-chat": 4096,
    "togethercomputer/llama-2-70b-chat": 4096,
    "togethercomputer/StripedHyena-Hessian-7B": 32768,
    "togethercomputer/LLaMA-2-7B-32K": 32768,
    "mistralai/Mistral-7B-Instruct-v0.2": 32768,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 32768,
    "zero-one-ai/Yi-34B-Chat": 4096,
    "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO": 32768,
    "NousResearch/Nous-Hermes-2-Yi-34B": 32768,
}


IRRETRIEVABLE_SITES = [
    "wsj.com",
    "english.alarabiya.net",
    "consilium.europa.eu",
    "abc.net.au",
    "thehill.com",
    "democracynow.org",
    "fifa.com",
    "si.com",
    "aa.com.tr",
    "thestreet.com",
    "newsweek.com",
    "spokesman.com",
    "aninews.in",
    "commonslibrary.parliament.uk",
    "cybernews.com",
    "lineups.com",
    "expressnews.com",
    "news-herald.com",
    "c-span.org/video",
    "investors.com",
    "finance.yahoo.com",  # This site has a “read more” button.
    "metaculus.com",  # newspaper4k cannot parse metaculus pages well
    "houstonchronicle.com",
    "unrwa.org",
    "njspotlightnews.org",
    "crisisgroup.org",
    "vanguardngr.com",  # protected by Cloudflare
    "ahram.org.eg",  # protected by Cloudflare
    "reuters.com",  # blocked by Javascript and CAPTCHA
    "carnegieendowment.org",
    "casino.org",
    "legalsportsreport.com",
    "thehockeynews.com",
    "yna.co.kr",
    "carrefour.com",
    "carnegieeurope.eu",
    "arabianbusiness.com",
    "inc.com",
    "joburg.org.za",
    "timesofindia.indiatimes.com",
    "seekingalpha.com",
    "producer.com",  # protected by Cloudflare
    "oecd.org",
    "almayadeen.net",  # protected by Cloudflare
    "manifold.markets",  # prevent data contamination
    "goodjudgment.com",  # prevent data contamination
    "infer-pub.com",  # prevent data contamination
    "www.gjopen.com",  # prevent data contamination
    "polymarket.com",  # prevent data contamination
    "betting.betfair.com",  # protected by Cloudflare
    "news.com.au",  # blocks crawler
    "predictit.org",  # prevent data contamination
    "atozsports.com",
    "barrons.com",
    "forex.com",
    "www.cnbc.com/quotes",  # stock market data: prevent data contamination
    "montrealgazette.com",
    "bangkokpost.com",
    "editorandpublisher.com",
    "realcleardefense.com",
    "axios.com",
    "mensjournal.com",
    "warriormaven.com",
    "tapinto.net",
    "indianexpress.com",
    "science.org",
    "businessdesk.co.nz",
    "mmanews.com",
    "jdpower.com",
    "hrexchangenetwork.com",
    "arabnews.com",
    "nationalpost.com",
    "bizjournals.com",
    "thejakartapost.com",
]