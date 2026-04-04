"""
Function Calling Robustness Eval v1

Adaptive evaluation designed based on failure modes discovered from running BFCL
on gpt-4o. Tests two key weaknesses:

1. IRRELEVANCE DISCRIMINATION: Model over-eagerly calls semantically-similar-but-
   wrong tools instead of abstaining. We test with near-miss tool descriptions
   that are plausible but wrong for the given query.

2. MULTI-STEP STATE TRACKING: Model loses track of state across multi-turn
   tool-calling sequences (file operations, API chaining). We test with
   sequences where each step depends on the result of the previous one.

These failure modes were identified via:
- BFCL irrelevance category: 10% error rate on gpt-4o
- BFCL multi_turn_base: 20% error rate on gpt-4o
- Scout scan: no eval-awareness detected (clean transcripts)
"""

import json

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    CORRECT,
    INCORRECT,
)
from inspect_ai.solver import TaskState, generate, use_tools
from inspect_ai.tool import Tool, tool


# ---------------------------------------------------------------------------
# Near-miss tools: plausible but wrong for the query
# ---------------------------------------------------------------------------


@tool
def get_weather_forecast():
    async def run(location: str, days: int = 3) -> str:
        """Get the weather forecast for a location.

        Args:
            location: City or region name
            days: Number of days to forecast (1-7)

        Returns:
            Weather forecast data
        """
        return json.dumps(
            {
                "location": location,
                "days": days,
                "forecast": [
                    {"day": 1, "temp_f": 75, "condition": "sunny"},
                    {"day": 2, "temp_f": 72, "condition": "partly cloudy"},
                    {"day": 3, "temp_f": 68, "condition": "rain"},
                ],
            }
        )

    return run


@tool
def get_air_quality():
    async def run(location: str) -> str:
        """Get the air quality index for a location.

        Args:
            location: City or region name

        Returns:
            Air quality data including AQI and pollutant levels
        """
        return json.dumps(
            {"location": location, "aqi": 42, "category": "Good", "pm25": 8.3}
        )

    return run


@tool
def search_restaurants():
    async def run(cuisine: str, location: str) -> str:
        """Search for restaurants by cuisine type and location.

        Args:
            cuisine: Type of cuisine (e.g. 'italian', 'mexican')
            location: City or area

        Returns:
            List of matching restaurants
        """
        return json.dumps(
            {
                "results": [
                    {
                        "name": f"Best {cuisine.title()} Place",
                        "rating": 4.5,
                        "address": f"123 Main St, {location}",
                    }
                ]
            }
        )

    return run


@tool
def calculate_distance():
    async def run(origin: str, destination: str) -> str:
        """Calculate driving distance between two cities.

        Args:
            origin: Starting city
            destination: Ending city

        Returns:
            Distance in miles and estimated driving time
        """
        return json.dumps(
            {"origin": origin, "destination": destination, "miles": 215, "hours": 3.5}
        )

    return run


@tool
def get_stock_price():
    async def run(ticker: str) -> str:
        """Get the current stock price for a ticker symbol.

        Args:
            ticker: Stock ticker symbol (e.g. 'AAPL')

        Returns:
            Current price and daily change
        """
        return json.dumps(
            {"ticker": ticker, "price": 185.42, "change_pct": 1.2, "volume": 52000000}
        )

    return run


@tool
def get_company_financials():
    async def run(ticker: str) -> str:
        """Get financial summary for a publicly traded company.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Revenue, profit, and key financial metrics
        """
        return json.dumps(
            {
                "ticker": ticker,
                "revenue_ttm": "394.3B",
                "net_income_ttm": "97.0B",
                "pe_ratio": 29.5,
                "market_cap": "2.87T",
            }
        )

    return run


# ---------------------------------------------------------------------------
# State-tracking tools for multi-step sequences
# ---------------------------------------------------------------------------

_INVENTORY: dict[str, int] = {}
_ORDER_LOG: list[dict] = []


@tool
def check_inventory():
    async def run(product_id: str) -> str:
        """Check current inventory levels for a product.

        Args:
            product_id: The product identifier

        Returns:
            Current stock level and reorder status
        """
        stock = _INVENTORY.get(product_id, 0)
        return json.dumps(
            {
                "product_id": product_id,
                "stock": stock,
                "reorder_needed": stock < 10,
            }
        )

    return run


@tool
def place_reorder():
    async def run(product_id: str, quantity: int) -> str:
        """Place a reorder for a product that's running low.

        Args:
            product_id: The product identifier
            quantity: Number of units to order

        Returns:
            Order confirmation
        """
        order = {
            "order_id": f"ORD-{len(_ORDER_LOG) + 1001}",
            "product_id": product_id,
            "quantity": quantity,
            "status": "confirmed",
        }
        _ORDER_LOG.append(order)
        _INVENTORY[product_id] = _INVENTORY.get(product_id, 0) + quantity
        return json.dumps(order)

    return run


@tool
def get_order_status():
    async def run(order_id: str) -> str:
        """Get the status of a specific order.

        Args:
            order_id: The order identifier

        Returns:
            Order details and current status
        """
        for order in _ORDER_LOG:
            if order["order_id"] == order_id:
                return json.dumps(order)
        return json.dumps({"error": f"Order {order_id} not found"})

    return run


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


@scorer(metrics=[accuracy()])
def function_call_scorer() -> Scorer:
    """Score based on whether the model correctly used or abstained from tools."""

    async def score(state: TaskState, target: Target) -> Score:
        target_val = target.text.strip().lower()

        # Check if model made any tool calls
        made_tool_call = False
        tool_names_called = []
        for msg in state.messages:
            if msg.role == "assistant":
                # Check tool_calls attribute (inspect-ai format)
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        made_tool_call = True
                        # In inspect-ai, tc.function is the tool name string
                        fn = getattr(tc, "function", "")
                        if isinstance(fn, str) and fn:
                            tool_names_called.append(fn)
                        elif isinstance(fn, dict):
                            tool_names_called.append(fn.get("name", ""))
                # Also check content blocks for tool_use type
                content = getattr(msg, "content", "")
                if isinstance(content, list):
                    for block in content:
                        if hasattr(block, "type") and getattr(block, "type", "") == "tool_use":
                            made_tool_call = True
                            name = getattr(block, "name", "")
                            if name:
                                tool_names_called.append(name)

        if target_val == "no_tool_call":
            # Model should NOT have called any tool
            if not made_tool_call:
                return Score(value=CORRECT, explanation="Correctly abstained from tool use")
            return Score(
                value=INCORRECT,
                explanation=f"Should have abstained but called: {tool_names_called}",
            )

        if target_val.startswith("called:"):
            # Model should have called a specific tool
            expected_tool = target_val.split(":", 1)[1].strip()
            if expected_tool in tool_names_called:
                return Score(value=CORRECT, explanation=f"Correctly called {expected_tool}")
            if made_tool_call:
                return Score(
                    value=INCORRECT,
                    explanation=f"Called {tool_names_called} instead of {expected_tool}",
                )
            return Score(value=INCORRECT, explanation=f"Should have called {expected_tool} but made no tool call")

        if target_val.startswith("contains:"):
            # Check if the final answer contains a specific string
            expected = target_val.split(":", 1)[1].strip()
            full_text = state.output.completion.lower()
            if expected.lower() in full_text:
                return Score(value=CORRECT, explanation=f"Output contains '{expected}'")
            return Score(
                value=INCORRECT,
                explanation=f"Output does not contain '{expected}'",
            )

        return Score(value=INCORRECT, explanation="Unknown target format")

    return score


# ---------------------------------------------------------------------------
# Dataset: Adaptive samples targeting discovered failure modes
# ---------------------------------------------------------------------------

NEAR_MISS_IRRELEVANCE_SAMPLES = [
    # Query asks about something close but different from available tools
    Sample(
        id="nearmiss_nutrient_vs_water",
        input="What nutrients does a succulent need to thrive indoors?",
        target="no_tool_call",
        metadata={
            "category": "near_miss_irrelevance",
            "explanation": "Tools include weather/air quality but nothing about plant nutrition",
            "failure_mode": "semantic_near_miss",
        },
    ),
    Sample(
        id="nearmiss_traffic_vs_distance",
        input="What's the current traffic situation on I-95 northbound?",
        target="no_tool_call",
        metadata={
            "category": "near_miss_irrelevance",
            "explanation": "calculate_distance exists but doesn't report traffic",
            "failure_mode": "semantic_near_miss",
        },
    ),
    Sample(
        id="nearmiss_recipe_vs_restaurant",
        input="Can you give me a recipe for homemade pasta?",
        target="no_tool_call",
        metadata={
            "category": "near_miss_irrelevance",
            "explanation": "search_restaurants is food-related but doesn't provide recipes",
            "failure_mode": "semantic_near_miss",
        },
    ),
    Sample(
        id="nearmiss_crypto_vs_stock",
        input="What's the current price of Bitcoin?",
        target="no_tool_call",
        metadata={
            "category": "near_miss_irrelevance",
            "explanation": "get_stock_price exists but Bitcoin isn't a stock ticker",
            "failure_mode": "semantic_near_miss",
        },
    ),
    Sample(
        id="nearmiss_health_vs_air",
        input="What are the symptoms of seasonal allergies?",
        target="no_tool_call",
        metadata={
            "category": "near_miss_irrelevance",
            "explanation": "air quality is allergy-adjacent but tool doesn't cover symptoms",
            "failure_mode": "semantic_near_miss",
        },
    ),
    Sample(
        id="nearmiss_history_vs_financials",
        input="When was Apple Inc. founded and by whom?",
        target="no_tool_call",
        metadata={
            "category": "near_miss_irrelevance",
            "explanation": "Company financials tool exists for AAPL but doesn't have founding info",
            "failure_mode": "semantic_near_miss",
        },
    ),
    Sample(
        id="nearmiss_rental_vs_distance",
        input="What's the average rent for a 2-bedroom apartment in Austin, TX?",
        target="no_tool_call",
        metadata={
            "category": "near_miss_irrelevance",
            "explanation": "Location-related tools exist but none cover rental prices",
            "failure_mode": "semantic_near_miss",
        },
    ),
    # Genuine tool-use cases (controls — model SHOULD call tools)
    Sample(
        id="genuine_weather",
        input="What's the weather forecast for Seattle this week?",
        target="called:get_weather_forecast",
        metadata={"category": "genuine_use", "explanation": "Directly matches available tool"},
    ),
    Sample(
        id="genuine_stock",
        input="What's MSFT stock trading at right now?",
        target="called:get_stock_price",
        metadata={"category": "genuine_use", "explanation": "Direct stock price query"},
    ),
    Sample(
        id="genuine_distance",
        input="How far is it to drive from Chicago to Detroit?",
        target="called:calculate_distance",
        metadata={"category": "genuine_use", "explanation": "Direct distance query"},
    ),
]

# Round 2: harder near-miss cases designed after discovering gpt-4o calls
# get_stock_price for Bitcoin (domain-adjacent confusion)
HARDER_NEAR_MISS_SAMPLES = [
    Sample(
        id="hard_nearmiss_etf_vs_stock",
        input="What's the expense ratio of the Vanguard S&P 500 ETF?",
        target="no_tool_call",
        metadata={
            "category": "hard_near_miss",
            "explanation": "get_stock_price/get_company_financials exist but don't return ETF expense ratios",
            "failure_mode": "domain_adjacent",
        },
    ),
    Sample(
        id="hard_nearmiss_stock_news_vs_price",
        input="What's the latest news about Tesla's CEO?",
        target="no_tool_call",
        metadata={
            "category": "hard_near_miss",
            "explanation": "Stock tools exist for TSLA but don't cover company news",
            "failure_mode": "domain_adjacent",
        },
    ),
    Sample(
        id="hard_nearmiss_forex_vs_stock",
        input="What's the current EUR/USD exchange rate?",
        target="no_tool_call",
        metadata={
            "category": "hard_near_miss",
            "explanation": "get_stock_price handles stocks not forex pairs",
            "failure_mode": "domain_adjacent",
        },
    ),
    Sample(
        id="hard_nearmiss_uv_vs_air",
        input="What's the UV index in Miami today? Should I wear sunscreen?",
        target="no_tool_call",
        metadata={
            "category": "hard_near_miss",
            "explanation": "get_air_quality is environmental but doesn't cover UV index",
            "failure_mode": "domain_adjacent",
        },
    ),
    Sample(
        id="hard_nearmiss_delivery_vs_distance",
        input="How long will my Amazon package take to arrive in Denver?",
        target="no_tool_call",
        metadata={
            "category": "hard_near_miss",
            "explanation": "calculate_distance computes driving distance, not delivery estimates",
            "failure_mode": "domain_adjacent",
        },
    ),
    Sample(
        id="hard_nearmiss_food_delivery_vs_restaurant",
        input="Can you order me a pizza from DoorDash?",
        target="no_tool_call",
        metadata={
            "category": "hard_near_miss",
            "explanation": "search_restaurants finds restaurants but can't place delivery orders",
            "failure_mode": "domain_adjacent",
        },
    ),
    Sample(
        id="hard_nearmiss_commodity_vs_stock",
        input="What's the current price of gold per ounce?",
        target="no_tool_call",
        metadata={
            "category": "hard_near_miss",
            "explanation": "get_stock_price handles stocks, not commodities",
            "failure_mode": "domain_adjacent",
        },
    ),
    Sample(
        id="hard_nearmiss_humidity_vs_weather",
        input="What's the indoor humidity level recommended for my houseplants?",
        target="no_tool_call",
        metadata={
            "category": "hard_near_miss",
            "explanation": "get_weather_forecast covers outdoor weather not indoor plant care",
            "failure_mode": "domain_adjacent",
        },
    ),
]


MULTI_STEP_SAMPLES = [
    Sample(
        id="multistep_check_then_reorder",
        input=(
            "Check the inventory for product SKU-A100. If it needs reordering "
            "(stock below 10), place a reorder for 50 units. Then confirm the "
            "order was placed by checking its status. Tell me the final order ID."
        ),
        target="contains:ORD-1001",
        metadata={
            "category": "multi_step_state",
            "explanation": "Requires 3 sequential tool calls with state dependency",
            "failure_mode": "state_tracking",
        },
    ),
    Sample(
        id="multistep_weather_then_restaurant",
        input=(
            "Check the weather in Portland. If it's going to rain in the next 3 days, "
            "find me an indoor-friendly Italian restaurant there instead of a picnic spot."
        ),
        target="called:search_restaurants",
        metadata={
            "category": "multi_step_conditional",
            "explanation": "Must check weather first, then conditionally call restaurant search",
            "failure_mode": "conditional_chaining",
        },
    ),
    Sample(
        id="multistep_stock_analysis",
        input=(
            "I'm considering investing in GOOGL. First check its current stock price, "
            "then look at its financial summary. Based on both, tell me the P/E ratio "
            "and whether the stock seems reasonably valued."
        ),
        target="contains:29.5",
        metadata={
            "category": "multi_step_synthesis",
            "explanation": "Must call two tools and synthesize results",
            "failure_mode": "result_synthesis",
        },
    ),
]

ALL_SAMPLES = NEAR_MISS_IRRELEVANCE_SAMPLES + HARDER_NEAR_MISS_SAMPLES + MULTI_STEP_SAMPLES


@task
def function_calling_robustness():
    """Eval targeting function-calling failure modes discovered from BFCL analysis."""
    near_miss_tools = [
        get_weather_forecast(),
        get_air_quality(),
        search_restaurants(),
        calculate_distance(),
        get_stock_price(),
        get_company_financials(),
    ]

    state_tools = [
        check_inventory(),
        place_reorder(),
        get_order_status(),
    ]

    all_tools = near_miss_tools + state_tools

    return Task(
        dataset=MemoryDataset(ALL_SAMPLES),
        solver=[use_tools(all_tools), generate()],
        scorer=function_call_scorer(),
        message_limit=15,
        token_limit=4096,
    )
