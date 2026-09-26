from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from backend.functions.weather import (
    get_current_weather,
    get_today_forecast,
    get_week_forecast,
    find_weather_extremes,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

MODEL = os.getenv(
    "ZOE_WEATHER_MODEL",
    "openai/gpt-oss-20b",
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY3"
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY3 is not set."
    )

client = Groq(
    api_key=GROQ_API_KEY
)


# Maximum LLM reasoning rounds.
MAX_AGENT_STEPS = int(
    os.getenv(
        "ZOE_WEATHER_MAX_STEPS",
        "5",
    )
)

# Maximum total weather tool executions.
MAX_TOOL_CALLS = int(
    os.getenv(
        "ZOE_WEATHER_MAX_TOOL_CALLS",
        "8",
    )
)

# Maximum previous conversation messages passed to the model.
MAX_CONTEXT_MESSAGES = int(
    os.getenv(
        "ZOE_WEATHER_MAX_CONTEXT_MESSAGES",
        "8",
    )
)

# Default weather location.
DEFAULT_LOCATION = os.getenv(
    "ZOE_DEFAULT_WEATHER_LOCATION",
    "Chittagong",
)


# ============================================================
# TOOLS
# ============================================================

TOOLS = {
    "get_current_weather": get_current_weather,
    "get_today_forecast": get_today_forecast,
    "get_week_forecast": get_week_forecast,
    "find_weather_extremes": find_weather_extremes,
}


# ============================================================
# GROQ TOOLS
# ============================================================

GROQ_TOOLS = [

    # ========================================================
    # CURRENT WEATHER
    # ========================================================

    {
        "type": "function",
        "function": {
            "name": "get_current_weather",

            "description": (
                "Get current weather conditions for a location. "
                "Use ONLY when the user is asking about weather "
                "right now or current conditions. "
                "Do not use this for tomorrow, future days, "
                "today's forecast, tonight, or this week."
            ),

            "parameters": {
                "type": "object",

                "properties": {

                    "location": {
                        "type": "string",

                        "description": (
                            "City, town, region, or location. "
                            f"If omitted, use {DEFAULT_LOCATION}."
                        ),
                    },
                },

                "required": [],
            },
        },
    },

    # ========================================================
    # TODAY FORECAST
    # ========================================================

    {
        "type": "function",
        "function": {
            "name": "get_today_forecast",

            "description": (
                "Get today's forecast and detailed hourly "
                "weather information. "
                "Use for questions about today, this morning, "
                "this afternoon, tonight, rain today, today's "
                "temperature, umbrella decisions for today, "
                "or other time-specific conditions today."
            ),

            "parameters": {
                "type": "object",

                "properties": {

                    "location": {
                        "type": "string",

                        "description": (
                            f"Weather location. "
                            f"If omitted, use {DEFAULT_LOCATION}."
                        ),
                    },
                },

                "required": [],
            },
        },
    },

    # ========================================================
    # WEEK FORECAST
    # ========================================================

    {
        "type": "function",
        "function": {
            "name": "get_week_forecast",

            "description": (
                "Get the seven-day weather forecast with "
                "detailed daily and hourly information. "
                "Use for tomorrow, future days, Friday, "
                "the weekend, next few days, or this week. "
                "Also use this when the user asks about a "
                "specific future day."
            ),

            "parameters": {
                "type": "object",

                "properties": {

                    "location": {
                        "type": "string",

                        "description": (
                            f"Weather location. "
                            f"If omitted, use {DEFAULT_LOCATION}."
                        ),
                    },
                },

                "required": [],
            },
        },
    },

    # ========================================================
    # WEATHER EXTREMES
    # ========================================================

    {
        "type": "function",
        "function": {
            "name": "find_weather_extremes",

            "description": (
                "Analyze the seven-day forecast and identify "
                "the hottest, coldest, wettest, and windiest "
                "days. "
                "Use when the user explicitly asks which day "
                "is hottest, coldest, wettest, rainiest, or "
                "windiest."
            ),

            "parameters": {
                "type": "object",

                "properties": {

                    "location": {
                        "type": "string",

                        "description": (
                            f"Weather location. "
                            f"If omitted, use {DEFAULT_LOCATION}."
                        ),
                    },
                },

                "required": [],
            },
        },
    },
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = f"""
You are ZOE's specialized Weather Agent.

Your job is to answer weather questions using real weather
data returned by the available weather tools.

You are NOT the main ZOE personality.

You must retrieve weather data when the question requires
current or forecast information.

Default location:
{DEFAULT_LOCATION}


============================================================
CORE RULES
============================================================

1. Never invent weather information.

2. Use the user's specified location when one is provided.

3. If no location is provided, use {DEFAULT_LOCATION}.

4. Use Celsius unless the user explicitly requests another unit.

5. Give concise, useful answers.

6. Do not dump raw JSON or every available weather metric.

7. Distinguish current conditions from forecasts.

8. Forecasts are predictions, not guarantees.

9. Never claim exact future conditions beyond what the
   returned forecast supports.

10. Never greet the user.


============================================================
TOOL SELECTION
============================================================

CURRENT WEATHER:

Use get_current_weather for:

- "What's the weather?"
- "What's it like outside?"
- "How hot is it?"
- "Is it raining right now?"
- "What's the temperature?"
- "How does it feel outside?"

These mean CURRENT conditions unless the user specifies
a future time.


TODAY:

Use get_today_forecast for:

- "What's the weather today?"
- "Will it rain today?"
- "Will it rain this afternoon?"
- "What's tonight looking like?"
- "How hot will it get today?"
- "Do I need an umbrella today?"

Use the hourly data when the user specifies a time period.


FUTURE DAYS:

Use get_week_forecast for:

- "What's tomorrow like?"
- "Will it rain tomorrow?"
- "What's Friday looking like?"
- "What about this weekend?"
- "What's the next few days like?"
- "What's the weather this week?"

If the user asks about a specific future day, use the
seven-day forecast and inspect that day.


EXTREMES:

Use find_weather_extremes for:

- "What's the hottest day?"
- "What's the coldest day?"
- "Which day will be rainiest?"
- "When will it be windiest?"

Do not manually estimate extremes when the dedicated
extremes tool is available.


============================================================
COMPARISONS
============================================================

If the user asks to compare multiple days or periods,
retrieve forecast data and perform the comparison.

Examples:

"Is Saturday hotter than Friday?"

"Which is better for going outside, Friday or Sunday?"

"Which day has the least rain?"

Use the relevant forecast data.

Do not invent missing values.


============================================================
ANSWERING
============================================================

After receiving weather data:

- Extract only information relevant to the question.
- Prefer temperature, feels-like temperature, rain,
  precipitation probability, wind, humidity, clouds,
  visibility, and UV when relevant.
- Mention timing when it materially affects the answer.
- Keep routine weather answers short.

Example:

User:
"What's the weather?"

Good:

"It's 29°C and partly cloudy in Chittagong. It feels like
33°C, with humid conditions and light winds."

Bad:

A giant dump containing every available API field.


For an umbrella question:

"Rain is possible tomorrow afternoon, with a 65% chance
during the period, so I'd take an umbrella."


============================================================
FAILURES
============================================================

If a weather tool returns an error:

- Do not fabricate weather.
- Explain briefly that current weather data could not be
  retrieved.
- If another tool can genuinely answer the same question,
  use it only when appropriate.
- Do not repeatedly call a failing tool without a reason.


============================================================
CONTEXT
============================================================

Conversation context may be provided by ZOE.

Use context only when it is relevant to resolving the
current weather request.

Do not assume an old location still applies if the user
explicitly specifies a new location.

Do not pretend to remember information that is not present
in the provided context.
"""


# ============================================================
# ARGUMENT NORMALIZATION
# ============================================================

def normalize_arguments(
    tool_name: str,
    arguments: Any,
) -> dict[str, Any]:
    """
    Validate and sanitize model-generated tool arguments.
    """

    # --------------------------------------------------------
    # Parse JSON if necessary.
    # --------------------------------------------------------

    if isinstance(arguments, str):

        raw = arguments.strip()

        if not raw:

            arguments = {}

        else:

            try:

                arguments = json.loads(
                    raw
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON arguments "
                    f"for tool '{tool_name}'."
                ) from exc

    # --------------------------------------------------------
    # None -> empty object.
    # --------------------------------------------------------

    if arguments is None:

        arguments = {}

    # --------------------------------------------------------
    # Arguments must be a dictionary.
    # --------------------------------------------------------

    if not isinstance(
        arguments,
        dict,
    ):

        raise ValueError(
            f"Arguments for tool '{tool_name}' "
            "must be an object."
        )

    # --------------------------------------------------------
    # Validate tool.
    # --------------------------------------------------------

    if tool_name not in TOOLS:

        raise ValueError(
            f"Unknown weather tool: "
            f"{tool_name}"
        )

    # --------------------------------------------------------
    # Only accepted argument currently is location.
    # --------------------------------------------------------

    cleaned: dict[str, Any] = {}

    if "location" in arguments:

        location = arguments[
            "location"
        ]

        if location is not None:

            if not isinstance(
                location,
                str,
            ):

                raise ValueError(
                    "Weather location must "
                    "be a string."
                )

            location = location.strip()

            if location:

                cleaned[
                    "location"
                ] = location

    # --------------------------------------------------------
    # Explicitly inject the default location.
    #
    # This makes the behavior deterministic instead of
    # relying entirely on the model to remember the default.
    # --------------------------------------------------------

    if "location" not in cleaned:

        cleaned[
            "location"
        ] = DEFAULT_LOCATION

    return cleaned


# ============================================================
# TOOL EXECUTION
# ============================================================

def execute_tool(
    tool_name: str,
    arguments: Any,
) -> Any:
    """
    Execute one weather tool safely.
    """

    function = TOOLS.get(
        tool_name
    )

    if function is None:

        return {
            "success": False,
            "error": (
                f"Unknown weather tool: "
                f"{tool_name}"
            ),
        }

    try:

        cleaned_arguments = (
            normalize_arguments(
                tool_name,
                arguments,
            )
        )

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
        }

    try:

        result = function(
            **cleaned_arguments
        )

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
        }

    # --------------------------------------------------------
    # Treat None as an actual tool failure.
    # --------------------------------------------------------

    if result is None:

        return {
            "success": False,
            "error": (
                f"Weather tool '{tool_name}' "
                "returned no data."
            ),
        }

    return result


# ============================================================
# RESULT STATUS
# ============================================================

def result_succeeded(
    result: Any,
) -> bool:
    """
    Determine whether a weather tool returned usable data.
    """

    if result is None:

        return False

    if isinstance(
        result,
        dict,
    ):

        if result.get(
            "success"
        ) is False:

            return False

    return True


# ============================================================
# CONTEXT CLEANING
# ============================================================

def clean_context(
    context: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """
    Keep only safe user/assistant conversation messages.
    """

    if not context:

        return []

    cleaned: list[
        dict[str, str]
    ] = []

    for message in context[
        -MAX_CONTEXT_MESSAGES:
    ]:

        if not isinstance(
            message,
            dict,
        ):

            continue

        role = message.get(
            "role"
        )

        content = message.get(
            "content"
        )

        if role not in {
            "user",
            "assistant",
        }:

            continue

        if not isinstance(
            content,
            str,
        ):

            continue

        content = content.strip()

        if not content:

            continue

        cleaned.append(
            {
                "role": role,
                "content": content,
            }
        )

    return cleaned


# ============================================================
# BUILD MESSAGES
# ============================================================

def build_messages(
    query: str,
    context: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build the model conversation.
    """

    messages: list[
        dict[str, Any]
    ] = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },

    ]

    messages.extend(
        clean_context(context)
    )

    messages.append(
        {
            "role": "user",
            "content": query.strip(),
        }
    )

    return messages


# ============================================================
# SERIALIZE TOOL RESULT
# ============================================================

def serialize_tool_result(
    result: Any,
) -> str:
    """
    Serialize tool data compactly for the model.
    """

    try:

        return json.dumps(
            result,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
            default=str,
        )

    except Exception:

        return str(result)


# ============================================================
# TOOL CALL SIGNATURE
# ============================================================

def tool_call_signature(
    tool_name: str,
    arguments: Any,
) -> str:
    """
    Create a stable signature for duplicate-call detection.
    """

    try:

        normalized = (
            normalize_arguments(
                tool_name,
                arguments,
            )
        )

        return (
            tool_name
            + ":"
            + json.dumps(
                normalized,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
        )

    except Exception:

        return (
            tool_name
            + ":"
            + str(arguments)
        )


# ============================================================
# WEATHER AGENT
# ============================================================

class WeatherAgent:
    """
    Specialized weather reasoning agent.

    The agent:

    - receives the current request
    - optionally receives limited context
    - chooses the appropriate weather tool
    - executes the tool
    - analyzes returned data
    - produces the final concise answer
    """

    def __init__(
        self,
        model: str = MODEL,
    ):

        self.model = model

    # ========================================================
    # RUN
    # ========================================================

    def run(
        self,
        query: str,
        context: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Execute a weather request.
        """

        # ----------------------------------------------------
        # Validate query.
        # ----------------------------------------------------

        if (
            not query
            or not query.strip()
        ):

            return (
                "I need a weather question to "
                "check."
            )

        messages = build_messages(
            query=query,
            context=context,
        )

        tool_calls_used = 0

        steps_used = 0

        previous_calls: set[
            str
        ] = set()

        last_tool_error: str | None = None

        # ====================================================
        # AGENT LOOP
        # ====================================================

        while (
            steps_used
            < MAX_AGENT_STEPS
        ):

            steps_used += 1

            print()
            print(
                "[Weather Agent]"
                f" Step {steps_used}/"
                f"{MAX_AGENT_STEPS}"
            )

            # ------------------------------------------------
            # Model request.
            # ------------------------------------------------

            try:

                response = (
                    client
                    .chat
                    .completions
                    .create(

                        model=self.model,

                        messages=messages,

                        tools=GROQ_TOOLS,

                        tool_choice="auto",

                        temperature=0.1,

                        max_completion_tokens=1000,
                    )
                )

            except Exception as exc:

                print(
                    "[Weather Agent]"
                    f" Model error: {exc}"
                )

                if last_tool_error:

                    return (
                        "I couldn't retrieve "
                        "the weather data right now."
                    )

                return (
                    "I couldn't retrieve "
                    "the weather information "
                    "right now."
                )

            # ------------------------------------------------
            # Validate response.
            # ------------------------------------------------

            try:

                message = (
                    response
                    .choices[0]
                    .message
                )

            except Exception as exc:

                print(
                    "[Weather Agent]"
                    f" Invalid response: {exc}"
                )

                return (
                    "I couldn't process the "
                    "weather response."
                )

            tool_calls = (
                message.tool_calls
                or []
            )

            # =================================================
            # FINAL ANSWER
            # =================================================

            if not tool_calls:

                content = (
                    message.content
                    or ""
                ).strip()

                if content:

                    return content

                if last_tool_error:

                    return (
                        "I couldn't retrieve "
                        "reliable weather data "
                        "right now."
                    )

                return (
                    "I couldn't determine "
                    "the weather information "
                    "you need."
                )

            # =================================================
            # TOOL BUDGET CHECK
            # =================================================

            remaining_budget = (
                MAX_TOOL_CALLS
                - tool_calls_used
            )

            if remaining_budget <= 0:

                return (
                    "I couldn't complete "
                    "the weather analysis."
                )

            # ------------------------------------------------
            # Preserve assistant tool calls.
            # ------------------------------------------------

            assistant_message: dict[
                str,
                Any,
            ] = {

                "role": "assistant",

                "content": (
                    message.content
                    or ""
                ),

                "tool_calls": [],
            }

            for tool_call in tool_calls:

                assistant_message[
                    "tool_calls"
                ].append(
                    {
                        "id": tool_call.id,

                        "type": "function",

                        "function": {
                            "name": (
                                tool_call
                                .function
                                .name
                            ),

                            "arguments": (
                                tool_call
                                .function
                                .arguments
                            ),
                        },
                    }
                )

            messages.append(
                assistant_message
            )

            # =================================================
            # EXECUTE TOOL CALLS
            # =================================================

            for tool_call in tool_calls:

                # --------------------------------------------
                # Enforce global budget.
                # --------------------------------------------

                if (
                    tool_calls_used
                    >= MAX_TOOL_CALLS
                ):

                    return (
                        "I couldn't complete "
                        "the weather analysis."
                    )

                tool_calls_used += 1

                tool_name = (
                    tool_call
                    .function
                    .name
                )

                raw_arguments = (
                    tool_call
                    .function
                    .arguments
                )

                print(
                    "[Weather Agent]"
                    f" Tool #{tool_calls_used}: "
                    f"{tool_name}"
                )

                print(
                    "[Weather Agent]"
                    f" Arguments: "
                    f"{raw_arguments}"
                )

                # --------------------------------------------
                # Detect duplicate tool calls.
                # --------------------------------------------

                signature = (
                    tool_call_signature(
                        tool_name,
                        raw_arguments,
                    )
                )

                if signature in previous_calls:

                    result = {
                        "success": False,
                        "error": (
                            "The same weather "
                            "tool call was already "
                            "executed during this "
                            "request."
                        ),
                    }

                    last_tool_error = (
                        result["error"]
                    )

                    print(
                        "[Weather Agent]"
                        " Duplicate tool call blocked."
                    )

                else:

                    previous_calls.add(
                        signature
                    )

                    # ----------------------------------------
                    # Execute.
                    # ----------------------------------------

                    result = execute_tool(
                        tool_name,
                        raw_arguments,
                    )

                    if not result_succeeded(
                        result
                    ):

                        if isinstance(
                            result,
                            dict,
                        ):

                            last_tool_error = (
                                str(
                                    result.get(
                                        "error",
                                        "Weather tool failed.",
                                    )
                                )
                            )

                        else:

                            last_tool_error = (
                                "Weather tool failed."
                            )

                    else:

                        last_tool_error = None

                    print(
                        "[Weather Agent]"
                        f" Result: {result}"
                    )

                # --------------------------------------------
                # Return result to model.
                # --------------------------------------------

                messages.append(
                    {
                        "role": "tool",

                        "tool_call_id": (
                            tool_call.id
                        ),

                        "content": (
                            serialize_tool_result(
                                result
                            )
                        ),
                    }
                )

        # ====================================================
        # STEP LIMIT
        # ====================================================

        return (
            "I couldn't complete the "
            "weather analysis."
        )


# ============================================================
# SINGLETON
# ============================================================

weather_agent = WeatherAgent()


# ============================================================
# PUBLIC ZOE INTERFACE
# ============================================================

def run_weather_agent(
    query: str,
    context: list[dict[str, Any]] | None = None,
) -> str:
    """
    Public interface used by ZOE Runtime.
    """

    return weather_agent.run(
        query=query,
        context=context,
    )


# ============================================================
# STANDALONE CLI
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print(
        "                 ZOE WEATHER AGENT"
    )
    print("=" * 70)
    print()

    print(
        f"Model: {MODEL}"
    )

    print(
        "Groq API: key 1"
    )

    print(
        f"Default location: "
        f"{DEFAULT_LOCATION}"
    )

    print(
        f"Max steps: "
        f"{MAX_AGENT_STEPS}"
    )

    print(
        f"Max tool calls: "
        f"{MAX_TOOL_CALLS}"
    )

    print()

    print(
        "Examples:"
    )

    print(
        "  What's the weather?"
    )

    print(
        "  Will it rain this afternoon?"
    )

    print(
        "  What's Friday looking like?"
    )

    print(
        "  What's the hottest day this week?"
    )

    print()

    print(
        "Type 'exit' or 'quit' to stop."
    )

    print()

    while True:

        try:

            query = input(
                "You: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\nExiting..."
            )

            break

        except EOFError:

            print(
                "\nExiting..."
            )

            break

        if not query:

            continue

        if query.lower() in {
            "exit",
            "quit",
        }:

            break

        try:

            response = (
                run_weather_agent(
                    query
                )
            )

            print(
                f"ZOE: {response}"
            )

            print()

        except Exception as exc:

            print(
                f"ERROR: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            print()