
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from groq import Groq

from backend.functions.gcal import (
    get_today_events,
    get_events,
    add_event,
    delete_event,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# CONFIG
# ============================================================

MODEL = os.getenv(
    "ZOE_CALENDAR_MODEL",
    "openai/gpt-oss-120b",
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY3",
)

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY3 is not set."
    )

client = Groq(
    api_key=GROQ_API_KEY,
)

TIMEZONE = "Asia/Dhaka"

ZOE_TIMEZONE = ZoneInfo(
    TIMEZONE
)

MAX_AGENT_STEPS = int(
    os.getenv(
        "ZOE_CALENDAR_MAX_STEPS",
        "8",
    )
)

MAX_TOOL_CALLS = int(
    os.getenv(
        "ZOE_CALENDAR_MAX_TOOL_CALLS",
        "20",
    )
)

FUZZY_MATCH_THRESHOLD = 0.55
AMBIGUITY_MARGIN = 0.12


# ============================================================
# TIME
# ============================================================

def get_current_datetime() -> datetime:
    return datetime.now(
        ZOE_TIMEZONE
    )


def get_current_datetime_string() -> str:
    return get_current_datetime().isoformat(
        timespec="seconds"
    )


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(
    text: str,
) -> str:

    if not text:
        return ""

    text = str(
        text
    ).lower().strip()

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
    )

    stop_words = {
        "my",
        "the",
        "a",
        "an",
        "event",
        "events",
        "appointment",
        "calendar",
        "meeting",
        "session",
        "please",
        "can",
        "you",
        "could",
        "would",
        "remove",
        "delete",
        "cancel",
        "clear",
        "from",
        "today",
        "tomorrow",
        "thing",
        "things",
        "one",
    }

    return " ".join(
        word
        for word in text.split()
        if word not in stop_words
    )


# ============================================================
# SIMILARITY
# ============================================================

def similarity(
    a: str,
    b: str,
) -> float:

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


# ============================================================
# EVENT MATCHING
# ============================================================

def event_matches_query(
    query: str,
    event: dict[str, Any],
) -> float:

    event_title = str(
        event.get(
            "title",
            "",
        )
        or ""
    )

    if not event_title:
        return 0.0

    q = normalize_text(
        query
    )

    title = normalize_text(
        event_title
    )

    if not q or not title:
        return 0.0

    if q == title:
        return 1.0

    if q in title:
        return 0.90

    if title in q:
        return 0.85

    query_words = set(
        q.split()
    )

    title_words = set(
        title.split()
    )

    if query_words and title_words:

        overlap = (
            len(
                query_words
                & title_words
            )
            / len(query_words)
        )

        if overlap >= 0.75:
            return max(
                0.75,
                similarity(q, title),
            )

        if overlap >= 0.50:
            return max(
                0.60,
                similarity(q, title),
            )

    return similarity(
        q,
        title,
    )


# ============================================================
# EVENT ID
# ============================================================

def get_event_id(
    event: dict[str, Any],
) -> str | None:

    for key in (
        "event_id",
        "id",
        "google_event_id",
    ):

        value = event.get(
            key
        )

        if value:
            return str(value)

    return None


# ============================================================
# EVENT LINK
# ============================================================

def get_event_link(
    result: Any,
) -> str | None:

    if not isinstance(
        result,
        dict,
    ):
        return None

    for key in (
        "htmlLink",
        "html_link",
        "url",
        "link",
    ):

        value = result.get(
            key
        )

        if value:
            value = str(
                value
            ).strip()

            if value:
                return value

    for key in (
        "event",
        "result",
        "data",
    ):

        nested = result.get(
            key
        )

        link = get_event_link(
            nested
        )

        if link:
            return link

    return None


# ============================================================
# EVENT DISPLAY
# ============================================================

def event_summary(
    event: dict[str, Any],
) -> dict[str, Any]:

    summary = {
        "id": event.get("id"),
        "title": event.get("title"),
        "time": event.get("time"),
        "datetime": event.get("datetime"),
        "end_datetime": event.get(
            "end_datetime"
        ),
        "status": event.get(
            "status"
        ),
        "calendar": event.get(
            "calendar"
        ),
    }

    link = get_event_link(
        event
    )

    if link:
        summary["url"] = link

    return summary


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def normalize_event_result(
    result: Any,
) -> dict[str, Any]:

    if result is None:
        return {
            "success": True,
            "events": [],
        }

    if isinstance(
        result,
        dict,
    ):
        return result

    if isinstance(
        result,
        list,
    ):
        return {
            "success": True,
            "events": result,
        }

    return {
        "success": True,
        "events": result,
    }


# ============================================================
# TOOL: TODAY
# ============================================================

def tool_get_today_events() -> dict[str, Any]:
    """
    Fetch today's personal calendar events.
    """

    return normalize_event_result(
        get_today_events(
            "personal"
        )
    )


# ============================================================
# TOOL: DATE RANGE
# ============================================================

def tool_get_events(
    start_datetime: str,
    end_datetime: str,
) -> dict[str, Any]:
    """
    Fetch personal calendar events in an arbitrary
    timezone-aware datetime range.
    """

    if not start_datetime:
        raise ValueError(
            "start_datetime is required."
        )

    if not end_datetime:
        raise ValueError(
            "end_datetime is required."
        )

    try:
        start = datetime.fromisoformat(
            start_datetime
        )

        end = datetime.fromisoformat(
            end_datetime
        )

    except ValueError as exc:

        raise ValueError(
            "start_datetime and end_datetime "
            "must be valid ISO-8601 datetimes."
        ) from exc

    if start.tzinfo is None:
        raise ValueError(
            "start_datetime must include a timezone."
        )

    if end.tzinfo is None:
        raise ValueError(
            "end_datetime must include a timezone."
        )

    if end <= start:
        raise ValueError(
            "end_datetime must be after "
            "start_datetime."
        )

    return normalize_event_result(
        get_events(
            start_datetime=start,
            end_datetime=end,
            calendar="personal",
        )
    )


# ============================================================
# TOOL: ADD
# ============================================================

def tool_add_event(
    title: str,
    start_datetime: str,
    end_datetime: str | None = None,
    description: str | None = None,
) -> Any:

    if not title or not str(title).strip():
        raise ValueError(
            "title is required."
        )

    if not start_datetime:
        raise ValueError(
            "start_datetime is required."
        )

    try:
        start = datetime.fromisoformat(
            str(start_datetime).strip()
        )

    except ValueError as exc:

        raise ValueError(
            "Invalid start_datetime. "
            "Expected timezone-aware ISO-8601."
        ) from exc

    if start.tzinfo is None:
        raise ValueError(
            "start_datetime must include "
            "a timezone."
        )

    end = None

    if end_datetime:

        try:
            end = datetime.fromisoformat(
                str(end_datetime).strip()
            )

        except ValueError as exc:

            raise ValueError(
                "Invalid end_datetime. "
                "Expected timezone-aware ISO-8601."
            ) from exc

        if end.tzinfo is None:
            raise ValueError(
                "end_datetime must include "
                "a timezone."
            )

        if end <= start:
            raise ValueError(
                "end_datetime must be after "
                "start_datetime."
            )

    return add_event(
        title=str(title).strip(),
        start_datetime=start,
        end_datetime=end,
        description=description,
    )


# ============================================================
# TOOL: DELETE
# ============================================================

def tool_delete_event(
    query: str,
) -> dict[str, Any]:

    if not query or not str(query).strip():
        raise ValueError(
            "query is required."
        )

    query = str(
        query
    ).strip()

    raw_events = get_today_events(
        "personal"
    )

    normalized = normalize_event_result(
        raw_events
    )

    events = normalized.get(
        "events",
        [],
    )

    if not events:

        return {
            "success": False,
            "deleted": False,
            "message": (
                "There are no personal "
                "calendar events today."
            ),
        }

    scored_events = []

    for event in events:

        if not isinstance(
            event,
            dict,
        ):
            continue

        score = event_matches_query(
            query,
            event,
        )

        scored_events.append(
            (
                score,
                event,
            )
        )

    if not scored_events:

        return {
            "success": False,
            "deleted": False,
            "message": (
                "No usable events were found."
            ),
        }

    scored_events.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_score, best_event = (
        scored_events[0]
    )

    if best_score < FUZZY_MATCH_THRESHOLD:

        return {
            "success": False,
            "deleted": False,
            "ambiguous": False,
            "message": (
                f"No event matched "
                f"{query!r} with sufficient confidence."
            ),
            "available_events": [
                event_summary(event)
                for _, event in scored_events
            ],
        }

    plausible = [
        (score, event)
        for score, event
        in scored_events
        if score >= FUZZY_MATCH_THRESHOLD
    ]

    if len(plausible) > 1:

        second_score, _ = plausible[1]

        if (
            best_score - second_score
            < AMBIGUITY_MARGIN
        ):

            return {
                "success": False,
                "deleted": False,
                "ambiguous": True,
                "message": (
                    f"Multiple events could "
                    f"match {query!r}."
                ),
                "matches": [
                    event_summary(event)
                    for _, event in plausible
                ],
            }

    event_id = get_event_id(
        best_event
    )

    if not event_id:

        return {
            "success": False,
            "deleted": False,
            "message": (
                "A matching event was found, "
                "but its Google Calendar ID "
                "could not be determined."
            ),
            "matched_event":
                event_summary(
                    best_event
                ),
        }

    delete_result = delete_event(
        event_id,
        calendar="personal",
    )

    if isinstance(
        delete_result,
        dict,
    ):

        if (
            delete_result.get(
                "success"
            )
            is False
        ):

            return {
                "success": False,
                "deleted": False,
                "event_id": event_id,
                "title": best_event.get(
                    "title"
                ),
                "delete_result":
                    delete_result,
            }

    result = {
        "success": True,
        "deleted": True,
        "event_id": event_id,
        "title": best_event.get(
            "title"
        ),
        "time": best_event.get(
            "time"
        ),
        "datetime": best_event.get(
            "datetime"
        ),
    }

    link = get_event_link(
        delete_result
    )

    if link:
        result["url"] = link

    return result


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOLS = {
    "get_today_events":
        tool_get_today_events,

    "get_events":
        tool_get_events,

    "add_event":
        tool_add_event,

    "delete_event":
        tool_delete_event,
}


# ============================================================
# GROQ TOOLS
# ============================================================

GROQ_TOOLS = [

    {
        "type": "function",

        "function": {

            "name":
                "get_today_events",

            "description": (
                "Retrieve all PERSONAL calendar events "
                "occurring TODAY."
            ),

            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },

    {
        "type": "function",

        "function": {

            "name":
                "get_events",

            "description": (
                "Retrieve PERSONAL calendar events "
                "within a specified datetime range. "
                "Use this for dates other than today, "
                "including tomorrow, a specific date, "
                "or a future date range."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "start_datetime": {
                        "type": "string",
                        "description": (
                            "Timezone-aware ISO-8601 "
                            "datetime using +06:00."
                        ),
                    },

                    "end_datetime": {
                        "type": "string",
                        "description": (
                            "Timezone-aware ISO-8601 "
                            "datetime using +06:00."
                        ),
                    },
                },

                "required": [
                    "start_datetime",
                    "end_datetime",
                ],
            },
        },
    },

    {
        "type": "function",

        "function": {

            "name":
                "add_event",

            "description": (
                "Create ONE event on the PERSONAL "
                "Google Calendar. "
                "start_datetime must be timezone-aware "
                "ISO-8601 using +06:00. "
                "Do not invent an end time."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "title": {
                        "type": "string",
                    },

                    "start_datetime": {
                        "type": "string",
                    },

                    "end_datetime": {
                        "type": "string",
                    },

                    "description": {
                        "type": "string",
                    },
                },

                "required": [
                    "title",
                    "start_datetime",
                ],
            },
        },
    },

    {
        "type": "function",

        "function": {

            "name":
                "delete_event",

            "description": (
                "Delete ONE PERSONAL calendar event "
                "from TODAY by natural-language description. "
                "Never guess when the match is ambiguous."
            ),

            "parameters": {

                "type": "object",

                "properties": {

                    "query": {
                        "type": "string",
                    },
                },

                "required": [
                    "query",
                ],
            },
        },
    },
]


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's Personal Calendar Action Agent.

You control ONLY the PERSONAL Google Calendar.

You are an EXECUTION AGENT, not a conversational assistant.


============================================================
CURRENT TIME
============================================================

Current datetime:
{current_datetime}

Current date:
{current_date}

Current time:
{current_time}

Current day:
{current_day}

Timezone:
Asia/Dhaka / UTC+06:00


============================================================
AVAILABLE CAPABILITIES
============================================================

You can:

- inspect today's personal events
- inspect personal events over arbitrary date ranges
- create personal events
- delete personal events from today


============================================================
CALENDAR SEPARATION
============================================================

This agent is ONLY for the personal calendar.

Do NOT access:

- football
- F1

Those calendars have separate read-only agents.


============================================================
DATE INTERPRETATION
============================================================

Convert natural-language dates using the supplied current date.

Examples:

today
tomorrow
Friday
next Friday
next week

must become concrete dates.

Always use +06:00.


============================================================
CREATION
============================================================

Preserve the user's title.

Do not invent:

- attendees
- location
- description
- duration
- end time

If the user says:

"Meeting tomorrow at 5"

create:

title = Meeting
start = tomorrow at 5 PM

Do not invent an end time.


============================================================
MULTI-STEP EXECUTION
============================================================

Complete the entire objective.

If the user requests:

"Add physics at 5 and math at 7"

create both.

If one independent operation fails,
continue with the others.


============================================================
DELETION
============================================================

Deletion operates only on TODAY'S personal events.

If multiple events are requested:

"Delete physics and chemistry"

attempt them independently.

If a match is ambiguous,
do not guess.

If no sufficiently strong match exists,
do not guess.


============================================================
NO INVENTION
============================================================

Never invent:

- event IDs
- event titles
- dates
- times
- URLs
- calendar state


============================================================
NO CONVERSATIONAL FILLER
============================================================

You are not the final conversational layer.

Use tools and finish the requested operation.

Do not produce unnecessary conversational text.
"""


# ============================================================
# PROMPT
# ============================================================

def build_system_prompt() -> str:

    now = get_current_datetime()

    return SYSTEM_PROMPT.format(
        current_datetime=now.isoformat(
            timespec="seconds"
        ),
        current_date=now.strftime(
            "%Y-%m-%d"
        ),
        current_time=now.strftime(
            "%H:%M:%S"
        ),
        current_day=now.strftime(
            "%A"
        ),
    )


def build_messages(
    user_request: str,
) -> list[dict[str, Any]]:

    return [
        {
            "role": "system",
            "content":
                build_system_prompt(),
        },
        {
            "role": "user",
            "content":
                user_request.strip(),
        },
    ]


# ============================================================
# ARGUMENT NORMALIZATION
# ============================================================

def normalize_arguments(
    name: str,
    arguments: Any,
) -> dict[str, Any]:

    if arguments is None:
        arguments = {}

    if isinstance(
        arguments,
        str,
    ):

        arguments = arguments.strip()

        if not arguments:
            arguments = {}

        else:

            try:
                arguments = json.loads(
                    arguments
                )

            except json.JSONDecodeError as exc:

                raise ValueError(
                    f"Invalid JSON arguments "
                    f"for tool {name!r}."
                ) from exc

    if not isinstance(
        arguments,
        dict,
    ):
        raise ValueError(
            f"Arguments for {name!r} "
            "must be a JSON object."
        )

    allowed_arguments = {

        "get_today_events":
            set(),

        "get_events": {
            "start_datetime",
            "end_datetime",
        },

        "add_event": {
            "title",
            "start_datetime",
            "end_datetime",
            "description",
        },

        "delete_event": {
            "query",
        },
    }

    if name not in allowed_arguments:
        raise ValueError(
            f"Unknown calendar tool: {name!r}"
        )

    unknown = [
        key
        for key in arguments
        if key not in allowed_arguments[name]
    ]

    if unknown:
        raise ValueError(
            f"Unknown argument(s) for {name!r}: "
            f"{', '.join(map(str, unknown))}"
        )

    required_arguments = {

        "get_today_events":
            set(),

        "get_events": {
            "start_datetime",
            "end_datetime",
        },

        "add_event": {
            "title",
            "start_datetime",
        },

        "delete_event": {
            "query",
        },
    }

    missing = [
        key
        for key in required_arguments[name]
        if (
            key not in arguments
            or arguments[key] is None
            or (
                isinstance(
                    arguments[key],
                    str,
                )
                and not arguments[key].strip()
            )
        )
    ]

    if missing:
        raise ValueError(
            f"Missing required argument(s) "
            f"for {name!r}: "
            f"{', '.join(missing)}"
        )

    return dict(arguments)


# ============================================================
# EXECUTE TOOL
# ============================================================

def execute_tool(
    name: str,
    arguments: Any,
) -> Any:

    if name not in TOOLS:
        raise ValueError(
            f"Unknown calendar tool: {name!r}"
        )

    cleaned = normalize_arguments(
        name,
        arguments,
    )

    return TOOLS[name](
        **cleaned
    )


# ============================================================
# LINKS
# ============================================================

def collect_operation_links(
    operations: list[dict[str, Any]],
) -> list[str]:

    links = []

    for operation in operations:

        link = operation.get(
            "url"
        )

        if not link:
            link = get_event_link(
                operation.get(
                    "result"
                )
            )

        if link and link not in links:
            links.append(
                str(link)
            )

    return links


# ============================================================
# FINAL RESULT
# ============================================================

def build_result(
    operations: list[dict[str, Any]],
) -> dict[str, Any]:

    result = {
        "success": bool(
            operations
        )
        and all(
            operation.get(
                "success",
                False,
            )
            for operation
            in operations
        ),

        "action": (
            "single_step"
            if len(operations) == 1
            else "multi_step"
        ),

        "operations":
            operations,
    }

    links = collect_operation_links(
        operations
    )

    if len(links) == 1:
        result["url"] = links[0]

    elif len(links) > 1:
        result["urls"] = links

    return result


# ============================================================
# AGENT
# ============================================================

class CalendarAgent:

    def __init__(
        self,
        model: str = MODEL,
    ):
        self.model = model

    def run(
        self,
        user_request: str,
    ) -> Any:

        if not user_request.strip():

            return {
                "success": False,
                "error":
                    "Calendar request "
                    "cannot be empty.",
            }

        messages = build_messages(
            user_request
        )

        operations = []

        total_tool_calls = 0

        for step in range(
            1,
            MAX_AGENT_STEPS + 1,
        ):

            print(
                f"[Calendar Agent] "
                f"Step {step}/{MAX_AGENT_STEPS}"
            )

            try:

                response = (
                    client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=GROQ_TOOLS,
                        tool_choice="auto",
                        temperature=0,
                        max_completion_tokens=1600,
                    )
                )

            except Exception as exc:

                return {
                    "success": False,
                    "error":
                        f"Groq request failed: {exc}",
                    "operations":
                        operations,
                }

            if not response.choices:

                return {
                    "success": False,
                    "error":
                        "Invalid model response.",
                    "operations":
                        operations,
                }

            message = response.choices[0].message

            tool_calls = (
                getattr(
                    message,
                    "tool_calls",
                    None,
                )
                or []
            )

            if not tool_calls:

                if operations:
                    return build_result(
                        operations
                    )

                return {
                    "success": False,
                    "error":
                        "Calendar Agent did not "
                        "produce a tool call.",
                    "model_response":
                        (
                            message.content
                            or ""
                        ).strip(),
                }

            assistant_message = {
                "role": "assistant",
                "content":
                    message.content or "",
                "tool_calls": [],
            }

            for tool_call in tool_calls:

                assistant_message[
                    "tool_calls"
                ].append(
                    {
                        "id":
                            tool_call.id,
                        "type":
                            "function",
                        "function": {
                            "name":
                                tool_call.function.name,
                            "arguments":
                                tool_call.function.arguments,
                        },
                    }
                )

            messages.append(
                assistant_message
            )

            for tool_call in tool_calls:

                if (
                    total_tool_calls
                    >= MAX_TOOL_CALLS
                ):

                    return {
                        "success": False,
                        "action":
                            "tool_limit",
                        "error":
                            "Maximum tool calls reached.",
                        "operations":
                            operations,
                    }

                total_tool_calls += 1

                name = (
                    tool_call.function.name
                )

                raw_arguments = (
                    tool_call.function.arguments
                )

                print(
                    f"[Calendar Agent] "
                    f"Tool #{total_tool_calls}: "
                    f"{name}"
                )

                try:

                    arguments = (
                        normalize_arguments(
                            name,
                            raw_arguments,
                        )
                    )

                    result = execute_tool(
                        name,
                        arguments,
                    )

                    success = not (
                        isinstance(
                            result,
                            dict,
                        )
                        and result.get(
                            "success"
                        )
                        is False
                    )

                    operation = {
                        "tool": name,
                        "arguments":
                            arguments,
                        "success":
                            success,
                        "result":
                            result,
                    }

                    link = get_event_link(
                        result
                    )

                    if link:
                        operation["url"] = link

                    operations.append(
                        operation
                    )

                except Exception as exc:

                    result = {
                        "success": False,
                        "error": str(exc),
                    }

                    operations.append(
                        {
                            "tool": name,
                            "arguments":
                                raw_arguments,
                            "success": False,
                            "error": str(exc),
                        }
                    )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id":
                            tool_call.id,
                        "content":
                            json.dumps(
                                result,
                                ensure_ascii=False,
                                default=str,
                            ),
                    }
                )

        return {
            "success": False,
            "action":
                "max_steps",
            "error":
                "Calendar Agent reached "
                "maximum reasoning steps.",
            "operations":
                operations,
        }


# ============================================================
# SINGLETON
# ============================================================

calendar_agent = CalendarAgent()


# ============================================================
# ZOE INTERFACE
# ============================================================

def run_calendar_agent(
    query: str,
) -> Any:

    return calendar_agent.run(
        user_request=query
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "ZOE Personal Calendar Agent"
    )

    while True:

        try:

            query = input(
                "\nYou > "
            ).strip()

            if query.lower() in {
                "exit",
                "quit",
            }:
                break

            if not query:
                continue

            result = run_calendar_agent(
                query
            )

            print(
                json.dumps(
                    result,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )

        except KeyboardInterrupt:
            break

        except Exception as exc:

            print(
                f"ERROR: {exc}"
            )
