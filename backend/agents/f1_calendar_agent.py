from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from groq import Groq

from backend.functions.gcal import (
    get_today_events,
    get_events,
    get_next_f1_race,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

MODEL = os.getenv(
    "ZOE_F1_CALENDAR_MODEL",
    os.getenv(
        "ZOE_CALENDAR_MODEL",
        "openai/gpt-oss-120b",
    ),
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


# ============================================================
# CONFIGURATION
# ============================================================

TIMEZONE = "Asia/Dhaka"

ZOE_TIMEZONE = ZoneInfo(
    TIMEZONE
)

MAX_COMPLETION_TOKENS = 150

# Number of upcoming F1 events to return.
NEXT_EVENT_COUNT = 5

# How far into the future to search.
# 120 days gives enough room around gaps in the F1 calendar.
EVENT_LOOKAHEAD_DAYS = 120


# ============================================================
# TIME
# ============================================================

def now() -> datetime:
    return datetime.now(
        ZOE_TIMEZONE
    )


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize(
    value: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        value.strip().lower(),
    )


# ============================================================
# COMPACT OUTPUT
# ============================================================

def _compact_event(
    event: dict[str, Any],
) -> dict[str, Any]:

    return {
        "id": event.get("id"),
        "title": event.get("title"),
        "datetime": event.get("datetime"),
        "end_datetime": event.get(
            "end_datetime"
        ),
        "time": event.get("time"),
        "location": event.get(
            "location"
        ),
        "status": event.get(
            "status"
        ),
        "url": event.get(
            "html_link"
        ),
    }


# ============================================================
# DIRECT FAST PATHS
# ============================================================

def _is_next_race(
    query: str,
) -> bool:

    q = _normalize(query)

    return q in {
        "next race",
        "next f1 race",
        "what is the next race",
        "what's the next race",
        "when is the next race",
        "what is the next f1 race",
        "what's the next f1 race",
        "when is the next f1 race",
    }


def _is_today(
    query: str,
) -> bool:

    q = _normalize(query)

    return q in {
        "today",
        "f1 today",
        "today's f1",
        "todays f1",
        "f1 events today",
        "today's race",
        "todays race",
    }


def _next_race() -> dict[str, Any]:

    event = get_next_f1_race()

    if event is None:

        return {
            "success": False,
            "calendar": "f1",
            "operation": "next_race",
            "error":
                "No upcoming F1 race found.",
        }

    race = _compact_event(
        event
    )

    return {
        "success": True,
        "calendar": "f1",
        "operation": "next_race",
        "race": race,
    }


def _today() -> dict[str, Any]:

    events = get_today_events(
        "f1"
    )

    if events is None:
        events = []

    if isinstance(events, dict):
        events = events.get(
            "events",
            [],
        )

    return {
        "success": True,
        "calendar": "f1",
        "operation": "today",
        "events": [
            _compact_event(event)
            for event in events
        ],
    }


# ============================================================
# NEXT 5 F1 EVENTS
# ============================================================

def _next_events(
    count: int = NEXT_EVENT_COUNT,
) -> dict[str, Any]:

    count = max(
        1,
        min(
            int(count),
            NEXT_EVENT_COUNT,
        ),
    )

    current = now()

    end = current + timedelta(
        days=EVENT_LOOKAHEAD_DAYS
    )

    try:

        events = get_events(
            start_datetime=current,
            end_datetime=end,
            calendar="f1",
        )

    except Exception as exc:

        return {
            "success": False,
            "calendar": "f1",
            "operation": "next_events",
            "events": [],
            "error": str(exc),
        }

    if not events:

        return {
            "success": False,
            "calendar": "f1",
            "operation": "next_events",
            "events": [],
            "error":
                "No upcoming F1 events found.",
        }

    # --------------------------------------------------------
    # SORT EVENTS CHRONOLOGICALLY
    # --------------------------------------------------------

    def event_datetime(
        event: dict[str, Any],
    ) -> datetime:

        value = event.get(
            "datetime"
        )

        if isinstance(
            value,
            datetime,
        ):

            parsed = value

            if parsed.tzinfo is None:
                parsed = parsed.replace(
                    tzinfo=ZOE_TIMEZONE
                )

            return parsed.astimezone(
                ZOE_TIMEZONE
            )

        if not value:

            return datetime.max.replace(
                tzinfo=ZOE_TIMEZONE
            )

        try:

            parsed = datetime.fromisoformat(
                str(value).replace(
                    "Z",
                    "+00:00",
                )
            )

            if parsed.tzinfo is None:

                parsed = parsed.replace(
                    tzinfo=ZOE_TIMEZONE
                )

            return parsed.astimezone(
                ZOE_TIMEZONE
            )

        except (
            TypeError,
            ValueError,
        ):

            return datetime.max.replace(
                tzinfo=ZOE_TIMEZONE
            )

    events = sorted(
        events,
        key=event_datetime,
    )

    # --------------------------------------------------------
    # RETURN ONLY NEXT N EVENTS
    # --------------------------------------------------------

    events = events[:count]

    compact_events = [
        _compact_event(event)
        for event in events
    ]

    return {
        "success": True,
        "calendar": "f1",
        "operation": "next_events",
        "count": len(compact_events),
        "events": compact_events,
    }


# ============================================================
# LLM PATH
# ============================================================

SYSTEM_PROMPT = """
You are ZOE's F1 calendar query parser.

Your only job is to convert a natural-language F1 calendar
request into ONE calendar operation.

Available operations:

TODAY
NEXT_EVENTS

Return ONLY JSON.

For TODAY:
{"operation":"today"}

For NEXT_EVENTS:
{"operation":"next_events"}

Rules:

- Use TODAY for requests specifically about F1 events today.
- Use NEXT_EVENTS for requests about upcoming F1 events.
- NEXT_EVENTS always returns the next 5 F1 calendar events.
- Requests such as "upcoming F1 events", "next events",
  "next few events", "future F1 events", "F1 calendar",
  "F1 schedule", "this week", or "next week"
  should use NEXT_EVENTS.
- If the user asks for upcoming F1 events without specifying
  a number, use NEXT_EVENTS.
- Never return RANGE.
- Never return days.
- Never answer the user.
- Never invent calendar data.
- Return ONLY valid JSON.
"""


def _run_llm(
    query: str,
) -> dict[str, Any]:

    current = now()

    prompt = (
        f"Current datetime: "
        f"{current.isoformat(timespec='minutes')}\n"
        f"Current day: "
        f"{current.strftime('%A')}\n"
        f"Timezone: Asia/Dhaka (+06:00)\n\n"
        f"Request: {query.strip()}"
    )

    response = (
        client.chat.completions.create(
            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content":
                        SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content":
                        prompt,
                },
            ],

            temperature=0,

            max_completion_tokens=(
                MAX_COMPLETION_TOKENS
            ),

            response_format={
                "type": "json_object"
            },
        )
    )

    content = (
        response.choices[0]
        .message
        .content
        or "{}"
    )

    try:

        result = json.loads(
            content
        )

    except json.JSONDecodeError:

        return {
            "operation": "next_events",
        }

    if not isinstance(
        result,
        dict,
    ):

        return {
            "operation": "next_events",
        }

    return result


# ============================================================
# AGENT
# ============================================================

class F1CalendarAgent:

    def __init__(
        self,
        model: str = MODEL,
    ):

        self.model = model

    def run(
        self,
        user_request: str,
    ) -> dict[str, Any]:

        query = user_request.strip()

        if not query:

            return {
                "success": False,
                "calendar": "f1",
                "error":
                    "Empty F1 request.",
            }

        # ----------------------------------------------------
        # FAST PATH: NEXT RACE
        # ----------------------------------------------------

        if _is_next_race(
            query
        ):

            return _next_race()

        # ----------------------------------------------------
        # FAST PATH: TODAY
        # ----------------------------------------------------

        if _is_today(
            query
        ):

            return _today()

        # ----------------------------------------------------
        # ONE SMALL LLM CALL
        # ----------------------------------------------------

        try:

            parsed = _run_llm(
                query
            )

        except Exception as exc:

            return {
                "success": False,
                "calendar": "f1",
                "error":
                    str(exc),
            }

        operation = str(
            parsed.get(
                "operation",
                "",
            )
        ).lower().strip()

        # ----------------------------------------------------
        # TODAY
        # ----------------------------------------------------

        if operation == "today":

            return _today()

        # ----------------------------------------------------
        # NEXT 5 EVENTS
        # ----------------------------------------------------

        if operation == "next_events":

            return _next_events()

        # ----------------------------------------------------
        # INVALID OPERATION
        # ----------------------------------------------------

        return {
            "success": False,
            "calendar": "f1",
            "error":
                "Unsupported F1 calendar operation.",
        }


# ============================================================
# PUBLIC API
# ============================================================

f1_calendar_agent = (
    F1CalendarAgent()
)


def run_f1_agent(
    query: str,
) -> dict[str, Any]:

    return f1_calendar_agent.run(
        query
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    while True:

        try:

            query = input(
                "\nF1 > "
            ).strip()

            if query.lower() in {
                "exit",
                "quit",
            }:

                break

            if not query:
                continue

            print(
                json.dumps(
                    run_f1_agent(query),
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