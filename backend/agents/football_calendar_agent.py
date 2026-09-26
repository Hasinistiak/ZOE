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
    get_next_football_match,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


MODEL = os.getenv(
    "ZOE_FOOTBALL_CALENDAR_MODEL",
    os.getenv(
        "ZOE_CALENDAR_MODEL",
        "openai/gpt-oss-120b",
    ),
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


# ============================================================
# CONFIGURATION
# ============================================================

TIMEZONE = "Asia/Dhaka"

ZOE_TIMEZONE = ZoneInfo(
    TIMEZONE
)

MAX_COMPLETION_TOKENS = 150

# Number of upcoming Real Madrid matches to return.
NEXT_MATCH_COUNT = 5

# How far into the future to search for those matches.
# 120 days gives enough room around international breaks.
MATCH_LOOKAHEAD_DAYS = 120


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


def _extract_url(
    description: Any,
) -> str | None:

    if not description:
        return None

    match = re.search(
        r"https?://[^\s<>\"']+",
        str(description),
    )

    if not match:
        return None

    return match.group(0)


# ============================================================
# COMPACT EVENT FORMAT
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
        "status": event.get("status"),
        "url": _extract_url(
            event.get("description")
        ),
    }


# ============================================================
# FAST-PATH DETECTION
# ============================================================

def _is_next_match(
    query: str,
) -> bool:

    q = _normalize(query)

    patterns = (
        r"\bnext match\b",
        r"\bnext game\b",
        r"\bnext real madrid match\b",
        r"\bnext real madrid game\b",
        r"\bwhat is the next match\b",
        r"\bwhat's the next match\b",
        r"\bwhen is the next match\b",
        r"\bwhat is the next game\b",
        r"\bwhat's the next game\b",
        r"\bwhen is the next game\b",
        r"\bwhat is the next real madrid match\b",
        r"\bwhat's the next real madrid match\b",
        r"\bwhen is the next real madrid match\b",
        r"\bwhat is the next real madrid game\b",
        r"\bwhat's the next real madrid game\b",
        r"\bwhen is the next real madrid game\b",
        r"\bwhat is the next football match\b",
        r"\bwhat's the next football match\b",
        r"\bwhen is the next football match\b",
    )

    return any(
        re.search(
            pattern,
            q,
        )
        for pattern in patterns
    )


def _is_today(
    query: str,
) -> bool:

    q = _normalize(query)

    patterns = (
        r"^today$",
        r"^football today$",
        r"^today's football$",
        r"^todays football$",
        r"^today's match$",
        r"^todays match$",
        r"^today's game$",
        r"^todays game$",
        r"^real madrid match today$",
        r"^real madrid game today$",
        r"^real madrid today$",
        r"^football match today$",
        r"^football game today$",
    )

    return any(
        re.search(
            pattern,
            q,
        )
        for pattern in patterns
    )


# ============================================================
# DIRECT OPERATIONS
# ============================================================

def _next_match() -> dict[str, Any]:

    event = get_next_football_match()

    if not event:
        return {
            "success": False,
            "calendar": "football",
            "operation": "next_match",
            "error": (
                "No upcoming Real Madrid "
                "match was found."
            ),
        }

    return {
        "success": True,
        "calendar": "football",
        "operation": "next_match",
        "match": _compact_event(
            event
        ),
    }


def _today() -> dict[str, Any]:

    events = get_today_events(
        "football"
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
        "calendar": "football",
        "operation": "today",
        "events": [
            _compact_event(event)
            for event in events
        ],
    }


# ============================================================
# NEXT 5 REAL MADRID MATCHES
# ============================================================

def _next_matches(
    count: int = NEXT_MATCH_COUNT,
) -> dict[str, Any]:

    count = max(
        1,
        min(
            int(count),
            NEXT_MATCH_COUNT,
        ),
    )

    current = now()

    end = current + timedelta(
        days=MATCH_LOOKAHEAD_DAYS
    )

    events = get_events(
        start_datetime=current,
        end_datetime=end,
        calendar="football",
    )

    if not events:
        return {
            "success": False,
            "calendar": "football",
            "operation": "next_matches",
            "matches": [],
            "error": (
                "No upcoming Real Madrid "
                "matches were found."
            ),
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
    # RETURN ONLY NEXT N MATCHES
    # --------------------------------------------------------

    matches = [
        _compact_event(event)
        for event in events[:count]
    ]

    return {
        "success": True,
        "calendar": "football",
        "operation": "next_matches",
        "count": len(matches),
        "matches": matches,
    }


# ============================================================
# SMALL LLM PARSER
# ============================================================

PARSER_PROMPT = """
You are ZOE's Real Madrid Match Calendar query parser.

IMPORTANT:

This calendar contains ONLY Real Madrid matches.

There are NO other football clubs.

Do not distinguish between clubs.

Do not return another team's matches.

Do not invent a team.

Every football request is implicitly about Real Madrid.

Convert the user's request into ONE operation.

Available operations:

TODAY
{"operation":"today"}

NEXT_MATCHES
{"operation":"next_matches"}

Rules:

- Use TODAY for requests specifically about Real Madrid matches today.
- Use NEXT_MATCHES for requests about upcoming Real Madrid matches.
- NEXT_MATCHES always returns the next 5 Real Madrid matches.
- Requests such as "upcoming matches", "next matches",
  "next few matches", "future matches",
  "Real Madrid fixtures", "this week", or "next week"
  should use NEXT_MATCHES.
- If the user asks for upcoming Real Madrid matches without
  specifying a number, use NEXT_MATCHES.
- Never return next_match.
- Never return range.
- Never return days.
- Never answer the user.
- Never return calendar data.
- Never invent match information.
- Return ONLY valid JSON.
"""


def _parse_request(
    query: str,
) -> dict[str, Any]:

    current = now()

    prompt = (
        f"{PARSER_PROMPT}\n"
        f"Current datetime: "
        f"{current.isoformat(timespec='seconds')}\n"
        f"Current date: "
        f"{current.strftime('%Y-%m-%d')}\n"
        f"Current day: "
        f"{current.strftime('%A')}\n"
        f"Timezone: Asia/Dhaka (+06:00)\n\n"
        f"User request:\n"
        f"{query.strip()}"
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        response_format={
            "type": "json_object",
        },
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

    except json.JSONDecodeError as exc:
        raise ValueError(
            "Real Madrid calendar parser "
            "returned invalid JSON."
        ) from exc

    if not isinstance(
        result,
        dict,
    ):
        raise ValueError(
            "Real Madrid calendar parser "
            "returned invalid data."
        )

    return result


# ============================================================
# AGENT
# ============================================================

class FootballCalendarAgent:

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
                "calendar": "football",
                "error": (
                    "Football calendar request "
                    "cannot be empty."
                ),
            }

        # ----------------------------------------------------
        # FAST PATH: NEXT REAL MADRID MATCH
        # ----------------------------------------------------

        if _is_next_match(query):
            return _next_match()

        # ----------------------------------------------------
        # FAST PATH: REAL MADRID MATCHES TODAY
        # ----------------------------------------------------

        if _is_today(query):
            return _today()

        # ----------------------------------------------------
        # ONE SMALL LLM CALL
        # ----------------------------------------------------

        try:

            parsed = _parse_request(
                query
            )

        except Exception as exc:

            return {
                "success": False,
                "calendar": "football",
                "error": str(exc),
            }

        operation = str(
            parsed.get(
                "operation",
                "",
            )
        ).lower().strip()

        # ----------------------------------------------------
        # PARSED TODAY
        # ----------------------------------------------------

        if operation == "today":
            return _today()

        # ----------------------------------------------------
        # PARSED NEXT MATCHES
        # ----------------------------------------------------

        if operation == "next_matches":
            return _next_matches()

        # ----------------------------------------------------
        # INVALID OPERATION
        # ----------------------------------------------------

        return {
            "success": False,
            "calendar": "football",
            "error": (
                "Unsupported Real Madrid "
                "football calendar operation."
            ),
        }


# ============================================================
# PUBLIC INTERFACE
# ============================================================

football_calendar_agent = (
    FootballCalendarAgent()
)


def run_football_calendar_agent(
    query: str,
) -> dict[str, Any]:

    return football_calendar_agent.run(
        query
    )


# Backwards compatibility

run_football_agent = (
    run_football_calendar_agent
)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    while True:

        try:

            query = input(
                "\nFootball > "
            ).strip()

            if query.lower() in {
                "exit",
                "quit",
            }:
                break

            if not query:
                continue

            result = (
                run_football_calendar_agent(
                    query
                )
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