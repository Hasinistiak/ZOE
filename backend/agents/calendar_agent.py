from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from groq import Groq

from backend.tools.gcal import (
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

MODEL = os.getenv("ZOE_CALENDAR_MODEL", "openai/gpt-oss-120b")

GROQ_API_KEY = os.getenv("GROQ_API_KEY3")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY3 is not set.")

client = Groq(api_key=GROQ_API_KEY)

TIMEZONE = "Asia/Dhaka"
ZOE_TIMEZONE = ZoneInfo(TIMEZONE)

MAX_AGENT_STEPS = int(os.getenv("ZOE_CALENDAR_MAX_STEPS", "8"))
MAX_TOOL_CALLS = int(os.getenv("ZOE_CALENDAR_MAX_TOOL_CALLS", "20"))

# Hard cap on how many events a single delete_events call may remove.
MAX_BULK_DELETE = int(os.getenv("ZOE_CALENDAR_MAX_BULK_DELETE", "10"))

# How many prior chat messages (user/assistant) to pass to the model.
MAX_HISTORY_MESSAGES = 6

# How many days ahead the date table in the system prompt covers.
DATE_TABLE_DAYS = 15

# Raise this if read-only answers come back empty (reasoning model).
MAX_COMPLETION_TOKENS = int(os.getenv("ZOE_CALENDAR_MAX_TOKENS", "1600"))

# Set ZOE_CALENDAR_DEBUG=1 to include raw operations in the returned dict.
DEBUG = os.getenv("ZOE_CALENDAR_DEBUG", "").lower() in {"1", "true", "yes"}

FUZZY_MATCH_THRESHOLD = 0.55
AMBIGUITY_MARGIN = 0.12

# Tools that only read the calendar.
READ_TOOLS = {"get_today_events", "get_events"}


# ============================================================
# TIME
# ============================================================

def get_current_datetime() -> datetime:
    return datetime.now(ZOE_TIMEZONE)


def get_current_datetime_string() -> str:
    return get_current_datetime().isoformat(timespec="seconds")


def build_date_table(now: datetime, days: int = DATE_TABLE_DAYS) -> str:
    """
    Explicit weekday -> date lookup so the model never has to
    compute "what day is the 10th" or "when is next Friday".
    """

    lines = []

    for offset in range(days):
        d = (now + timedelta(days=offset)).date()

        if offset == 0:
            tag = "  <- TODAY"
        elif offset == 1:
            tag = "  <- TOMORROW"
        else:
            tag = ""

        lines.append(f"{d.strftime('%A')}, {d.isoformat()}{tag}")

    return "\n".join(lines)


# ============================================================
# DISPLAY HELPERS (used for compact summaries)
# ============================================================

def parse_dt(value: Any) -> tuple[datetime | None, bool]:
    """
    Parse an ISO string into a Dhaka-timezone datetime.
    Returns (datetime | None, is_all_day).
    Date-only strings (all-day events) are flagged all-day.
    """

    if not value:
        return None, False

    text = str(value).strip()

    try:
        if len(text) == 10:  # YYYY-MM-DD
            dt = datetime.fromisoformat(text).replace(tzinfo=ZOE_TIMEZONE)
            return dt, True

        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))

    except ValueError:
        return None, False

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZOE_TIMEZONE)
    else:
        dt = dt.astimezone(ZOE_TIMEZONE)

    return dt, False


def fmt_time(dt: datetime) -> str:
    return dt.strftime("%I:%M %p").lstrip("0")


def fmt_day(d: date) -> str:
    return f"{d.strftime('%a %b')} {d.day}"


def day_label(d: date, today: date) -> str:
    if d == today:
        return f"Today ({fmt_day(d)})"

    if d == today + timedelta(days=1):
        return f"Tomorrow ({fmt_day(d)})"

    if d == today - timedelta(days=1):
        return f"Yesterday ({fmt_day(d)})"

    return fmt_day(d)


def event_start(event: dict[str, Any]) -> tuple[datetime | None, bool]:
    return parse_dt(
        event.get("datetime") or event.get("start_datetime") or event.get("start")
    )


def event_end(event: dict[str, Any]) -> datetime | None:
    dt, _ = parse_dt(event.get("end_datetime") or event.get("end"))
    return dt


def format_event_line(event: dict[str, Any]) -> str:
    title = str(event.get("title") or "Untitled event")
    start, all_day = event_start(event)

    if start is None:
        fallback = event.get("time")
        return f"{fallback} {title}".strip() if fallback else title

    if all_day:
        return f"All day: {title}"

    end = event_end(event)

    if end and end > start:
        return f"{fmt_time(start)}-{fmt_time(end)} {title}"

    return f"{fmt_time(start)} {title}"


def format_events_by_day(
    events: list[dict[str, Any]],
    today: date,
) -> str:
    """Group events by date, chronological, one line per day."""

    buckets: dict[date, list[tuple[datetime, dict[str, Any]]]] = {}
    undated: list[dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            continue

        start, _ = event_start(event)

        if start is None:
            undated.append(event)
            continue

        buckets.setdefault(start.date(), []).append((start, event))

    parts = []

    for d in sorted(buckets):
        items = sorted(buckets[d], key=lambda pair: pair[0])
        lines = "; ".join(format_event_line(e) for _, e in items)
        parts.append(f"{day_label(d, today)}: {lines}")

    if undated:
        parts.append("; ".join(format_event_line(e) for e in undated))

    return "\n".join(parts)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

_STOP_WORDS = {
    "my", "the", "a", "an", "event", "events", "appointment",
    "calendar", "meeting", "session", "please", "can", "you",
    "could", "would", "remove", "delete", "cancel", "clear",
    "from", "today", "tomorrow", "thing", "things", "one",
}


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(w for w in text.split() if w not in _STOP_WORDS)


# ============================================================
# SIMILARITY
# ============================================================

def similarity(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


# ============================================================
# EVENT MATCHING
# ============================================================

def event_matches_query(query: str, event: dict[str, Any]) -> float:
    event_title = str(event.get("title", "") or "")

    if not event_title:
        return 0.0

    q = normalize_text(query)
    title = normalize_text(event_title)

    if not q or not title:
        return 0.0

    if q == title:
        return 1.0

    if q in title:
        return 0.90

    if title in q:
        return 0.85

    query_words = set(q.split())
    title_words = set(title.split())

    if query_words and title_words:
        overlap = len(query_words & title_words) / len(query_words)

        if overlap >= 0.75:
            return max(0.75, similarity(q, title))

        if overlap >= 0.50:
            return max(0.60, similarity(q, title))

    return similarity(q, title)


# ============================================================
# EVENT ID
# ============================================================

def get_event_id(event: dict[str, Any]) -> str | None:
    for key in ("event_id", "id", "google_event_id"):
        value = event.get(key)

        if value:
            return str(value)

    return None


# ============================================================
# EVENT LINK
# ============================================================

def get_event_link(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None

    for key in ("htmlLink", "html_link", "url", "link"):
        value = result.get(key)

        if value:
            value = str(value).strip()

            if value:
                return value

    for key in ("event", "result", "data"):
        link = get_event_link(result.get(key))

        if link:
            return link

    return None


# ============================================================
# EVENT DISPLAY
# ============================================================

def event_summary(event: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "id": event.get("id"),
        "title": event.get("title"),
        "time": event.get("time"),
        "datetime": event.get("datetime"),
        "end_datetime": event.get("end_datetime"),
        "status": event.get("status"),
        "calendar": event.get("calendar"),
    }

    link = get_event_link(event)

    if link:
        summary["url"] = link

    return summary


# ============================================================
# RESULT NORMALIZATION
# ============================================================

def normalize_event_result(result: Any) -> dict[str, Any]:
    if result is None:
        return {"success": True, "events": []}

    if isinstance(result, dict):
        return result

    return {"success": True, "events": result}


# ============================================================
# SEEN-ID TRACKING
#
# The model may only delete by ID if that ID was returned by a
# calendar lookup during THIS request. This makes hallucinated
# IDs harmless. State is per-run (never a module global), so
# overlapping requests cannot share it.
# ============================================================

def remember_events(result: Any, seen_ids: dict[str, str]) -> None:
    if not isinstance(result, dict):
        return

    events = result.get("events") or []

    if not isinstance(events, list):
        return

    for event in events:
        if not isinstance(event, dict):
            continue

        event_id = get_event_id(event)

        if event_id:
            seen_ids[event_id] = str(event.get("title") or "Untitled event")


# ============================================================
# TOOL: TODAY
# ============================================================

def tool_get_today_events() -> dict[str, Any]:
    """Fetch today's personal calendar events."""

    return normalize_event_result(get_today_events("personal"))


# ============================================================
# TOOL: DATE RANGE
# ============================================================

def tool_get_events(
    start_datetime: str,
    end_datetime: str,
) -> dict[str, Any]:
    """Fetch personal events in a timezone-aware datetime range."""

    if not start_datetime:
        raise ValueError("start_datetime is required.")

    if not end_datetime:
        raise ValueError("end_datetime is required.")

    try:
        start = datetime.fromisoformat(str(start_datetime).strip())
        end = datetime.fromisoformat(str(end_datetime).strip())

    except ValueError as exc:
        raise ValueError(
            "start_datetime and end_datetime must be valid "
            "ISO-8601 datetimes."
        ) from exc

    if start.tzinfo is None:
        raise ValueError("start_datetime must include a timezone.")

    if end.tzinfo is None:
        raise ValueError("end_datetime must include a timezone.")

    if end <= start:
        raise ValueError("end_datetime must be after start_datetime.")

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
        raise ValueError("title is required.")

    if not start_datetime:
        raise ValueError("start_datetime is required.")

    try:
        start = datetime.fromisoformat(str(start_datetime).strip())

    except ValueError as exc:
        raise ValueError(
            "Invalid start_datetime. Expected timezone-aware ISO-8601."
        ) from exc

    if start.tzinfo is None:
        raise ValueError("start_datetime must include a timezone.")

    end = None

    if end_datetime:
        try:
            end = datetime.fromisoformat(str(end_datetime).strip())

        except ValueError as exc:
            raise ValueError(
                "Invalid end_datetime. Expected timezone-aware ISO-8601."
            ) from exc

        if end.tzinfo is None:
            raise ValueError("end_datetime must include a timezone.")

        if end <= start:
            raise ValueError("end_datetime must be after start_datetime.")

    return add_event(
        title=str(title).strip(),
        start_datetime=start,
        end_datetime=end,
        description=description,
    )


# ============================================================
# TOOL: DELETE (single event, TODAY, fuzzy query)
# ============================================================

def tool_delete_event(query: str) -> dict[str, Any]:

    if not query or not str(query).strip():
        raise ValueError("query is required.")

    query = str(query).strip()

    normalized = normalize_event_result(get_today_events("personal"))
    events = normalized.get("events", [])

    if not events:
        return {
            "success": False,
            "deleted": False,
            "message": "There are no personal calendar events today.",
        }

    scored_events = []

    for event in events:
        if not isinstance(event, dict):
            continue

        scored_events.append((event_matches_query(query, event), event))

    if not scored_events:
        return {
            "success": False,
            "deleted": False,
            "message": "No usable events were found.",
        }

    scored_events.sort(key=lambda item: item[0], reverse=True)

    best_score, best_event = scored_events[0]

    if best_score < FUZZY_MATCH_THRESHOLD:
        return {
            "success": False,
            "deleted": False,
            "ambiguous": False,
            "message": (
                f"No event matched {query!r} with sufficient confidence."
            ),
            "available_events": [
                event_summary(event) for _, event in scored_events
            ],
        }

    plausible = [
        (score, event)
        for score, event in scored_events
        if score >= FUZZY_MATCH_THRESHOLD
    ]

    if len(plausible) > 1:
        second_score, _ = plausible[1]

        if best_score - second_score < AMBIGUITY_MARGIN:
            return {
                "success": False,
                "deleted": False,
                "ambiguous": True,
                "message": f"Multiple events could match {query!r}.",
                "matches": [event_summary(event) for _, event in plausible],
            }

    event_id = get_event_id(best_event)

    if not event_id:
        return {
            "success": False,
            "deleted": False,
            "message": (
                "A matching event was found, but its Google Calendar "
                "ID could not be determined."
            ),
            "matched_event": event_summary(best_event),
        }

    delete_result = delete_event(event_id, calendar="personal")

    # gcal.delete_event returns a bool (False on 404).
    if delete_result is False:
        return {
            "success": False,
            "deleted": False,
            "event_id": event_id,
            "title": best_event.get("title"),
            "message": "Event no longer exists or was already deleted.",
        }

    if isinstance(delete_result, dict) and delete_result.get("success") is False:
        return {
            "success": False,
            "deleted": False,
            "event_id": event_id,
            "title": best_event.get("title"),
            "delete_result": delete_result,
        }

    result = {
        "success": True,
        "deleted": True,
        "event_id": event_id,
        "title": best_event.get("title"),
        "time": best_event.get("time"),
        "datetime": best_event.get("datetime"),
    }

    link = get_event_link(delete_result)

    if link:
        result["url"] = link

    return result


# ============================================================
# TOOL: DELETE BY ID (any date, bulk)
# ============================================================

def tool_delete_events_by_id(
    event_ids: list[str],
    seen_ids: dict[str, str],
) -> dict[str, Any]:

    if not isinstance(event_ids, list) or not event_ids:
        raise ValueError("event_ids must be a non-empty list.")

    # De-duplicate while preserving order.
    unique_ids: list[str] = []

    for raw in event_ids:
        event_id = str(raw).strip()

        if event_id and event_id not in unique_ids:
            unique_ids.append(event_id)

    if not unique_ids:
        raise ValueError("event_ids must contain at least one ID.")

    if len(unique_ids) > MAX_BULK_DELETE:
        return {
            "success": False,
            "error": (
                f"Refusing to delete {len(unique_ids)} events in one "
                f"call (limit {MAX_BULK_DELETE}). Narrow the request."
            ),
            "results": [],
        }

    results = []

    for event_id in unique_ids:

        if event_id not in seen_ids:
            results.append({
                "event_id": event_id,
                "deleted": False,
                "error": (
                    "ID was not returned by a calendar lookup in "
                    "this request."
                ),
            })
            continue

        try:
            deleted = delete_event(event_id, calendar="personal")
            error = None

        except Exception as exc:
            deleted = False
            error = str(exc)

        entry = {
            "event_id": event_id,
            "title": seen_ids[event_id],
            "deleted": bool(deleted),
        }

        if not deleted:
            entry["error"] = error or "Event not found or already deleted."

        results.append(entry)

    return {
        "success": all(r["deleted"] for r in results),
        "results": results,
    }


# ============================================================
# TOOL REGISTRY
# ============================================================

# delete_events is dispatched separately in execute_tool because
# it needs the per-run seen_ids state.
TOOLS = {
    "get_today_events": tool_get_today_events,
    "get_events": tool_get_events,
    "add_event": tool_add_event,
    "delete_event": tool_delete_event,
    "delete_events": tool_delete_events_by_id,
}


# ============================================================
# GROQ TOOLS
# ============================================================

GROQ_TOOLS = [

    {
        "type": "function",
        "function": {
            "name": "get_today_events",
            "description": (
                "Retrieve all PERSONAL calendar events occurring TODAY. "
                "Results include event IDs."
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
            "name": "get_events",
            "description": (
                "Retrieve PERSONAL calendar events within a specified "
                "datetime range. Use this for dates other than today, "
                "including tomorrow, a specific date, or a future date "
                "range. Results include event IDs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_datetime": {
                        "type": "string",
                        "description": (
                            "Timezone-aware ISO-8601 datetime using +06:00."
                        ),
                    },
                    "end_datetime": {
                        "type": "string",
                        "description": (
                            "Timezone-aware ISO-8601 datetime using +06:00."
                        ),
                    },
                },
                "required": ["start_datetime", "end_datetime"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "add_event",
            "description": (
                "Create ONE event on the PERSONAL Google Calendar. "
                "start_datetime must be timezone-aware ISO-8601 using "
                "+06:00. Do not invent an end time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "start_datetime": {"type": "string"},
                    "end_datetime": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["title", "start_datetime"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "delete_event",
            "description": (
                "Delete ONE PERSONAL calendar event from TODAY by "
                "natural-language description. Never guess when the "
                "match is ambiguous. For any other date, or for "
                "several events at once, use delete_events instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        },
    },

    {
        "type": "function",
        "function": {
            "name": "delete_events",
            "description": (
                "Delete one or more PERSONAL calendar events by exact "
                "event ID. IDs must come from get_events or "
                "get_today_events results in this request. Use this for "
                "any date other than today and for bulk deletes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "event_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["event_ids"],
            },
        },
    },
]


# ============================================================
# SYSTEM PROMPT
# NOTE: this string goes through str.format(); do not add
# literal curly braces to it.
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

Current day of week:
{current_day}

Timezone:
Asia/Dhaka / UTC+06:00


============================================================
DATE LOOKUP TABLE (authoritative)
============================================================

Use this table to resolve every weekday name and day offset.
Never compute weekdays yourself; read them from here.

{date_table}

Rules:

- "today" and "tomorrow" are marked in the table.
- A bare weekday ("Friday", "this Friday") means the nearest
  matching row in the table, counting today if it matches.
- "next <weekday>" means the matching row in the week AFTER the
  nearest one (7 days later than the nearest match).
- "next week" means the 7 days starting the coming Monday.
- "this weekend" means the coming Saturday and Sunday
  (Friday and Saturday are the weekend in Bangladesh, but only
  use that if the user says so; default to Saturday and Sunday).
- A bare date like "the 10th" means the nearest upcoming row
  with that day number.
- If a date is outside the table, compute it carefully from the
  table's last row.
- Always use +06:00 in tool arguments.


============================================================
AVAILABLE CAPABILITIES
============================================================

You can:

- inspect today's personal events
- inspect personal events over arbitrary date ranges
- create personal events
- delete personal events on any date


============================================================
CALENDAR SEPARATION
============================================================

This agent is ONLY for the personal calendar.

Do NOT access:

- football
- F1

Those calendars have separate read-only agents.


============================================================
CONVERSATION CONTEXT
============================================================

Earlier messages in this conversation may be supplied.
Use them to resolve references such as "those", "them",
"that one" or "the classes". If a reference still cannot be
resolved to specific events, delete nothing.


============================================================
QUERYING
============================================================

For a whole day, fetch from 00:00:00 of that day to 00:00:00
of the following day.

For "today", use get_today_events.

For "this week" / "next week" or any multi-day range, make ONE
get_events call covering the full range.

For "next <thing>" / "when is my <thing>" questions, make ONE
get_events call starting at the CURRENT datetime (not tomorrow,
not midnight) and ending 14 days later. If nothing matches,
make ONE more call covering the following 30 days.


============================================================
ANSWERING READ-ONLY QUESTIONS
============================================================

After looking up events for a question (not for add/delete),
reply with a SHORT plain-text answer to exactly what was asked.

- Answer only the question. Do not list unrelated events.
- Give weekday, date and time, e.g. Tue Oct 13, 4:00 PM.
- "Next X": give only the first matching event, plus the one
  after it only if it is clearly useful.
- Match by meaning: a biology class includes titles that
  contain Biology. An exam is not a class unless the user
  said exam.
- If nothing matches, say so in one sentence and say what
  range you searched.
- One to three sentences. No markdown, no preamble.


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

Single event today:
use delete_event with a query.

Any other date, or several events at once:
1. call get_events for the exact range
   (call it once per date if the dates differ)
2. decide which returned events the user clearly referred to
3. call delete_events with those exact IDs

Only delete IDs that appeared in a lookup result.

"Classes" or "lessons" means the class-type events returned by
the lookup for that date. If the user referred to a whole date
without naming events, and it is not clear which events they
mean, do not delete anything.

Never delete an event the user did not clearly refer to.
If a match is ambiguous, do not guess.
If no sufficiently strong match exists, do not guess.


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

For add/delete requests, use tools and finish the operation,
then stop. Do not produce conversational text for them.
"""


# ============================================================
# PROMPT
# ============================================================

def build_system_prompt() -> str:
    now = get_current_datetime()

    return SYSTEM_PROMPT.format(
        current_datetime=now.isoformat(timespec="seconds"),
        current_date=now.strftime("%Y-%m-%d"),
        current_time=now.strftime("%H:%M:%S"),
        current_day=now.strftime("%A"),
        date_table=build_date_table(now),
    )


def clean_history(
    history: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """
    Keep only plain user/assistant text turns, most recent last.
    """

    if not history:
        return []

    cleaned: list[dict[str, str]] = []

    for item in history:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role not in ("user", "assistant"):
            continue

        if not isinstance(content, str) or not content.strip():
            continue

        cleaned.append({"role": role, "content": content.strip()})

    return cleaned[-MAX_HISTORY_MESSAGES:]


def build_messages(
    user_request: str,
    history: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:

    return [
        {"role": "system", "content": build_system_prompt()},
        *clean_history(history),
        {"role": "user", "content": user_request.strip()},
    ]


# ============================================================
# ARGUMENT NORMALIZATION
# ============================================================

ALLOWED_ARGUMENTS = {
    "get_today_events": set(),
    "get_events": {"start_datetime", "end_datetime"},
    "add_event": {"title", "start_datetime", "end_datetime", "description"},
    "delete_event": {"query"},
    "delete_events": {"event_ids"},
}

REQUIRED_ARGUMENTS = {
    "get_today_events": set(),
    "get_events": {"start_datetime", "end_datetime"},
    "add_event": {"title", "start_datetime"},
    "delete_event": {"query"},
    "delete_events": {"event_ids"},
}


def normalize_arguments(name: str, arguments: Any) -> dict[str, Any]:

    if arguments is None:
        arguments = {}

    if isinstance(arguments, str):
        arguments = arguments.strip()

        if not arguments:
            arguments = {}

        else:
            try:
                arguments = json.loads(arguments)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON arguments for tool {name!r}."
                ) from exc

    if not isinstance(arguments, dict):
        raise ValueError(f"Arguments for {name!r} must be a JSON object.")

    if name not in ALLOWED_ARGUMENTS:
        raise ValueError(f"Unknown calendar tool: {name!r}")

    unknown = [k for k in arguments if k not in ALLOWED_ARGUMENTS[name]]

    if unknown:
        raise ValueError(
            f"Unknown argument(s) for {name!r}: "
            f"{', '.join(map(str, unknown))}"
        )

    def is_missing(value: Any) -> bool:
        if value is None:
            return True

        if isinstance(value, str):
            return not value.strip()

        if isinstance(value, list):
            return len(value) == 0

        return False

    missing = [
        key
        for key in REQUIRED_ARGUMENTS[name]
        if key not in arguments or is_missing(arguments[key])
    ]

    if missing:
        raise ValueError(
            f"Missing required argument(s) for {name!r}: "
            f"{', '.join(missing)}"
        )

    return dict(arguments)


# ============================================================
# EXECUTE TOOL
# ============================================================

def execute_tool(
    name: str,
    arguments: Any,
    seen_ids: dict[str, str],
) -> Any:

    if name not in TOOLS:
        raise ValueError(f"Unknown calendar tool: {name!r}")

    cleaned = normalize_arguments(name, arguments)

    if name == "delete_events":
        return tool_delete_events_by_id(seen_ids=seen_ids, **cleaned)

    result = TOOLS[name](**cleaned)

    if name in READ_TOOLS:
        remember_events(result, seen_ids)

    return result


# ============================================================
# COMPACT TOOL RESULTS (what the CALENDAR MODEL sees)
#
# Raw Google event dicts are large. The model only needs
# id/title/day/start/end to answer questions or pick deletes.
# ============================================================

def compact_event(event: dict[str, Any]) -> dict[str, Any]:
    start, _ = event_start(event)

    out = {
        "id": get_event_id(event),
        "title": event.get("title"),
        "start": event.get("datetime") or event.get("time"),
    }

    if start:
        out["day"] = start.strftime("%a %Y-%m-%d")

    if event.get("end_datetime"):
        out["end"] = event.get("end_datetime")

    return out


def compact_for_model(name: str, result: Any) -> Any:
    if not isinstance(result, dict):
        return result

    if name in READ_TOOLS:
        events = result.get("events") or []

        return {
            "success": result.get("success", True),
            "count": len(events),
            "events": [
                compact_event(e) for e in events if isinstance(e, dict)
            ],
        }

    if name == "add_event":
        return {
            "success": result.get("success", True),
            "created": True,
        }

    if name == "delete_event":
        slim = dict(result)

        for key in ("available_events", "matches"):
            if key in slim:
                slim[key] = [
                    compact_event(e) for e in slim[key]
                    if isinstance(e, dict)
                ]

        return slim

    return result


# ============================================================
# SUMMARIES (what the MAIN BRAIN sees)
#
# Mutations (add/delete) are summarized deterministically from
# tool results: no LLM, no hallucination risk. Read-only
# questions use the model's own short answer, with this as the
# fallback.
# ============================================================

def _titles(events: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"{e.get('title') or 'Untitled'}"
        for e in events
        if isinstance(e, dict)
    )


def summarize_operation(
    op: dict[str, Any],
    today: date,
) -> tuple[str, bool]:
    """Returns (sentence(s), needs_clarification)."""

    tool = op.get("tool")
    args = op.get("arguments") if isinstance(op.get("arguments"), dict) else {}
    result = op.get("result") if isinstance(op.get("result"), dict) else {}

    # ---- failed before/while executing ----------------------
    if "error" in op and not result:
        return f"Failed ({tool}): {op['error']}", False

    # ---- read: today -----------------------------------------
    if tool == "get_today_events":
        events = result.get("events") or []

        if not events:
            return f"No events today ({fmt_day(today)}).", False

        return (
            f"{len(events)} event(s) today ({fmt_day(today)}):\n"
            + format_events_by_day(events, today),
            False,
        )

    # ---- read: range -----------------------------------------
    if tool == "get_events":
        events = result.get("events") or []
        start, _ = parse_dt(args.get("start_datetime"))
        end, _ = parse_dt(args.get("end_datetime"))

        if start and end:
            last_day = (end - timedelta(seconds=1)).date()

            if start.date() == last_day:
                span = day_label(start.date(), today)
            else:
                span = f"{fmt_day(start.date())} to {fmt_day(last_day)}"
        else:
            span = "that range"

        if not events:
            return f"No events on {span}.", False

        return (
            f"{len(events)} event(s) on {span}:\n"
            + format_events_by_day(events, today),
            False,
        )

    # ---- create ----------------------------------------------
    if tool == "add_event":
        title = args.get("title", "Event")

        if not op.get("success", False):
            err = result.get("error") or op.get("error") or "unknown error"
            return f"Could not add '{title}': {err}", False

        start, _ = parse_dt(args.get("start_datetime"))

        if start:
            return (
                f"Added '{title}' on {day_label(start.date(), today)} "
                f"at {fmt_time(start)}.",
                False,
            )

        return f"Added '{title}'.", False

    # ---- delete (single, today) ------------------------------
    if tool == "delete_event":
        if result.get("deleted"):
            when = result.get("time")
            when_txt = f" ({when})" if when else ""
            return f"Deleted '{result.get('title')}'{when_txt}.", False

        if result.get("ambiguous"):
            names = _titles(result.get("matches") or [])
            return (
                f"Several events match '{args.get('query')}': {names}. "
                f"Which one did you mean?",
                True,
            )

        available = result.get("available_events") or []
        msg = result.get("message") or "Nothing was deleted."

        if available:
            return f"{msg} Today's events: {_titles(available)}.", True

        return msg, False

    # ---- delete (by id, bulk) --------------------------------
    if tool == "delete_events":
        rows = result.get("results") or []
        done = [r.get("title") or r.get("event_id") for r in rows if r.get("deleted")]
        failed = [r for r in rows if not r.get("deleted")]

        parts = []

        if done:
            parts.append(
                f"Deleted {len(done)} event(s): " + ", ".join(map(str, done)) + "."
            )

        if failed:
            reasons = "; ".join(
                f"{r.get('title') or r.get('event_id')}: "
                f"{r.get('error', 'not deleted')}"
                for r in failed
            )
            parts.append(f"Not deleted: {reasons}")

        if not parts:
            parts.append(result.get("error") or "Nothing was deleted.")

        return " ".join(parts), False

    return f"{tool}: done." if op.get("success") else f"{tool}: failed.", False


def build_summary(operations: list[dict[str, Any]]) -> tuple[str, bool]:
    today = get_current_datetime().date()

    has_mutation = any(op.get("tool") not in READ_TOOLS for op in operations)

    lines: list[str] = []
    needs_clarification = False

    for op in operations:
        # Lookups done only to find IDs to delete are noise.
        if has_mutation and op.get("tool") in READ_TOOLS:
            continue

        text, clarify = summarize_operation(op, today)

        if text:
            lines.append(text)

        needs_clarification = needs_clarification or clarify

    return "\n".join(lines), needs_clarification


# ============================================================
# LINKS
# ============================================================

def collect_operation_links(operations: list[dict[str, Any]]) -> list[str]:
    links: list[str] = []

    for operation in operations:
        link = operation.get("url")

        if not link:
            link = get_event_link(operation.get("result"))

        if link and link not in links:
            links.append(str(link))

    return links


# ============================================================
# FINAL RESULT (compact; this is what goes to the main brain)
# ============================================================

def build_result(
    operations: list[dict[str, Any]],
    answer: str | None = None,
) -> dict[str, Any]:

    summary, needs_clarification = build_summary(operations)

    # Read-only question: the model's short answer replaces the dump.
    if answer:
        summary = answer
        needs_clarification = False

    result: dict[str, Any] = {
        "success": bool(operations) and all(
            op.get("success", False) for op in operations
        ),
        "summary": summary,
    }

    if needs_clarification:
        result["needs_clarification"] = True

    links = collect_operation_links(operations)

    # Links only matter for created events; keep payload tiny.
    if len(links) == 1:
        result["url"] = links[0]

    elif len(links) > 1:
        result["urls"] = links

    if DEBUG:
        result["operations"] = operations

    return result


def build_error(
    message: str,
    operations: list[dict[str, Any]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Compact failure result, still including what did happen."""

    operations = operations or []

    parts = []

    if operations:
        done_summary, _ = build_summary(operations)

        if done_summary:
            parts.append(done_summary)

    parts.append(message)

    result: dict[str, Any] = {
        "success": False,
        "summary": "\n".join(parts),
    }

    result.update(extra)

    if DEBUG:
        result["operations"] = operations

    return result


# ============================================================
# AGENT
# ============================================================

class CalendarAgent:

    def __init__(self, model: str = MODEL):
        self.model = model

    def run(
        self,
        user_request: str,
        history: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:

        if not user_request or not user_request.strip():
            return build_error("Calendar request cannot be empty.")

        messages = build_messages(user_request, history)

        operations: list[dict[str, Any]] = []

        # Per-run: IDs the model has actually seen from lookups.
        seen_ids: dict[str, str] = {}

        total_tool_calls = 0

        for step in range(1, MAX_AGENT_STEPS + 1):

            print(f"[Calendar Agent] Step {step}/{MAX_AGENT_STEPS}")

            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=GROQ_TOOLS,
                    tool_choice="auto",
                    temperature=0,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                )

            except Exception as exc:
                return build_error(
                    f"Calendar model request failed: {exc}",
                    operations,
                )

            if not response.choices:
                return build_error("Invalid model response.", operations)

            message = response.choices[0].message

            tool_calls = getattr(message, "tool_calls", None) or []

            if not tool_calls:

                text = (message.content or "").strip()

                if operations:
                    read_only = all(
                        op.get("tool") in READ_TOOLS for op in operations
                    )

                    # Read-only question: the model's short answer wins.
                    # Falls back to the deterministic summary if empty.
                    return build_result(
                        operations,
                        answer=text if (read_only and text) else None,
                    )

                return build_error(
                    "Calendar agent did not perform any action."
                    + (f" Model said: {text}" if text else "")
                )

            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })

            for tool_call in tool_calls:

                if total_tool_calls >= MAX_TOOL_CALLS:
                    return build_error(
                        "Maximum tool calls reached.",
                        operations,
                    )

                total_tool_calls += 1

                name = tool_call.function.name
                raw_arguments = tool_call.function.arguments

                print(f"[Calendar Agent] Tool #{total_tool_calls}: {name}")

                try:
                    arguments = normalize_arguments(name, raw_arguments)

                    result = execute_tool(name, arguments, seen_ids)

                    success = not (
                        isinstance(result, dict)
                        and result.get("success") is False
                    )

                    operation = {
                        "tool": name,
                        "arguments": arguments,
                        "success": success,
                        "result": result,
                    }

                    link = get_event_link(result)

                    if link:
                        operation["url"] = link

                    operations.append(operation)

                    model_payload = compact_for_model(name, result)

                except Exception as exc:

                    model_payload = {"success": False, "error": str(exc)}

                    operations.append({
                        "tool": name,
                        "arguments": raw_arguments,
                        "success": False,
                        "error": str(exc),
                    })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(
                        model_payload,
                        ensure_ascii=False,
                        default=str,
                    ),
                })

        return build_error(
            "Calendar agent reached the maximum number of steps.",
            operations,
        )


# ============================================================
# SINGLETON
# ============================================================

calendar_agent = CalendarAgent()


# ============================================================
# ZOE INTERFACE
# ============================================================

def run_calendar_agent(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Returns a compact dict:
        {
          "success": bool,
          "summary": "plain-text answer for the main brain",
          "needs_clarification": True,   # only when relevant
          "url" / "urls": ...            # only when an event link exists
        }

    history: optional recent chat turns from ZOE's router, e.g.
        [{"role": "user", "content": "how's tomorrow looking"},
         {"role": "assistant", "content": "Tomorrow: 5:00 PM Physics; ..."}]
    Store result["summary"] as the assistant turn so references
    like "those" can be resolved.
    """

    return calendar_agent.run(user_request=query, history=history)


def run_calendar_agent_text(
    query: str,
    history: list[dict[str, Any]] | None = None,
) -> str:
    """Convenience wrapper: just the summary string."""

    return run_calendar_agent(query, history=history).get("summary", "")


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("ZOE Personal Calendar Agent")

    now = get_current_datetime()
    print(f"Now: {now.strftime('%A, %Y-%m-%d %H:%M')} ({TIMEZONE})")

    chat_history: list[dict[str, str]] = []

    while True:

        try:
            query = input("\nYou > ").strip()

            if query.lower() in {"exit", "quit"}:
                break

            if not query:
                continue

            result = run_calendar_agent(query, history=chat_history)

            # Store the SUMMARY, not raw JSON: shorter, and the model
            # can resolve "those" / "that one" from it.
            chat_history.append({"role": "user", "content": query})
            chat_history.append({
                "role": "assistant",
                "content": result.get("summary", ""),
            })

            print(result.get("summary", ""))

            if DEBUG:
                print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

        except KeyboardInterrupt:
            break

        except Exception as exc:
            print(f"ERROR: {exc}")