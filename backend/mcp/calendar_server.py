"""
ZOE Personal Calendar MCP server (stdio).

Thin MCP wrapper over backend.tools.gcal. Only the PERSONAL calendar is
exposed; football / F1 stay out of reach of anything using this server.

Run:  python -m backend.mcp.calendar_server
(Make sure gcal auth already works; an OAuth prompt inside a stdio
server can't be completed.)

Env:
    ZOE_CALENDAR_MAX_BULK_DELETE   max events per delete_events call (default 10)

IMPORTANT: stdout is the MCP transport. Anything in gcal that print()s would
corrupt it, so imports and every tool call run with stdout redirected to stderr.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from mcp.server.fastmcp import FastMCP

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
log = logging.getLogger("zoe.calendar")

with contextlib.redirect_stdout(sys.stderr):
    from backend.tools.gcal import (
        add_event as _gcal_add,
        delete_event as _gcal_delete,
        get_events as _gcal_events,
        get_today_events as _gcal_today,
    )

TIMEZONE = "Asia/Dhaka"
TZ = ZoneInfo(TIMEZONE)
CALENDAR = "personal"
MAX_BULK_DELETE = int(os.getenv("ZOE_CALENDAR_MAX_BULK_DELETE", "10"))
DATE_TABLE_DAYS = 15

# ============================================================
# HELPERS
# ============================================================


def _safe(fn):
    """Structured errors + stdout protection for every tool call."""

    @functools.wraps(fn)  # keeps the real signature visible to FastMCP
    def wrapper(*a, **kw):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                return fn(*a, **kw)
        except Exception as exc:
            log.exception("tool %s failed", fn.__name__)
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    return wrapper


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        if len(text) == 10:  # all-day: YYYY-MM-DD
            return datetime.fromisoformat(text).replace(tzinfo=TZ)
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.replace(tzinfo=TZ) if dt.tzinfo is None else dt.astimezone(TZ)


def _aware(value: str | None, field: str) -> datetime:
    if not value or not str(value).strip():
        raise ValueError(f"{field} is required.")
    try:
        dt = datetime.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field} must be a valid ISO-8601 datetime.") from exc
    if dt.tzinfo is None:
        raise ValueError(f"{field} must include a timezone (use +06:00).")
    return dt


def _event_id(event: dict) -> str | None:
    for key in ("event_id", "id", "google_event_id"):
        if event.get(key):
            return str(event[key])
    return None


def _link(result: Any) -> str | None:
    if not isinstance(result, dict):
        return None
    for key in ("htmlLink", "html_link", "url", "link"):
        if result.get(key) and str(result[key]).strip():
            return str(result[key]).strip()
    for key in ("event", "result", "data"):
        found = _link(result.get(key))
        if found:
            return found
    return None


def _compact(event: dict) -> dict[str, Any]:
    """Raw Google events are huge; the model only needs these fields."""
    start_raw = event.get("datetime") or event.get("start_datetime") or event.get("start")
    start = _parse_dt(start_raw)
    out: dict[str, Any] = {
        "id": _event_id(event),
        "title": event.get("title"),
        "start": event.get("datetime") or event.get("time"),
        "end": event.get("end_datetime") or event.get("end"),
        "time": event.get("time"),
        "day": start.strftime("%a %Y-%m-%d") if start else None,
    }
    return {k: v for k, v in out.items() if v is not None}


def _events_payload(result: Any) -> dict[str, Any]:
    if result is None:
        events: list = []
    elif isinstance(result, dict):
        if result.get("success") is False:
            return result
        events = result.get("events") or []
    else:
        events = result
    if not isinstance(events, list):
        events = []
    compact = [_compact(e) for e in events if isinstance(e, dict)]
    return {"success": True, "count": len(compact), "events": compact}


# ============================================================
# MCP SERVER
# ============================================================

mcp = FastMCP("zoe-calendar")


@mcp.tool()
@_safe
def get_current_datetime() -> dict[str, Any]:
    """Current datetime in Asia/Dhaka (UTC+06:00) plus a weekday->date table
    for the next two weeks. Use it to resolve 'Friday', 'next week', 'the 10th'."""
    now = datetime.now(TZ)
    table = []
    for offset in range(DATE_TABLE_DAYS):
        d = (now + timedelta(days=offset)).date()
        tag = " <- TODAY" if offset == 0 else " <- TOMORROW" if offset == 1 else ""
        table.append(f"{d.strftime('%A')}, {d.isoformat()}{tag}")
    return {
        "success": True,
        "now": now.isoformat(timespec="seconds"),
        "timezone": TIMEZONE,
        "date_table": table,
    }


@mcp.tool()
@_safe
def get_today_events() -> dict[str, Any]:
    """Get all PERSONAL calendar events happening today. Results include event IDs."""
    return _events_payload(_gcal_today(CALENDAR))


@mcp.tool()
@_safe
def get_events(start_datetime: str, end_datetime: str) -> dict[str, Any]:
    """Get PERSONAL calendar events in a datetime range (any date, not just today).

    Both arguments are timezone-aware ISO-8601 using +06:00, e.g.
    2026-11-16T00:00:00+06:00. For a whole day use 00:00 of that day to 00:00
    of the next. Results include event IDs.
    """
    start = _aware(start_datetime, "start_datetime")
    end = _aware(end_datetime, "end_datetime")
    if end <= start:
        raise ValueError("end_datetime must be after start_datetime.")
    return _events_payload(
        _gcal_events(start_datetime=start, end_datetime=end, calendar=CALENDAR)
    )


@mcp.tool()
@_safe
def add_event(
    title: str,
    start_datetime: str,
    end_datetime: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Create ONE event on the PERSONAL Google Calendar.

    start_datetime must be timezone-aware ISO-8601 using +06:00. Do not
    invent an end time, description or attendees the user didn't give.
    """
    if not title or not title.strip():
        raise ValueError("title is required.")
    start = _aware(start_datetime, "start_datetime")
    end = _aware(end_datetime, "end_datetime") if end_datetime else None
    if end and end <= start:
        raise ValueError("end_datetime must be after start_datetime.")

    result = _gcal_add(
        title=title.strip(),
        start_datetime=start,
        end_datetime=end,
        description=description,
    )
    if isinstance(result, dict) and result.get("success") is False:
        return result

    out: dict[str, Any] = {"success": True, "created": True, "title": title.strip()}
    link = _link(result)
    if link:
        out["url"] = link
    return out


@mcp.tool()
@_safe
def delete_events(event_ids: list[str]) -> dict[str, Any]:
    """Delete one or more PERSONAL calendar events by exact event ID.

    IDs must come from get_events / get_today_events results. Never guess an
    ID. Works for any date and for bulk deletes (capped per call).
    """
    unique = list(dict.fromkeys(str(i).strip() for i in event_ids if str(i).strip()))
    if not unique:
        raise ValueError("event_ids must contain at least one ID.")
    if len(unique) > MAX_BULK_DELETE:
        return {
            "success": False,
            "error": (
                f"Refusing to delete {len(unique)} events in one call "
                f"(limit {MAX_BULK_DELETE}). Narrow the request."
            ),
            "results": [],
        }

    results = []
    for eid in unique:
        try:
            r = _gcal_delete(eid, calendar=CALENDAR)
            ok = r is not False and not (isinstance(r, dict) and r.get("success") is False)
            err = None if ok else "Event not found or already deleted."
        except Exception as exc:
            ok, err = False, str(exc)
        entry: dict[str, Any] = {"event_id": eid, "deleted": ok}
        if err:
            entry["error"] = err
        results.append(entry)

    return {"success": all(r["deleted"] for r in results), "results": results}


if __name__ == "__main__":
    log.info("zoe-calendar starting")
    mcp.run()  # stdio