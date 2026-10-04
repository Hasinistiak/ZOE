from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# ============================================================
# CONFIGURATION
# ============================================================

SCOPES = [
    "https://www.googleapis.com/auth/calendar",
]

CREDENTIALS_FILE = "backend/credentials.json"
TOKEN_FILE = "token.json"

TIMEZONE = "Asia/Dhaka"

ZOE_TIMEZONE = ZoneInfo(
    TIMEZONE
)


# ============================================================
# CALENDARS
# ============================================================

CALENDARS = {
    "personal": "primary",

    "f1": (
        "547241534d3a78d4a14e4b1fff9d952775ff1db8084292659975bce461ef5e4e@group.calendar.google.com"
    ),

    "football": (
        "vbgut32b56uoqg1glv7gbent7nq2t92v@import.calendar.google.com"
    ),
}


# ============================================================
# GOOGLE CALENDAR SERVICE
# ============================================================

def get_calendar_service():
    """
    Return an authenticated Google Calendar API service.
    """

    creds = None

    if os.path.exists(TOKEN_FILE):

        creds = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES,
        )

    if not creds or not creds.valid:

        if (
            creds
            and creds.expired
            and creds.refresh_token
        ):

            creds.refresh(
                Request()
            )

        else:

            if not os.path.exists(
                CREDENTIALS_FILE
            ):

                raise FileNotFoundError(
                    "Google Calendar credentials "
                    f"not found: {CREDENTIALS_FILE}"
                )

            flow = (
                InstalledAppFlow
                .from_client_secrets_file(
                    CREDENTIALS_FILE,
                    SCOPES,
                )
            )

            creds = flow.run_local_server(
                port=0
            )

        with open(
            TOKEN_FILE,
            "w",
        ) as token:

            token.write(
                creds.to_json()
            )

    return build(
        "calendar",
        "v3",
        credentials=creds,
        cache_discovery=False,
    )


# ============================================================
# CALENDAR RESOLUTION
# ============================================================

def resolve_calendar_id(
    calendar: str,
) -> str:
    """
    Resolve a friendly calendar name or raw ID.
    """

    if not calendar:
        raise ValueError(
            "Calendar name cannot be empty."
        )

    calendar_key = (
        calendar.strip().lower()
    )

    if calendar_key in CALENDARS:
        return CALENDARS[
            calendar_key
        ]

    if calendar.strip() in CALENDARS.values():
        return calendar.strip()

    raise ValueError(
        f"Unknown calendar '{calendar}'. "
        "Available calendars: "
        + ", ".join(
            CALENDARS.keys()
        )
    )


# ============================================================
# EVENT PARSING
# ============================================================

def _parse_datetime(
    value: str | None,
) -> datetime | None:

    if not value:
        return None

    return datetime.fromisoformat(
        value.replace(
            "Z",
            "+00:00",
        )
    )


def _get_event_status(
    start_datetime: datetime,
    end_datetime: datetime | None,
    current: datetime,
) -> str:

    if end_datetime:

        if end_datetime <= current:
            return "completed"

        if (
            start_datetime
            <= current
            < end_datetime
        ):
            return "ongoing"

        return "upcoming"

    return (
        "completed"
        if start_datetime <= current
        else "upcoming"
    )


def _parse_event(
    event: dict[str, Any],
    calendar_name: str,
    calendar_id: str,
    current: datetime | None = None,
) -> dict[str, Any] | None:
    """
    Normalize a Google Calendar event.

    Deliberately excludes descriptions and other large
    metadata because calendar agents do not need them.
    """

    if current is None:
        current = datetime.now(
            ZOE_TIMEZONE
        )

    start = event.get(
        "start",
        {},
    )

    end = event.get(
        "end",
        {},
    )

    # --------------------------------------------------------
    # Ignore all-day events
    # --------------------------------------------------------

    if "date" in start:
        return None

    start_datetime = _parse_datetime(
        start.get(
            "dateTime"
        )
    )

    if not start_datetime:
        return None

    start_datetime = (
        start_datetime.astimezone(
            ZOE_TIMEZONE
        )
    )

    end_datetime = _parse_datetime(
        end.get(
            "dateTime"
        )
    )

    if end_datetime:
        end_datetime = (
            end_datetime.astimezone(
                ZOE_TIMEZONE
            )
        )

    status = _get_event_status(
        start_datetime,
        end_datetime,
        current,
    )

    if (
        start_datetime.date()
        == current.date()
    ):

        day = "today"

    elif (
        start_datetime.date()
        == (
            current.date()
            + timedelta(days=1)
        )
    ):

        day = "tomorrow"

    else:

        day = start_datetime.strftime(
            "%Y-%m-%d"
        )

    return {
        "id": event.get(
            "id"
        ),

        "title": event.get(
            "summary",
            "Untitled event",
        ),

        "location": event.get(
            "location",
            "",
        ),

        "description": event.get(
        "description",
    ),

        "time": start_datetime.strftime(
            "%I:%M %p"
        ).lstrip("0"),

        "datetime": start_datetime.isoformat(),

        "end_datetime": (
            end_datetime.isoformat()
            if end_datetime
            else None
        ),

        "day": day,

        "status": status,

        "calendar": calendar_name,

        "calendar_id": calendar_id,

        "html_link": event.get(
            "htmlLink"
        ),
    }


# ============================================================
# GENERIC EVENT FETCHER
# ============================================================

def get_events(
    start_datetime: datetime,
    end_datetime: datetime,
    calendar: str | None = None,
) -> list[dict[str, Any]]:
    """
    Fetch normalized events from one or all configured
    calendars.
    """

    if start_datetime.tzinfo is None:
        raise ValueError(
            "start_datetime must be timezone-aware."
        )

    if end_datetime.tzinfo is None:
        raise ValueError(
            "end_datetime must be timezone-aware."
        )

    if end_datetime <= start_datetime:
        raise ValueError(
            "end_datetime must be after start_datetime."
        )

    service = get_calendar_service()

    current = datetime.now(
        ZOE_TIMEZONE
    )

    # --------------------------------------------------------
    # Resolve calendars
    # --------------------------------------------------------

    if calendar is None:

        calendars = list(
            CALENDARS.items()
        )

    else:

        calendar_key = (
            calendar.strip().lower()
        )

        calendar_id = (
            resolve_calendar_id(
                calendar_key
            )
        )

        calendars = [
            (
                calendar_key,
                calendar_id,
            )
        ]

    all_events: list[
        dict[str, Any]
    ] = []

    # --------------------------------------------------------
    # Fetch
    # --------------------------------------------------------

    for (
        calendar_name,
        calendar_id,
    ) in calendars:

        try:

            response = (
                service.events()
                .list(
                    calendarId=calendar_id,

                    timeMin=(
                        start_datetime
                        .astimezone(
                            ZOE_TIMEZONE
                        )
                        .isoformat()
                    ),

                    timeMax=(
                        end_datetime
                        .astimezone(
                            ZOE_TIMEZONE
                        )
                        .isoformat()
                    ),

                    singleEvents=True,

                    orderBy="startTime",

                    maxResults=250,

                    fields=(
                        "items("
                        "id,"
                        "summary,"
                        "location,"
                        "description,"
                        "start,"
                        "end,"
                        "htmlLink"
                        ")"
                    ),
                )
                .execute()
            )

        except HttpError as exc:

            print(
                "[ZOE CALENDAR] "
                f"Failed to fetch "
                f"{calendar_name}: {exc}"
            )

            continue

        events = response.get(
            "items",
            [],
        )

        for event in events:

            parsed = _parse_event(
                event=event,
                calendar_name=calendar_name,
                calendar_id=calendar_id,
                current=current,
            )

            if parsed is not None:
                all_events.append(
                    parsed
                )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    all_events.sort(
        key=lambda event:
            event.get(
                "datetime",
                "9999-12-31",
            )
    )

    return all_events


# ============================================================
# TODAY
# ============================================================

def get_today_events(
    calendar: str | None = None,
) -> list[dict[str, Any]]:

    current = datetime.now(
        ZOE_TIMEZONE
    )

    start_of_today = current.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )

    start_of_tomorrow = (
        start_of_today
        + timedelta(days=1)
    )

    return get_events(
        start_datetime=start_of_today,
        end_datetime=start_of_tomorrow,
        calendar=calendar,
    )


# ============================================================
# CALENDAR-SPECIFIC FETCHERS
# ============================================================

def get_personal_events(
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> list[dict[str, Any]]:

    return _get_calendar_events_with_defaults(
        calendar="personal",
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )


def get_football_events(
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> list[dict[str, Any]]:

    return _get_calendar_events_with_defaults(
        calendar="football",
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )


def get_f1_events(
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> list[dict[str, Any]]:

    return _get_calendar_events_with_defaults(
        calendar="f1",
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )


def _get_calendar_events_with_defaults(
    calendar: str,
    start_datetime: datetime | None,
    end_datetime: datetime | None,
) -> list[dict[str, Any]]:

    current = datetime.now(
        ZOE_TIMEZONE
    )

    if start_datetime is None:
        start_datetime = current

    if end_datetime is None:
        end_datetime = (
            start_datetime
            + timedelta(days=7)
        )

    return get_events(
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        calendar=calendar,
    )


# ============================================================
# NEXT F1 RACE
# ============================================================

def get_next_f1_race() -> (
    dict[str, Any] | None
):
    """
    Return only the next actual F1 race.

    Practice, qualifying, and sprint events are ignored.
    """

    current = datetime.now(
        ZOE_TIMEZONE
    )

    events = get_events(
        start_datetime=current,
        end_datetime=(
            current
            + timedelta(days=180)
        ),
        calendar="f1",
    )

    for event in events:

        title = (
            event.get("title")
            or ""
        ).strip().lower()

        if not title:
            continue

        # Events that are not the actual race.
        excluded = (
            "practice",
            "free practice",
            "fp1",
            "fp2",
            "fp3",
            "qualifying",
            "sprint",
        )

        if any(
            item in title
            for item in excluded
        ):
            continue

        # Accept titles containing either:
        #   race
        #   grand prix
        #   gp
        #
        # This handles common calendar naming styles.
        is_race = (
            "race" in title
            or "grand prix" in title
            or " gp" in title
        )

        if not is_race:
            continue

        return event

    return None


# ============================================================
# NEXT FOOTBALL MATCH
# ============================================================

def get_next_football_match() -> (
    dict[str, Any] | None
):
    """
    Return the next upcoming football calendar event.
    """

    current = datetime.now(
        ZOE_TIMEZONE
    )

    events = get_events(
        start_datetime=current,
        end_datetime=(
            current
            + timedelta(days=60)
        ),
        calendar="football",
    )

    for event in events:

        if (
            event.get("status")
            == "upcoming"
        ):

            return event

    return None


# ============================================================
# SINGLE EVENT
# ============================================================

def get_event(
    event_id: str,
    calendar: str = "personal",
) -> dict[str, Any] | None:

    if not event_id:
        raise ValueError(
            "event_id cannot be empty."
        )

    service = get_calendar_service()

    calendar_key = (
        calendar.strip().lower()
    )

    calendar_id = resolve_calendar_id(
        calendar_key
    )

    try:

        event = (
            service.events()
            .get(
                calendarId=calendar_id,
                eventId=event_id,
            )
            .execute()
        )

    except HttpError as exc:

        if getattr(
            exc.resp,
            "status",
            None,
        ) == 404:

            return None

        raise

    return _parse_event(
        event=event,
        calendar_name=calendar_key,
        calendar_id=calendar_id,
    )


# ============================================================
# ADD EVENT
# ============================================================

def add_event(
    title: str,
    start_datetime: datetime,
    end_datetime: datetime | None = None,
    description: str | None = None,
) -> dict[str, Any]:

    if not title or not title.strip():
        raise ValueError(
            "title cannot be empty."
        )

    if start_datetime.tzinfo is None:
        raise ValueError(
            "start_datetime must be timezone-aware."
        )

    if end_datetime is None:
        end_datetime = (
            start_datetime
            + timedelta(hours=1)
        )

    if end_datetime.tzinfo is None:
        raise ValueError(
            "end_datetime must be timezone-aware."
        )

    if end_datetime <= start_datetime:
        raise ValueError(
            "end_datetime must be after start_datetime."
        )

    service = get_calendar_service()

    event = {
        "summary": title.strip(),

        "description": (
            description or ""
        ),

        "start": {
            "dateTime":
                start_datetime.isoformat(),

            "timeZone":
                str(
                    start_datetime.tzinfo
                ),
        },

        "end": {
            "dateTime":
                end_datetime.isoformat(),

            "timeZone":
                str(
                    end_datetime.tzinfo
                ),
        },
    }

    created_event = (
        service.events()
        .insert(
            calendarId=CALENDARS[
                "personal"
            ],
            body=event,
        )
        .execute()
    )

    return {
        "id": created_event.get(
            "id"
        ),

        "title": created_event.get(
            "summary"
        ),

        "datetime": (
            created_event
            .get("start", {})
            .get("dateTime")
        ),

        "end_datetime": (
            created_event
            .get("end", {})
            .get("dateTime")
        ),

        "calendar": "personal",

        "calendar_id": CALENDARS[
            "personal"
        ],

        "link": created_event.get(
            "htmlLink"
        ),
    }


# ============================================================
# DELETE EVENT
# ============================================================

def delete_event(
    event_id: str,
    calendar: str = "personal",
) -> bool:

    if not event_id:
        raise ValueError(
            "event_id cannot be empty."
        )

    service = get_calendar_service()

    calendar_id = resolve_calendar_id(
        calendar
    )

    try:

        (
            service.events()
            .delete(
                calendarId=calendar_id,
                eventId=event_id,
            )
            .execute()
        )

    except HttpError as exc:

        if getattr(
            exc.resp,
            "status",
            None,
        ) == 404:

            return False

        raise

    return True


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("\n=== TODAY ===")

    for event in get_today_events():

        print(
            event["calendar"],
            "|",
            event["title"],
            "|",
            event["time"],
            "|",
            event["status"],
        )

    print("\n=== NEXT F1 RACE ===")

    next_race = (
        get_next_f1_race()
    )

    print(
        next_race
    )

    print("\n=== NEXT FOOTBALL ===")

    next_match = (
        get_next_football_match()
    )

    print(
        next_match
    )