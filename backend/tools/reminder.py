from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from todoist_api_python.api import TodoistAPI


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

TODOIST_API_TOKEN = os.getenv("TODOIST_API_TOKEN")

if not TODOIST_API_TOKEN:
    raise RuntimeError(
        "TODOIST_API_TOKEN is not set. "
        "Add it to your .env file."
    )

api = TodoistAPI(TODOIST_API_TOKEN)


# ============================================================
# TODOIST TASK RETRIEVAL
# ============================================================


def _get_all_tasks():
    """
    Flatten Todoist's paginated task response.
    """

    tasks = []

    for page in api.get_tasks():
        tasks.extend(page)

    return tasks


# ============================================================
# SERIALIZATION
# ============================================================


def _serialize_due_date(task) -> str | None:
    """
    Serialize Todoist's due.date value.
    """

    if not task.due:
        return None

    value = getattr(
        task.due,
        "date",
        None,
    )

    if value is None:
        return None

    return (
        value.isoformat()
        if hasattr(value, "isoformat")
        else str(value)
    )


def _serialize_due_datetime(task) -> str | None:
    """
    Return Todoist's exact datetime when available.

    Todoist's SDK can expose timed tasks through either:

        due.datetime

    or:

        due.date
    """

    if not task.due:
        return None

    # --------------------------------------------------------
    # Preferred representation
    # --------------------------------------------------------

    value = getattr(
        task.due,
        "datetime",
        None,
    )

    if value is not None:

        serialized = (
            value.isoformat()
            if hasattr(value, "isoformat")
            else str(value)
        )

        if "T" in serialized:
            return serialized

    # --------------------------------------------------------
    # Fallback representation
    # --------------------------------------------------------

    value = getattr(
        task.due,
        "date",
        None,
    )

    if value is not None:

        serialized = (
            value.isoformat()
            if hasattr(value, "isoformat")
            else str(value)
        )

        if "T" in serialized:
            return serialized

    return None


def _serialize_task(task) -> dict[str, Any]:
    """
    Convert a Todoist task into a ZOE-friendly dictionary.
    """

    due_datetime = _serialize_due_datetime(
        task
    )

    return {
        "id": str(task.id),

        "name": task.content,

        "due": (
            task.due.string
            if task.due
            else None
        ),

        "due_date": _serialize_due_date(
            task
        ),

        "due_datetime": due_datetime,

        "has_exact_time": (
            due_datetime is not None
        ),

        "completed": bool(
            getattr(
                task,
                "is_completed",
                False,
            )
        ),
    }


# ============================================================
# CREATE REMINDER
# ============================================================


def create_reminder(
    name: str,
    due_string: str | None = None,
):
    """
    Create exactly one Todoist task.

    IMPORTANT:

    This function does NOT interpret natural-language time.

    The Reminder Agent is responsible for resolving:

        "in 5 minutes"
        "tomorrow at 8 PM"
        "Friday at 7:30 PM"

    into an appropriate Todoist due value.

    This function simply passes the resolved value to Todoist.
    """

    # --------------------------------------------------------
    # Validate name
    # --------------------------------------------------------

    if not name:

        raise ValueError(
            "Reminder name is required."
        )

    name = str(
        name
    ).strip()

    if not name:

        raise ValueError(
            "Reminder name is required."
        )

    # --------------------------------------------------------
    # Normalize due string
    # --------------------------------------------------------

    if due_string is not None:

        due_string = str(
            due_string
        ).strip()

        if not due_string:
            due_string = None

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print()
    print(
        "[REMINDER TOOL] Creating reminder"
    )

    print(
        f"[REMINDER TOOL] Name: {name}"
    )

    print(
        f"[REMINDER TOOL] Due: {due_string}"
    )

    # --------------------------------------------------------
    # Create Todoist task
    # --------------------------------------------------------

    try:

        task = api.add_task(
            content=name,
            due_string=due_string,
        )

        serialized = _serialize_task(
            task
        )

        result = {
            "success": True,
            **serialized,
        }

        print(
            "[REMINDER TOOL] Created:"
        )

        print(
            result
        )

        return result

    except Exception as exc:

        print(
            "[REMINDER TOOL] Creation failed:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "success": False,
            "error": str(exc),
        }


# ============================================================
# GET REMINDERS
# ============================================================


def get_reminders():
    """
    Return all active Todoist reminders.
    """

    try:

        tasks = _get_all_tasks()

        reminders = []

        for task in tasks:

            if getattr(
                task,
                "is_completed",
                False,
            ):
                continue

            reminders.append(
                _serialize_task(
                    task
                )
            )

        return {
            "success": True,
            "reminders": reminders,
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
            "reminders": [],
        }


# ============================================================
# DELETE REMINDER
# ============================================================


def delete_reminder(
    name: str,
    allow_partial_match: bool = False,
):
    """
    Delete exactly one Todoist reminder.

    Exact matching is always preferred.

    Partial matching is only performed when explicitly
    requested by the caller.
    """

    if not name:

        return {
            "success": False,
            "error": (
                "Reminder name is required."
            ),
        }

    try:

        tasks = _get_all_tasks()

        target = (
            str(name)
            .strip()
            .lower()
        )

        # ----------------------------------------------------
        # Exact match
        # ----------------------------------------------------

        exact_matches = [
            task
            for task in tasks
            if (
                task.content
                .strip()
                .lower()
                == target
            )
        ]

        if len(exact_matches) == 1:

            task = exact_matches[0]

        elif len(exact_matches) > 1:

            return {
                "success": False,
                "error": (
                    f"Multiple reminders exactly "
                    f"match '{name}'; "
                    "refusing to guess."
                ),
            }

        # ----------------------------------------------------
        # Partial match
        # ----------------------------------------------------

        elif allow_partial_match:

            partial_matches = [
                task
                for task in tasks
                if target
                in task.content.lower()
            ]

            if not partial_matches:

                return {
                    "success": False,
                    "error": (
                        f"Could not find reminder: "
                        f"{name}"
                    ),
                }

            if len(partial_matches) > 1:

                names = ", ".join(
                    task.content
                    for task in partial_matches
                )

                return {
                    "success": False,
                    "error": (
                        f"Multiple reminders partially "
                        f"match '{name}' ({names}); "
                        "refusing to guess."
                    ),
                }

            task = partial_matches[0]

        # ----------------------------------------------------
        # No match
        # ----------------------------------------------------

        else:

            return {
                "success": False,
                "error": (
                    f"Could not find reminder: "
                    f"{name}"
                ),
            }

        # ----------------------------------------------------
        # Delete
        # ----------------------------------------------------

        api.delete_task(
            task.id
        )

        return {
            "success": True,
            "id": str(task.id),
            "name": task.content,
            "message": "Reminder deleted",
        }

    except Exception as exc:

        return {
            "success": False,
            "error": str(exc),
        }
