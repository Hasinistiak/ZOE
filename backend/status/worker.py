from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import os
import threading
import time
from typing import Any, Callable

from .events import EventBus, StatusEvent
from .state import StatusState


# ============================================================
# CONFIGURATION
# ============================================================


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float = 0.05,
) -> float:
    try:
        value = float(
            os.getenv(name, str(default))
        )
    except (TypeError, ValueError):
        value = default

    return max(minimum, value)


def _env_int(
    name: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    try:
        value = int(
            os.getenv(name, str(default))
        )
    except (TypeError, ValueError):
        value = default

    return max(minimum, value)


DEFAULT_INTERVALS = {
    "calendar": _env_float(
        "ZOE_STATUS_CALENDAR_INTERVAL",
        300,
    ),

    "tasks": _env_float(
        "ZOE_STATUS_TASK_INTERVAL",
        60,
    ),

    "mail": _env_float(
        "ZOE_STATUS_MAIL_INTERVAL",
        60,
    ),

    "weather": _env_float(
        "ZOE_STATUS_WEATHER_INTERVAL",
        1800,
    ),

    "spotify": _env_float(
        "ZOE_STATUS_SPOTIFY_INTERVAL",
        30,
    ),

    "system": _env_float(
        "ZOE_STATUS_SYSTEM_INTERVAL",
        30,
    ),
}


def _load_thresholds() -> tuple[int, ...]:
    raw = os.getenv(
        "ZOE_STATUS_EVENT_THRESHOLDS",
        "60,30,15,5",
    )

    values: set[int] = set()

    for value in raw.split(","):
        value = value.strip()

        if not value:
            continue

        try:
            parsed = int(value)
        except ValueError:
            continue

        if parsed >= 0:
            values.add(parsed)

    if not values:
        return (60, 30, 15, 5)

    return tuple(
        sorted(
            values,
            reverse=True,
        )
    )


REMINDER_THRESHOLDS = _load_thresholds()


TASK_OVERDUE_GRACE_MINUTES = _env_int(
    "ZOE_STATUS_TASK_OVERDUE_GRACE_MINUTES",
    30,
)


# ============================================================
# DEFAULT PROVIDERS
# ============================================================


def _default_sources() -> dict[
    str,
    Callable[[], Any],
]:
    def calendar() -> Any:
        from backend.functions.gcal import (
            get_today_events,
        )

        return get_today_events()

    def tasks() -> Any:
        from backend.functions.reminder import (
            get_reminders,
        )

        result = get_reminders()

        if isinstance(result, dict):
            return result.get(
                "reminders",
                [],
            )

        return result

    def mail() -> Any:
        from backend.functions.mail import (
            check_emails,
        )

        return check_emails(
            max_results=25,
            unread_only=True,
            since_seconds=60,
        )

    def weather() -> Any:
        from backend.functions.weather import (
            get_current_weather,
        )

        return get_current_weather()

    def spotify() -> Any:
        from backend.functions.spotify import get_spotify

        sp = get_spotify()

        return (
            sp.current_playback()
            or {}
        )

    def system() -> Any:
        """
        Basic host system observation.

        psutil is intentionally imported lazily so importing the status
        worker does not require psutil until system monitoring is enabled.
        """

        try:
            import psutil
        except ImportError:
            return {}

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "memory_percent": float(
                memory.percent
            ),
            "disk_percent": float(
                disk.percent
            ),
            "cpu_percent": float(
                psutil.cpu_percent(
                    interval=None
                )
            ),
        }

    return {
        "calendar": calendar,
        "tasks": tasks,
        "mail": mail,
        "weather": weather,
        "spotify": spotify,
        "system": system,
    }


# ============================================================
# NORMALIZATION HELPERS
# ============================================================


def _items(
    value: Any,
) -> list[dict[str, Any]]:
    """
    Normalize provider output into a list of dictionaries.
    """

    if isinstance(value, dict):
        value = value.get(
            "items",
            value.get(
                "reminders",
                value.get(
                    "events",
                    [],
                ),
            ),
        )

    if not isinstance(value, list):
        return []

    return [
        item
        for item in value
        if isinstance(item, dict)
    ]


def _key(
    item: dict[str, Any],
) -> str:
    """
    Stable-ish identity for provider items.

    Provider IDs should always win.
    """

    return str(
        item.get("id")
        or item.get("uri")
        or item.get("title")
        or item.get("name")
        or item
    )


def _parse_datetime(
    value: Any,
    now: datetime,
) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00",
            )
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(
            tzinfo=now.tzinfo
            or timezone.utc
        )

    return parsed


def _minutes_until(
    target: datetime,
    now: datetime,
) -> int:
    """
    Convert a timedelta to human-oriented integer minutes.

    Truncates toward zero instead of using floor division.

    Examples:

        +59 sec  ->  0
        +61 sec  ->  1
        -59 sec  ->  0
        -61 sec  -> -1
    """

    seconds = (
        target - now
    ).total_seconds()

    return int(seconds / 60)


# ============================================================
# STATUS WORKER
# ============================================================


class StatusWorker:
    """
    24/7 environmental observation layer.

    Responsibilities:

      - poll external state
      - maintain StatusState
      - detect structural changes
      - detect temporal conditions
      - emit structured StatusEvents
      - report provider health

    It does NOT decide whether ZOE should speak.

    The Brain owns that decision.
    """

    def __init__(
        self,
        event_bus: EventBus | None = None,
        state: StatusState | None = None,
        sources: dict[
            str,
            Callable[[], Any],
        ] | None = None,
        intervals: dict[
            str,
            float,
        ] | None = None,
        now: Callable[[], datetime] | None = None,
        poll_sleep: float = 1.0,
        recent_change_limit: int = 25,
    ) -> None:
        self.event_bus = (
            event_bus
            or EventBus()
        )

        self.state = (
            state
            or StatusState()
        )

        self.sources = (
            _default_sources()
            if sources is None
            else dict(sources)
        )

        self.intervals = {
            **DEFAULT_INTERVALS,
            **(
                intervals
                or {}
            ),
        }

        self.intervals = {
            source: max(
                0.05,
                float(interval),
            )
            for source, interval
            in self.intervals.items()
        }

        self.now = (
            now
            or (
                lambda:
                datetime.now(
                    timezone.utc
                )
            )
        )

        self.poll_sleep = max(
            0.05,
            poll_sleep,
        )

        self.recent_change_limit = max(
            1,
            recent_change_limit,
        )

        self._stop = threading.Event()

        self._thread: threading.Thread | None = None

        self._last_refresh: dict[
            str,
            float,
        ] = {}

        # Last successfully observed provider value.
        self._previous: dict[
            str,
            Any,
        ] = {}

        # Reminder markers that have already fired.
        #
        # Each marker is:
        #
        #     (source, item_key, occurrence, threshold)
        #
        self._thresholds_fired: set[
            tuple[str, str, str, int]
        ] = set()

        # Current source health.
        self._source_status: dict[
            str,
            str,
        ] = {}

        self._lock = threading.RLock()

    # ========================================================
    # LIFECYCLE
    # ========================================================

    @property
    def running(self) -> bool:
        thread = self._thread

        return bool(
            thread
            and thread.is_alive()
        )

    def start(self) -> None:
        with self._lock:
            if self.running:
                return

            self._stop.clear()

            self._thread = threading.Thread(
                target=self._run,
                name="zoe-status",
                daemon=True,
            )

            self._thread.start()

            print(
                "[STATUS] Worker started"
            )

    def stop(
        self,
        timeout: float = 5.0,
    ) -> None:
        self._stop.set()

        thread = self._thread

        if (
            thread
            and thread
            is not threading.current_thread()
        ):
            thread.join(
                timeout=timeout
            )

        print(
            "[STATUS] Worker stopped"
        )

    # ========================================================
    # PUBLIC STATE API
    # ========================================================

    def snapshot(self) -> dict[str, Any]:
        return self.state.snapshot()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    def _run(self) -> None:
        first_cycle = True

        while not self._stop.is_set():
            monotonic_now = time.monotonic()

            # Use exactly one clock value for this worker cycle.
            cycle_now = self.now()

            self.state.update_clock(
                cycle_now
            )

            for source in list(
                self.sources
            ):
                interval = self.intervals.get(
                    source,
                    300.0,
                )

                last = self._last_refresh.get(
                    source
                )

                due = (
                    first_cycle
                    or last is None
                    or (
                        monotonic_now
                        - last
                        >= interval
                    )
                )

                if not due:
                    continue

                self._last_refresh[source] = (
                    monotonic_now
                )

                self.refresh(
                    source,
                    initial=first_cycle,
                    now=cycle_now,
                )

            first_cycle = False

            self._stop.wait(
                self.poll_sleep
            )

    # ========================================================
    # REFRESH
    # ========================================================

    def refresh(
        self,
        source: str,
        *,
        initial: bool = False,
        now: datetime | None = None,
    ) -> list[StatusEvent]:
        provider = self.sources.get(
            source
        )

        if provider is None:
            return []

        current_now = (
            now
            or self.now()
        )

        self.state.update_clock(
            current_now
        )

        # ----------------------------------------------------
        # Provider execution
        # ----------------------------------------------------

        try:
            value = provider()

        except Exception as exc:
            return self._handle_source_failure(
                source,
                exc,
                current_now,
            )

        # ----------------------------------------------------
        # Compare against last successful state
        # ----------------------------------------------------

        with self._lock:
            previous_exists = (
                source
                in self._previous
            )

            previous = deepcopy(
                self._previous.get(
                    source
                )
            )

        events: list[StatusEvent] = []

        if (
            not initial
            and previous_exists
        ):
            events.extend(
                self._detect_changes(
                    source,
                    previous,
                    value,
                    current_now,
                )
            )

        # ----------------------------------------------------
        # Commit successful state
        # ----------------------------------------------------

        with self._lock:
            self._previous[source] = (
                deepcopy(value)
            )

            was_unavailable = (
                self._source_status.get(
                    source
                )
                == "unavailable"
            )

            self._source_status[source] = (
                "current"
            )

        self._set_source(
            source,
            value,
            "current",
        )

        # ----------------------------------------------------
        # Source recovery
        # ----------------------------------------------------

        if was_unavailable:
            events.insert(
                0,
                StatusEvent(
                    type="source_recovered",
                    source=source,
                    data={
                        "source": source,
                    },
                    importance="normal",
                    identity=(
                        f"{source}:recovered"
                    ),
                ),
            )

        # ----------------------------------------------------
        # Publish
        # ----------------------------------------------------

        for event in events:
            self._record_and_publish(
                event
            )

        return events

    # ========================================================
    # SOURCE FAILURE
    # ========================================================

    def _handle_source_failure(
        self,
        source: str,
        exc: Exception,
        now: datetime,
    ) -> list[StatusEvent]:
        with self._lock:
            was_unavailable = (
                self._source_status.get(
                    source
                )
                == "unavailable"
            )

            previous = deepcopy(
                self._previous.get(
                    source
                )
            )

            self._source_status[source] = (
                "unavailable"
            )

        # Keep the last valid state intact.
        self._set_source(
            source,
            previous,
            "unavailable",
            str(exc),
        )

        print(
            "[STATUS] "
            f"{source} unavailable: "
            f"{type(exc).__name__}: {exc}"
        )

        # Do not repeatedly emit the same failure.
        if was_unavailable:
            return []

        event = StatusEvent(
            type="source_unavailable",
            source=source,
            data={
                "source": source,
                "error": str(exc),
            },
            importance="normal",
            identity=(
                f"{source}:unavailable"
            ),
        )

        self._record_and_publish(
            event
        )

        return [event]

    # ========================================================
    # STATE UPDATE
    # ========================================================

    def _set_source(
        self,
        source: str,
        value: Any,
        status: str,
        error: str = "",
    ) -> None:
        with self.state._lock:
            source_state: dict[
                str,
                Any,
            ] = {
                "status": status,
                "updated_at": self.state.timestamp,
            }

            if error:
                source_state["error"] = error

            self.state.sources[source] = (
                source_state
            )

            # Never destroy valid state because a provider
            # temporarily failed.
            if status == "unavailable":
                return

            if source == "calendar":
                self._set_calendar_state(
                    value
                )

            elif source == "tasks":
                self._set_task_state(
                    value
                )

            elif source == "mail":
                self._set_mail_state(
                    value
                )

            elif source == "weather":
                self.state.weather = (
                    deepcopy(value)
                    if isinstance(
                        value,
                        dict,
                    )
                    else {}
                )

            elif source == "spotify":
                self.state.spotify = (
                    deepcopy(value)
                    if isinstance(
                        value,
                        dict,
                    )
                    else {}
                )

            elif source == "system":
                self.state.system = (
                    deepcopy(value)
                    if isinstance(
                        value,
                        dict,
                    )
                    else {}
                )

    def _set_calendar_state(
        self,
        value: Any,
    ) -> None:
        events = _items(value)

        # State timestamp is already available, but parsing uses
        # the current runtime clock.
        now = self.now()

        def event_sort_key(
            event: dict[str, Any],
        ) -> datetime:
            parsed = _parse_datetime(
                event.get("datetime")
                or event.get("start"),
                now,
            )

            if parsed is not None:
                return parsed

            # Make this comparable with both UTC and local aware
            # datetimes.
            return datetime.max.replace(
                tzinfo=timezone.utc
            )

        events = sorted(
            (
                deepcopy(event)
                for event in events
            ),
            key=event_sort_key,
        )

        self.state.calendar = {
            "events": events
        }

        self.state.next_event = (
            deepcopy(events[0])
            if events
            else None
        )

    def _set_task_state(
        self,
        value: Any,
    ) -> None:
        tasks = _items(value)

        # Calculate overdue state from one clock value.
        now = self.now()

        self.state.tasks = deepcopy(
            tasks
        )

        overdue: list[
            dict[str, Any]
        ] = []

        for task in tasks:
            if task.get("completed"):
                continue

            if self._task_is_overdue(
                task,
                now=now,
            ):
                overdue.append(
                    deepcopy(task)
                )

        self.state.overdue_tasks = overdue

    def _set_mail_state(
        self,
        value: Any,
    ) -> None:
        """
        Normalize Gmail provider output into StatusState.mail.

        The Gmail provider returns:

            {
                "success": bool,
                "count": int,
                "emails": [...]
            }

        Keep the provider's useful metadata while also exposing
        a convenient unread count for the Brain/UI.
        """

        if not isinstance(
            value,
            dict,
        ):
            self.state.mail = {}

            return

        emails = value.get(
            "emails",
            [],
        )

        if not isinstance(
            emails,
            list,
        ):
            emails = []

        emails = [
            deepcopy(email)
            for email in emails
            if isinstance(
                email,
                dict,
            )
        ]

        unread_count = sum(
            1
            for email in emails
            if email.get(
                "unread",
                False,
            )
        )

        self.state.mail = {
            "success": bool(
                value.get(
                    "success",
                    False,
                )
            ),

            "count": len(emails),

            "unread_count":
                unread_count,

            "emails": emails,

            "updated_at":
                self.state.timestamp,
        }

        if value.get("error"):
            self.state.mail["error"] = (
                value["error"]
            )

    # ========================================================
    # CHANGE DETECTION
    # ========================================================

    def _detect_changes(
        self,
        source: str,
        old: Any,
        new: Any,
        now: datetime,
    ) -> list[StatusEvent]:
        """
        Detect both structural changes and time-based conditions.

        Temporal conditions are deliberately evaluated on every
        successful refresh.

        This is critical for reminders because time can pass without
        the provider's returned data changing.
        """

        if source == "calendar":
            return self._calendar_events(
                old,
                new,
                now,
            )

        if source == "tasks":
            return self._task_events(
                old,
                new,
                now,
            )

        if source == "mail":
            return self._mail_events(
                old,
                new,
            )

        if source == "weather":
            return self._weather_events(
                old,
                new,
            )

        if source == "spotify":
            return self._spotify_events(
                old,
                new,
            )

        if source == "system":
            return self._system_events(
                old,
                new,
            )

        return []

    # ========================================================
    # CALENDAR
    # ========================================================

    def _calendar_events(
        self,
        old: Any,
        new: Any,
        now: datetime,
    ) -> list[StatusEvent]:
        before = {
            _key(item): item
            for item in _items(old)
        }

        after = {
            _key(item): item
            for item in _items(new)
        }

        result: list[StatusEvent] = []

        # ----------------------------------------------------
        # Structural changes
        # ----------------------------------------------------

        for key, item in after.items():
            if key not in before:
                result.append(
                    StatusEvent(
                        type="event_added",
                        source="calendar",
                        data=deepcopy(item),
                        importance="normal",
                        identity=(
                            f"{key}:event_added"
                        ),
                    )
                )

        for key, item in before.items():
            if key not in after:
                result.append(
                    StatusEvent(
                        type="event_removed",
                        source="calendar",
                        data=deepcopy(item),
                        importance="normal",
                        identity=(
                            f"{key}:event_removed"
                        ),
                    )
                )

        for key in (
            after.keys()
            & before.keys()
        ):
            if after[key] != before[key]:
                result.append(
                    StatusEvent(
                        type="event_changed",
                        source="calendar",
                        data=deepcopy(
                            after[key]
                        ),
                        importance="normal",
                        identity=(
                            f"{key}:event_changed"
                        ),
                    )
                )

        # ----------------------------------------------------
        # Temporal reminders
        #
        # IMPORTANT:
        # This runs every calendar refresh even if the provider
        # returned exactly the same data.
        # ----------------------------------------------------

        result.extend(
            self._approaching_events(
                after,
                now,
            )
        )

        # ----------------------------------------------------
        # Remove reminder markers for events that no longer
        # exist or whose occurrence changed.
        # ----------------------------------------------------

        self._prune_calendar_markers(
            after
        )

        return result

    def _approaching_events(
        self,
        events: dict[
            str,
            dict[str, Any],
        ],
        now: datetime,
    ) -> list[StatusEvent]:
        candidates: list[
            tuple[
                int,
                str,
                str,
                dict[str, Any],
            ]
        ] = []

        for key, event in events.items():
            if event.get("completed"):
                continue

            start = _parse_datetime(
                event.get("datetime")
                or event.get("start"),
                now,
            )

            if start is None:
                continue

            minutes = _minutes_until(
                start,
                now,
            )

            if minutes < 0:
                continue

            occurrence = str(
                event.get("datetime")
                or event.get("start")
                or ""
            )

            candidates.append(
                (
                    minutes,
                    key,
                    occurrence,
                    event,
                )
            )

        candidates.sort(
            key=lambda item: item[0]
        )

        result: list[StatusEvent] = []

        for (
            minutes,
            key,
            occurrence,
            event,
        ) in candidates:
            eligible = [
                threshold
                for threshold
                in REMINDER_THRESHOLDS
                if minutes <= threshold
            ]

            if not eligible:
                continue

            threshold = min(
                eligible
            )

            marker = (
                "calendar",
                key,
                occurrence,
                threshold,
            )

            if marker in self._thresholds_fired:
                continue

            self._thresholds_fired.add(
                marker
            )

            result.append(
                StatusEvent(
                    type="event_approaching",
                    source="calendar",
                    data={
                        **deepcopy(event),
                        "minutes_until": minutes,
                        "threshold": threshold,
                    },
                    importance="high",
                    identity=(
                        f"{key}:"
                        f"{occurrence}:"
                        f"approaching:"
                        f"{threshold}"
                    ),
                )
            )

        return result

    def _prune_calendar_markers(
        self,
        events: dict[
            str,
            dict[str, Any],
        ],
    ) -> None:
        valid_occurrences = {
            (
                key,
                str(
                    event.get("datetime")
                    or event.get("start")
                    or ""
                ),
            )
            for key, event in events.items()
        }

        with self._lock:
            self._thresholds_fired = {
                marker
                for marker
                in self._thresholds_fired
                if (
                    marker[0] != "calendar"
                    or (
                        marker[1],
                        marker[2],
                    )
                    in valid_occurrences
                )
            }

    # ========================================================
    # TASKS
    # ========================================================

    def _task_events(
        self,
        old: Any,
        new: Any,
        now: datetime,
    ) -> list[StatusEvent]:
        before = {
            _key(item): item
            for item in _items(old)
        }

        after = {
            _key(item): item
            for item in _items(new)
        }

        result: list[StatusEvent] = []

        # ----------------------------------------------------
        # Added
        # ----------------------------------------------------

        for key, item in after.items():
            if key not in before:
                result.append(
                    StatusEvent(
                        type="task_added",
                        source="tasks",
                        data=deepcopy(item),
                        importance="normal",
                        identity=(
                            f"{key}:task_added"
                        ),
                    )
                )

        # ----------------------------------------------------
        # Removed
        # ----------------------------------------------------

        for key, item in before.items():
            if key not in after:
                result.append(
                    StatusEvent(
                        type="task_removed",
                        source="tasks",
                        data=deepcopy(item),
                        importance="normal",
                        identity=(
                            f"{key}:task_removed"
                        ),
                    )
                )

        # ----------------------------------------------------
        # Existing task changes
        # ----------------------------------------------------

        for key in (
            after.keys()
            & before.keys()
        ):
            old_task = before[key]
            new_task = after[key]

            if (
                not old_task.get("completed")
                and new_task.get("completed")
            ):
                result.append(
                    StatusEvent(
                        type="task_completed",
                        source="tasks",
                        data=deepcopy(
                            new_task
                        ),
                        importance="normal",
                        identity=(
                            f"{key}:"
                            f"task_completed"
                        ),
                    )
                )

            elif (
                old_task != new_task
                and not new_task.get("completed")
            ):
                result.append(
                    StatusEvent(
                        type="task_changed",
                        source="tasks",
                        data=deepcopy(
                            new_task
                        ),
                        importance="normal",
                        identity=(
                            f"{key}:task_changed"
                        ),
                    )
                )

        # ----------------------------------------------------
        # Temporal task events
        # ----------------------------------------------------

        result.extend(
            self._task_due_events(
                after,
                now,
            )
        )

        self._prune_task_markers(
            after
        )

        return result

    def _task_is_overdue(
        self,
        task: dict[str, Any],
        *,
        now: datetime | None = None,
    ) -> bool:
        if task.get("overdue"):
            return True

        if task.get("is_overdue"):
            return True

        current_now = (
            now
            or self.now()
        )

        minutes = self._task_minutes_until(
            task,
            current_now,
        )

        return (
            minutes is not None
            and minutes < 0
        )

    def _task_minutes_until(
        self,
        task: dict[str, Any],
        now: datetime | None = None,
    ) -> int | None:
        raw = (
            task.get("due_datetime")
            or task.get("due_at")
        )

        if not raw:
            return None

        current_now = (
            now
            or self.now()
        )

        due = _parse_datetime(
            raw,
            current_now,
        )

        if due is None:
            return None

        return _minutes_until(
            due,
            current_now,
        )

    def _task_due_events(
        self,
        tasks: dict[
            str,
            dict[str, Any],
        ],
        now: datetime,
    ) -> list[StatusEvent]:
        result: list[StatusEvent] = []

        for key, task in tasks.items():
            if task.get("completed"):
                continue

            minutes = self._task_minutes_until(
                task,
                now,
            )

            if minutes is None:
                continue

            due_identity = str(
                task.get("due_datetime")
                or task.get("due_at")
            )

            # ------------------------------------------------
            # Due
            # ------------------------------------------------

            if minutes <= 0:
                marker = (
                    "tasks",
                    key,
                    due_identity,
                    0,
                )

                if marker not in (
                    self._thresholds_fired
                ):
                    self._thresholds_fired.add(
                        marker
                    )

                    result.append(
                        StatusEvent(
                            type="task_due",
                            source="tasks",
                            data={
                                **deepcopy(
                                    task
                                ),
                                "minutes_until": minutes,
                            },
                            importance="high",
                            identity=(
                                f"{key}:"
                                f"{due_identity}:"
                                f"task_due"
                            ),
                        )
                    )

            # ------------------------------------------------
            # Overdue after grace period
            # ------------------------------------------------

            if (
                minutes
                <= -TASK_OVERDUE_GRACE_MINUTES
            ):
                marker = (
                    "tasks",
                    key,
                    due_identity,
                    -TASK_OVERDUE_GRACE_MINUTES,
                )

                if marker not in (
                    self._thresholds_fired
                ):
                    self._thresholds_fired.add(
                        marker
                    )

                    result.append(
                        StatusEvent(
                            type="task_overdue",
                            source="tasks",
                            data={
                                **deepcopy(
                                    task
                                ),
                                "minutes_overdue": abs(
                                    minutes
                                ),
                            },
                            importance="high",
                            identity=(
                                f"{key}:"
                                f"{due_identity}:"
                                f"task_overdue"
                            ),
                        )
                    )

        return result

    def _prune_task_markers(
        self,
        tasks: dict[
            str,
            dict[str, Any],
        ],
    ) -> None:
        valid = {
            (
                key,
                str(
                    task.get("due_datetime")
                    or task.get("due_at")
                    or ""
                ),
            )
            for key, task in tasks.items()
        }

        with self._lock:
            self._thresholds_fired = {
                marker
                for marker
                in self._thresholds_fired
                if (
                    marker[0] != "tasks"
                    or (
                        marker[1],
                        marker[2],
                    )
                    in valid
                )
            }

    # ========================================================
    # MAIL
    # ========================================================

    def _mail_events(
        self,
        old: Any,
        new: Any,
    ) -> list[StatusEvent]:
        """
        Detect Gmail changes.

        The current Gmail provider returns a recent-message
        window rather than a complete mailbox snapshot.

        Therefore:

          - new IDs are treated as newly observed emails
          - unread-state changes are detected
          - disappearing messages are NOT treated as deleted

        This prevents false "email removed" notifications when
        an older message simply falls outside the provider's
        recent-results window.
        """

        if not isinstance(
            old,
            dict,
        ):
            old = {}

        if not isinstance(
            new,
            dict,
        ):
            new = {}

        before_emails = old.get(
            "emails",
            [],
        )

        after_emails = new.get(
            "emails",
            [],
        )

        if not isinstance(
            before_emails,
            list,
        ):
            before_emails = []

        if not isinstance(
            after_emails,
            list,
        ):
            after_emails = []

        before = {
            str(email.get("id")): email
            for email in before_emails
            if isinstance(
                email,
                dict,
            )
            and email.get("id")
        }

        after = {
            str(email.get("id")): email
            for email in after_emails
            if isinstance(
                email,
                dict,
            )
            and email.get("id")
        }

        result: list[StatusEvent] = []

        # ----------------------------------------------------
        # New emails
        # ----------------------------------------------------

        for key, email in after.items():

            if key not in before:

                result.append(
                    StatusEvent(
                        type="email_received",
                        source="mail",
                        data=deepcopy(
                            email
                        ),
                        importance=(
                            "high"
                            if email.get(
                                "unread",
                                False,
                            )
                            else "normal"
                        ),
                        identity=(
                            f"{key}:"
                            "email_received"
                        ),
                    )
                )

        # ----------------------------------------------------
        # Existing email changes
        # ----------------------------------------------------

        for key in (
            after.keys()
            & before.keys()
        ):
            old_email = before[key]
            new_email = after[key]

            old_unread = bool(
                old_email.get(
                    "unread",
                    False,
                )
            )

            new_unread = bool(
                new_email.get(
                    "unread",
                    False,
                )
            )

            # ------------------------------------------------
            # Read -> unread
            # ------------------------------------------------

            if (
                not old_unread
                and new_unread
            ):
                result.append(
                    StatusEvent(
                        type="email_unread",
                        source="mail",
                        data=deepcopy(
                            new_email
                        ),
                        importance="high",
                        identity=(
                            f"{key}:"
                            "email_unread"
                        ),
                    )
                )

            # ------------------------------------------------
            # Other metadata change
            # ------------------------------------------------

            elif (
                old_email
                != new_email
            ):
                result.append(
                    StatusEvent(
                        type="email_changed",
                        source="mail",
                        data=deepcopy(
                            new_email
                        ),
                        importance="normal",
                        identity=(
                            f"{key}:"
                            "email_changed"
                        ),
                    )
                )

        # ----------------------------------------------------
        # Aggregate unread-count change
        # ----------------------------------------------------

        old_unread_count = sum(
            1
            for email in before_emails
            if isinstance(
                email,
                dict,
            )
            and email.get(
                "unread",
                False,
            )
        )

        new_unread_count = sum(
            1
            for email in after_emails
            if isinstance(
                email,
                dict,
            )
            and email.get(
                "unread",
                False,
            )
        )

        if (
            old_unread_count
            != new_unread_count
        ):
            # Do not emit a second generic event if we already
            # emitted specific email_unread events. The specific
            # event carries better information for the Brain.
            if not any(
                event.type
                == "email_unread"
                for event in result
            ):
                result.append(
                    StatusEvent(
                        type="email_unread_count_changed",
                        source="mail",
                        data={
                            "old_unread_count":
                                old_unread_count,

                            "new_unread_count":
                                new_unread_count,
                        },
                        importance="normal",
                        identity=(
                            "mail:"
                            "unread_count_changed"
                        ),
                    )
                )

        return result

    # ========================================================
    # WEATHER
    # ========================================================

    def _weather_events(
        self,
        old: Any,
        new: Any,
    ) -> list[StatusEvent]:
        if not isinstance(
            old,
            dict,
        ):
            return []

        if not isinstance(
            new,
            dict,
        ):
            return []

        old_condition = str(
            old.get(
                "condition",
                "",
            )
        ).lower()

        new_condition = str(
            new.get(
                "condition",
                "",
            )
        ).lower()

        old_temp = old.get(
            "temperature_c"
        )

        new_temp = new.get(
            "temperature_c"
        )

        meaningful = (
            old_condition
            != new_condition
            or (
                isinstance(
                    old_temp,
                    (int, float),
                )
                and isinstance(
                    new_temp,
                    (int, float),
                )
                and abs(
                    new_temp
                    - old_temp
                )
                >= 2
            )
        )

        if not meaningful:
            return []

        return [
            StatusEvent(
                type="weather_changed",
                source="weather",
                data=deepcopy(new),
                importance="normal",
                identity="weather_changed",
            )
        ]

    # ========================================================
    # SPOTIFY
    # ========================================================

    def _spotify_events(
        self,
        old: Any,
        new: Any,
    ) -> list[StatusEvent]:
        if not isinstance(
            old,
            dict,
        ):
            return []

        if not isinstance(
            new,
            dict,
        ):
            return []

        old_item = (
            old.get("item")
            or {}
        ).get("id")

        new_item = (
            new.get("item")
            or {}
        ).get("id")

        old_playing = bool(
            old.get("is_playing")
        )

        new_playing = bool(
            new.get("is_playing")
        )

        result: list[StatusEvent] = []

        if (
            old_item != new_item
            and new_item
        ):
            result.append(
                StatusEvent(
                    type="track_changed",
                    source="spotify",
                    data=deepcopy(new),
                    importance="low",
                    identity=(
                        "spotify:"
                        f"track_changed:"
                        f"{new_item}"
                    ),
                )
            )

        if (
            old_playing
            != new_playing
        ):
            event_type = (
                "spotify_started"
                if new_playing
                else "spotify_stopped"
            )

            result.append(
                StatusEvent(
                    type=event_type,
                    source="spotify",
                    data=deepcopy(new),
                    importance="low",
                    identity=event_type,
                )
            )

        return result

    # ========================================================
    # SYSTEM
    # ========================================================

    def _system_events(
        self,
        old: Any,
        new: Any,
    ) -> list[StatusEvent]:
        if not isinstance(
            old,
            dict,
        ):
            return []

        if not isinstance(
            new,
            dict,
        ):
            return []

        events: list[StatusEvent] = []

        old_memory = old.get(
            "memory_percent"
        )

        new_memory = new.get(
            "memory_percent"
        )

        old_disk = old.get(
            "disk_percent"
        )

        new_disk = new.get(
            "disk_percent"
        )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        if (
            isinstance(
                new_memory,
                (int, float),
            )
            and new_memory >= 90
            and (
                not isinstance(
                    old_memory,
                    (int, float),
                )
                or old_memory < 90
            )
        ):
            events.append(
                StatusEvent(
                    type="system_memory_high",
                    source="system",
                    data=deepcopy(new),
                    importance="high",
                    identity=(
                        "system:"
                        "memory_high"
                    ),
                )
            )

        # ----------------------------------------------------
        # Disk
        # ----------------------------------------------------

        if (
            isinstance(
                new_disk,
                (int, float),
            )
            and new_disk >= 90
            and (
                not isinstance(
                    old_disk,
                    (int, float),
                )
                or old_disk < 90
            )
        ):
            events.append(
                StatusEvent(
                    type="system_disk_high",
                    source="system",
                    data=deepcopy(new),
                    importance="high",
                    identity=(
                        "system:"
                        "disk_high"
                    ),
                )
            )

        return events

    # ========================================================
    # EVENT PUBLISHING
    # ========================================================

    def _record_and_publish(
        self,
        event: StatusEvent,
    ) -> None:
        self.state.append_change(
            event.to_dict(),
            limit=self.recent_change_limit,
        )

        print(
            "[STATUS] "
            "Change detected "
            f"type={event.type} "
            f"source={event.source}"
        )

        self.event_bus.publish(
            event
        )