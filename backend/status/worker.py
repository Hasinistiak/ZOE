
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
import threading
import time
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from typing import Any, Callable

from .events import EventBus, StatusEvent
from .state import StatusState


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION HELPERS
# ============================================================


def _env_float(
    name: str,
    default: float,
    *,
    minimum: float = 0.05,
) -> float:
    try:
        value = float(os.getenv(name, str(default)))
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
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default

    return max(minimum, value)


def _env_bool(
    name: str,
    default: bool,
) -> bool:
    raw = os.getenv(name)

    if raw is None:
        return default

    return raw.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# ============================================================
# POLLING INTERVALS
# ============================================================


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


# ============================================================
# PROVIDER TIMEOUTS
# ============================================================

DEFAULT_PROVIDER_TIMEOUTS = {
    "calendar": _env_float(
        "ZOE_STATUS_CALENDAR_TIMEOUT",
        10,
    ),

    "tasks": _env_float(
        "ZOE_STATUS_TASK_TIMEOUT",
        5,
    ),

    "mail": _env_float(
        "ZOE_STATUS_MAIL_TIMEOUT",
        10,
    ),

    "weather": _env_float(
        "ZOE_STATUS_WEATHER_TIMEOUT",
        8,
    ),

    "spotify": _env_float(
        "ZOE_STATUS_SPOTIFY_TIMEOUT",
        5,
    ),

    "system": _env_float(
        "ZOE_STATUS_SYSTEM_TIMEOUT",
        3,
    ),
}


# ============================================================
# REMINDER THRESHOLDS
# ============================================================


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


# ============================================================
# TASK OVERDUE
# ============================================================


TASK_OVERDUE_GRACE_MINUTES = _env_int(
    "ZOE_STATUS_TASK_OVERDUE_GRACE_MINUTES",
    30,
)


# ============================================================
# SYSTEM THRESHOLDS
# ============================================================


SYSTEM_MEMORY_HIGH_THRESHOLD = _env_float(
    "ZOE_STATUS_MEMORY_HIGH_THRESHOLD",
    90,
    minimum=0,
)


SYSTEM_MEMORY_RECOVERY_THRESHOLD = _env_float(
    "ZOE_STATUS_MEMORY_RECOVERY_THRESHOLD",
    80,
    minimum=0,
)


SYSTEM_DISK_HIGH_THRESHOLD = _env_float(
    "ZOE_STATUS_DISK_HIGH_THRESHOLD",
    90,
    minimum=0,
)


SYSTEM_DISK_RECOVERY_THRESHOLD = _env_float(
    "ZOE_STATUS_DISK_RECOVERY_THRESHOLD",
    80,
    minimum=0,
)


# ============================================================
# SCHEDULER
# ============================================================


DEFAULT_POLL_SLEEP = _env_float(
    "ZOE_STATUS_POLL_SLEEP",
    0.5,
)


# ============================================================
# SOURCE HEALTH
# ============================================================


@dataclass
class SourceHealth:
    status: str = "unknown"
    updated_at: str | None = None
    last_success_at: str | None = None
    last_failure_at: str | None = None
    latency_ms: float | None = None
    failure_count: int = 0
    error: str = ""


# ============================================================
# DEFAULT PROVIDERS
# ============================================================


def _default_sources() -> dict[
    str,
    Callable[[], Any],
]:
    def calendar() -> Any:
        from backend.tools.gcal import (
            get_today_events,
        )

        return get_today_events()

    def tasks() -> Any:
        from backend.tools.reminder import (
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
        from backend.tools.mail import (
            check_emails,
        )

        return check_emails(
            max_results=25,
            unread_only=True,
            since_seconds=60,
        )

    def weather() -> Any:
        from backend.tools.weather import (
            get_current_weather,
        )

        return get_current_weather()

    def spotify() -> Any:
        from backend.tools.spotify import (
            get_spotify,
        )

        sp = get_spotify()

        return (
            sp.current_playback()
            or {}
        )

    def system() -> Any:
        """
        Basic host system observation.

        psutil is imported lazily so importing this module does
        not require psutil until system monitoring is enabled.
        """

        try:
            import psutil
        except ImportError:
            return {}

        memory = psutil.virtual_memory()

        system_drive = os.environ.get(
            "SystemDrive",
            "C:",
        )

        disk_path = (
            system_drive + "\\"
            if os.name == "nt"
            else "/"
        )

        disk = psutil.disk_usage(
            disk_path
        )

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

    Provider IDs are preferred.

    Mutable human-facing fields are only fallbacks.
    """

    for field in (
        "id",
        "uri",
        "uid",
        "event_id",
        "task_id",
        "message_id",
    ):
        value = item.get(field)

        if value not in (
            None,
            "",
        ):
            return str(value)

    for field in (
        "title",
        "name",
        "subject",
    ):
        value = item.get(field)

        if value not in (
            None,
            "",
        ):
            return str(value)

    return repr(
        sorted(
            (
                str(key),
                repr(value),
            )
            for key, value in item.items()
        )
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
            tzinfo=(
                now.tzinfo
                or timezone.utc
            )
        )

    return parsed


def _minutes_until(
    target: datetime,
    now: datetime,
) -> int:
    """
    Convert timedelta to human-oriented integer minutes.

    Truncates toward zero.

    Examples:

        +59 sec -> 0
        +61 sec -> 1
        -59 sec -> 0
        -61 sec -> -1
    """

    seconds = (
        target - now
    ).total_seconds()

    return int(
        seconds / 60
    )


def _timestamp(
    value: datetime,
) -> str:
    return value.isoformat()


# ============================================================
# STATUS WORKER
# ============================================================


class StatusWorker:
    """
    Production-oriented 24/7 environmental observation layer.

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
        timeouts: dict[
            str,
            float,
        ] | None = None,
        now: Callable[[], datetime] | None = None,
        poll_sleep: float = DEFAULT_POLL_SLEEP,
        recent_change_limit: int = 25,
        max_workers: int | None = None,
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

        self.timeouts = {
            **DEFAULT_PROVIDER_TIMEOUTS,
            **(
                timeouts
                or {}
            ),
        }

        self.timeouts = {
            source: max(
                0.05,
                float(timeout),
            )
            for source, timeout
            in self.timeouts.items()
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
            float(poll_sleep),
        )

        self.recent_change_limit = max(
            1,
            recent_change_limit,
        )

        worker_count = (
            max_workers
            if max_workers is not None
            else max(
                1,
                len(self.sources),
            )
        )

        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="zoe-status-provider",
        )

        self._stop = threading.Event()

        self._thread: threading.Thread | None = None

        self._last_refresh: dict[
            str,
            float,
        ] = {}

        self._previous: dict[
            str,
            Any,
        ] = {}

        self._thresholds_fired: set[
            tuple[str, str, str, int]
        ] = set()

        self._source_status: dict[
            str,
            str,
        ] = {}

        self._health: dict[
            str,
            SourceHealth,
        ] = {
            source: SourceHealth()
            for source in self.sources
        }

        self._refresh_locks: dict[
            str,
            threading.Lock,
        ] = {
            source: threading.Lock()
            for source in self.sources
        }

        self._lock = threading.RLock()

        # ----------------------------------------------------
        # System alert latch state.
        #
        # True means an alert has already been emitted and
        # recovery threshold must be crossed before another
        # alert can occur.
        # ----------------------------------------------------

        self._memory_high = False
        self._disk_high = False

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

            logger.info(
                "Status worker started"
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
                timeout=max(
                    0.0,
                    timeout,
                )
            )

        logger.info(
            "Status worker stopped"
        )

    def close(
        self,
        timeout: float = 5.0,
    ) -> None:
        """
        Stop the worker and release the provider executor.

        Safe to call more than once.
        """

        self.stop(
            timeout=timeout
        )

        try:
            self._executor.shutdown(
                wait=False,
                cancel_futures=True,
            )
        except TypeError:
            # Python versions without cancel_futures.
            self._executor.shutdown(
                wait=False
            )

    def __enter__(self) -> StatusWorker:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        self.close()

    # ========================================================
    # PUBLIC STATE API
    # ========================================================

    def snapshot(self) -> dict[str, Any]:
        return self.state.snapshot()

    def source_health(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Return a safe snapshot of provider health.
        """

        with self._lock:
            return {
                source: {
                    "status": health.status,
                    "updated_at": health.updated_at,
                    "last_success_at": (
                        health.last_success_at
                    ),
                    "last_failure_at": (
                        health.last_failure_at
                    ),
                    "latency_ms": (
                        health.latency_ms
                    ),
                    "failure_count": (
                        health.failure_count
                    ),
                    "error": health.error,
                }
                for source, health
                in self._health.items()
            }

    # ========================================================
    # MAIN LOOP
    # ========================================================

    def _run(self) -> None:
        first_cycle = True

        while not self._stop.is_set():
            cycle_started = time.monotonic()

            try:
                cycle_now = self.now()

                self.state.update_clock(
                    cycle_now
                )

                due_sources: list[str] = []

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
                            cycle_started
                            - last
                            >= interval
                        )
                    )

                    if due:
                        due_sources.append(
                            source
                        )

                # ------------------------------------------------
                # Schedule due providers concurrently.
                # ------------------------------------------------

                futures: dict[
                    str,
                    Future[Any],
                ] = {}

                for source in due_sources:
                    self._last_refresh[source] = (
                        cycle_started
                    )

                    try:
                        futures[source] = (
                            self._executor.submit(
                                self.refresh,
                                source,
                                initial=first_cycle,
                                now=cycle_now,
                            )
                        )
                    except Exception as exc:
                        logger.exception(
                            "Failed to schedule provider %s",
                            source,
                        )

                        self._handle_source_failure(
                            source,
                            exc,
                            cycle_now,
                        )

                # ------------------------------------------------
                # Wait for submitted providers.
                #
                # The refresh method itself owns the provider
                # timeout. This wait only collects completed
                # refresh results and therefore does not block
                # indefinitely.
                # ------------------------------------------------

                for source, future in futures.items():
                    try:
                        future.result(
                            timeout=(
                                self.timeouts.get(
                                    source,
                                    10.0,
                                )
                                + 1.0
                            )
                        )

                    except FutureTimeoutError:
                        # Normally refresh() should already have
                        # converted provider timeout into a source
                        # failure. This protects the scheduler
                        # against unexpected hangs in refresh().
                        logger.error(
                            "Refresh handling timed out for source %s",
                            source,
                        )

                    except Exception:
                        logger.exception(
                            "Unexpected refresh error for source %s",
                            source,
                        )

                first_cycle = False

            except Exception:
                logger.exception(
                    "Unexpected status-worker cycle failure"
                )

            # ----------------------------------------------------
            # Wake periodically so stop() remains responsive.
            # ----------------------------------------------------

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

        refresh_lock = self._refresh_locks.setdefault(
            source,
            threading.Lock(),
        )

        # Prevent simultaneous manual/automatic refreshes
        # of the same provider.
        acquired = refresh_lock.acquire(
            blocking=False
        )

        if not acquired:
            return []

        try:
            current_now = (
                now
                or self.now()
            )

            self.state.update_clock(
                current_now
            )

            # ------------------------------------------------
            # Provider execution with hard timeout.
            # ------------------------------------------------

            started = time.monotonic()

            future = self._executor.submit(
                provider
            )

            timeout = self.timeouts.get(
                source,
                10.0,
            )

            try:
                value = future.result(
                    timeout=timeout
                )

            except FutureTimeoutError:
                future.cancel()

                elapsed_ms = (
                    time.monotonic()
                    - started
                ) * 1000

                return self._handle_source_failure(
                    source,
                    TimeoutError(
                        f"Provider timed out after "
                        f"{timeout:.2f}s"
                    ),
                    current_now,
                    latency_ms=elapsed_ms,
                )

            except Exception as exc:
                elapsed_ms = (
                    time.monotonic()
                    - started
                ) * 1000

                return self._handle_source_failure(
                    source,
                    exc,
                    current_now,
                    latency_ms=elapsed_ms,
                )

            elapsed_ms = (
                time.monotonic()
                - started
            ) * 1000

            # ------------------------------------------------
            # Compare against last successful state.
            # ------------------------------------------------

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

            # ------------------------------------------------
            # Commit successful state.
            # ------------------------------------------------

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

                health = self._health.setdefault(
                    source,
                    SourceHealth(),
                )

                health.status = "current"
                health.updated_at = _timestamp(
                    current_now
                )
                health.last_success_at = _timestamp(
                    current_now
                )
                health.latency_ms = elapsed_ms
                health.error = ""

            self._set_source(
                source,
                value,
                "current",
            )

            # ------------------------------------------------
            # Recovery.
            # ------------------------------------------------

            if was_unavailable:
                events.insert(
                    0,
                    StatusEvent(
                        type="source_recovered",
                        source=source,
                        data={
                            "source": source,
                            "latency_ms": elapsed_ms,
                        },
                        importance="normal",
                        identity=(
                            f"{source}:recovered"
                        ),
                    ),
                )

            # ------------------------------------------------
            # Publish.
            # ------------------------------------------------

            for event in events:
                self._record_and_publish(
                    event
                )

            return events

        finally:
            refresh_lock.release()

    # ========================================================
    # SOURCE FAILURE
    # ========================================================

    def _handle_source_failure(
        self,
        source: str,
        exc: Exception,
        now: datetime,
        *,
        latency_ms: float | None = None,
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

            health = self._health.setdefault(
                source,
                SourceHealth(),
            )

            health.status = "unavailable"
            health.updated_at = _timestamp(
                now
            )
            health.last_failure_at = _timestamp(
                now
            )
            health.latency_ms = latency_ms
            health.failure_count += 1
            health.error = str(exc)

        # Keep the last valid state intact.
        self._set_source(
            source,
            previous,
            "unavailable",
            str(exc),
        )

        logger.warning(
            "Status source %s unavailable: %s: %s",
            source,
            type(exc).__name__,
            exc,
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
                "error_type": type(exc).__name__,
                "latency_ms": latency_ms,
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
        """
        Update source state.

        This retains compatibility with the current StatusState
        implementation by using its existing internal lock.
        """

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
                    value,
                    self.state.timestamp_datetime
                    if hasattr(
                        self.state,
                        "timestamp_datetime",
                    )
                    else self.now(),
                )

            elif source == "tasks":
                self._set_task_state(
                    value,
                    self.state.timestamp_datetime
                    if hasattr(
                        self.state,
                        "timestamp_datetime",
                    )
                    else self.now(),
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

    # ========================================================
    # CALENDAR STATE
    # ========================================================

    def _set_calendar_state(
        self,
        value: Any,
        now: datetime | None = None,
    ) -> None:
        events = _items(value)

        current_now = (
            now
            or self.now()
        )

        def event_sort_key(
            event: dict[str, Any],
        ) -> datetime:
            parsed = _parse_datetime(
                event.get("datetime")
                or event.get("start"),
                current_now,
            )

            if parsed is not None:
                return parsed

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

        # ----------------------------------------------------
        # next_event must be a future event.
        # ----------------------------------------------------

        next_event = None

        for event in events:
            start = _parse_datetime(
                event.get("datetime")
                or event.get("start"),
                current_now,
            )

            if start is None:
                continue

            if start >= current_now:
                next_event = event
                break

        self.state.next_event = (
            deepcopy(next_event)
            if next_event is not None
            else None
        )

    # ========================================================
    # TASK STATE
    # ========================================================

    def _set_task_state(
        self,
        value: Any,
        now: datetime | None = None,
    ) -> None:
        tasks = _items(value)

        current_now = (
            now
            or self.now()
        )

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
                now=current_now,
            ):
                overdue.append(
                    deepcopy(task)
                )

        self.state.overdue_tasks = overdue

    # ========================================================
    # MAIL STATE
    # ========================================================

    def _set_mail_state(
        self,
        value: Any,
    ) -> None:
        """
        Normalize Gmail provider output into StatusState.mail.
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

            "unread_count": unread_count,

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
        # Added
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

        # ----------------------------------------------------
        # Removed
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Changed
        # ----------------------------------------------------

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
        # Temporal reminders.
        #
        # Evaluated on every successful refresh.
        # ----------------------------------------------------

        result.extend(
            self._approaching_events(
                after,
                now,
            )
        )

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

            # ------------------------------------------------
            # Correct threshold selection.
            #
            # Example:
            #
            # 59 min -> 60 threshold
            # 29 min -> 30 threshold
            # 14 min -> 15 threshold
            # 4 min  -> 5 threshold
            #
            # We need the LARGEST threshold that the current
            # time has crossed, not the smallest.
            # ------------------------------------------------

            eligible = [
                threshold
                for threshold
                in REMINDER_THRESHOLDS
                if minutes <= threshold
            ]

            if not eligible:
                continue

            threshold = max(
                eligible
            )

            marker = (
                "calendar",
                key,
                occurrence,
                threshold,
            )

            with self._lock:
                already_fired = (
                    marker
                    in self._thresholds_fired
                )

                if not already_fired:
                    self._thresholds_fired.add(
                        marker
                    )

            if already_fired:
                continue

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
                            "task_completed"
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

                with self._lock:
                    already_fired = (
                        marker
                        in self._thresholds_fired
                    )

                    if not already_fired:
                        self._thresholds_fired.add(
                            marker
                        )

                if not already_fired:
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
                                "task_due"
                            ),
                        )
                    )

            # ------------------------------------------------
            # Overdue after grace.
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

                with self._lock:
                    already_fired = (
                        marker
                        in self._thresholds_fired
                    )

                    if not already_fired:
                        self._thresholds_fired.add(
                            marker
                        )

                if not already_fired:
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
                                "task_overdue"
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

        Gmail currently returns a recent-message window rather
        than a complete mailbox snapshot.

        Therefore:

          - new IDs are treated as newly observed emails
          - unread transitions are detected
          - disappearing messages are NOT treated as deleted
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
            and not any(
                event.type
                == "email_unread"
                for event in result
            )
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
                        "track_changed:"
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
        # Memory high alert.
        # ----------------------------------------------------

        if (
            isinstance(
                new_memory,
                (int, float),
            )
            and new_memory
            >= SYSTEM_MEMORY_HIGH_THRESHOLD
            and not self._memory_high
        ):
            self._memory_high = True

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

        elif (
            isinstance(
                new_memory,
                (int, float),
            )
            and new_memory
            <= SYSTEM_MEMORY_RECOVERY_THRESHOLD
            and self._memory_high
        ):
            self._memory_high = False

            events.append(
                StatusEvent(
                    type="system_memory_recovered",
                    source="system",
                    data=deepcopy(new),
                    importance="normal",
                    identity=(
                        "system:"
                        "memory_recovered"
                    ),
                )
            )

        # ----------------------------------------------------
        # Disk high alert.
        # ----------------------------------------------------

        if (
            isinstance(
                new_disk,
                (int, float),
            )
            and new_disk
            >= SYSTEM_DISK_HIGH_THRESHOLD
            and not self._disk_high
        ):
            self._disk_high = True

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

        elif (
            isinstance(
                new_disk,
                (int, float),
            )
            and new_disk
            <= SYSTEM_DISK_RECOVERY_THRESHOLD
            and self._disk_high
        ):
            self._disk_high = False

            events.append(
                StatusEvent(
                    type="system_disk_recovered",
                    source="system",
                    data=deepcopy(new),
                    importance="normal",
                    identity=(
                        "system:"
                        "disk_recovered"
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
        try:
            self.state.append_change(
                event.to_dict(),
                limit=self.recent_change_limit,
            )
        except Exception:
            logger.exception(
                "Failed to record status event %s",
                event.type,
            )

        logger.info(
            "Status event: type=%s source=%s importance=%s",
            event.type,
            event.source,
            event.importance,
        )

        try:
            self.event_bus.publish(
                event
            )
        except Exception:
            logger.exception(
                "Failed to publish status event %s",
                event.type,
            )
