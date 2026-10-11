from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from typing import Any, Callable

from .events import EventBus, StatusEvent
from .state import StatusState

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION HELPERS
# ============================================================


def _env_float(name: str, default: float, *, minimum: float = 0.05) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(minimum, value)


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_set(name: str, default: str) -> frozenset[str]:
    raw = os.getenv(name, default)
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _resolve_timezone() -> tzinfo:
    name = os.getenv("ZOE_TIMEZONE", "").strip()
    if name:
        try:
            from zoneinfo import ZoneInfo

            return ZoneInfo(name)
        except Exception:
            # On Windows this usually means the `tzdata` package is missing.
            logger.warning(
                "Unknown or unavailable ZOE_TIMEZONE %r; using host timezone",
                name,
            )
    return datetime.now().astimezone().tzinfo or timezone.utc


LOCAL_TZ = _resolve_timezone()


# ============================================================
# POLLING INTERVALS / TIMEOUTS
# ============================================================

DEFAULT_INTERVALS = {
    "calendar": _env_float("ZOE_STATUS_CALENDAR_INTERVAL", 300),
    "tasks": _env_float("ZOE_STATUS_TASK_INTERVAL", 60),
    "mail": _env_float("ZOE_STATUS_MAIL_INTERVAL", 60),
    "weather": _env_float("ZOE_STATUS_WEATHER_INTERVAL", 1800),
    "spotify": _env_float("ZOE_STATUS_SPOTIFY_INTERVAL", 30),
    "system": _env_float("ZOE_STATUS_SYSTEM_INTERVAL", 30),
}

DEFAULT_PROVIDER_TIMEOUTS = {
    "calendar": _env_float("ZOE_STATUS_CALENDAR_TIMEOUT", 10),
    "tasks": _env_float("ZOE_STATUS_TASK_TIMEOUT", 5),
    "mail": _env_float("ZOE_STATUS_MAIL_TIMEOUT", 10),
    "weather": _env_float("ZOE_STATUS_WEATHER_TIMEOUT", 8),
    "spotify": _env_float("ZOE_STATUS_SPOTIFY_TIMEOUT", 5),
    "system": _env_float("ZOE_STATUS_SYSTEM_TIMEOUT", 3),
}


# ============================================================
# REMINDER / TASK THRESHOLDS
# ============================================================


def _load_thresholds() -> tuple[int, ...]:
    raw = os.getenv("ZOE_STATUS_EVENT_THRESHOLDS", "60,30,15,5")
    values: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            parsed = int(part)
        except ValueError:
            continue
        if parsed >= 0:
            values.add(parsed)
    if not values:
        return (60, 30, 15, 5)
    return tuple(sorted(values, reverse=True))


REMINDER_THRESHOLDS = _load_thresholds()

TASK_OVERDUE_GRACE_MINUTES = _env_int("ZOE_STATUS_TASK_OVERDUE_GRACE_MINUTES", 30)


# ============================================================
# SYSTEM THRESHOLDS (recovery is clamped so it can never exceed "high")
# ============================================================

SYSTEM_MEMORY_HIGH_THRESHOLD = _env_float(
    "ZOE_STATUS_MEMORY_HIGH_THRESHOLD", 90, minimum=0
)
SYSTEM_MEMORY_RECOVERY_THRESHOLD = min(
    _env_float("ZOE_STATUS_MEMORY_RECOVERY_THRESHOLD", 80, minimum=0),
    SYSTEM_MEMORY_HIGH_THRESHOLD,
)
SYSTEM_DISK_HIGH_THRESHOLD = _env_float(
    "ZOE_STATUS_DISK_HIGH_THRESHOLD", 90, minimum=0
)
SYSTEM_DISK_RECOVERY_THRESHOLD = min(
    _env_float("ZOE_STATUS_DISK_RECOVERY_THRESHOLD", 80, minimum=0),
    SYSTEM_DISK_HIGH_THRESHOLD,
)


# ============================================================
# MAIL
#
# Mail is tracked by message ID, not by a "since" window, so a slow or
# failed poll can never silently drop messages. The lookback only has to
# be comfortably larger than the polling interval.
# ============================================================

MAIL_MAX_RESULTS = _env_int("ZOE_STATUS_MAIL_MAX_RESULTS", 25, minimum=1)
MAIL_UNREAD_ONLY = _env_bool("ZOE_STATUS_MAIL_UNREAD_ONLY", False)
MAIL_LOOKBACK_SECONDS = max(
    _env_int("ZOE_STATUS_MAIL_LOOKBACK_SECONDS", 3600),
    int(DEFAULT_INTERVALS["mail"] * 3),
)
MAIL_SEEN_LIMIT = 5000


# ============================================================
# CHANGE DETECTION
# ============================================================

# Provider fields that change by themselves with the passage of time and
# must not be reported as "the item changed".
VOLATILE_FIELDS = _env_set(
    "ZOE_STATUS_VOLATILE_FIELDS",
    "minutes_until,minutes_overdue,overdue,is_overdue,relative_time,time_until",
)

DEFAULT_POLL_SLEEP = _env_float("ZOE_STATUS_POLL_SLEEP", 0.5)


# ============================================================
# TYPES
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


class ProviderError(RuntimeError):
    """A provider returned an explicit failure instead of raising."""


_UNSET: Any = object()


# ============================================================
# DEFAULT PROVIDERS
# ============================================================


def _default_sources() -> dict[str, Callable[[], Any]]:
    def calendar() -> Any:
        from backend.tools.gcal import get_today_events

        return get_today_events()

    def tasks() -> Any:
        from backend.tools.reminder import get_reminders

        result = get_reminders()
        if isinstance(result, dict):
            return result.get("reminders", [])
        return result

    def mail() -> Any:
        from backend.tools.mail import check_emails

        return check_emails(
            max_results=MAIL_MAX_RESULTS,
            unread_only=MAIL_UNREAD_ONLY,
            since_seconds=MAIL_LOOKBACK_SECONDS,
        )

    def weather() -> Any:
        from backend.tools.weather import get_current_weather

        return get_current_weather()

    def spotify() -> Any:
        from backend.tools.spotify import get_spotify

        return get_spotify().current_playback() or {}

    primed: list[bool] = []

    def system() -> Any:
        """
        Basic host observation. psutil is imported lazily so this module
        does not require it unless system monitoring is actually used.
        """
        try:
            import psutil
        except ImportError:
            return {}

        memory = psutil.virtual_memory()

        if os.name == "nt":
            disk_path = os.environ.get("SystemDrive", "C:") + "\\"
        else:
            disk_path = "/"
        disk = psutil.disk_usage(disk_path)

        # The very first cpu_percent(interval=None) call always returns 0.0,
        # so the first sample blocks briefly to produce a real number.
        cpu = psutil.cpu_percent(interval=None if primed else 0.1)
        primed.append(True)

        return {
            "memory_percent": float(memory.percent),
            "disk_percent": float(disk.percent),
            "cpu_percent": float(cpu),
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

_ID_FIELDS = ("id", "uri", "uid", "event_id", "task_id", "message_id")
_NAME_FIELDS = ("title", "name", "subject")
_TIME_FIELDS = ("datetime", "start", "due_datetime", "due_at")


def _items(value: Any) -> list[dict[str, Any]]:
    """Normalize provider output into a list of dictionaries."""
    if isinstance(value, dict):
        value = value.get(
            "items", value.get("reminders", value.get("events", []))
        )
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _key(item: dict[str, Any]) -> str:
    """
    Identity for a provider item.

    Provider IDs are preferred. When falling back to a human-facing name,
    the item's time is appended so recurring items sharing a title do not
    collapse into a single dictionary entry.
    """
    for field in _ID_FIELDS:
        value = item.get(field)
        if value not in (None, ""):
            return str(value)

    for field in _NAME_FIELDS:
        value = item.get(field)
        if value not in (None, ""):
            when = next(
                (str(item[t]) for t in _TIME_FIELDS if item.get(t)), ""
            )
            return f"{value}@{when}" if when else str(value)

    return repr(sorted((str(k), repr(v)) for k, v in item.items()))


def _index(value: Any) -> dict[str, dict[str, Any]]:
    return {_key(item): item for item in _items(value)}


def _stable(item: dict[str, Any]) -> dict[str, Any]:
    """Item view with time-derived fields removed, for change comparison."""
    return {k: v for k, v in item.items() if k not in VOLATILE_FIELDS}


def _parse_datetime(value: Any, now: datetime) -> datetime | None:
    if not value:
        return None

    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        # Date-only strings are all-day items; they have no time of day.
        if len(text) <= 10:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=now.tzinfo or LOCAL_TZ)
    return parsed


def _seconds_until(target: datetime, now: datetime) -> float:
    return (target - now).total_seconds()


def _minutes_until(target: datetime, now: datetime) -> int:
    """Whole minutes, truncated toward zero (+61s -> 1, -61s -> -1)."""
    return int(_seconds_until(target, now) / 60)


def _timestamp(value: datetime) -> str:
    return value.isoformat()


def _event_start(event: dict[str, Any], now: datetime) -> datetime | None:
    return _parse_datetime(event.get("datetime") or event.get("start"), now)


def _occurrence(event: dict[str, Any]) -> str:
    return str(event.get("datetime") or event.get("start") or "")


def _task_due(task: dict[str, Any], now: datetime) -> datetime | None:
    return _parse_datetime(task.get("due_datetime") or task.get("due_at"), now)


def _task_occurrence(task: dict[str, Any]) -> str:
    return str(task.get("due_datetime") or task.get("due_at") or "")


def _mail_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict):
        return []
    emails = value.get("emails", [])
    if not isinstance(emails, list):
        return []
    return [e for e in emails if isinstance(e, dict)]


# ============================================================
# STATUS WORKER
# ============================================================


class StatusWorker:
    def __init__(
        self,
        event_bus: EventBus | None = None,
        state: StatusState | None = None,
        sources: dict[str, Callable[[], Any]] | None = None,
        intervals: dict[str, float] | None = None,
        timeouts: dict[str, float] | None = None,
        now: Callable[[], datetime] | None = None,
        poll_sleep: float = DEFAULT_POLL_SLEEP,
        recent_change_limit: int = 25,
        max_workers: int | None = None,
    ) -> None:
        self.event_bus = event_bus or EventBus()
        self.state = state or StatusState()

        self.sources = (
            _default_sources() if sources is None else dict(sources)
        )

        self.intervals = {
            source: max(0.05, float(interval))
            for source, interval in {
                **DEFAULT_INTERVALS,
                **(intervals or {}),
            }.items()
        }
        self.timeouts = {
            source: max(0.05, float(timeout))
            for source, timeout in {
                **DEFAULT_PROVIDER_TIMEOUTS,
                **(timeouts or {}),
            }.items()
        }

        self.now = now or (lambda: datetime.now(LOCAL_TZ))
        self.poll_sleep = max(0.05, float(poll_sleep))
        self.recent_change_limit = max(1, recent_change_limit)

        # Refresh pool only. Provider calls run on their own daemon threads
        # (see _start_provider), so refresh tasks can never starve them.
        self._refresh_executor = ThreadPoolExecutor(
            max_workers=(
                max_workers
                if max_workers is not None
                else max(1, len(self.sources))
            ),
            thread_name_prefix="zoe-status-refresh",
        )

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

        self._last_refresh: dict[str, float] = {}
        self._pending: dict[str, Future[Any]] = {}
        self._inflight: dict[str, Future[Any]] = {}

        # Last successful raw value per source (replaced, never mutated).
        self._previous: dict[str, Any] = {}

        # (source, item key, occurrence, threshold-or-label)
        self._thresholds_fired: set[tuple[str, str, str, int | str]] = set()

        self._source_status: dict[str, str] = {}
        self._health: dict[str, SourceHealth] = {
            source: SourceHealth() for source in self.sources
        }
        self._refresh_locks: dict[str, threading.Lock] = {
            source: threading.Lock() for source in self.sources
        }

        self._lock = threading.RLock()

        # Alert latches: True once an alert fired, until recovery.
        self._latches: dict[str, bool] = {}

        # Message IDs already announced, oldest first, bounded.
        self._seen_mail: OrderedDict[str, None] = OrderedDict()

        # The only place we touch StatusState internals.
        state_lock = getattr(self.state, "_lock", None)
        self._state_lock = (
            state_lock if state_lock is not None else threading.RLock()
        )

    # ========================================================
    # LIFECYCLE
    # ========================================================

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("StatusWorker is closed")
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="zoe-status", daemon=True
            )
            self._thread.start()
        logger.info("Status worker started")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, timeout))
        logger.info("Status worker stopped")

    def close(self, timeout: float = 5.0) -> None:
        """Stop the worker and release the refresh pool. Idempotent."""
        self.stop(timeout=timeout)
        with self._lock:
            self._closed = True
        try:
            self._refresh_executor.shutdown(wait=False, cancel_futures=True)
        except TypeError:  # Python < 3.9
            self._refresh_executor.shutdown(wait=False)

    def __enter__(self) -> StatusWorker:
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    # ========================================================
    # PUBLIC STATE API
    # ========================================================

    def snapshot(self) -> dict[str, Any]:
        return self.state.snapshot()

    def source_health(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                source: {
                    "status": h.status,
                    "updated_at": h.updated_at,
                    "last_success_at": h.last_success_at,
                    "last_failure_at": h.last_failure_at,
                    "latency_ms": h.latency_ms,
                    "failure_count": h.failure_count,
                    "error": h.error,
                }
                for source, h in self._health.items()
            }

    # ========================================================
    # MAIN LOOP
    #
    # Each tick: schedule due providers (non-blocking), then evaluate
    # time-based conditions against cached state. Reminder accuracy is
    # therefore bounded by poll_sleep, not by provider intervals, and a
    # slow provider can never delay a reminder.
    # ========================================================

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                tick = self.now()
                self.state.update_clock(tick)
                self._schedule_due_sources()
                self._evaluate_time(tick)
            except Exception:
                logger.exception("Unexpected status-worker cycle failure")

            self._stop.wait(self.poll_sleep)

    def _schedule_due_sources(self) -> None:
        for source in list(self.sources):
            pending = self._pending.get(source)
            if pending is not None and not pending.done():
                continue

            last = self._last_refresh.get(source)
            interval = self.intervals.get(source, 300.0)
            if last is not None and time.monotonic() - last < interval:
                continue

            self._last_refresh[source] = time.monotonic()

            try:
                self._pending[source] = self._refresh_executor.submit(
                    self._safe_refresh, source, last is None
                )
            except RuntimeError:
                return  # executor shut down
            except Exception as exc:
                logger.exception("Failed to schedule provider %s", source)
                self._handle_source_failure(source, exc, self.now())

    def _safe_refresh(self, source: str, initial: bool) -> None:
        try:
            self.refresh(source, initial=initial)
        except Exception:
            logger.exception("Unexpected refresh error for source %s", source)

    # ========================================================
    # TIME-BASED EVALUATION
    # ========================================================

    def _evaluate_time(self, now: datetime) -> None:
        with self._lock:
            calendar = self._previous.get("calendar")
            tasks = self._previous.get("tasks")

        self._update_time_derived_state(calendar, tasks, now)

        events: list[StatusEvent] = []
        if calendar is not None:
            events.extend(self._approaching_events(_index(calendar), now))
        if tasks is not None:
            events.extend(self._task_due_events(_index(tasks), now))

        for event in events:
            self._record_and_publish(event)

    def _update_time_derived_state(
        self, calendar: Any, tasks: Any, now: datetime
    ) -> None:
        """next_event and overdue_tasks depend on the clock, not on polls."""
        if calendar is not None:
            next_event = self._compute_next_event(_items(calendar), now)
            with self._state_lock:
                if self.state.next_event != next_event:
                    self.state.next_event = deepcopy(next_event)

        if tasks is not None:
            overdue = [
                deepcopy(task)
                for task in _items(tasks)
                if not task.get("completed")
                and self._task_is_overdue(task, now=now)
            ]
            with self._state_lock:
                if self.state.overdue_tasks != overdue:
                    self.state.overdue_tasks = overdue

    # ========================================================
    # PROVIDER EXECUTION
    # ========================================================

    def _start_provider(
        self, source: str, provider: Callable[[], Any]
    ) -> Future[Any] | None:
        """
        Run the provider on a dedicated daemon thread.

        Returns None when a previous call for this source is still running
        (it timed out earlier and is hung): we never stack up threads
        behind a stuck provider, and a hung call cannot block process exit.
        """
        with self._lock:
            running = self._inflight.get(source)
            if running is not None and not running.done():
                return None
            future: Future[Any] = Future()
            self._inflight[source] = future

        def target() -> None:
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(provider())
            except BaseException as exc:  # noqa: BLE001
                future.set_exception(exc)

        try:
            threading.Thread(
                target=target, name=f"zoe-status-{source}", daemon=True
            ).start()
        except Exception as exc:
            future.set_exception(exc)

        return future

    @staticmethod
    def _validate(source: str, value: Any) -> None:
        if (
            source == "mail"
            and isinstance(value, dict)
            and value.get("success") is False
        ):
            raise ProviderError(
                str(value.get("error") or "mail provider reported failure")
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
    ) -> list[StatusEvent] | None:
        """
        Refresh one source and return the events it produced.

        Returns None (not an empty list) if the refresh was skipped because
        another refresh of the same source is already running. Raises
        KeyError for an unknown source.
        """
        provider = self.sources.get(source)
        if provider is None:
            raise KeyError(f"Unknown status source: {source!r}")

        refresh_lock = self._refresh_locks.setdefault(
            source, threading.Lock()
        )
        if not refresh_lock.acquire(blocking=False):
            logger.debug("Refresh of %s already running; skipped", source)
            return None

        try:
            return self._refresh_locked(source, provider, initial, now)
        finally:
            refresh_lock.release()

    def _refresh_locked(
        self,
        source: str,
        provider: Callable[[], Any],
        initial: bool,
        now: datetime | None,
    ) -> list[StatusEvent]:
        current_now = now or self.now()
        self.state.update_clock(current_now)

        started = time.monotonic()
        timeout = self.timeouts.get(source, 10.0)

        def elapsed_ms() -> float:
            return (time.monotonic() - started) * 1000

        future = self._start_provider(source, provider)
        if future is None:
            return self._handle_source_failure(
                source,
                TimeoutError("previous provider call is still running"),
                current_now,
            )

        try:
            value = future.result(timeout=timeout)
            self._validate(source, value)
        except FutureTimeoutError:
            return self._handle_source_failure(
                source,
                TimeoutError(f"Provider timed out after {timeout:.2f}s"),
                current_now,
                latency_ms=elapsed_ms(),
            )
        except Exception as exc:
            return self._handle_source_failure(
                source, exc, current_now, latency_ms=elapsed_ms()
            )

        latency = elapsed_ms()

        # ----------------------------------------------------
        # Compare against last successful value.
        # The first successful value for a source (even if earlier
        # attempts failed) is a baseline, not a set of changes.
        # ----------------------------------------------------

        with self._lock:
            previous_exists = source in self._previous
            previous = self._previous.get(source)

        baseline = initial or not previous_exists

        if baseline:
            events = self._baseline_events(source, value)
        else:
            events = self._detect_changes(source, previous, value, current_now)

        # ----------------------------------------------------
        # Commit. Marker seeding, cache swap and pruning happen under one
        # lock so the time evaluator never sees new data without its
        # markers (which would replay already-crossed reminders).
        # ----------------------------------------------------

        with self._lock:
            if baseline:
                self._seed_markers(source, value, current_now)

            self._previous[source] = deepcopy(value)
            self._prune_markers(source, value)

            was_unavailable = self._source_status.get(source) == "unavailable"
            self._source_status[source] = "current"

            health = self._health.setdefault(source, SourceHealth())
            health.status = "current"
            health.updated_at = _timestamp(current_now)
            health.last_success_at = _timestamp(current_now)
            health.latency_ms = latency
            health.error = ""

        self._write_state(source, "current", now=current_now, value=value)

        if was_unavailable:
            events.insert(
                0,
                StatusEvent(
                    type="source_recovered",
                    source=source,
                    data={"source": source, "latency_ms": latency},
                    importance="normal",
                    identity=f"{source}:recovered",
                ),
            )

        for event in events:
            self._record_and_publish(event)

        return events

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
            was_unavailable = self._source_status.get(source) == "unavailable"
            self._source_status[source] = "unavailable"

            health = self._health.setdefault(source, SourceHealth())
            health.status = "unavailable"
            health.updated_at = _timestamp(now)
            health.last_failure_at = _timestamp(now)
            health.latency_ms = latency_ms
            health.failure_count += 1
            health.error = str(exc)

        # Last valid data stays in StatusState; only the status changes.
        self._write_state(source, "unavailable", now=now, error=str(exc))

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
            identity=f"{source}:unavailable",
        )
        self._record_and_publish(event)
        return [event]

    # ========================================================
    # STATE UPDATE
    # ========================================================

    def _write_state(
        self,
        source: str,
        status: str,
        *,
        now: datetime,
        value: Any = _UNSET,
        error: str = "",
    ) -> None:
        with self._state_lock:
            entry: dict[str, Any] = {
                "status": status,
                "updated_at": self.state.timestamp,
            }
            if error:
                entry["error"] = error
            self.state.sources[source] = entry

            # Never destroy valid state because a provider failed.
            if status == "unavailable" or value is _UNSET:
                return

            self._apply_source_value(source, value, now)

    def _apply_source_value(
        self, source: str, value: Any, now: datetime
    ) -> None:
        if source == "calendar":
            self._set_calendar_state(value, now)
        elif source == "tasks":
            self._set_task_state(value, now)
        elif source == "mail":
            self._set_mail_state(value)
        elif source in ("weather", "spotify", "system"):
            setattr(
                self.state,
                source,
                deepcopy(value) if isinstance(value, dict) else {},
            )

    @staticmethod
    def _compute_next_event(
        events: list[dict[str, Any]], now: datetime
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_start: datetime | None = None
        for event in events:
            if event.get("completed"):
                continue
            start = _event_start(event, now)
            if start is None or start < now:
                continue
            if best_start is None or start < best_start:
                best, best_start = event, start
        return best

    def _set_calendar_state(self, value: Any, now: datetime) -> None:
        far_future = datetime.max.replace(tzinfo=timezone.utc)

        def sort_key(event: dict[str, Any]) -> datetime:
            return _event_start(event, now) or far_future

        events = sorted((deepcopy(e) for e in _items(value)), key=sort_key)

        self.state.calendar = {"events": events}
        next_event = self._compute_next_event(events, now)
        self.state.next_event = (
            deepcopy(next_event) if next_event is not None else None
        )

    def _set_task_state(self, value: Any, now: datetime) -> None:
        tasks = _items(value)
        self.state.tasks = deepcopy(tasks)
        self.state.overdue_tasks = [
            deepcopy(task)
            for task in tasks
            if not task.get("completed")
            and self._task_is_overdue(task, now=now)
        ]

    def _set_mail_state(self, value: Any) -> None:
        if not isinstance(value, dict):
            self.state.mail = {}
            return

        emails = [deepcopy(e) for e in _mail_list(value)]
        unread = sum(1 for e in emails if e.get("unread", False))

        mail: dict[str, Any] = {
            "success": bool(value.get("success", False)),
            "count": len(emails),
            # Unread among the fetched window, not the whole mailbox.
            "unread_count": unread,
            "emails": emails,
            "updated_at": self.state.timestamp,
        }
        if value.get("error"):
            mail["error"] = value["error"]
        self.state.mail = mail

    # ========================================================
    # BASELINE (first successful snapshot of a source)
    # ========================================================

    def _baseline_events(self, source: str, value: Any) -> list[StatusEvent]:
        if source == "mail":
            # Existing mail is not "received".
            self._remember_mail_ids(
                str(e["id"]) for e in _mail_list(value) if e.get("id")
            )
            return []

        if source == "system":
            # A condition that is already bad at boot is still worth an alert.
            return self._system_events(value)

        return []

    def _seed_markers(self, source: str, value: Any, now: datetime) -> None:
        """Mark everything already crossed as fired, without emitting."""
        if source == "calendar":
            self._approaching_events(_index(value), now, seed=True)
        elif source == "tasks":
            self._task_due_events(_index(value), now, seed=True)

    # ========================================================
    # CHANGE DETECTION
    # ========================================================

    def _detect_changes(
        self, source: str, old: Any, new: Any, now: datetime
    ) -> list[StatusEvent]:
        if source == "calendar":
            return self._diff_collections(
                "calendar",
                old,
                new,
                added="event_added",
                removed="event_removed",
                changed="event_changed",
            )
        if source == "tasks":
            return self._diff_collections(
                "tasks",
                old,
                new,
                added="task_added",
                removed="task_removed",
                changed="task_changed",
                classify=self._classify_task_change,
            )
        if source == "mail":
            return self._mail_events(old, new)
        if source == "weather":
            return self._weather_events(old, new)
        if source == "spotify":
            return self._spotify_events(old, new)
        if source == "system":
            return self._system_events(new)
        return []

    @staticmethod
    def _item_event(
        kind: str, source: str, key: str, item: dict[str, Any]
    ) -> StatusEvent:
        return StatusEvent(
            type=kind,
            source=source,
            data=deepcopy(item),
            importance="normal",
            identity=f"{key}:{kind}",
        )

    def _diff_collections(
        self,
        source: str,
        old: Any,
        new: Any,
        *,
        added: str,
        removed: str,
        changed: str,
        classify: Callable[[dict[str, Any], dict[str, Any]], str | None]
        | None = None,
    ) -> list[StatusEvent]:
        before = _index(old)
        after = _index(new)
        result: list[StatusEvent] = []

        for key, item in after.items():
            if key not in before:
                result.append(self._item_event(added, source, key, item))

        for key, item in before.items():
            if key not in after:
                result.append(self._item_event(removed, source, key, item))

        for key in sorted(after.keys() & before.keys()):
            old_item, new_item = before[key], after[key]
            if _stable(old_item) == _stable(new_item):
                continue
            kind = classify(old_item, new_item) if classify else changed
            if kind:
                result.append(self._item_event(kind, source, key, new_item))

        return result

    @staticmethod
    def _classify_task_change(
        old_task: dict[str, Any], new_task: dict[str, Any]
    ) -> str | None:
        if not old_task.get("completed") and new_task.get("completed"):
            return "task_completed"
        if not new_task.get("completed"):
            return "task_changed"
        return None

    # ========================================================
    # CALENDAR REMINDERS
    # ========================================================

    def _claim_marker(self, marker: tuple[str, str, str, int | str]) -> bool:
        """Atomically record a marker. True if this caller claimed it."""
        with self._lock:
            if marker in self._thresholds_fired:
                return False
            self._thresholds_fired.add(marker)
            return True

    def _approaching_events(
        self,
        events: dict[str, dict[str, Any]],
        now: datetime,
        *,
        seed: bool = False,
    ) -> list[StatusEvent]:
        """
        Fire the TIGHTEST threshold that still covers the time remaining.

        With thresholds 60/30/15/5 and 14 minutes left, thresholds 60, 30
        and 15 all cover it; 15 is the one that is accurate to announce.
        In seed mode every covered threshold is marked as fired and nothing
        is emitted, so a restart does not replay old reminders.
        """
        result: list[StatusEvent] = []

        for key, event in events.items():
            if event.get("completed"):
                continue

            start = _event_start(event, now)
            if start is None:
                continue

            seconds = _seconds_until(start, now)
            if seconds < 0:
                continue

            eligible = [t for t in REMINDER_THRESHOLDS if seconds <= t * 60]
            if not eligible:
                continue

            occurrence = _occurrence(event)

            if seed:
                for threshold in eligible:
                    self._claim_marker(("calendar", key, occurrence, threshold))
                continue

            threshold = min(eligible)
            if not self._claim_marker(("calendar", key, occurrence, threshold)):
                continue

            result.append(
                StatusEvent(
                    type="event_approaching",
                    source="calendar",
                    data={
                        **deepcopy(event),
                        "minutes_until": _minutes_until(start, now),
                        "threshold": threshold,
                    },
                    importance="high",
                    identity=f"{key}:{occurrence}:approaching:{threshold}",
                )
            )

        result.sort(key=lambda e: e.data.get("minutes_until", 0))
        return result

    # ========================================================
    # TASK DUE / OVERDUE
    # ========================================================

    def _task_is_overdue(
        self, task: dict[str, Any], *, now: datetime | None = None
    ) -> bool:
        if task.get("overdue") or task.get("is_overdue"):
            return True
        current_now = now or self.now()
        due = _task_due(task, current_now)
        return due is not None and _seconds_until(due, current_now) < 0

    def _task_due_events(
        self,
        tasks: dict[str, dict[str, Any]],
        now: datetime,
        *,
        seed: bool = False,
    ) -> list[StatusEvent]:
        result: list[StatusEvent] = []
        grace_seconds = TASK_OVERDUE_GRACE_MINUTES * 60

        for key, task in tasks.items():
            if task.get("completed"):
                continue

            due = _task_due(task, now)
            if due is None:
                continue

            seconds = _seconds_until(due, now)
            if seconds > 0:
                continue

            occurrence = _task_occurrence(task)
            overdue_hit = seconds <= -grace_seconds

            claimed_due = self._claim_marker(("tasks", key, occurrence, "due"))
            claimed_overdue = overdue_hit and self._claim_marker(
                ("tasks", key, occurrence, "overdue")
            )

            if seed:
                continue

            minutes = _minutes_until(due, now)

            if claimed_overdue:
                # If it is already overdue the first time we see it, the
                # "due" notice is stale; announce only the overdue state.
                result.append(
                    StatusEvent(
                        type="task_overdue",
                        source="tasks",
                        data={
                            **deepcopy(task),
                            "minutes_overdue": abs(minutes),
                        },
                        importance="high",
                        identity=f"{key}:{occurrence}:task_overdue",
                    )
                )
            elif claimed_due and not overdue_hit:
                result.append(
                    StatusEvent(
                        type="task_due",
                        source="tasks",
                        data={**deepcopy(task), "minutes_until": minutes},
                        importance="high",
                        identity=f"{key}:{occurrence}:task_due",
                    )
                )

        return result

    # ========================================================
    # MARKER PRUNING
    # ========================================================

    def _prune_markers(self, source: str, value: Any) -> None:
        if source == "calendar":
            valid = {(k, _occurrence(e)) for k, e in _index(value).items()}
        elif source == "tasks":
            valid = {(k, _task_occurrence(t)) for k, t in _index(value).items()}
        else:
            return

        with self._lock:
            self._thresholds_fired = {
                m
                for m in self._thresholds_fired
                if m[0] != source or (m[1], m[2]) in valid
            }

    # ========================================================
    # MAIL
    # ========================================================

    def _remember_mail_ids(self, ids: Any) -> set[str]:
        """Record message IDs; return the ones never seen before."""
        newly_seen: set[str] = set()
        with self._lock:
            for message_id in ids:
                if message_id not in self._seen_mail:
                    newly_seen.add(message_id)
                self._seen_mail[message_id] = None
                self._seen_mail.move_to_end(message_id)
            while len(self._seen_mail) > MAIL_SEEN_LIMIT:
                self._seen_mail.popitem(last=False)
        return newly_seen

    def _mail_events(self, old: Any, new: Any) -> list[StatusEvent]:
        """
        Mail providers return a recent window, not the full mailbox, so:

          - a message is "received" only the first time its ID is seen
          - messages leaving the window are NOT treated as deleted
          - read -> unread transitions are detected for IDs in both windows
        """
        before_emails = _mail_list(old)
        after_emails = _mail_list(new)

        before = {str(e["id"]): e for e in before_emails if e.get("id")}
        after = {str(e["id"]): e for e in after_emails if e.get("id")}

        newly_seen = self._remember_mail_ids(after.keys())
        result: list[StatusEvent] = []

        for key, email in after.items():
            if key in newly_seen:
                unread = bool(email.get("unread", False))
                result.append(
                    StatusEvent(
                        type="email_received",
                        source="mail",
                        data=deepcopy(email),
                        importance="high" if unread else "normal",
                        identity=f"{key}:email_received",
                    )
                )
                continue

            old_email = before.get(key)
            if old_email is None:
                continue  # re-entered the window; already announced

            if not old_email.get("unread", False) and email.get(
                "unread", False
            ):
                result.append(
                    StatusEvent(
                        type="email_unread",
                        source="mail",
                        data=deepcopy(email),
                        importance="high",
                        identity=f"{key}:email_unread",
                    )
                )
            elif _stable(old_email) != _stable(email):
                result.append(
                    StatusEvent(
                        type="email_changed",
                        source="mail",
                        data=deepcopy(email),
                        importance="normal",
                        identity=f"{key}:email_changed",
                    )
                )

        old_count = sum(1 for e in before_emails if e.get("unread", False))
        new_count = sum(1 for e in after_emails if e.get("unread", False))

        if old_count != new_count and not any(
            e.type == "email_unread" for e in result
        ):
            result.append(
                StatusEvent(
                    type="email_unread_count_changed",
                    source="mail",
                    data={
                        "old_unread_count": old_count,
                        "new_unread_count": new_count,
                    },
                    importance="normal",
                    identity="mail:unread_count_changed",
                )
            )

        return result

    # ========================================================
    # WEATHER
    # ========================================================

    def _weather_events(self, old: Any, new: Any) -> list[StatusEvent]:
        if not isinstance(old, dict) or not isinstance(new, dict):
            return []

        old_condition = str(old.get("condition", "")).lower()
        new_condition = str(new.get("condition", "")).lower()
        old_temp = old.get("temperature_c")
        new_temp = new.get("temperature_c")

        condition_changed = bool(
            old_condition and new_condition and old_condition != new_condition
        )
        temp_changed = (
            isinstance(old_temp, (int, float))
            and isinstance(new_temp, (int, float))
            and abs(new_temp - old_temp) >= 2
        )

        if not (condition_changed or temp_changed):
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

    def _spotify_events(self, old: Any, new: Any) -> list[StatusEvent]:
        if not isinstance(old, dict) or not isinstance(new, dict):
            return []

        def item_id(payload: dict[str, Any]) -> Any:
            item = payload.get("item")
            return item.get("id") if isinstance(item, dict) else None

        old_item, new_item = item_id(old), item_id(new)
        old_playing = bool(old.get("is_playing"))
        new_playing = bool(new.get("is_playing"))

        result: list[StatusEvent] = []

        if new_item and old_item != new_item:
            result.append(
                StatusEvent(
                    type="track_changed",
                    source="spotify",
                    data=deepcopy(new),
                    importance="low",
                    identity=f"spotify:track_changed:{new_item}",
                )
            )

        if old_playing != new_playing:
            kind = "spotify_started" if new_playing else "spotify_stopped"
            result.append(
                StatusEvent(
                    type=kind,
                    source="spotify",
                    data=deepcopy(new),
                    importance="low",
                    identity=kind,
                )
            )

        return result

    # ========================================================
    # SYSTEM
    # ========================================================

    def _system_events(self, new: Any) -> list[StatusEvent]:
        if not isinstance(new, dict):
            return []

        return [
            *self._latched_alert(
                "memory",
                new,
                new.get("memory_percent"),
                SYSTEM_MEMORY_HIGH_THRESHOLD,
                SYSTEM_MEMORY_RECOVERY_THRESHOLD,
            ),
            *self._latched_alert(
                "disk",
                new,
                new.get("disk_percent"),
                SYSTEM_DISK_HIGH_THRESHOLD,
                SYSTEM_DISK_RECOVERY_THRESHOLD,
            ),
        ]

    def _latched_alert(
        self,
        name: str,
        data: dict[str, Any],
        value: Any,
        high: float,
        recovery: float,
    ) -> list[StatusEvent]:
        """
        Hysteresis alert: fire once at `high`, then stay quiet until the
        value drops to `recovery`.
        """
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return []

        with self._lock:
            latched = self._latches.get(name, False)
            if value >= high and not latched:
                self._latches[name] = True
                kind, importance = "high", "high"
            elif value <= recovery and latched:
                self._latches[name] = False
                kind, importance = "recovered", "normal"
            else:
                return []

        return [
            StatusEvent(
                type=f"system_{name}_{kind}",
                source="system",
                data=deepcopy(data),
                importance=importance,
                identity=f"system:{name}_{kind}",
            )
        ]

    # ========================================================
    # EVENT PUBLISHING
    # ========================================================

    def _record_and_publish(self, event: StatusEvent) -> None:
        try:
            self.state.append_change(
                event.to_dict(), limit=self.recent_change_limit
            )
        except Exception:
            logger.exception("Failed to record status event %s", event.type)

        logger.info(
            "Status event: type=%s source=%s importance=%s",
            event.type,
            event.source,
            event.importance,
        )

        try:
            self.event_bus.publish(event)
        except Exception:
            logger.exception("Failed to publish status event %s", event.type)