from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import threading


@dataclass
class StatusState:
    """
    Current observable state of ZOE's environment.

    StatusState contains facts.

    StatusWorker:
        observes providers
        updates this state
        detects changes
        emits events

    Brain:
        decides whether those events matter
        decides whether ZOE should speak
        decides what action to take
    """

    timestamp: str = ""
    time: str = ""
    date: str = ""

    calendar: dict[str, Any] = field(
        default_factory=dict
    )

    next_event: dict[str, Any] | None = None

    tasks: list[dict[str, Any]] = field(
        default_factory=list
    )

    overdue_tasks: list[dict[str, Any]] = field(
        default_factory=list
    )

    mail: dict[str, Any] = field(
        default_factory=dict
    )

    weather: dict[str, Any] = field(
        default_factory=dict
    )

    spotify: dict[str, Any] = field(
        default_factory=dict
    )

    system: dict[str, Any] = field(
        default_factory=dict
    )

    alerts: list[dict[str, Any]] = field(
        default_factory=list
    )

    recent_changes: list[dict[str, Any]] = field(
        default_factory=list
    )

    sources: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )

    _lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
        compare=False,
    )

    def snapshot(self) -> dict[str, Any]:
        """
        Return a fully detached snapshot suitable for APIs/UI/Brain.

        deepcopy happens while holding the state lock so callers never
        receive a partially updated structure.
        """

        with self._lock:
            return deepcopy(
                {
                    "timestamp": self.timestamp,
                    "time": self.time,
                    "date": self.date,
                    "calendar": self.calendar,
                    "next_event": self.next_event,
                    "tasks": self.tasks,
                    "overdue_tasks": self.overdue_tasks,
                    "mail": self.mail,
                    "weather": self.weather,
                    "spotify": self.spotify,
                    "system": self.system,
                    "alerts": self.alerts,
                    "recent_changes": self.recent_changes,
                    "sources": self.sources,
                }
            )

    def update_clock(
        self,
        now: datetime,
    ) -> None:
        """
        Update the runtime clock.

        Timezone-aware values retain their supplied timezone.

        Naive datetimes are interpreted as UTC rather than silently using
        the host machine's timezone.
        """

        with self._lock:
            current = now

            if current.tzinfo is None:
                current = current.replace(
                    tzinfo=timezone.utc
                )

            self.timestamp = current.isoformat()
            self.time = current.strftime(
                "%H:%M:%S"
            )
            self.date = current.date().isoformat()

    def append_change(
        self,
        event: dict[str, Any],
        *,
        limit: int,
    ) -> None:
        """
        Add an event to recent_changes atomically.
        """

        with self._lock:
            self.recent_changes.append(
                deepcopy(event)
            )

            self.recent_changes = (
                self.recent_changes[-max(1, limit):]
            )