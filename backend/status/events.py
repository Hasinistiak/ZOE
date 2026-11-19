from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable
import threading


@dataclass(frozen=True)
class StatusEvent:
    """
    Immutable event emitted by the environmental status layer.

    StatusEvents describe something that happened or became relevant.
    They do not contain decisions about whether ZOE should speak.
    """

    type: str
    source: str
    data: dict[str, Any]
    importance: str = "normal"
    identity: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "source": self.source,
            "data": self.data,
            "importance": self.importance,
            "identity": self.identity,
            "timestamp": self.timestamp,
        }


class EventBus:
    """
    Small thread-safe in-process transport for structured runtime events.

    Subscribers are called synchronously, but the subscriber list itself
    is copied before callbacks execute. Therefore EventBus locks are never
    held while arbitrary subscriber code is running.
    """

    def __init__(self) -> None:
        self._subscribers: list[
            Callable[[StatusEvent], None]
        ] = []

        self._lock = threading.RLock()

    def subscribe(
        self,
        callback: Callable[[StatusEvent], None],
    ) -> None:
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(
        self,
        callback: Callable[[StatusEvent], None],
    ) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def publish(
        self,
        event: StatusEvent,
    ) -> None:
        with self._lock:
            subscribers = list(self._subscribers)

        for callback in subscribers:
            try:
                callback(event)

            except Exception as exc:
                import traceback

                subscriber_name = getattr(
                    callback,
                    "__name__",
                    repr(callback),
                )

                print(
                    "[STATUS] Subscriber failed: "
                    f"event_type={event.type} "
                    f"event_identity={event.identity} "
                    f"subscriber={subscriber_name} "
                    f"error={type(exc).__name__}: {exc}"
                )

                traceback.print_exc()