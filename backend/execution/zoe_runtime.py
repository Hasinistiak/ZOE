from __future__ import annotations

import os
import queue
import re
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

from backend.brain.main import (
    BrainResponse,
    create_zoe_brain,
)

from backend.context.agents import (
    AGENTS,
    execute_agent,
)

from backend.execution.runtime_commands import (
    execute_runtime_command,
    validate_runtime_command,
)

from backend.status import (
    EventBus,
    StatusEvent,
    StatusWorker,
)

from backend.memory.session import (
    SessionManager,
)


# ============================================================
# ZOE RUNTIME
# ============================================================


class ZoeRuntime:

    # ========================================================
    # STATES
    # ========================================================

    STATE_IDLE = "idle"
    STATE_LISTENING = "listening"
    STATE_THINKING = "thinking"
    STATE_EXECUTING = "executing"
    STATE_SPEAKING = "speaking"
    STATE_NOTIFYING = "notifying"
    STATE_MUTED = "muted"
    STATE_ERROR = "error"

    VALID_STATES = {
        STATE_IDLE,
        STATE_LISTENING,
        STATE_THINKING,
        STATE_EXECUTING,
        STATE_SPEAKING,
        STATE_NOTIFYING,
        STATE_MUTED,
        STATE_ERROR,
    }

    # ========================================================
    # ACTIVITY
    # ========================================================

    ACTIVITY_LIMIT = 60

    ACTIVITY_EVENT_TYPES = {
        # Runtime
        "runtime_ready",

        # Status
        "status_event",
        "status_event_dropped",
        "status_ignored",
        "status_error",
        "status_worker_error",

        # Reminders
        "reminder_triggered",
        "reminder_scheduler_error",

        # Agents
        "agent_batch_started",
        "agent_started",
        "agent_progress",
        "agent_completed",
        "agent_failed",
        "agent_batch_completed",

        # Agent synthesis
        "agent_synthesis_started",
        "agent_synthesis_completed",
        "agent_synthesis_error",

        # Runtime state
        "state_changed",

        # Web
        "web_open",
    }

    # ========================================================
    # SESSION
    # ========================================================

    SESSION_CONTEXT_LIMIT = 6

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        on_output: Callable[[str], None] | None = None,
        on_speech: Callable[
            [str, Callable[[], None] | None],
            None,
        ] | None = None,
        on_interrupt: Callable[[], None] | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        session_manager: SessionManager | None = None,
    ):

        # ====================================================
        # CORE SERVICES
        # ====================================================

        self.event_bus = EventBus()

        # ====================================================
        # SESSION
        # ====================================================

        self.session_manager = (
            session_manager
            if session_manager is not None
            else SessionManager()
        )

        # ====================================================
        # STATUS
        # ====================================================

        self.status_worker = StatusWorker(
            event_bus=self.event_bus,
        )

        # ====================================================
        # BRAIN
        # ====================================================

        self.brain = create_zoe_brain(
            status_provider=self.status_worker.snapshot,
        )

        # ====================================================
        # CALLBACKS
        # ====================================================

        self.on_output = on_output
        self.on_speech = on_speech
        self.on_interrupt = on_interrupt
        self.on_event = on_event

        # ====================================================
        # SHUTDOWN
        # ====================================================

        self._shutdown = False
        self._shutdown_lock = threading.RLock()

        # ====================================================
        # STATE
        # ====================================================

        self._state_lock = threading.RLock()

        self._state = self.STATE_IDLE

        self._state_reason = ""

        self._state_since = datetime.now(
            timezone.utc
        )

        # ====================================================
        # WEB URL STATE
        # ====================================================

        self._url_lock = threading.RLock()

        self._url: str | None = None

        # ====================================================
        # BRAIN LOCK
        # ====================================================

        self._brain_lock = threading.Lock()

        # ====================================================
        # FOREGROUND LOCK
        # ====================================================

        self._foreground_lock = threading.Lock()

        # ====================================================
        # INTERACTION COMPLETION
        #
        # Each interaction gets its own completion Event.
        #
        # Output means:
        #     "ZOE produced something."
        #
        # Completion means:
        #     "This specific interaction has its final answer."
        #
        # This distinction prevents Telegram/agent races.
        # ====================================================

        self._completion_lock = threading.RLock()

        self._completion_waiters: dict[
            str,
            dict[str, Any],
        ] = {}

        # ====================================================
        # SESSION LOCK
        # ====================================================

        self._session_lock = threading.RLock()

        # ====================================================
        # OUTPUT LOCK
        # ====================================================

        self._output_lock = threading.Lock()

        # ====================================================
        # EVENT LOCK
        # ====================================================

        self._event_lock = threading.Lock()

        self._event_sequence = 0

        # ====================================================
        # ACTIVITY HISTORY
        # ====================================================

        self._activity_lock = threading.RLock()

        self._activities: deque[
            dict[str, Any]
        ] = deque(
            maxlen=self.ACTIVITY_LIMIT
        )

        # ====================================================
        # STATUS EVENT QUEUE
        # ====================================================

        self._status_queue: queue.Queue[
            StatusEvent
        ] = queue.Queue(
            maxsize=100
        )

        self._status_shutdown = threading.Event()

        self._status_dispatcher: (
            threading.Thread | None
        ) = None

        self._status_dispatch_lock = threading.Lock()

        # ====================================================
        # AGENT THREADS
        # ====================================================

        self._agent_threads: list[
            threading.Thread
        ] = []

        self._agent_threads_lock = threading.Lock()

        self._agent_shutdown = threading.Event()

        # ====================================================
        # AGENT BATCHES
        # ====================================================

        self._agent_batches: dict[
            str,
            dict[str, Any],
        ] = {}

        self._agent_batches_lock = threading.Lock()

        # ====================================================
        # EVENT BUS
        # ====================================================

        self.event_bus.subscribe(
            self._on_status_event
        )

        # ====================================================
        # START STATUS DISPATCHER
        # ====================================================

        self._start_status_dispatcher()

        # ====================================================
        # START BACKGROUND STATUS SYSTEM
        # ====================================================

        status_enabled = os.getenv(
            "ZOE_STATUS_ENABLED",
            "1",
        ).lower() not in {
            "0",
            "false",
            "no",
        }

        if status_enabled:

            try:

                self.status_worker.start()

            except Exception as exc:

                print(
                    "[ZOE RUNTIME] "
                    "Status worker failed to start: "
                    f"{type(exc).__name__}: {exc}"
                )

                self._emit_event(
                    "status_worker_error",
                    {
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

            # ------------------------------------------------
            # Initial task refresh
            # ------------------------------------------------

            try:

                self.status_worker.refresh(
                    "tasks",
                    initial=True,
                )

            except Exception as exc:

                print(
                    "[ZOE RUNTIME] "
                    "Initial task refresh failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        print(
            "[ZOE RUNTIME] Ready."
        )

        self._emit_event(
            "runtime_ready",
            {
                "state": self._state,
            },
        )

    # ========================================================
    # INTERACTION COMPLETION
    # ========================================================

    def _create_completion_waiter(
        self,
        interaction_id: str,
    ) -> threading.Event:

        event = threading.Event()

        with self._completion_lock:

            self._completion_waiters[
                interaction_id
            ] = {
                "event": event,
                "answer": "",
                "error": None,
            }

        return event

    # ========================================================

    def _resolve_completion(
        self,
        interaction_id: str,
        answer: str = "",
        error: str | None = None,
    ) -> None:

        answer = str(
            answer or ""
        ).strip()

        with self._completion_lock:

            waiter = self._completion_waiters.get(
                interaction_id
            )

            if waiter is None:

                return

            if waiter["event"].is_set():

                return

            waiter["answer"] = answer
            waiter["error"] = error

            waiter["event"].set()

    # ========================================================

    def _wait_for_completion(
        self,
        interaction_id: str,
        timeout: float | None = None,
    ) -> str:

        with self._completion_lock:

            waiter = self._completion_waiters.get(
                interaction_id
            )

            if waiter is None:

                return ""

            event = waiter["event"]

        completed = event.wait(
            timeout=timeout
        )

        with self._completion_lock:

            waiter = self._completion_waiters.get(
                interaction_id
            )

            if waiter is None:

                return ""

            answer = str(
                waiter.get(
                    "answer",
                    "",
                )
                or ""
            ).strip()

            error = waiter.get(
                "error"
            )

            # ------------------------------------------------
            # Only remove the waiter once it actually
            # completed.
            #
            # If timeout occurs, keep it alive so a background
            # agent can resolve it later.
            # ------------------------------------------------

            if completed:

                self._completion_waiters.pop(
                    interaction_id,
                    None,
                )

        if error:

            raise RuntimeError(
                str(error)
            )

        return answer

    # ========================================================

    def _discard_completion_waiter(
        self,
        interaction_id: str,
    ) -> None:

        with self._completion_lock:

            self._completion_waiters.pop(
                interaction_id,
                None,
            )

    # ========================================================

    @staticmethod
    def _is_agent_interaction(
        response: BrainResponse,
    ) -> bool:

        return (
            isinstance(
                response,
                BrainResponse,
            )
            and response.action == "agent"
        )

    # ========================================================
    # WEB URL
    # ========================================================

    def get_url(
        self,
    ) -> str | None:

        """
        Return the currently requested web URL.

        The frontend exposes this through /api/state and
        decides whether a WebPage should be opened.
        """

        with self._url_lock:

            return self._url

    # ========================================================

    def set_url(
        self,
        url: str | None,
    ) -> None:

        """
        Store a web URL for the frontend.

        Only HTTP/HTTPS URLs are accepted.
        """

        if url is None:

            with self._url_lock:

                self._url = None

            return

        url = str(
            url
        ).strip()

        if not url:

            with self._url_lock:

                self._url = None

            return

        if not url.startswith(
            (
                "http://",
                "https://",
            )
        ):

            print(
                "[ZOE RUNTIME] "
                "Ignoring invalid web URL: "
                f"{url!r}"
            )

            return

        with self._url_lock:

            self._url = url

        print(
            "[ZOE RUNTIME] "
            f"Web URL set: {url}"
        )

        self._emit_event(
            "web_open",
            {
                "url": url,
            },
        )

    # ========================================================

    def clear_url(
        self,
    ) -> None:

        """
        Clear the currently stored web URL.

        This is intentionally NOT called automatically after
        set_url(). The frontend polls /api/state every 150 ms,
        so clearing immediately could cause the UI to miss it.
        """

        with self._url_lock:

            self._url = None

    # ========================================================

    @staticmethod
    def _extract_response_url(
        response: BrainResponse,
    ) -> str | None:

        """
        Extract a URL from BrainResponse.

        Preferred source:
            response.url

        Compatibility fallback:
            URL embedded inside response.answer
        """

        explicit_url = getattr(
            response,
            "url",
            None,
        )

        if explicit_url:

            explicit_url = str(
                explicit_url
            ).strip()

            if explicit_url.startswith(
                (
                    "http://",
                    "https://",
                )
            ):

                return explicit_url

        answer = getattr(
            response,
            "answer",
            None,
        )

        if not answer:

            return None

        match = re.search(
            r'https?://[^\s<>"\']+',
            str(answer),
        )

        if not match:

            return None

        return match.group(
            0
        ).rstrip(
            ".,!?;:)]}"
        )

    # ========================================================
    # SESSION CONTEXT
    # ========================================================

    def _get_session_context(
        self,
    ) -> list[dict[str, Any]]:

        with self._session_lock:

            try:

                messages = (
                    self.session_manager.get_recent_messages(
                        limit=self.SESSION_CONTEXT_LIMIT,
                    )
                )

            except Exception as exc:

                print(
                    "[ZOE RUNTIME] "
                    "Could not retrieve session context: "
                    f"{type(exc).__name__}: {exc}"
                )

                return []

        if not messages:

            return []

        if not isinstance(
            messages,
            list,
        ):

            try:

                messages = list(
                    messages
                )

            except Exception:

                return []

        return [
            message
            for message in messages
            if isinstance(
                message,
                dict,
            )
        ]

    # ========================================================

    def _add_user_to_session(
        self,
        message: str,
    ) -> None:

        if not message:

            return

        with self._session_lock:

            try:

                self.session_manager.add_user_message(
                    message
                )

            except Exception as exc:

                print(
                    "[ZOE RUNTIME] "
                    "Could not save user session message: "
                    f"{type(exc).__name__}: {exc}"
                )

    # ========================================================

    def _add_assistant_to_session(
        self,
        text: str,
    ) -> None:

        if not text:

            return

        text = str(
            text
        ).strip()

        if not text:

            return

        with self._session_lock:

            try:

                self.session_manager.add_assistant_message(
                    text
                )

            except Exception as exc:

                print(
                    "[ZOE RUNTIME] "
                    "Could not save assistant session message: "
                    f"{type(exc).__name__}: {exc}"
                )

    # ========================================================

    def _capture_session_context(
        self,
    ) -> list[dict[str, Any]]:

        context = self._get_session_context()

        return [
            dict(message)
            for message in context
        ]

    # ========================================================
    # SHUTDOWN STATUS
    # ========================================================

    def is_shutdown(
        self,
    ) -> bool:

        with self._shutdown_lock:

            return self._shutdown

    # ========================================================
    # STATE
    # ========================================================

    def get_state(
        self,
    ) -> str:

        with self._state_lock:

            return self._state

    # ========================================================

    def get_state_snapshot(
        self,
    ) -> dict[str, Any]:

        with self._state_lock:

            state = self._state
            reason = self._state_reason
            since = self._state_since

        return {
            "state": state,
            "reason": reason,
            "since": since.isoformat(),
        }

    # ========================================================

    def _set_state(
        self,
        state: str,
        reason: str = "",
    ) -> None:

        if state not in self.VALID_STATES:

            raise ValueError(
                f"Invalid ZOE runtime state: {state!r}"
            )

        now = datetime.now(
            timezone.utc
        )

        with self._state_lock:

            previous = self._state

            if (
                previous == state
                and self._state_reason == reason
            ):

                return

            self._state = state
            self._state_reason = reason
            self._state_since = now

        self._emit_event(
            "state_changed",
            {
                "state": state,
                "previous_state": previous,
                "reason": reason,
                "timestamp": now.isoformat(),
            },
        )

    # ========================================================
    # STATUS DISPATCHER
    # ========================================================

    def _start_status_dispatcher(
        self,
    ) -> None:

        with self._status_dispatch_lock:

            if (
                self._status_dispatcher
                and self._status_dispatcher.is_alive()
            ):

                return

            self._status_dispatcher = threading.Thread(
                target=self._dispatch_status_events,
                daemon=True,
                name="zoe-status-events",
            )

            self._status_dispatcher.start()

    # ========================================================

    def _on_status_event(
        self,
        event: StatusEvent,
    ) -> None:

        if self.is_shutdown():

            return

        self._emit_event(
            "status_event",
            {
                "event_type": event.type,
                "source": event.source,
                "data": event.data or {},
            },
        )

        if self.is_shutdown():

            return

        try:

            self._status_queue.put_nowait(
                event
            )

        except queue.Full:

            print(
                "[STATUS] "
                "Notification queue full; "
                "dropping event."
            )

            self._emit_event(
                "status_event_dropped",
                {
                    "event_type": event.type,
                    "source": event.source,
                },
            )

    # ========================================================

    def _dispatch_status_events(
        self,
    ) -> None:

        print(
            "[STATUS] Event dispatcher started."
        )

        while not self._status_shutdown.is_set():

            try:

                event = self._status_queue.get(
                    timeout=0.5
                )

            except queue.Empty:

                continue

            try:

                self._process_status_event(
                    event
                )

            except Exception as exc:

                print(
                    "[STATUS] "
                    "Unhandled event processing error: "
                    f"{type(exc).__name__}: {exc}"
                )

                self._emit_event(
                    "status_error",
                    {
                        "event_type": event.type,
                        "source": event.source,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )

            finally:

                self._status_queue.task_done()

        print(
            "[STATUS] Event dispatcher stopped."
        )

    # ========================================================
    # STATUS EVENTS
    # ========================================================

    def _process_status_event(
        self,
        event: StatusEvent,
    ) -> None:

        if self.is_shutdown():

            return

        if event.type in {
            "reminder_due",
            "reminder_overdue",
        }:

            self._deliver_reminder(
                event
            )

            return

        status_snapshot = (
            self.status_worker.snapshot()
        )

        session_context = (
            self._capture_session_context()
        )

        self._set_state(
            self.STATE_NOTIFYING,
            reason=event.type,
        )

        try:

            with self._brain_lock:

                response = (
                    self.brain.handle_status_event(
                        event.to_dict(),
                        status=status_snapshot,
                        session_context=session_context,
                    )
                )

        except Exception:

            self._set_state(
                self.STATE_ERROR,
                reason="status_brain_error",
            )

            raise

        # Shutdown may race with a slow brain call. Do not emit a
        # notification after the runtime has already begun shutting down.
        if self.is_shutdown():

            return

        if not isinstance(
            response,
            BrainResponse,
        ):

            raise TypeError(
                "Brain returned an invalid "
                "response for a status event."
            )

        if response.action == "ignore":

            print(
                "[STATUS] Brain decision: IGNORE "
                f"| type={event.type} "
                f"| reason="
                f"{response.reason or 'not useful enough'}"
            )

            self._emit_event(
                "status_ignored",
                {
                    "event_type": event.type,
                    "source": event.source,
                    "reason": response.reason or "",
                },
            )

            self._set_state(
                self.STATE_IDLE
            )

            return

        if response.action != "answer":

            print(
                "[STATUS] "
                "Unexpected Brain action: "
                f"{response.action!r}"
            )

            self._set_state(
                self.STATE_IDLE
            )

            return

        if not response.answer:

            self._set_state(
                self.STATE_IDLE
            )

            return

        answer = response.answer.strip()

        if not answer:

            self._set_state(
                self.STATE_IDLE
            )

            return

        response_url = (
            self._extract_response_url(
                response
            )
        )

        if response_url:

            self.set_url(
                response_url
            )

        self._add_assistant_to_session(
            answer
        )

        self._emit_output(
            answer,
            speak_output=response.speak,
            interrupt=response.interrupt,
            event_type="notification",
            metadata={
                "source": event.source,
                "status_event": event.type,
            },
        )

        print(
            "[STATUS] "
            "Proactive notification dispatched "
            f"| type={event.type} "
            f"| importance={response.importance} "
            f"| speak={response.speak} "
            f"| interrupt={response.interrupt}"
        )

    # ========================================================
    # REMINDER DELIVERY
    # ========================================================

    def _deliver_reminder(
        self,
        event: StatusEvent,
    ) -> None:

        if self.is_shutdown():

            return

        data = event.data or {}

        name = str(
            data.get(
                "name",
                "Reminder",
            )
        ).strip()

        if not name:

            name = "Reminder"

        reminder_id = str(
            data.get(
                "id",
                data.get(
                    "task_id",
                    uuid.uuid4().hex,
                ),
            )
        )

        if event.type == "reminder_due":

            text = (
                f"Reminder: {name}."
            )

        elif event.type == "reminder_overdue":

            minutes = data.get(
                "minutes_overdue",
                0,
            )

            try:

                minutes = int(
                    minutes
                )

            except (
                TypeError,
                ValueError,
            ):

                minutes = 0

            if minutes <= 0:

                text = (
                    f"Reminder: {name}."
                )

            elif minutes == 1:

                text = (
                    f"Reminder: {name}. "
                    "It's one minute overdue."
                )

            else:

                text = (
                    f"Reminder: {name}. "
                    f"It's {minutes} minutes overdue."
                )

        else:

            print(
                "[REMINDER] "
                f"Unknown reminder event: "
                f"{event.type!r}"
            )

            return

        self._set_state(
            self.STATE_NOTIFYING,
            reason="reminder",
        )

        self._emit_event(
            "reminder_triggered",
            {
                "reminder_id": reminder_id,
                "name": name,
                "event_type": event.type,
                "text": text,
            },
        )

        print()
        print(
            "=" * 60
        )
        print(
            "[ZOE RUNTIME → REMINDER]"
        )
        print(
            "=" * 60
        )
        print(
            f"Event : {event.type}"
        )
        print(
            f"ID    : {reminder_id}"
        )
        print(
            f"Name  : {name}"
        )
        print(
            f"Text  : {text}"
        )
        print(
            "=" * 60
        )

        self._add_assistant_to_session(
            text
        )

        self._emit_output(
            text,
            speak_output=True,
            interrupt=True,
            event_type="reminder",
            metadata={
                "reminder_id": reminder_id,
                "reminder_name": name,
                "reminder_event": event.type,
            },
        )

    # ========================================================
    # PUBLIC FOREGROUND API
    # ========================================================

    def run(
        self,
        message: str,
        speak_output: bool = True,
        wait_for_completion: bool = False,
        timeout: float | None = None,
    ) -> str:

        if self.is_shutdown():

            raise RuntimeError(
                "ZOE Runtime is shut down."
            )

        if not message or not message.strip():

            raise ValueError(
                "Cannot process an empty message."
            )

        message = message.strip()

        interaction_id = uuid.uuid4().hex

        # ----------------------------------------------------
        # Completion waiter must exist BEFORE the brain runs.
        # ----------------------------------------------------

        if wait_for_completion:

            self._create_completion_waiter(
                interaction_id
            )

        self._emit_event(
            "interaction_started",
            {
                "interaction_id": interaction_id,
                "message": message,
            },
        )

        answer = ""

        try:

            with self._foreground_lock:

                if self.is_shutdown():

                    raise RuntimeError(
                        "ZOE Runtime is shut down."
                    )

                print()
                print(
                    "=" * 60
                )
                print(
                    "[ZOE RUNTIME]"
                )
                print(
                    "=" * 60
                )
                print(
                    f"Interaction: {interaction_id}"
                )
                print(
                    f"User: {message}"
                )
                print()

                self._set_state(
                    self.STATE_THINKING,
                    reason="foreground_request",
                )

                self._emit_event(
                    "transcript",
                    {
                        "interaction_id": interaction_id,
                        "text": message,
                        "final": True,
                    },
                )

                # =============================================
                # SESSION
                # =============================================

                session_context = (
                    self._capture_session_context()
                )

                self._add_user_to_session(
                    message
                )

                try:

                    with self._brain_lock:

                        response = self.brain.run(
                            message=message,
                            session_context=session_context,
                        )

                    answer = self._handle_response(
                        response=response,
                        speak_output=speak_output,
                        interaction_id=interaction_id,
                    )

                except Exception as exc:

                    self._set_state(
                        self.STATE_ERROR,
                        reason="foreground_error",
                    )

                    self._emit_event(
                        "interaction_error",
                        {
                            "interaction_id": interaction_id,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                        },
                    )

                    self._resolve_completion(
                        interaction_id,
                        error=(
                            f"{type(exc).__name__}: {exc}"
                        ),
                    )

                    raise

            # =================================================
            # WAIT FOR REAL COMPLETION
            # =================================================

            if wait_for_completion:

                completed_answer = (
                    self._wait_for_completion(
                        interaction_id=interaction_id,
                        timeout=timeout,
                    )
                )

                if completed_answer:

                    answer = completed_answer

                elif self._is_agent_interaction(
                    response
                ):

                    print(
                        "[ZOE RUNTIME] "
                        f"Completion timeout for "
                        f"interaction={interaction_id}"
                    )

            # =================================================
            # INTERACTION FINISHED
            # =================================================

            self._emit_event(
                "interaction_finished",
                {
                    "interaction_id": interaction_id,
                    "answer": answer,
                    "completed": bool(answer),
                },
            )

            # =================================================
            # STATE
            # =================================================

            if self.get_state() not in {
                self.STATE_EXECUTING,
                self.STATE_SPEAKING,
                self.STATE_NOTIFYING,
            }:

                self._set_state(
                    self.STATE_IDLE
                )

            return answer

        except Exception:

            self._resolve_completion(
                interaction_id,
                error="ZOE failed to process the interaction.",
            )

            raise

        finally:

            if wait_for_completion:

                with self._completion_lock:

                    waiter = self._completion_waiters.get(
                        interaction_id
                    )

                    if (
                        waiter is not None
                        and waiter["event"].is_set()
                    ):

                        self._completion_waiters.pop(
                            interaction_id,
                            None,
                        )

    # ========================================================
    # RESPONSE HANDLER
    # ========================================================

    def _handle_response(
        self,
        response: BrainResponse,
        speak_output: bool,
        interaction_id: str,
    ) -> str:

        if not isinstance(
            response,
            BrainResponse,
        ):

            raise TypeError(
                "Brain returned an invalid "
                "response object."
            )

        # ====================================================
        # ANSWER
        # ====================================================

        if response.action == "answer":



            answer = response.answer.strip()

            response_url = (
                self._extract_response_url(
                    response
                )
            )

            if response_url:

                self.set_url(
                    response_url
                )

            self._add_assistant_to_session(
                answer
            )

            self._emit_event(
                "response",
                {
                    "interaction_id": interaction_id,
                    "text": answer,
                    "speak": (
                        speak_output
                        and response.speak
                    ),
                    "interrupt": response.interrupt,
                },
            )

            print(
                "[ZOE RUNTIME] "
                "Final answer ready."
            )

            self._emit_output(
                answer,
                speak_output=(
                    speak_output
                    and response.speak
                ),
                interrupt=response.interrupt,
                event_type="response",
                metadata={
                    "interaction_id": interaction_id,
                },
            )

            self._resolve_completion(
                interaction_id,
                answer=answer,
            )

            return answer

        # ====================================================
        # AGENT
        # ====================================================

        if response.action == "agent":

            agent_calls = response.agents

            if not agent_calls:

                raise RuntimeError(
                    "Brain selected an agent "
                    "without a query."
                )

            normalized_calls: list[
                dict[str, Any]
            ] = []

            for index, call in enumerate(
                agent_calls
            ):

                if not isinstance(
                    call,
                    dict,
                ):

                    raise RuntimeError(
                        "Malformed agent call."
                    )

                agent = (
                    call.get(
                        "agent",
                        "",
                    )
                    or ""
                ).strip()

                query = (
                    call.get(
                        "query",
                        "",
                    )
                    or ""
                ).strip()

                self._validate_agent(
                    agent
                )

                if not query:

                    raise RuntimeError(
                        "Brain selected an agent "
                        "without a query."
                    )

                normalized_calls.append(
                    {
                        "index": index,
                        "agent": agent,
                        "query": query,
                    }
                )

            session_context = (
                self._capture_session_context()
            )

            batch_id = self._start_agents(
                calls=normalized_calls,
                speak_output=(
                    speak_output
                    and response.speak
                ),
                interaction_id=interaction_id,
                session_context=session_context,
            )

            self._set_state(
                self.STATE_EXECUTING,
                reason="background_agents",
            )

            answer = str(
                response.answer or ""
            ).strip()

            if not answer:

                print(
                    "[ZOE RUNTIME] "
                    f"Background work started "
                    f"| batch={batch_id} "
                    f"| agents={len(normalized_calls)} "
                    f"| acknowledgement=no"
                )

                return ""

            response_url = (
                self._extract_response_url(
                    response
                )
            )

            if response_url:

                self.set_url(
                    response_url
                )

            self._add_assistant_to_session(
                answer
            )

            self._emit_event(
                "response",
                {
                    "interaction_id": interaction_id,
                    "text": answer,
                    "speak": (
                        speak_output
                        and response.speak
                    ),
                    "interrupt": response.interrupt,
                    "batch_id": batch_id,
                    "background": True,
                    "acknowledgement": True,
                },
            )

            self._emit_output(
                answer,
                speak_output=(
                    speak_output
                    and response.speak
                ),
                interrupt=response.interrupt,
                event_type="response",
                metadata={
                    "interaction_id": interaction_id,
                    "batch_id": batch_id,
                    "background": True,
                    "acknowledgement": True,
                },
            )

            print(
                "[ZOE RUNTIME] "
                f"Background work started "
                f"| batch={batch_id} "
                f"| agents={len(normalized_calls)} "
                f"| acknowledgement=yes"
            )

            # The acknowledgement is NOT completion.
            return answer

        # ====================================================
        # RUNTIME
        # ====================================================

        if response.action == "runtime":

            if not response.runtime:

                raise RuntimeError(
                    "Brain selected 'runtime' "
                    "without a runtime command."
                )

            command = response.runtime.get(
                "command",
                "",
            )

            if not command:

                raise RuntimeError(
                    "Brain selected 'runtime' "
                    "but command is empty."
                )

            if not validate_runtime_command(
                command
            ):

                raise RuntimeError(
                    "Brain selected unknown "
                    f"runtime command: {command!r}"
                )

            result = execute_runtime_command(
                command
            )

            answer = ""

            if response.answer:

                answer = response.answer.strip()

            if answer:

                response_url = (
                    self._extract_response_url(
                        response
                    )
                )

                if response_url:

                    self.set_url(
                        response_url
                    )

                self._add_assistant_to_session(
                    answer
                )

                self._emit_event(
                    "response",
                    {
                        "interaction_id": interaction_id,
                        "text": answer,
                        "speak": (
                            speak_output
                            and response.speak
                        ),
                        "interrupt": response.interrupt,
                        "runtime_command": command,
                    },
                )

                self._emit_output(
                    answer,
                    speak_output=(
                        speak_output
                        and response.speak
                    ),
                    interrupt=response.interrupt,
                    event_type="response",
                    metadata={
                        "interaction_id": interaction_id,
                        "runtime_command": command,
                    },
                )

            self._emit_event(
                "runtime_command_executed",
                {
                    "interaction_id": interaction_id,
                    "command": command,
                    "result": result,
                },
            )

            self._resolve_completion(
                interaction_id,
                answer=answer,
            )

            print(
                "[ZOE RUNTIME] "
                f"Runtime command executed: {command}"
            )

            return answer

        # ====================================================
        # IGNORE
        # ====================================================

        if response.action == "ignore":

            print(
                "[ZOE RUNTIME] "
                "Brain chose IGNORE."
            )

            self._emit_event(
                "interaction_ignored",
                {
                    "interaction_id": interaction_id,
                    "reason": response.reason or "",
                },
            )

            self._resolve_completion(
                interaction_id,
                answer="",
            )

            return ""

        # ====================================================
        # UNKNOWN
        # ====================================================

        raise RuntimeError(
            "Unknown Brain action: "
            f"{response.action!r}"
        )

    # ========================================================
    # AGENTS
    # ========================================================

    def _start_agents(
        self,
        calls: list[dict[str, Any]],
        speak_output: bool,
        interaction_id: str,
        session_context: list[dict[str, Any]] | None = None,
    ) -> str:

        if self.is_shutdown():

            raise RuntimeError(
                "Cannot start agents after "
                "runtime shutdown."
            )

        if not calls:

            raise RuntimeError(
                "Cannot start an empty agent batch."
            )

        batch_id = uuid.uuid4().hex

        started_at = datetime.now(
            timezone.utc
        ).isoformat()

        batch = {
            "batch_id": batch_id,
            "interaction_id": interaction_id,
            "expected": len(calls),
            "results": {},
            "jobs": {},
            "speak_output": bool(
                speak_output
            ),
            "started_at": started_at,
            "phase": "executing",
            "session_context": [
                dict(message)
                for message in (
                    session_context or []
                )
            ],
        }

        with self._agent_batches_lock:

            self._agent_batches[
                batch_id
            ] = batch

        self._emit_event(
            "agent_batch_started",
            {
                "batch_id": batch_id,
                "interaction_id": interaction_id,
                "count": len(calls),
                "agents": [
                    {
                        "agent": call["agent"],
                        "query": call["query"],
                        "job_id": (
                            f"{batch_id}_{call['index']}"
                        ),
                    }
                    for call in calls
                ],
                "started_at": started_at,
            },
        )

        print()
        print(
            "=" * 60
        )
        print(
            "[ZOE RUNTIME → BACKGROUND AGENTS]"
        )
        print(
            "=" * 60
        )

        print(
            f"Batch       : {batch_id}"
        )

        print(
            f"Interaction : {interaction_id}"
        )

        print(
            f"Count       : {len(calls)}"
        )

        for call in calls:

            # Shutdown can race with the loop above (especially when
            # event callbacks trigger shutdown). Stop creating new work.
            if self.is_shutdown():

                break

            index = call["index"]
            agent = call["agent"]
            query = call["query"]

            job_id = (
                f"{batch_id}_{index}"
            )

            with self._agent_batches_lock:

                current_batch = (
                    self._agent_batches.get(
                        batch_id
                    )
                )

                if current_batch is not None:

                    current_batch[
                        "jobs"
                    ][index] = {
                        "job_id": job_id,
                        "batch_id": batch_id,
                        "interaction_id": interaction_id,
                        "agent": agent,
                        "query": query,
                        "status": "starting",
                        "started_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    }

            print(
                f"Agent : {agent}"
            )

            print(
                f"Job   : {job_id}"
            )

            print(
                f"Query : {query}"
            )

            self._emit_event(
                "agent_started",
                {
                    "batch_id": batch_id,
                    "job_id": job_id,
                    "interaction_id": interaction_id,
                    "agent": agent,
                    "query": query,
                    "description": (
                        self._describe_agent(
                            agent,
                            query,
                        )
                    ),
                },
            )

            thread = threading.Thread(
                target=self._run_agent,
                args=(
                    batch_id,
                    index,
                    agent,
                    query,
                ),
                daemon=True,
                name=f"zoe-agent-{agent}-{index}",
            )

            with self._agent_threads_lock:

                self._agent_threads.append(
                    thread
                )

            try:

                thread.start()

            except Exception as exc:

                print(
                    "[ZOE RUNTIME] "
                    f"Could not start agent thread "
                    f"{agent}: {exc}"
                )

                self._mark_agent_start_failure(
                    batch_id=batch_id,
                    index=index,
                    agent=agent,
                    query=query,
                    error=exc,
                )

        print()

        self._cleanup_agent_threads()

        return batch_id

    # ========================================================
    # AGENT DESCRIPTION
    # ========================================================

    @staticmethod
    def _describe_agent(
        agent: str,
        query: str,
    ) -> str:

        if query:

            return query

        return f"Running {agent}"

    # ========================================================
    # AGENT START FAILURE
    # ========================================================

    def _mark_agent_start_failure(
        self,
        batch_id: str,
        index: int,
        agent: str,
        query: str,
        error: Exception,
    ) -> None:

        job_id = (
            f"{batch_id}_{index}"
        )

        self._emit_event(
            "agent_failed",
            {
                "batch_id": batch_id,
                "job_id": job_id,
                "interaction_id": self._get_batch_interaction_id(
                    batch_id
                ),
                "agent": agent,
                "query": query,
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        self._handle_agent_completion(
            batch_id=batch_id,
            index=index,
            agent=agent,
            query=query,
            result={
                "success": False,
                "error": str(error),
            },
        )

    # ========================================================
    # RUN AGENT
    # ========================================================

    def _run_agent(
        self,
        batch_id: str,
        index: int,
        agent: str,
        query: str,
    ) -> None:

        job_id = (
            f"{batch_id}_{index}"
        )

        try:

            if (
                self.is_shutdown()
                or self._agent_shutdown.is_set()
            ):

                return

            self._update_agent_job(
                batch_id=batch_id,
                index=index,
                status="running",
            )

            print(
                f"[ZOE AGENT] Started: "
                f"{agent} "
                f"| job={job_id}"
            )

            result = execute_agent(
                agent,
                query=query,
            )

            if (
                self.is_shutdown()
                or self._agent_shutdown.is_set()
            ):

                return

            self._update_agent_job(
                batch_id=batch_id,
                index=index,
                status="completed",
            )

            print(
                f"[ZOE AGENT] Completed: "
                f"{agent} "
                f"| job={job_id}"
            )

            self._emit_event(
                "agent_completed",
                {
                    "batch_id": batch_id,
                    "job_id": job_id,
                    "interaction_id": self._get_batch_interaction_id(
                        batch_id
                    ),
                    "agent": agent,
                    "query": query,
                    "status": "completed",
                },
            )

            self._handle_agent_completion(
                batch_id=batch_id,
                index=index,
                agent=agent,
                query=query,
                result=result,
            )

        except Exception as exc:

            self._update_agent_job(
                batch_id=batch_id,
                index=index,
                status="failed",
            )

            print(
                f"[ZOE AGENT] Failed: "
                f"{agent} "
                f"| job={job_id} "
                f"| {type(exc).__name__}: {exc}"
            )

            self._emit_event(
                "agent_failed",
                {
                    "batch_id": batch_id,
                    "job_id": job_id,
                    "interaction_id": self._get_batch_interaction_id(
                        batch_id
                    ),
                    "agent": agent,
                    "query": query,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

            try:

                self._handle_agent_completion(
                    batch_id=batch_id,
                    index=index,
                    agent=agent,
                    query=query,
                    result={
                        "success": False,
                        "error": str(exc),
                    },
                )

            except Exception as completion_error:

                print(
                    "[ZOE RUNTIME] "
                    "Could not process agent failure: "
                    f"{type(completion_error).__name__}: "
                    f"{completion_error}"
                )

                self._resolve_completion(
                    self._get_batch_interaction_id(
                        batch_id
                    ) or "",
                    error=(
                        f"Agent completion failed: "
                        f"{completion_error}"
                    ),
                )

        finally:

            self._cleanup_agent_threads()

    # ========================================================
    # UPDATE AGENT JOB
    # ========================================================

    def _update_agent_job(
        self,
        batch_id: str,
        index: int,
        status: str,
    ) -> None:

        with self._agent_batches_lock:

            batch = self._agent_batches.get(
                batch_id
            )

            if batch is None:

                return

            job = batch["jobs"].get(
                index
            )

            if job is None:

                return

            job["status"] = status

            if status == "running":

                job["running_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

            elif status in {
                "completed",
                "failed",
            }:

                job["finished_at"] = datetime.now(
                    timezone.utc
                ).isoformat()

    # ========================================================
    # GET BATCH INTERACTION
    # ========================================================

    def _get_batch_interaction_id(
        self,
        batch_id: str,
    ) -> str | None:

        with self._agent_batches_lock:

            batch = self._agent_batches.get(
                batch_id
            )

            if batch is None:

                return None

            return batch.get(
                "interaction_id"
            )

    # ========================================================
    # AGENT BATCH CLEANUP
    # ========================================================

    def _complete_agent_batch(
        self,
        batch_id: str,
        interaction_id: str,
        results: list[dict[str, Any]],
        expected: int,
        started_at: str,
        phase: str,
    ) -> None:

        with self._agent_batches_lock:

            batch = self._agent_batches.get(
                batch_id
            )

            if batch is not None:

                batch["phase"] = phase

        self._emit_event(
            "agent_batch_completed",
            {
                "batch_id": batch_id,
                "interaction_id": interaction_id,
                "count": len(results),
                "expected": expected,
                "started_at": started_at,
                "completed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
                "agents": [
                    {
                        "job_id": item["job_id"],
                        "agent": item["agent"],
                    }
                    for item in results
                ],
            },
        )

        with self._agent_batches_lock:

            self._agent_batches.pop(
                batch_id,
                None,
            )

    # ========================================================
    # AGENT COMPLETION
    # ========================================================

    def _handle_agent_completion(
        self,
        batch_id: str,
        index: int,
        agent: str,
        query: str,
        result: Any,
    ) -> None:

        if self.is_shutdown():

            return

        job_id = (
            f"{batch_id}_{index}"
        )

        with self._agent_batches_lock:

            batch = self._agent_batches.get(
                batch_id
            )

            if batch is None:

                print(
                    "[ZOE RUNTIME] "
                    f"Unknown agent batch: {batch_id}"
                )

                return

            if index in batch["results"]:

                return

            batch["results"][index] = {
                "job_id": job_id,
                "agent": agent,
                "query": query,
                "result": result,
            }

            result_count = len(
                batch["results"]
            )

            expected = int(
                batch["expected"]
            )

            interaction_id = batch[
                "interaction_id"
            ]

            if result_count < expected:

                self._emit_event(
                    "agent_progress",
                    {
                        "batch_id": batch_id,
                        "interaction_id": interaction_id,
                        "completed": result_count,
                        "expected": expected,
                        "pending": (
                            expected
                            - result_count
                        ),
                    },
                )

                return

            results = [
                batch["results"][key]
                for key in sorted(
                    batch["results"]
                )
            ]

            batch_speak_output = bool(
                batch["speak_output"]
            )

            started_at = batch[
                "started_at"
            ]

            session_context = [
                dict(message)
                for message in (
                    batch.get(
                        "session_context",
                        [],
                    )
                )
            ]

            batch["phase"] = "synthesizing"

        self._emit_event(
            "agent_synthesis_started",
            {
                "batch_id": batch_id,
                "interaction_id": interaction_id,
                "agent_count": len(results),
            },
        )

        print()
        print(
            "=" * 60
        )
        print(
            "[ZOE RUNTIME → BRAIN]"
        )
        print(
            "=" * 60
        )

        print(
            f"Batch : {batch_id}"
        )

        print(
            "Agents:",
            [
                item["agent"]
                for item in results
            ],
        )

        self._set_state(
            self.STATE_THINKING,
            reason="agent_synthesis",
        )

        try:

            with self._brain_lock:

                response = (
                    self.brain.handle_agent_results(
                        results=results,
                        session_context=session_context,
                    )
                )

        except Exception as exc:

            if self.is_shutdown():

                return

            self._set_state(
                self.STATE_ERROR,
                reason="agent_synthesis_error",
            )

            self._emit_event(
                "agent_synthesis_error",
                {
                    "batch_id": batch_id,
                    "interaction_id": interaction_id,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )

            print(
                "[ZOE RUNTIME] "
                "Agent synthesis failed: "
                f"{type(exc).__name__}: {exc}"
            )

            error_text = (
                "I couldn't complete the background work. "
                f"{type(exc).__name__}: {exc}"
            )

            self._resolve_completion(
                interaction_id,
                error=error_text,
            )

            self._complete_agent_batch(
                batch_id=batch_id,
                interaction_id=interaction_id,
                results=results,
                expected=expected,
                started_at=started_at,
                phase="failed",
            )

            return

        self._emit_event(
            "agent_synthesis_completed",
            {
                "batch_id": batch_id,
                "interaction_id": interaction_id,
            },
        )

        # A shutdown can begin immediately after synthesis. The batch
        # completion is no longer useful once the runtime is offline.
        if self.is_shutdown():

            return

        # ====================================================
        # VALIDATE SYNTHESIS RESPONSE
        # ====================================================

        if not isinstance(
            response,
            BrainResponse,
        ):

            error_text = (
                "Brain returned an invalid response "
                "for completed agent work."
            )

            self._set_state(
                self.STATE_ERROR,
                reason="invalid_agent_response",
            )

            self._emit_event(
                "agent_synthesis_error",
                {
                    "batch_id": batch_id,
                    "interaction_id": interaction_id,
                    "error": error_text,
                    "error_type": "TypeError",
                },
            )

            self._resolve_completion(
                interaction_id,
                error=error_text,
            )

            self._complete_agent_batch(
                batch_id=batch_id,
                interaction_id=interaction_id,
                results=results,
                expected=expected,
                started_at=started_at,
                phase="failed",
            )

            return

        if response.action != "answer":

            error_text = (
                "Brain returned invalid action "
                f"{response.action!r} "
                "for completed agent work."
            )

            self._set_state(
                self.STATE_ERROR,
                reason="invalid_agent_action",
            )

            self._emit_event(
                "agent_synthesis_error",
                {
                    "batch_id": batch_id,
                    "interaction_id": interaction_id,
                    "error": error_text,
                    "error_type": "RuntimeError",
                },
            )

            self._resolve_completion(
                interaction_id,
                error=error_text,
            )

            self._complete_agent_batch(
                batch_id=batch_id,
                interaction_id=interaction_id,
                results=results,
                expected=expected,
                started_at=started_at,
                phase="failed",
            )

            return

        if not response.answer:

            error_text = (
                "Brain returned an empty answer "
                "for completed agent work."
            )

            self._set_state(
                self.STATE_ERROR,
                reason="empty_agent_answer",
            )

            self._emit_event(
                "agent_synthesis_error",
                {
                    "batch_id": batch_id,
                    "interaction_id": interaction_id,
                    "error": error_text,
                    "error_type": "RuntimeError",
                },
            )

            self._resolve_completion(
                interaction_id,
                error=error_text,
            )

            self._complete_agent_batch(
                batch_id=batch_id,
                interaction_id=interaction_id,
                results=results,
                expected=expected,
                started_at=started_at,
                phase="failed",
            )

            return

        answer = response.answer.strip()

        if not answer:

            error_text = (
                "Brain returned an empty answer "
                "for completed agent work."
            )

            self._set_state(
                self.STATE_ERROR,
                reason="empty_agent_answer",
            )

            self._resolve_completion(
                interaction_id,
                error=error_text,
            )

            self._complete_agent_batch(
                batch_id=batch_id,
                interaction_id=interaction_id,
                results=results,
                expected=expected,
                started_at=started_at,
                phase="failed",
            )

            return

        # ====================================================
        # WEB URL
        # ====================================================

        response_url = (
            self._extract_response_url(
                response
            )
        )

        if response_url:

            self.set_url(
                response_url
            )

        # ====================================================
        # SESSION
        # ====================================================

        self._add_assistant_to_session(
            answer
        )

        print()
        print(
            "=" * 60
        )
        print(
            "[ZOE AGENT → USER]"
        )
        print(
            "=" * 60
        )

        print(
            answer
        )

        print()

        # ====================================================
        # FINAL RESPONSE EVENT
        # ====================================================

        self._emit_event(
            "response",
            {
                "interaction_id": interaction_id,
                "text": answer,
                "speak": (
                    batch_speak_output
                    and response.speak
                ),
                "interrupt": response.interrupt,
                "batch_id": batch_id,
                "background": True,
                "agent_result": True,
                "final": True,
            },
        )

        self._emit_output(
            answer,
            speak_output=(
                batch_speak_output
                and response.speak
            ),
            interrupt=response.interrupt,
            event_type="response",
            metadata={
                "interaction_id": interaction_id,
                "batch_id": batch_id,
                "background": True,
                "agent_result": True,
                "final": True,
            },
        )

        # ====================================================
        # FINAL AGENT COMPLETION
        # ====================================================

        self._resolve_completion(
            interaction_id,
            answer=answer,
        )

        if not batch_speak_output:

            self._set_state(
                self.STATE_IDLE
            )

        self._complete_agent_batch(
            batch_id=batch_id,
            interaction_id=interaction_id,
            results=results,
            expected=expected,
            started_at=started_at,
            phase="completed",
        )

    # ========================================================
    # OUTPUT
    # ========================================================

    def _emit_output(
        self,
        text: str,
        speak_output: bool = True,
        interrupt: bool = False,
        event_type: str = "response",
        metadata: dict[str, Any] | None = None,
    ) -> None:

        if not text:

            return

        text = str(
            text
        ).strip()

        if not text:

            return

        metadata = (
            dict(metadata)
            if metadata
            else {}
        )

        # Metadata is supplemental and must never be able to overwrite
        # the canonical output fields. This prevents accidental corruption
        # when a caller supplies a colliding metadata key.
        event_data = {
            **metadata,
            "type": event_type,
            "text": text,
            "speak": bool(
                speak_output
            ),
            "interrupt": bool(
                interrupt
            ),
        }

        self._emit_event(
            "output",
            event_data,
        )

        with self._output_lock:

            if (
                interrupt
                and self.on_interrupt is not None
            ):

                try:

                    self.on_interrupt()

                except Exception as exc:

                    print(
                        "[ZOE RUNTIME] "
                        "Interrupt callback failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            if self.on_output is not None:

                try:

                    self.on_output(
                        text
                    )

                except Exception as exc:

                    print(
                        "[ZOE RUNTIME] "
                        "Output callback failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

            if (
                speak_output
                and self.on_speech is not None
            ):

                try:

                    self._set_state(
                        self.STATE_SPEAKING,
                        reason="tts",
                    )

                    def speech_complete() -> None:

                        if self.is_shutdown():

                            return

                        print(
                            "[ZOE RUNTIME] "
                            "TTS completion received."
                        )

                        self.set_speaking(
                            False
                        )

                    self.on_speech(
                        text,
                        speech_complete,
                    )

                except Exception as exc:

                    print(
                        "[ZOE RUNTIME] "
                        "Speech callback failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    self.set_speaking(
                        False
                    )

            elif speak_output:

                print(
                    "[ZOE RUNTIME] "
                    "Speech requested but no speech "
                    "callback is configured."
                )

    # ========================================================
    # EXTERNAL STATE HELPERS
    # ========================================================

    def set_listening(
        self,
        listening: bool,
    ) -> None:

        if self.is_shutdown():

            return

        if listening:

            self._set_state(
                self.STATE_LISTENING,
                reason="user_speech",
            )

        elif self.get_state() == self.STATE_LISTENING:

            self._set_state(
                self.STATE_IDLE
            )

    # ========================================================

    def set_speaking(
        self,
        speaking: bool,
    ) -> None:

        if self.is_shutdown():

            return

        if speaking:

            self._set_state(
                self.STATE_SPEAKING,
                reason="tts",
            )

            return

        current = self.get_state()

        if current != self.STATE_SPEAKING:

            return

        if self.has_active_agents():

            self._set_state(
                self.STATE_EXECUTING,
                reason="background_agents",
            )

            return

        self._set_state(
            self.STATE_IDLE,
            reason="tts_complete",
        )

    # ========================================================

    def set_muted(
        self,
        muted: bool,
    ) -> None:

        if muted:

            self._set_state(
                self.STATE_MUTED,
                reason="muted",
            )

        elif self.get_state() == self.STATE_MUTED:

            self._set_state(
                self.STATE_IDLE
            )

    # ========================================================
    # AGENT VALIDATION
    # ========================================================

    @staticmethod
    def _validate_agent(
        agent: str | None,
    ) -> None:

        if not agent:

            raise RuntimeError(
                "Brain selected an empty agent."
            )

        if agent not in AGENTS:

            raise RuntimeError(
                "Brain selected unknown agent: "
                f"{agent!r}"
            )

    # ========================================================
    # AGENT THREAD CLEANUP
    # ========================================================

    def _cleanup_agent_threads(
        self,
    ) -> None:

        with self._agent_threads_lock:

            if not self._agent_threads:

                return

            current = threading.current_thread()

            self._agent_threads = [
                thread
                for thread in self._agent_threads
                if thread.is_alive()
                and thread is not current
            ]

    # ========================================================
    # ACTIVE AGENTS
    # ========================================================

    def get_active_agents(
        self,
    ) -> list[str]:

        self._cleanup_agent_threads()

        with self._agent_threads_lock:

            return [
                thread.name
                for thread in self._agent_threads
                if thread.is_alive()
            ]

    # ========================================================

    def get_active_agent_jobs(
        self,
    ) -> list[dict[str, Any]]:

        with self._agent_batches_lock:

            jobs: list[
                dict[str, Any]
            ] = []

            for batch in self._agent_batches.values():

                for index, job in batch["jobs"].items():

                    if index in batch["results"]:

                        continue

                    jobs.append(
                        {
                            "job_id": job["job_id"],
                            "batch_id": batch["batch_id"],
                            "interaction_id": job[
                                "interaction_id"
                            ],
                            "agent": job["agent"],
                            "query": job["query"],
                            "description": (
                                self._describe_agent(
                                    job["agent"],
                                    job["query"],
                                )
                            ),
                            "status": job["status"],
                            "started_at": job["started_at"],
                        }
                    )

            return jobs

    # ========================================================

    def get_agent_batches(
        self,
    ) -> list[dict[str, Any]]:

        with self._agent_batches_lock:

            batches = []

            for batch in self._agent_batches.values():

                batches.append(
                    {
                        "batch_id": batch["batch_id"],
                        "interaction_id": batch[
                            "interaction_id"
                        ],
                        "expected": batch["expected"],
                        "completed": len(
                            batch["results"]
                        ),
                        "pending": (
                            batch["expected"]
                            - len(
                                batch["results"]
                            )
                        ),
                        "speak_output": batch[
                            "speak_output"
                        ],
                        "started_at": batch["started_at"],
                        "agents": [
                            {
                                "job_id": job["job_id"],
                                "agent": job["agent"],
                                "query": job["query"],
                                "status": job["status"],
                            }
                            for job in batch["jobs"].values()
                        ],
                    }
                )

            return batches

    # ========================================================
    # BACKGROUND WORK
    # ========================================================

    def get_background_work(
        self,
    ) -> dict[str, Any] | None:

        with self._agent_batches_lock:

            all_jobs: list[dict[str, Any]] = []

            for batch in self._agent_batches.values():

                phase = batch.get(
                    "phase",
                    "executing",
                )

                if phase not in {
                    "executing",
                    "synthesizing",
                }:

                    continue

                for index, job in batch["jobs"].items():

                    job_status = job.get(
                        "status",
                        "starting",
                    )

                    if (
                        job_status in {
                            "completed",
                            "failed",
                        }
                        and phase == "executing"
                    ):

                        continue

                    all_jobs.append(
                        {
                            "job_id": job["job_id"],
                            "batch_id": batch["batch_id"],
                            "interaction_id": job[
                                "interaction_id"
                            ],
                            "agent": job["agent"],
                            "query": job["query"],
                            "description": (
                                self._describe_agent(
                                    job["agent"],
                                    job["query"],
                                )
                            ),
                            "status": job_status,
                            "started_at": job["started_at"],
                            "phase": phase,
                        }
                    )

            if not all_jobs:

                return None

            phases = {
                job["phase"]
                for job in all_jobs
            }

            overall_phase = (
                "synthesizing"
                if "synthesizing" in phases
                else "executing"
            )

            return {
                "active": True,
                "phase": overall_phase,
                "count": len(all_jobs),
                "jobs": all_jobs,
            }

    # ========================================================

    def has_active_agents(
        self,
    ) -> bool:

        with self._agent_batches_lock:

            if self._agent_batches:

                return True

        self._cleanup_agent_threads()

        with self._agent_threads_lock:

            return any(
                thread.is_alive()
                for thread in self._agent_threads
            )

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(
        self,
    ) -> dict[str, Any]:

        try:

            status = (
                self.status_worker.snapshot()
            )

        except Exception as exc:

            print(
                "[ZOE RUNTIME] "
                "Status snapshot failed: "
                f"{type(exc).__name__}: {exc}"
            )

            status = {}

        return status

    # ========================================================
    # SCHEDULED REMINDERS
    # ========================================================

    def get_scheduled_reminders(
        self,
    ) -> list[dict[str, Any]]:

        try:

            snapshot = (
                self.status_worker.snapshot()
            )

        except Exception as exc:

            print(
                "[ZOE RUNTIME] "
                "Scheduled reminder snapshot failed: "
                f"{type(exc).__name__}: {exc}"
            )

            return []

        tasks = snapshot.get(
            "tasks",
            [],
        )

        if not isinstance(
            tasks,
            list,
        ):

            return []

        return [
            task
            for task in tasks
            if (
                isinstance(
                    task,
                    dict,
                )
                and task.get(
                    "due_datetime"
                )
            )
        ]

    # ========================================================
    # ACTIVITY
    # ========================================================

    def get_activity(
        self,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:

        with self._activity_lock:

            activities = list(
                self._activities
            )

        if limit is None:

            return activities

        try:

            limit = int(limit)

        except (
            TypeError,
            ValueError,
        ):

            limit = self.ACTIVITY_LIMIT

        limit = max(
            0,
            min(
                limit,
                self.ACTIVITY_LIMIT,
            ),
        )

        if limit == 0:

            return []

        return activities[-limit:]

    # ========================================================

    def _build_activity_entry(
        self,
        event: dict[str, Any],
    ) -> dict[str, Any] | None:

        event_type = event.get(
            "type",
            "",
        )

        data = event.get(
            "data",
            {},
        )

        if not isinstance(
            data,
            dict,
        ):

            data = {}

        timestamp = event.get(
            "timestamp",
            "",
        )

        if event_type not in self.ACTIVITY_EVENT_TYPES:

            return None

        activity = {
            "id": event.get(
                "id"
            ),
            "sequence": event.get(
                "sequence"
            ),
            "type": event_type,
            "timestamp": timestamp,
            "time": timestamp,
            "kind": "system",
            "surface": "generic",
            "label": "ZOE",
            "text": event_type,
            "data": data,
        }

        # ====================================================
        # RUNTIME READY
        # ====================================================

        if event_type == "runtime_ready":

            activity.update(
                {
                    "kind": "system",
                    "surface": "generic",
                    "label": "ZOE",
                    "text": "ZOE runtime online.",
                }
            )

        # ====================================================
        # WEB
        # ====================================================

        elif event_type == "web_open":

            url = str(
                data.get(
                    "url",
                    "",
                )
            ).strip()

            activity.update(
                {
                    "kind": "web",
                    "surface": "web",
                    "label": "Web",
                    "text": (
                        f"Opening {url}"
                        if url
                        else "Opening web page."
                    ),
                    "status": "opened",
                }
            )

        # ====================================================
        # STATUS
        # ====================================================

        elif event_type in {
            "status_event",
            "status_event_dropped",
            "status_ignored",
            "status_error",
            "status_worker_error",
        }:

            source = str(
                data.get(
                    "source",
                    "system",
                )
            ).lower()

            surface = source

            if surface not in {
                "calendar",
                "spotify",
                "weather",
                "reminder",
                "tasks",
            }:

                surface = "generic"

            event_name = str(
                data.get(
                    "event_type",
                    event_type,
                )
            )

            if event_type == "status_event":

                text = (
                    f"{source.title()} "
                    f"updated: {event_name}"
                )

            elif event_type == "status_ignored":

                text = (
                    f"Status event ignored: "
                    f"{event_name}"
                )

            elif event_type == "status_event_dropped":

                text = (
                    f"Status event dropped: "
                    f"{event_name}"
                )

            elif event_type == "status_error":

                text = (
                    f"Status processing error: "
                    f"{event_name}"
                )

            else:

                text = (
                    "Status worker error."
                )

            activity.update(
                {
                    "kind": "status",
                    "surface": surface,
                    "label": source.title(),
                    "text": text,
                }
            )

        # ====================================================
        # REMINDER
        # ====================================================

        elif event_type in {
            "reminder_triggered",
            "reminder_scheduler_error",
        }:

            if event_type == "reminder_triggered":

                name = str(
                    data.get(
                        "name",
                        "Reminder",
                    )
                ).strip()

                text = str(
                    data.get(
                        "text",
                        f"Reminder triggered: {name}",
                    )
                )

                activity.update(
                    {
                        "kind": "reminder",
                        "surface": "reminder",
                        "label": name,
                        "text": text,
                    }
                )

            else:

                activity.update(
                    {
                        "kind": "error",
                        "surface": "reminder",
                        "label": "Reminders",
                        "text": (
                            "Reminder scheduler error."
                        ),
                    }
                )

        # ====================================================
        # AGENT BATCH
        # ====================================================

        elif event_type == "agent_batch_started":

            count = data.get(
                "count",
                0,
            )

            activity.update(
                {
                    "kind": "agent",
                    "surface": "generic",
                    "label": "Agents",
                    "text": (
                        f"Started {count} "
                        f"background agent"
                        f"{'' if count == 1 else 's'}."
                    ),
                    "status": "running",
                }
            )

        elif event_type == "agent_started":

            agent = str(
                data.get(
                    "agent",
                    "Agent",
                )
            )

            description = str(
                data.get(
                    "description",
                    data.get(
                        "query",
                        f"Running {agent}",
                    ),
                )
            ).strip()

            activity.update(
                {
                    "kind": "agent",
                    "surface": "generic",
                    "label": agent,
                    "text": description,
                    "status": "running",
                }
            )

        elif event_type == "agent_progress":

            completed = data.get(
                "completed",
                0,
            )

            expected = data.get(
                "expected",
                0,
            )

            activity.update(
                {
                    "kind": "agent",
                    "surface": "generic",
                    "label": "Agents",
                    "text": (
                        f"Background work: "
                        f"{completed}/{expected} complete."
                    ),
                    "status": "running",
                }
            )

        elif event_type == "agent_completed":

            agent = str(
                data.get(
                    "agent",
                    "Agent",
                )
            )

            activity.update(
                {
                    "kind": "agent",
                    "surface": "generic",
                    "label": agent,
                    "text": (
                        f"{agent} completed."
                    ),
                    "status": "completed",
                }
            )

        elif event_type == "agent_failed":

            agent = str(
                data.get(
                    "agent",
                    "Agent",
                )
            )

            error = str(
                data.get(
                    "error",
                    "Unknown error",
                )
            )

            activity.update(
                {
                    "kind": "error",
                    "surface": "generic",
                    "label": agent,
                    "text": (
                        f"{agent} failed: {error}"
                    ),
                    "status": "failed",
                }
            )

        elif event_type == "agent_batch_completed":

            count = data.get(
                "count",
                0,
            )

            activity.update(
                {
                    "kind": "agent",
                    "surface": "generic",
                    "label": "Agents",
                    "text": (
                        f"Background batch completed "
                        f"({count} result"
                        f"{'' if count == 1 else 's'})."
                    ),
                    "status": "completed",
                }
            )

        # ====================================================
        # AGENT SYNTHESIS
        # ====================================================

        elif event_type == "agent_synthesis_started":

            count = data.get(
                "agent_count",
                0,
            )

            activity.update(
                {
                    "kind": "agent",
                    "surface": "generic",
                    "label": "ZOE",
                    "text": (
                        f"Synthesizing {count} "
                        f"agent result"
                        f"{'' if count == 1 else 's'}."
                    ),
                    "status": "running",
                }
            )

        elif event_type == "agent_synthesis_completed":

            activity.update(
                {
                    "kind": "agent",
                    "surface": "generic",
                    "label": "ZOE",
                    "text": (
                        "Agent results synthesized."
                    ),
                    "status": "completed",
                }
            )

        elif event_type == "agent_synthesis_error":

            error = str(
                data.get(
                    "error",
                    "Unknown error",
                )
            )

            activity.update(
                {
                    "kind": "error",
                    "surface": "generic",
                    "label": "ZOE",
                    "text": (
                        f"Agent synthesis failed: "
                        f"{error}"
                    ),
                    "status": "failed",
                }
            )

        # ====================================================
        # STATE
        # ====================================================

        elif event_type == "state_changed":

            state = str(
                data.get(
                    "state",
                    "unknown",
                )
            )

            reason = str(
                data.get(
                    "reason",
                    "",
                )
            ).strip()

            if reason:

                text = (
                    f"State → {state} "
                    f"({reason})"
                )

            else:

                text = (
                    f"State → {state}"
                )

            activity.update(
                {
                    "kind": "state",
                    "surface": "generic",
                    "label": "ZOE",
                    "text": text,
                    "status": state,
                }
            )

        return activity

    # ========================================================
    # SNAPSHOT
    # ========================================================

    def snapshot(
        self,
    ) -> dict[str, Any]:

        state = (
            self.get_state_snapshot()
        )

        status = (
            self.get_status()
        )

        with self._agent_batches_lock:

            active_batches = len(
                self._agent_batches
            )

        self._cleanup_agent_threads()

        with self._agent_threads_lock:

            active_threads = sum(
                1
                for thread in self._agent_threads
                if thread.is_alive()
            )

        return {
            "online": not self.is_shutdown(),

            "state": state["state"],
            "state_reason": state["reason"],
            "state_since": state["since"],

            # =================================================
            # WEB
            # =================================================

            "url": self.get_url(),

            "active_agents": active_threads,
            "active_agent_batches": active_batches,

            "agents": self.get_active_agents(),

            "active_jobs": (
                self.get_active_agent_jobs()
            ),

            "agent_batches": (
                self.get_agent_batches()
            ),

            "background_work": (
                self.get_background_work()
            ),

            "status": status,

            "activity": self.get_activity(
                self.ACTIVITY_LIMIT
            ),
        }

    # ========================================================
    # EVENT EMISSION
    # ========================================================

    def _emit_event(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> None:

        with self._event_lock:

            self._event_sequence += 1

            sequence = (
                self._event_sequence
            )

        event = {
            "id": uuid.uuid4().hex,
            "sequence": sequence,
            "type": event_type,
            "timestamp": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "data": (
                dict(data)
                if data
                else {}
            ),
        }

        if (
            event_type
            in self.ACTIVITY_EVENT_TYPES
        ):

            try:

                activity = (
                    self._build_activity_entry(
                        event
                    )
                )

                if activity is not None:

                    with self._activity_lock:

                        self._activities.append(
                            activity
                        )

            except Exception as exc:

                print(
                    "[ZOE RUNTIME] "
                    "Activity history update failed: "
                    f"{type(exc).__name__}: {exc}"
                )

        callback = self.on_event

        if callback is None:

            return

        try:

            callback(
                event
            )

        except Exception as exc:

            print(
                "[ZOE RUNTIME] "
                "Event callback failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(
        self,
    ) -> None:

        with self._shutdown_lock:

            if self._shutdown:

                return

            self._shutdown = True

        print(
            "[ZOE RUNTIME] "
            "Shutting down..."
        )

        # ====================================================
        # WAKE ALL COMPLETION WAITERS
        # ====================================================

        with self._completion_lock:

            for waiter in self._completion_waiters.values():

                waiter["error"] = (
                    "ZOE runtime is shutting down."
                )

                waiter["event"].set()

        self._emit_event(
            "runtime_shutdown_started"
        )

        self._set_state(
            self.STATE_IDLE,
            reason="shutdown",
        )

        try:

            self.event_bus.unsubscribe(
                self._on_status_event
            )

        except Exception as exc:

            print(
                "[ZOE RUNTIME] "
                "Event unsubscribe failed: "
                f"{type(exc).__name__}: {exc}"
            )

        try:

            self.status_worker.stop()

        except Exception as exc:

            print(
                "[ZOE RUNTIME] "
                "Status worker shutdown failed: "
                f"{type(exc).__name__}: {exc}"
            )

        self._status_shutdown.set()

        dispatcher = (
            self._status_dispatcher
        )

        if (
            dispatcher
            and dispatcher
            is not threading.current_thread()
        ):

            dispatcher.join(
                timeout=5.0
            )

        while True:

            try:

                self._status_queue.get_nowait()

                self._status_queue.task_done()

            except queue.Empty:

                break

        self._agent_shutdown.set()

        with self._agent_batches_lock:

            self._agent_batches.clear()

        self._cleanup_agent_threads()

        with self._agent_threads_lock:

            agent_threads = list(
                self._agent_threads
            )

        for thread in agent_threads:

            if (
                thread is not threading.current_thread()
                and thread.is_alive()
            ):

                print(
                    f"[ZOE RUNTIME] "
                    f"Waiting for agent thread: "
                    f"{thread.name}"
                )

                thread.join(
                    timeout=10.0
                )

                if thread.is_alive():

                    print(
                        f"[ZOE RUNTIME] "
                        f"WARNING: Agent thread did not "
                        f"terminate: {thread.name}"
                    )

        active_agents = (
            self.get_active_agents()
        )

        if active_agents:

            print(
                "[ZOE RUNTIME] "
                "Active background agents still running "
                "after join: "
                f"{active_agents}"
            )

        self._emit_event(
            "runtime_shutdown_complete"
        )

        # ----------------------------------------------------
        # Completion waiters can now be discarded safely.
        # ----------------------------------------------------

        with self._completion_lock:

            self._completion_waiters.clear()

        print(
            "[ZOE RUNTIME] "
            "Shutdown complete."
        )


# ============================================================
# FACTORY
# ============================================================


def create_zoe_runtime(
    on_output: Callable[[str], None] | None = None,
    on_speech: Callable[
        [str, Callable[[], None] | None],
        None,
    ] | None = None,
    on_interrupt: Callable[[], None] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    session_manager: SessionManager | None = None,
) -> ZoeRuntime:

    return ZoeRuntime(
        on_output=on_output,
        on_speech=on_speech,
        on_interrupt=on_interrupt,
        on_event=on_event,
        session_manager=session_manager,
    )


# ============================================================
# DIRECT TEST
# ============================================================


if __name__ == "__main__":

    def debug_output(
        text: str,
    ) -> None:

        print(
            f"[OUTPUT] {text}"
        )

    def debug_event(
        event: dict[str, Any],
    ) -> None:

        print(
            "[EVENT]",
            event["type"],
            event["data"],
        )

    runtime = create_zoe_runtime(
        on_output=debug_output,
        on_event=debug_event,
    )

    try:

        while True:

            message = input(
                "\nYou: "
            ).strip()

            if not message:

                continue

            if message.lower() in {
                "exit",
                "quit",
                "shutdown",
            }:

                break

            try:

                answer = runtime.run(
                    message
                )

                if answer:

                    print(
                        f"\nZOE: {answer}"
                    )

            except Exception as exc:

                print(
                    "\n[ZOE ERROR]"
                )

                print(
                    f"{type(exc).__name__}: {exc}"
                )

    except KeyboardInterrupt:

        print()

    finally:

        runtime.shutdown()