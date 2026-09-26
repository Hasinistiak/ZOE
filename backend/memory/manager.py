
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable

from .database import memory_database
from .learning import build_memory_learning_prompt
from .retriever import memory_retriever


# ============================================================
# CONFIG
# ============================================================

MEMORY_MODEL = os.getenv(
    "ZOE_MEMORY_MODEL",
    os.getenv("ZOE_MODEL", "gemini-3.5-flash-lite"),
)

MEMORY_MAX_EXISTING = int(
    os.getenv(
        "ZOE_MEMORY_MAX_EXISTING",
        "200",
    )
)

MEMORY_MAX_CONVERSATION_CHARS = int(
    os.getenv(
        "ZOE_MEMORY_MAX_CONVERSATION_CHARS",
        "8000",
    )
)

MEMORY_WORKERS = int(
    os.getenv(
        "ZOE_MEMORY_WORKERS",
        "1",
    )
)

# How long a key is temporarily considered unavailable after
# a rate-limit/quota failure.
#
# Daily project quota errors should normally stay quarantined
# until the process is restarted or the configured cooldown
# expires. The provider should ideally own this policy, but
# this fallback implementation keeps memory independently safe.
MEMORY_KEY_COOLDOWN_SECONDS = float(
    os.getenv(
        "ZOE_GEMINI_KEY_COOLDOWN",
        "3600",
    )
)


# ============================================================
# MANAGER
# ============================================================

class MemoryManager:
    """
    Maintains ZOE's long-term knowledge.

    Responsibilities:

        1. Receive completed conversation turns.
        2. Store raw conversational memories.
        3. Run durable-memory analysis asynchronously.
        4. Apply ADD / UPDATE / DELETE / IGNORE operations.

    The Brain remains responsible for deciding when memory
    should be retrieved.

    Gemini usage:

        - A shared analyzer/provider can be injected through
          `analyzer=...`.
        - If no analyzer is supplied, this class uses its own
          multi-key Gemini fallback implementation.

    Environment variables:

        GEMINI_API_KEY
        GEMINI_API_KEY2
        GEMINI_API_KEY3
        ...

    The first configured key is used normally.

    When Gemini returns a quota/rate-limit error, that key is
    temporarily quarantined and the next available key is tried.
    """

    VALID_ACTIONS = {
        "ADD",
        "UPDATE",
        "DELETE",
        "IGNORE",
    }

    VALID_MEMORY_TYPES = {
        "fact",
        "preference",
        "configuration",
        "project",
        "decision",
    }

    def __init__(
        self,
        database=None,
        retriever=None,
        analyzer: Callable[
            [str],
            str | dict[str, Any],
        ] | None = None,
    ) -> None:

        self.database = (
            database
            or memory_database
        )

        self.retriever = (
            retriever
            or memory_retriever
        )

        # Preferred path:
        #
        # The application should inject the same Gemini
        # provider used by the Brain.
        #
        # Example:
        #
        # MemoryManager(
        #     analyzer=gemini_provider.analyze_memory
        # )
        #
        self._analyzer = analyzer

        self._executor = ThreadPoolExecutor(
            max_workers=max(
                1,
                MEMORY_WORKERS,
            ),
            thread_name_prefix="zoe-memory",
        )

        self._futures: set[Future[Any]] = set()

        self._future_lock = threading.RLock()

        self._shutdown = False

        # ----------------------------------------------------
        # Fallback Gemini key pool.
        #
        # These are only used when an analyzer/provider was not
        # injected.
        # ----------------------------------------------------

        self._gemini_lock = threading.RLock()

        self._gemini_keys = self._load_gemini_keys()

        self._gemini_clients: dict[int, Any] = {}

        self._active_gemini_key = 0

        self._gemini_key_cooldowns: dict[int, float] = {}

    # ========================================================
    # GEMINI KEY DISCOVERY
    # ========================================================

    @staticmethod
    def _load_gemini_keys() -> list[str]:

        keys: list[str] = []

        # ----------------------------------------------------
        # Primary key.
        # ----------------------------------------------------

        primary = os.getenv(
            "GEMINI_API_KEY"
        )

        if primary:
            primary = primary.strip()

            if primary:
                keys.append(primary)

        # ----------------------------------------------------
        # Numbered keys.
        #
        # GEMINI_API_KEY2
        # GEMINI_API_KEY3
        # GEMINI_API_KEY4
        # ...
        #
        # Continue until the configured sequence ends.
        # ----------------------------------------------------

        index = 2

        while True:

            value = os.getenv(
                f"GEMINI_API_KEY{index}"
            )

            if value is None:

                # Allow gaps, but stop after a reasonable
                # number of consecutive missing keys.
                #
                # This prevents accidentally scanning forever.
                missing_count = index

                if missing_count > 100:
                    break

                # We cannot know whether a later key exists
                # without scanning. Continue through the
                # expected range.
                index += 1
                continue

            value = value.strip()

            if value:
                keys.append(value)

            index += 1

            if index > 100:
                break

        # ----------------------------------------------------
        # Remove duplicates while preserving order.
        # ----------------------------------------------------

        unique: list[str] = []
        seen: set[str] = set()

        for key in keys:

            if key in seen:
                continue

            seen.add(key)
            unique.append(key)

        return unique

    # ========================================================
    # GEMINI ERROR DETECTION
    # ========================================================

    @staticmethod
    def _is_quota_error(
        exc: BaseException,
    ) -> bool:

        text = str(exc).lower()

        quota_markers = (
            "429",
            "resource_exhausted",
            "resource exhausted",
            "quota exceeded",
            "rate limit",
            "rate_limit",
            "too many requests",
            "generaterequestsperday",
            "generaterequestsperminute",
            "requestsperday",
            "requestsperminute",
        )

        return any(
            marker in text
            for marker in quota_markers
        )

    # ========================================================
    # KEY AVAILABILITY
    # ========================================================

    def _key_available(
        self,
        index: int,
    ) -> bool:

        cooldown_until = (
            self._gemini_key_cooldowns.get(
                index,
                0.0,
            )
        )

        return (
            time.monotonic()
            >= cooldown_until
        )

    def _quarantine_gemini_key(
        self,
        index: int,
        exc: BaseException,
    ) -> None:

        cooldown = (
            MEMORY_KEY_COOLDOWN_SECONDS
        )

        with self._gemini_lock:

            self._gemini_key_cooldowns[index] = (
                time.monotonic()
                + cooldown
            )

        print(
            "[MEMORY] Gemini key "
            f"{index + 1} quarantined for "
            f"{cooldown:.0f}s: "
            f"{type(exc).__name__}: {exc}"
        )

    # ========================================================
    # KEY ORDER
    # ========================================================

    def _gemini_key_order(self) -> list[int]:

        with self._gemini_lock:

            count = len(
                self._gemini_keys
            )

            if count == 0:
                return []

            start = (
                self._active_gemini_key
                % count
            )

            order: list[int] = []

            for offset in range(count):

                index = (
                    start + offset
                ) % count

                if self._key_available(
                    index
                ):

                    order.append(
                        index
                    )

            return order

    # ========================================================
    # GEMINI CLIENT
    # ========================================================

    def _get_gemini_client(
        self,
        index: int,
    ) -> Any:

        with self._gemini_lock:

            if index in self._gemini_clients:

                return self._gemini_clients[
                    index
                ]

            try:

                from google import genai

            except ImportError as exc:

                raise RuntimeError(
                    "google-genai is required for "
                    "the default memory analyzer."
                ) from exc

            client = genai.Client(
                api_key=self._gemini_keys[index]
            )

            self._gemini_clients[index] = client

            return client

    # ========================================================
    # DEFAULT GEMINI ANALYZER
    # ========================================================

    def _default_analyzer(
        self,
        prompt: str,
    ) -> str:

        """
        Gemini-backed memory analyzer using the same
        GEMINI_API_KEY / GEMINI_API_KEY2 / ... convention.

        This is a fallback.

        Production ZOE should preferably inject the shared
        Gemini provider used by the Brain.
        """

        if not self._gemini_keys:

            raise RuntimeError(
                "No Gemini API keys are configured. "
                "Expected GEMINI_API_KEY, GEMINI_API_KEY2, ..."
            )

        try:

            from google.genai import types

        except ImportError as exc:

            raise RuntimeError(
                "google-genai is required for the "
                "default memory analyzer."
            ) from exc

        key_order = (
            self._gemini_key_order()
        )

        if not key_order:

            raise RuntimeError(
                "All configured Gemini API keys "
                "are currently unavailable."
            )

        last_error: BaseException | None = None

        for key_index in key_order:

            try:

                client = (
                    self._get_gemini_client(
                        key_index
                    )
                )

                response = (
                    client.models.generate_content(
                        model=MEMORY_MODEL,
                        contents=prompt,
                        config=(
                            types.GenerateContentConfig(
                                temperature=0.0,
                                response_mime_type=(
                                    "application/json"
                                ),
                            )
                        ),
                    )
                )

                response_text = getattr(
                    response,
                    "text",
                    None,
                )

                if not response_text:

                    raise RuntimeError(
                        "Memory analyzer returned "
                        "an empty response."
                    )

                # ------------------------------------------------
                # Successful key becomes the active key.
                # ------------------------------------------------

                with self._gemini_lock:

                    self._active_gemini_key = (
                        key_index
                    )

                    self._gemini_key_cooldowns.pop(
                        key_index,
                        None,
                    )

                if key_index != self._active_gemini_key:

                    print(
                        "[MEMORY] Gemini failover "
                        f"succeeded on key "
                        f"{key_index + 1}."
                    )

                return str(
                    response_text
                )

            except Exception as exc:

                last_error = exc

                if self._is_quota_error(
                    exc
                ):

                    self._quarantine_gemini_key(
                        key_index,
                        exc,
                    )

                    # Immediately try another key.
                    continue

                # ------------------------------------------------
                # Non-quota errors are not blindly retried
                # against every key.
                #
                # They may indicate:
                #   - malformed request
                #   - invalid model
                #   - invalid schema
                #   - programming error
                # ------------------------------------------------

                raise

        raise RuntimeError(
            "All available Gemini keys failed "
            "for memory analysis."
        ) from last_error

    # ========================================================
    # TIME
    # ========================================================

    @staticmethod
    def _now() -> str:

        return (
            datetime.now()
            .astimezone()
            .isoformat()
        )

    # ========================================================
    # SUBMIT TURN
    # ========================================================

    def submit_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str | None = None,
    ) -> Future[Any] | None:

        if self._shutdown:

            raise RuntimeError(
                "MemoryManager has been shut down."
            )

        session_id = str(
            session_id
        ).strip()

        user_message = str(
            user_message
        ).strip()

        if not session_id:

            raise ValueError(
                "Memory turn requires session_id."
            )

        if not user_message:

            return None

        assistant_message_clean = None

        if assistant_message is not None:

            assistant_message_clean = str(
                assistant_message
            ).strip()

            if not assistant_message_clean:

                assistant_message_clean = None

        # ----------------------------------------------------
        # Store raw messages synchronously.
        # ----------------------------------------------------

        timestamp = self._now()

        user_memory_id = (
            self.retriever.store(
                session_id=session_id,
                role="user",
                content=user_message,
                timestamp=timestamp,
                importance=0.5,
                memory_type="conversation",
                metadata={
                    "source": "conversation",
                    "turn_role": "user",
                },
            )
        )

        assistant_memory_id = None

        if assistant_message_clean:

            assistant_timestamp = (
                self._now()
            )

            assistant_memory_id = (
                self.retriever.store(
                    session_id=session_id,
                    role="assistant",
                    content=assistant_message_clean,
                    timestamp=assistant_timestamp,
                    importance=0.3,
                    memory_type="conversation",
                    metadata={
                        "source": "conversation",
                        "turn_role": "assistant",
                    },
                )
            )

        # ----------------------------------------------------
        # Build learning payload.
        # ----------------------------------------------------

        conversation = (
            f"User:\n{user_message}"
        )

        if assistant_message_clean:

            conversation += (
                "\n\nZOE:\n"
                f"{assistant_message_clean}"
            )

        if len(conversation) > (
            MEMORY_MAX_CONVERSATION_CHARS
        ):

            conversation = (
                conversation[
                    :MEMORY_MAX_CONVERSATION_CHARS
                ].rstrip()
                + "\n..."
            )

        # ----------------------------------------------------
        # Run durable-memory analysis asynchronously.
        # ----------------------------------------------------

        future = self._executor.submit(
            self._process_learning,
            session_id,
            conversation,
            user_memory_id,
            assistant_memory_id,
        )

        self._track_future(
            future
        )

        return future

    # ========================================================
    # FUTURE TRACKING
    # ========================================================

    def _track_future(
        self,
        future: Future[Any],
    ) -> None:

        with self._future_lock:

            self._futures.add(
                future
            )

        future.add_done_callback(
            self._future_finished
        )

    def _future_finished(
        self,
        future: Future[Any],
    ) -> None:

        with self._future_lock:

            self._futures.discard(
                future
            )

        try:

            result = future.result()

            if result is not None:

                print(
                    "[MEMORY] Learning complete: "
                    f"{result}"
                )

        except Exception as exc:

            print(
                "[MEMORY] Background learning failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # ========================================================
    # PROCESS LEARNING
    # ========================================================

    def _process_learning(
        self,
        session_id: str,
        conversation: str,
        user_memory_id: int,
        assistant_memory_id: int | None,
    ) -> dict[str, Any]:

        existing = self.all_active()

        existing = existing[
            :MEMORY_MAX_EXISTING
        ]

        prompt = build_memory_learning_prompt(
            conversation=conversation,
            existing_memory=existing,
        )

        raw_result = self._analyze(
            prompt
        )

        operations = self._parse_operations(
            raw_result
        )

        results: list[dict[str, Any]] = []

        for operation in operations:

            try:

                result = self.apply(
                    action=operation["action"],
                    subject=operation["subject"],
                    key=operation["key"],
                    value=operation["value"],
                    memory_type=operation[
                        "memory_type"
                    ],
                    confidence=operation[
                        "confidence"
                    ],
                    importance=operation[
                        "importance"
                    ],
                    session_id=session_id,
                    source_memory_id=user_memory_id,
                    metadata={
                        "source": "memory_learning",
                        "assistant_memory_id": (
                            assistant_memory_id
                        ),
                    },
                )

                results.append(result)

            except Exception as exc:

                print(
                    "[MEMORY] Invalid learning operation: "
                    f"{type(exc).__name__}: {exc}"
                )

        return {
            "session_id": session_id,
            "operations": results,
        }

    # ========================================================
    # ANALYZE
    # ========================================================

    def _analyze(
        self,
        prompt: str,
    ) -> str | dict[str, Any]:

        if self._analyzer is not None:

            return self._analyzer(
                prompt
            )

        return self._default_analyzer(
            prompt
        )

    # ========================================================
    # PARSE OPERATIONS
    # ========================================================

    @classmethod
    def _parse_operations(
        cls,
        raw_result: str | dict[str, Any],
    ) -> list[dict[str, Any]]:

        if isinstance(
            raw_result,
            dict,
        ):

            data = raw_result

        else:

            text = str(
                raw_result
            ).strip()

            if not text:

                return []

            text = cls._strip_json_fences(
                text
            )

            try:

                data = json.loads(
                    text
                )

            except json.JSONDecodeError:

                data = cls._extract_json_object(
                    text
                )

        if not isinstance(
            data,
            dict,
        ):

            raise ValueError(
                "Memory analyzer response must "
                "be a JSON object."
            )

        operations = data.get(
            "operations",
            [],
        )

        if operations is None:

            return []

        if not isinstance(
            operations,
            list,
        ):

            raise ValueError(
                "'operations' must be a list."
            )

        validated: list[
            dict[str, Any]
        ] = []

        for raw_operation in operations:

            operation = cls._validate_operation(
                raw_operation
            )

            if operation is not None:

                validated.append(
                    operation
                )

        return validated

    # ========================================================
    # JSON HELPERS
    # ========================================================

    @staticmethod
    def _strip_json_fences(
        text: str,
    ) -> str:

        text = text.strip()

        if text.startswith("```"):

            text = re.sub(
                r"^```(?:json)?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            )

            text = re.sub(
                r"\s*```$",
                "",
                text,
            )

        return text.strip()

    @staticmethod
    def _extract_json_object(
        text: str,
    ) -> dict[str, Any]:

        start = text.find("{")
        end = text.rfind("}")

        if (
            start < 0
            or end <= start
        ):

            raise ValueError(
                "Memory analyzer did not return valid JSON."
            )

        candidate = text[
            start:end + 1
        ]

        data = json.loads(
            candidate
        )

        if not isinstance(
            data,
            dict,
        ):

            raise ValueError(
                "Extracted memory JSON is not an object."
            )

        return data

    # ========================================================
    # OPERATION VALIDATION
    # ========================================================

    @classmethod
    def _validate_operation(
        cls,
        operation: Any,
    ) -> dict[str, Any] | None:

        if not isinstance(
            operation,
            dict,
        ):

            return None

        action = str(
            operation.get(
                "action",
                "IGNORE",
            )
        ).strip().upper()

        if action not in cls.VALID_ACTIONS:

            return None

        if action == "IGNORE":

            return {
                "action": "IGNORE",
                "subject": "",
                "key": "",
                "value": "",
                "memory_type": "fact",
                "confidence": 0.0,
                "importance": 0.0,
            }

        subject = str(
            operation.get(
                "subject",
                "",
            )
            or ""
        ).strip()

        key = str(
            operation.get(
                "key",
                "",
            )
            or ""
        ).strip()

        value = str(
            operation.get(
                "value",
                "",
            )
            or ""
        ).strip()

        if not subject or not key:

            return None

        if (
            action != "DELETE"
            and not value
        ):

            return None

        memory_type = str(
            operation.get(
                "memory_type",
                "fact",
            )
            or "fact"
        ).strip().lower()

        if memory_type not in cls.VALID_MEMORY_TYPES:

            memory_type = "fact"

        confidence = cls._number(
            operation.get(
                "confidence",
                1.0,
            ),
            default=1.0,
        )

        importance = cls._number(
            operation.get(
                "importance",
                0.5,
            ),
            default=0.5,
        )

        return {
            "action": action,
            "subject": subject,
            "key": key,
            "value": value,
            "memory_type": memory_type,
            "confidence": confidence,
            "importance": importance,
        }

    @staticmethod
    def _number(
        value: Any,
        default: float,
    ) -> float:

        try:

            value = float(value)

        except Exception:

            value = default

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )

    # ========================================================
    # APPLY
    # ========================================================

    def apply(
        self,
        action: str,
        *,
        subject: str = "",
        key: str = "",
        value: str = "",
        memory_type: str = "fact",
        confidence: float = 1.0,
        importance: float = 0.5,
        session_id: str | None = None,
        source_memory_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:

        action = str(
            action
        ).strip().upper()

        if action not in self.VALID_ACTIONS:

            raise ValueError(
                f"Invalid memory action: {action}"
            )

        if action == "IGNORE":

            return {
                "action": "IGNORE",
                "status": "ignored",
            }

        subject = str(
            subject
        ).strip()

        key = str(
            key
        ).strip()

        value = str(
            value
        ).strip()

        if not subject or not key:

            raise ValueError(
                "Durable memory requires "
                "subject and key."
            )

        if (
            action != "DELETE"
            and not value
        ):

            raise ValueError(
                "Durable memory requires value."
            )

        memory_type = str(
            memory_type
        ).strip().lower()

        if memory_type not in self.VALID_MEMORY_TYPES:

            memory_type = "fact"

        confidence = max(
            0.0,
            min(
                1.0,
                float(confidence),
            ),
        )

        importance = max(
            0.0,
            min(
                1.0,
                float(importance),
            ),
        )

        now = self._now()

        existing = (
            self.database.get_active_durable(
                subject,
                key,
            )
        )

        # ====================================================
        # ADD
        # ====================================================

        if action == "ADD":

            if existing is not None:

                if (
                    str(existing["value"])
                    == value
                ):

                    self.database.update_durable(
                        memory_id=int(
                            existing["id"]
                        ),
                        value=value,
                        confidence=confidence,
                        importance=importance,
                        timestamp=now,
                        metadata=metadata,
                    )

                    return {
                        "action": "UPDATE",
                        "status": "reinforced",
                        "id": int(
                            existing["id"]
                        ),
                    }

                action = "UPDATE"

            else:

                memory_id = (
                    self.database.insert_durable(
                        subject=subject,
                        key=key,
                        value=value,
                        memory_type=memory_type,
                        confidence=confidence,
                        importance=importance,
                        created_at=now,
                        updated_at=now,
                        valid_from=now,
                        source_session_id=session_id,
                        source_memory_id=source_memory_id,
                        metadata=metadata,
                    )
                )

                return {
                    "action": "ADD",
                    "status": "created",
                    "id": memory_id,
                    "subject": subject,
                    "key": key,
                    "value": value,
                }

        # ====================================================
        # UPDATE
        # ====================================================

        if action == "UPDATE":

            if existing is None:

                memory_id = (
                    self.database.insert_durable(
                        subject=subject,
                        key=key,
                        value=value,
                        memory_type=memory_type,
                        confidence=confidence,
                        importance=importance,
                        created_at=now,
                        updated_at=now,
                        valid_from=now,
                        source_session_id=session_id,
                        source_memory_id=source_memory_id,
                        metadata=metadata,
                    )
                )

                return {
                    "action": "ADD",
                    "status": "created_missing",
                    "id": memory_id,
                    "subject": subject,
                    "key": key,
                    "value": value,
                }

            old_value = str(
                existing["value"]
            )

            if old_value == value:

                self.database.update_durable(
                    memory_id=int(
                        existing["id"]
                    ),
                    value=value,
                    confidence=confidence,
                    importance=importance,
                    timestamp=now,
                    metadata=metadata,
                )

                return {
                    "action": "UPDATE",
                    "status": "reinforced",
                    "id": int(
                        existing["id"]
                    ),
                }

            # Preserve historical state.

            self.database.supersede_durable(
                memory_id=int(
                    existing["id"]
                ),
                timestamp=now,
            )

            new_id = (
                self.database.insert_durable(
                    subject=subject,
                    key=key,
                    value=value,
                    memory_type=memory_type,
                    confidence=confidence,
                    importance=importance,
                    created_at=now,
                    updated_at=now,
                    valid_from=now,
                    source_session_id=session_id,
                    source_memory_id=source_memory_id,
                    supersedes_id=int(
                        existing["id"]
                    ),
                    metadata=metadata,
                )
            )

            return {
                "action": "UPDATE",
                "status": "superseded_old",
                "old_id": int(
                    existing["id"]
                ),
                "new_id": new_id,
                "old_value": old_value,
                "new_value": value,
            }

        # ====================================================
        # DELETE
        # ====================================================

        if action == "DELETE":

            if existing is None:

                return {
                    "action": "DELETE",
                    "status": "not_found",
                }

            self.database.delete_durable(
                memory_id=int(
                    existing["id"]
                ),
                timestamp=now,
            )

            return {
                "action": "DELETE",
                "status": "deleted",
                "id": int(
                    existing["id"]
                ),
                "subject": subject,
                "key": key,
            }

        raise RuntimeError(
            "Memory manager reached "
            "an impossible state."
        )

    # ========================================================
    # CURRENT STATE
    # ========================================================

    def current_state(
        self,
        subject: str,
    ) -> dict[str, str]:

        rows = (
            self.database.search_durable(
                subject=subject
            )
        )

        return {
            str(row["memory_key"]):
            str(row["value"])
            for row in rows
        }

    # ========================================================
    # ALL KNOWLEDGE
    # ========================================================

    def all_active(
        self,
    ) -> list[dict[str, Any]]:

        rows = (
            self.database
            .get_all_active_durable()
        )

        return [
            dict(row)
            for row in rows
        ]

    # ========================================================
    # WAIT
    # ========================================================

    def wait_for_pending(
        self,
        timeout: float | None = None,
    ) -> None:

        with self._future_lock:

            futures = list(
                self._futures
            )

        for future in futures:

            future.result(
                timeout=timeout
            )

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(
        self,
        wait: bool = True,
    ) -> None:

        if self._shutdown:

            return

        self._shutdown = True

        self._executor.shutdown(
            wait=wait
        )


# ============================================================
# GLOBAL
# ============================================================

memory_manager = MemoryManager()
