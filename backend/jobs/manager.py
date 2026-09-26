from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable


# ============================================================
# JOB
# ============================================================

@dataclass
class AgentJob:

    job_id: str

    agent: str

    query: str

    status: str = "queued"

    result: Any = None

    error: str = ""

    created_at: str = ""

    started_at: str = ""

    completed_at: str = ""

    future: Future | None = None


# ============================================================
# JOB MANAGER
# ============================================================

class JobManager:
    """
    Manages asynchronous ZOE background jobs.

    Responsibilities:

        - Create job IDs
        - Run agents in background workers
        - Track job state
        - Store results
        - Handle failures
        - Notify ZOE when a job completes
        - Support cancellation
        - Cleanly shut down workers

    The JobManager does NOT decide what ZOE should say.
    """

    def __init__(
        self,
        max_workers: int = 4,
        on_complete: Callable[[AgentJob], None] | None = None,
        on_error: Callable[[AgentJob], None] | None = None,
    ):

        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="zoe-agent",
        )

        self.on_complete = on_complete

        self.on_error = on_error

        self.jobs: dict[str, AgentJob] = {}

        self._lock = threading.RLock()

        self._shutdown = False

        print(
            f"[ZOE JOB MANAGER] Ready | "
            f"workers={max_workers}"
        )

    # ========================================================
    # START
    # ========================================================

    def start(
        self,
        agent: str,
        query: str,
        function: Callable[..., Any],
    ) -> str:

        if self._shutdown:

            raise RuntimeError(
                "Cannot start a job after JobManager shutdown."
            )

        if not callable(function):

            raise TypeError(
                f"Agent {agent!r} is not callable."
            )

        job_id = (
            f"{agent}_{uuid.uuid4().hex[:8]}"
        )

        now = self._now()

        job = AgentJob(
            job_id=job_id,
            agent=agent,
            query=query,
            status="queued",
            created_at=now,
        )

        with self._lock:

            self.jobs[job_id] = job

        future = self.executor.submit(
            self._execute,
            job_id,
            function,
            query,
        )

        with self._lock:

            job.future = future

        print()
        print(
            "============================================================"
        )
        print(
            "[ZOE JOB STARTED]"
        )
        print(
            "============================================================"
        )
        print(
            f"Job   : {job_id}"
        )
        print(
            f"Agent : {agent}"
        )
        print(
            f"Query : {query}"
        )
        print()

        return job_id

    # ========================================================
    # EXECUTE
    # ========================================================

    def _execute(
        self,
        job_id: str,
        function: Callable[..., Any],
        query: str,
    ) -> None:

        job = self.get(job_id)

        if job is None:

            return

        with self._lock:

            job.status = "running"

            job.started_at = self._now()

        try:

            result = function(
                query=query
            )

            with self._lock:

                job.status = "completed"

                job.result = result

                job.completed_at = self._now()

            print()
            print(
                "============================================================"
            )
            print(
                "[ZOE JOB COMPLETED]"
            )
            print(
                "============================================================"
            )
            print(
                f"Job   : {job.job_id}"
            )
            print(
                f"Agent : {job.agent}"
            )
            print()

            if self.on_complete:

                try:

                    self.on_complete(job)

                except Exception as exc:

                    print(
                        "[ZOE JOB CALLBACK ERROR]",
                        exc,
                    )

        except Exception as exc:

            with self._lock:

                job.status = "failed"

                job.error = (
                    f"{type(exc).__name__}: {exc}"
                )

                job.completed_at = self._now()

            print()
            print(
                "============================================================"
            )
            print(
                "[ZOE JOB FAILED]"
            )
            print(
                "============================================================"
            )
            print(
                f"Job   : {job.job_id}"
            )
            print(
                f"Agent : {job.agent}"
            )
            print(
                f"Error : {job.error}"
            )
            print()

            if self.on_error:

                try:

                    self.on_error(job)

                except Exception as callback_exc:

                    print(
                        "[ZOE JOB ERROR CALLBACK FAILED]",
                        callback_exc,
                    )

    # ========================================================
    # GET
    # ========================================================

    def get(
        self,
        job_id: str,
    ) -> AgentJob | None:

        with self._lock:

            return self.jobs.get(
                job_id
            )

    # ========================================================
    # STATUS
    # ========================================================

    def status(
        self,
        job_id: str,
    ) -> str | None:

        job = self.get(
            job_id
        )

        if job is None:

            return None

        return job.status

    # ========================================================
    # LIST
    # ========================================================

    def list_jobs(
        self,
    ) -> list[AgentJob]:

        with self._lock:

            return list(
                self.jobs.values()
            )

    # ========================================================
    # CANCEL
    # ========================================================

    def cancel(
        self,
        job_id: str,
    ) -> bool:

        job = self.get(
            job_id
        )

        if job is None:

            return False

        if job.future is None:

            return False

        cancelled = job.future.cancel()

        if cancelled:

            with self._lock:

                job.status = "cancelled"

                job.completed_at = self._now()

        return cancelled

    # ========================================================
    # SHUTDOWN
    # ========================================================

    def shutdown(
        self,
        wait: bool = False,
    ) -> None:

        with self._lock:

            if self._shutdown:

                return

            self._shutdown = True

        self.executor.shutdown(
            wait=wait,
            cancel_futures=True,
        )

        print(
            "[ZOE JOB MANAGER] Shutdown complete."
        )

    # ========================================================
    # TIME
    # ========================================================

    @staticmethod
    def _now() -> str:

        return datetime.now(
            timezone.utc
        ).isoformat()