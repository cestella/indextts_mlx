"""Priority job queue with application grouping and heartbeat expiry."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Job:
    id: str
    model_type: str
    model: str
    application_id: str
    priority: int
    payload: dict
    result_path: Optional[str]
    status: str  # queued/running/done/failed/cancelled/expired
    created_at: float
    last_heartbeat: float
    queue_seq: int
    result: Optional[dict] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "model_type": self.model_type,
            "model": self.model,
            "application_id": self.application_id,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "last_heartbeat": self.last_heartbeat,
            "result_path": self.result_path,
            "result": self.result,
            "error": self.error,
        }


class JobQueue:
    def __init__(self, heartbeat_timeout: float = 300.0, max_size: int = 100) -> None:
        self._jobs: dict[str, Job] = {}
        self._seq = 0
        self._lock = threading.Lock()
        self._heartbeat_timeout = heartbeat_timeout
        self._max_size = max_size

    def submit(
        self,
        model_type: str,
        model: str,
        application_id: str,
        priority: int,
        payload: dict,
        result_path: Optional[str] = None,
    ) -> str:
        with self._lock:
            queued_count = sum(1 for j in self._jobs.values() if j.status == "queued")
            if queued_count >= self._max_size:
                raise ValueError(f"Queue full ({self._max_size} jobs)")
            job_id = str(uuid.uuid4())
            now = time.time()
            self._jobs[job_id] = Job(
                id=job_id,
                model_type=model_type,
                model=model,
                application_id=application_id,
                priority=priority,
                payload=payload,
                result_path=result_path,
                status="queued",
                created_at=now,
                last_heartbeat=now,
                queue_seq=self._seq,
            )
            self._seq += 1
            return job_id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                job.last_heartbeat = time.time()
            return job

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return False
            if job.status == "queued":
                job.status = "cancelled"
                return True
            return False

    def pick_next(
        self,
        current_model_key: Optional[tuple[str, str]],
        current_app_id: Optional[str],
    ) -> Optional[Job]:
        with self._lock:
            now = time.time()
            # Expire stale queued jobs
            for job in self._jobs.values():
                if (
                    job.status == "queued"
                    and (now - job.last_heartbeat) > self._heartbeat_timeout
                ):
                    job.status = "expired"

            candidates = [j for j in self._jobs.values() if j.status == "queued"]
            if not candidates:
                return None

            # Sort by (priority ASC, queue_seq ASC)
            candidates.sort(key=lambda j: (j.priority, j.queue_seq))

            top_priority = candidates[0].priority

            # Priority 0 = urgent, no grouping — just pick first
            if top_priority == 0:
                return candidates[0]

            # Filter to top priority band
            band = [j for j in candidates if j.priority == top_priority]

            # Prefer jobs matching current model AND app
            if current_model_key is not None and current_app_id is not None:
                same_model_and_app = [
                    j
                    for j in band
                    if (j.model_type, j.model) == current_model_key
                    and j.application_id == current_app_id
                ]
                if same_model_and_app:
                    return same_model_and_app[0]

            # Prefer jobs matching current app (even if model differs)
            if current_app_id is not None:
                same_app = [j for j in band if j.application_id == current_app_id]
                if same_app:
                    return same_app[0]

            # FIFO tiebreak
            return band[0]

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "running"

    def mark_done(self, job_id: str, result: dict) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "done"
                job.result = result

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job.status = "failed"
                job.error = error

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def has_urgent(self) -> bool:
        with self._lock:
            return any(j.status == "queued" and j.priority == 0 for j in self._jobs.values())
