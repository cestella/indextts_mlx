"""Persistent job queue backed by a JSON file. Thread-safe."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_isbn(isbn: str) -> str:
    """Strip hyphens/spaces → all digits, lowercase."""
    return isbn.replace("-", "").replace(" ", "").lower()


# Job status values
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"
INTERRUPTED = "interrupted"


class QueueManager:
    def __init__(self, audiobooks_dir: Path):
        self.audiobooks_dir = Path(audiobooks_dir)
        self._lock = threading.Lock()
        self._queue_file = self.audiobooks_dir / ".queue.json"
        self._jobs: list[dict] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if self._queue_file.exists():
            try:
                data = json.loads(self._queue_file.read_text())
                self._jobs = data.get("jobs", [])
                # Any job that was "running" when server died gets marked interrupted
                for job in self._jobs:
                    if job["status"] == RUNNING:
                        job["status"] = INTERRUPTED
                        job["error"] = "Server restarted while job was running"
                        job["finished_at"] = _now()
                self._save_locked()
            except Exception:
                self._jobs = []
        else:
            self.audiobooks_dir.mkdir(parents=True, exist_ok=True)
            self._jobs = []

    def _save_locked(self):
        """Write queue file; must be called while holding self._lock."""
        tmp = self._queue_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"jobs": self._jobs}, indent=2))
        tmp.replace(self._queue_file)

    def _save(self):
        with self._lock:
            self._save_locked()

    # ── public API ────────────────────────────────────────────────────────────

    def submit(
        self,
        isbn: str,
        epub_url: str,
        voice: str,
        steps: int = 10,
        temperature: float = 1.0,
        emotion: float = 1.0,
        cfg_rate: float = 0.7,
        token_target: int = 250,
        extra_opts: dict | None = None,
    ) -> dict:
        dir_name = normalize_isbn(isbn)
        job: dict = {
            "id": str(uuid.uuid4()),
            "isbn": isbn,
            "epub_url": epub_url,
            "voice": voice,
            "steps": steps,
            "temperature": temperature,
            "emotion": emotion,
            "cfg_rate": cfg_rate,
            "token_target": token_target,
            "extra_opts": extra_opts or {},
            "status": QUEUED,
            "stage": None,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
            "dir_name": dir_name,
            "title": None,
            "author": None,
            "m4b_path": None,
            "epub_path": None,
        }
        with self._lock:
            self._jobs.append(job)
            self._save_locked()
        return job

    def get_next_queued(self) -> Optional[dict]:
        with self._lock:
            for job in self._jobs:
                if job["status"] == QUEUED:
                    return dict(job)
        return None

    def update(self, job_id: str, **kwargs) -> None:
        with self._lock:
            for job in self._jobs:
                if job["id"] == job_id:
                    job.update(kwargs)
                    self._save_locked()
                    return

    def cancel(self, job_id: str) -> str:
        """Returns the previous status, or 'not_found'."""
        with self._lock:
            for job in self._jobs:
                if job["id"] == job_id:
                    prev = job["status"]
                    if prev in (QUEUED, RUNNING, INTERRUPTED):
                        job["status"] = CANCELLED
                        job["finished_at"] = _now()
                        self._save_locked()
                    return prev
        return "not_found"

    def all_jobs(self) -> list[dict]:
        with self._lock:
            return [dict(j) for j in self._jobs]

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._lock:
            for job in self._jobs:
                if job["id"] == job_id:
                    return dict(job)
        return None

    def active_job(self) -> Optional[dict]:
        with self._lock:
            for job in self._jobs:
                if job["status"] == RUNNING:
                    return dict(job)
        return None
