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

# Priority values (lower number = higher priority).
# URGENT_PRIORITY (0): preempts any non-urgent running job.
# DEFAULT_PRIORITY (10): used for all regular audiobook submissions.
# Podcast configs may override via their own priority field (default 0).
URGENT_PRIORITY = 0
DEFAULT_PRIORITY = 10

# Map previously-persisted priority values to the current scheme.
# History: old descending scheme (10/5/0) → ascending tiers (1/2/3) → current (0/9/10).
#
# IMPORTANT: 0 and 10 are omitted deliberately — they are valid in the current
# scheme (URGENT_PRIORITY=0, DEFAULT_PRIORITY=10) and must not be remapped.
# Only the previous ascending-tier values (1/2/3) and the old intermediate
# "interrupted" value (5) are unambiguously legacy.
_LEGACY_PRIORITY_MAP = {
    5: DEFAULT_PRIORITY - 1,  # old descending "interrupted" → current interrupted (9)
    1: URGENT_PRIORITY,       # previous ascending "podcast" tier → urgent (0)
    2: DEFAULT_PRIORITY - 1,  # previous ascending "interrupted" tier → interrupted (9)
    3: DEFAULT_PRIORITY,      # previous ascending "normal" tier → normal (10)
}


class QueueManager:
    def __init__(self, audiobooks_dir: Path):
        self.audiobooks_dir = Path(audiobooks_dir)
        self._lock = threading.Lock()
        self._queue_file = self.audiobooks_dir / ".queue.json"
        self._jobs: list[dict] = []
        self._next_seq: int = 0
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load(self):
        if self._queue_file.exists():
            try:
                data = json.loads(self._queue_file.read_text())
                self._jobs = data.get("jobs", [])
                # Any job that was "running" when the server died is re-queued
                # from the stage that can be detected from its directory contents,
                # so it will be picked up again before any already-queued jobs.
                for job in self._jobs:
                    # Migrate legacy priority values to the new tier constants.
                    # Old scheme: higher number = higher priority (10/5/0).
                    # New scheme: lower number = higher priority (1/2/3).
                    old_pri = job.get("priority")
                    if old_pri in _LEGACY_PRIORITY_MAP:
                        job["priority"] = _LEGACY_PRIORITY_MAP[old_pri]
                    elif old_pri is None:
                        job["priority"] = DEFAULT_PRIORITY

                    # Backfill queue_seq for jobs from older queue files.
                    if "queue_seq" not in job:
                        job["queue_seq"] = self._next_seq
                        self._next_seq += 1

                    if job["status"] == RUNNING:
                        job_dir = self.audiobooks_dir / (job.get("dir_name") or "")
                        start_stage, _ = _detect_stage(job_dir)
                        if start_stage is not None:
                            job["status"] = QUEUED
                            job["start_stage"] = start_stage
                            job["error"] = None
                            job["started_at"] = None
                            job["finished_at"] = None
                        else:
                            # Nothing actionable on disk (e.g. killed before writing
                            # anything); mark interrupted so user can decide.
                            job["status"] = INTERRUPTED
                            job["error"] = "Server restarted while job was running"
                            job["finished_at"] = _now()
                # Ensure _next_seq is above every existing seq so new jobs
                # always get a unique, higher sequence number.
                if self._jobs:
                    self._next_seq = max(
                        j.get("queue_seq", 0) for j in self._jobs
                    ) + 1
                self._save_locked()
            except Exception:
                self._jobs = []
        else:
            self.audiobooks_dir.mkdir(parents=True, exist_ok=True)
            self._jobs = []

    def _alloc_seq(self) -> int:
        """Return the next unique queue sequence number. Call under self._lock."""
        seq = self._next_seq
        self._next_seq += 1
        return seq

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
        direct_narration: bool = False,
        priority: int = DEFAULT_PRIORITY,
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
            "direct_narration": direct_narration,
            "priority": priority,
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
            job["queue_seq"] = self._alloc_seq()
            self._jobs.append(job)
            self._save_locked()
        return job

    def get_next_queued(self) -> Optional[dict]:
        with self._lock:
            queued = [j for j in self._jobs if j["status"] == QUEUED]
            if not queued:
                return None
            # Sort by ascending priority tier, then ascending queue_seq (FIFO).
            # Tier 1 (podcast) → tier 2 (interrupted) → tier 3 (normal).
            queued.sort(key=lambda j: (j.get("priority", DEFAULT_PRIORITY), j.get("queue_seq", 0)))
            return dict(queued[0])

    def enqueue_podcast_episode(self, episode: dict) -> dict:
        """Insert a pre-built podcast episode job dict into the queue."""
        with self._lock:
            episode["queue_seq"] = self._alloc_seq()
            self._jobs.append(episode)
            self._save_locked()
        return episode

    def requeue_interrupted(self, job_id: str, start_stage: str, priority: int) -> None:
        """Reset a running job back to QUEUED with the given priority."""
        with self._lock:
            for job in self._jobs:
                if job["id"] == job_id:
                    job["status"] = QUEUED
                    job["start_stage"] = start_stage
                    job["priority"] = priority
                    job["started_at"] = None
                    job["stage"] = None
                    job["error"] = None
                    self._save_locked()
                    return

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

    def resume(
        self,
        dir_name: str,
        isbn: str,
        voice: str | None,
        start_stage: str,
        steps: int = 10,
        temperature: float = 1.0,
        emotion: float = 1.0,
        cfg_rate: float = 0.7,
        token_target: int = 250,
        direct_narration: bool = False,
        priority: int = DEFAULT_PRIORITY,
    ) -> dict:
        """Queue a job that resumes from an existing directory at start_stage."""
        job: dict = {
            "id": str(uuid.uuid4()),
            "isbn": isbn,
            "epub_url": None,  # no download needed
            "voice": voice,
            "steps": steps,
            "temperature": temperature,
            "emotion": emotion,
            "cfg_rate": cfg_rate,
            "token_target": token_target,
            "extra_opts": {},
            "direct_narration": direct_narration,
            "priority": priority,
            "status": QUEUED,
            "stage": None,
            "start_stage": start_stage,  # worker skips earlier stages
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
            job["queue_seq"] = self._alloc_seq()
            self._jobs.append(job)
            self._save_locked()
        return job

    def _queued_sorted(self) -> list[dict]:
        """Return queued jobs in execution order. Must be called under self._lock."""
        queued = [j for j in self._jobs if j["status"] == QUEUED]
        queued.sort(key=lambda j: (j.get("priority", DEFAULT_PRIORITY), j.get("queue_seq", 0)))
        return queued

    def move_up(self, job_id: str) -> bool:
        """Move a queued job one position earlier in the execution order.

        Swaps the sort keys (priority + queue_seq) with the preceding queued
        job.  Returns True if the swap was made, False if the job was not found
        or is already first.
        """
        with self._lock:
            ordered = self._queued_sorted()
            idx = next((i for i, j in enumerate(ordered) if j["id"] == job_id), None)
            if idx is None or idx == 0:
                return False
            a, b = ordered[idx - 1], ordered[idx]
            a["priority"], b["priority"] = b["priority"], a["priority"]
            a["queue_seq"], b["queue_seq"] = b["queue_seq"], a["queue_seq"]
            self._save_locked()
            return True

    def move_down(self, job_id: str) -> bool:
        """Move a queued job one position later in the execution order.

        Swaps the sort keys (priority + queue_seq) with the following queued
        job.  Returns True if the swap was made, False if the job was not found
        or is already last.
        """
        with self._lock:
            ordered = self._queued_sorted()
            idx = next((i for i, j in enumerate(ordered) if j["id"] == job_id), None)
            if idx is None or idx >= len(ordered) - 1:
                return False
            a, b = ordered[idx], ordered[idx + 1]
            a["priority"], b["priority"] = b["priority"], a["priority"]
            a["queue_seq"], b["queue_seq"] = b["queue_seq"], a["queue_seq"]
            self._save_locked()
            return True

    def scan_dirs(self) -> list[dict]:
        """Scan audiobooks_dir for subdirs not already in the queue.

        Returns a list of dicts describing each detected directory and what
        stage the worker would resume from.
        """
        # Collect dir_names already tracked in the queue
        # Only dirs with an active (queued/running) job are excluded.
        # Interrupted and failed dirs should reappear so the user can resume them.
        with self._lock:
            tracked = {
                j["dir_name"]
                for j in self._jobs
                if j["status"] in (QUEUED, RUNNING) and j.get("dir_name")
            }

        results = []
        for subdir in sorted(self.audiobooks_dir.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith("."):
                continue
            dir_name = subdir.name
            if dir_name in tracked:
                continue

            start_stage, description = _detect_stage(subdir)
            if start_stage is None:
                continue  # nothing actionable here

            # Read metadata.json if present to surface isbn/title/author/settings
            meta = _read_metadata(subdir)

            results.append(
                {
                    "dir_name": dir_name,
                    "start_stage": start_stage,
                    "description": description,
                    "isbn": meta.get("isbn"),
                    "title": meta.get("title"),
                    "author": meta.get("author"),
                    "voice": meta.get("voice"),
                    "steps": meta.get("steps"),
                    "emotion": meta.get("emotion"),
                    "token_target": meta.get("token_target"),
                    "direct_narration": bool(meta.get("direct_narration", False)),
                }
            )
        return results


def _read_metadata(job_dir: Path) -> dict:
    """Read metadata.json from a job directory, returning {} on any error."""
    meta_path = job_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception:
        return {}


def _detect_stage(job_dir: Path) -> tuple[str | None, str]:
    """Inspect a job directory and return (start_stage, human_description).

    Returns (None, '') when the directory has nothing actionable (empty, or
    already finished).
    """
    # Already packaged?
    m4b_files = list(job_dir.glob("*.m4b"))
    if m4b_files:
        return None, "already done"

    chapters_txt = job_dir / "chapters_txt"
    chapters_mp3 = job_dir / "chapters_mp3"
    chapters_directed = job_dir / "chapters_directed"

    txt_files = sorted(chapters_txt.glob("*.txt")) if chapters_txt.is_dir() else []
    jsonl_files = sorted(chapters_directed.glob("*.jsonl")) if chapters_directed.is_dir() else []
    mp3_files = sorted(chapters_mp3.glob("*.mp3")) if chapters_mp3.is_dir() else []

    n_txt = len(txt_files)
    n_jsonl = len(jsonl_files)
    n_mp3 = len(mp3_files)

    # chapters_txt is always the ground truth: one .txt per chapter, written
    # atomically by the extract step before any other stage runs.
    total_chapters = n_txt

    directing_complete = n_txt > 0 and n_jsonl >= n_txt
    # Synthesize input is directed jsonl if directing is complete, else txt
    synth_total = n_jsonl if directing_complete else n_txt

    # Have mp3s — check if synthesis is complete or still in progress
    if n_mp3 and synth_total:
        if n_mp3 >= synth_total:
            return "packaging", f"{n_mp3} mp3s ready, packaging M4B"
        else:
            return (
                "synthesizing",
                f"{n_mp3}/{synth_total} chapters synthesized, resuming",
            )

    # Directing was started but not finished → resume directing
    if n_jsonl and not directing_complete:
        return (
            "directing",
            f"{n_jsonl}/{n_txt} chapters directed, resuming",
        )

    # Directing is fully complete → synthesize from directed output
    if directing_complete:
        return "synthesizing", f"{n_jsonl} directed chapters ready to synthesize"

    # Have txt but no directed/mp3 → start at directing stage so the worker
    # can decide whether to run classify-emotions (based on direct_narration flag)
    # or skip straight to synthesizing.
    if txt_files:
        return "directing", f"{n_txt} chapters extracted, ready to direct or synthesize"

    # Have an epub but no txt → extract first
    epub_files = list(job_dir.glob("*.epub"))
    if epub_files:
        return "extracting", f"EPUB present ({epub_files[0].name}), ready to extract"

    return None, "empty"
