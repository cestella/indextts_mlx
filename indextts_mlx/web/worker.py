"""Background worker thread: pulls jobs from the queue and runs the pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .queue_manager import QueueManager

from .queue_manager import CANCELLED, DONE, FAILED, RUNNING, _now


def _indextts_cmd() -> str:
    """Find the indextts executable in the same venv as us."""
    candidate = Path(sys.executable).parent / "indextts"
    if candidate.exists():
        return str(candidate)
    found = shutil.which("indextts")
    return found or "indextts"


class Worker(threading.Thread):
    def __init__(
        self,
        queue: "QueueManager",
        audiobooks_dir: Path,
        voices_dir: str | None,
        default_voice: str | None,
    ):
        super().__init__(daemon=True, name="indextts-worker")
        self.queue = queue
        self.audiobooks_dir = Path(audiobooks_dir)
        self.voices_dir = voices_dir
        self.default_voice = default_voice
        self._stop_event = threading.Event()
        self._cancel_event = threading.Event()
        self._current_proc: subprocess.Popen | None = None
        self._proc_lock = threading.Lock()
        self._current_job_id: str | None = None

    def stop(self):
        self._stop_event.set()
        self.cancel_current()

    def cancel_current(self):
        self._cancel_event.set()
        with self._proc_lock:
            if self._current_proc and self._current_proc.poll() is None:
                try:
                    self._current_proc.terminate()
                    time.sleep(0.5)
                    if self._current_proc.poll() is None:
                        self._current_proc.kill()
                except Exception:
                    pass

    def request_cancel(self, job_id: str) -> bool:
        if self._current_job_id == job_id:
            self.cancel_current()
            return True
        return False

    def run(self):
        cmd = _indextts_cmd()
        while not self._stop_event.is_set():
            job = self.queue.get_next_queued()
            if job is None:
                time.sleep(2)
                continue

            self._cancel_event.clear()
            self._current_job_id = job["id"]
            self.queue.update(job["id"], status=RUNNING, started_at=_now(), stage="starting")

            try:
                self._run_job(cmd, job)
            except Exception as exc:
                self.queue.update(
                    job["id"],
                    status=FAILED,
                    error=str(exc),
                    finished_at=_now(),
                    stage=None,
                )
            finally:
                self._current_job_id = None
                self._current_proc = None

    # ── pipeline stages ───────────────────────────────────────────────────────

    def _run_job(self, cmd: str, job: dict):
        jid = job["id"]
        isbn = job["isbn"]
        dir_name = job["dir_name"]
        job_dir = self.audiobooks_dir / dir_name
        job_dir.mkdir(parents=True, exist_ok=True)

        epub_path = job_dir / f"{dir_name}.epub"
        chapters_txt = job_dir / "chapters_txt"
        chapters_mp3 = job_dir / "chapters_mp3"
        chapters_txt.mkdir(exist_ok=True)
        chapters_mp3.mkdir(exist_ok=True)
        status_dir = job_dir / ".status"
        status_dir.mkdir(exist_ok=True)

        # Write metadata.json so future scans can recover isbn and settings
        # without needing the queue entry.  Merge with any existing metadata
        # (a previous run may have populated title/author via ebooklib).
        metadata_path = job_dir / "metadata.json"
        existing_meta = {}
        if metadata_path.exists():
            try:
                existing_meta = json.loads(metadata_path.read_text())
            except Exception:
                pass
        meta = {
            **existing_meta,
            "isbn": isbn,
            "voice": job.get("voice"),
            "steps": job.get("steps", 10),
            "temperature": job.get("temperature", 1.0),
            "emotion": job.get("emotion", 1.0),
            "cfg_rate": job.get("cfg_rate", 0.7),
            "token_target": job.get("token_target", 50),
            "epub_url": job.get("epub_url"),
        }
        tmp_meta = metadata_path.with_suffix(".json.tmp")
        tmp_meta.write_text(json.dumps(meta, indent=2))
        tmp_meta.replace(metadata_path)

        # start_stage controls which pipeline stages are skipped.
        # A resume job sets this to "extracting", "directing", "synthesizing", or "packaging".
        # A normal job has no start_stage (or "downloading").
        _STAGE_ORDER = ["downloading", "extracting", "directing", "synthesizing", "packaging"]
        start_stage = job.get("start_stage") or "downloading"
        def _should_skip(stage: str) -> bool:
            try:
                return _STAGE_ORDER.index(stage) < _STAGE_ORDER.index(start_stage)
            except ValueError:
                return False

        # ── 1. Download epub ─────────────────────────────────────────────────
        if not _should_skip("downloading"):
            if self._is_cancelled(jid):
                return
            self.queue.update(
                jid, stage="downloading", epub_path=str(epub_path.relative_to(self.audiobooks_dir))
            )
            self._run_proc(
                jid,
                ["wget", "-q", "-O", str(epub_path), job["epub_url"]],
                stage="downloading",
            )
            if not epub_path.exists() or epub_path.stat().st_size == 0:
                raise RuntimeError(f"Download failed or file is empty: {epub_path}")

        # ── 2. Extract epub chapters → txt ───────────────────────────────────
        if not _should_skip("extracting"):
            if self._is_cancelled(jid):
                return
            self.queue.update(jid, stage="extracting")
            self._run_proc(
                jid,
                [cmd, "extract", str(epub_path), str(chapters_txt)],
                stage="extracting",
            )

        # Best-effort: read title/author from epub if present
        if epub_path.exists():
            self._try_read_title(jid, epub_path)

        # ── 3. Direct narration (classify emotions + pauses) ─────────────────
        direct_narration = job.get("direct_narration", False)
        chapters_directed = job_dir / "chapters_directed"
        if direct_narration and not _should_skip("directing"):
            if self._is_cancelled(jid):
                return
            # Delete stale classify_status.json so UI doesn't show old progress
            stale_cls = status_dir / "classify_status.json"
            if stale_cls.exists():
                stale_cls.unlink(missing_ok=True)
            chapters_directed.mkdir(exist_ok=True)
            self.queue.update(jid, stage="directing")
            direct_cmd = [
                cmd,
                "classify-emotions",
                str(chapters_txt),
                str(chapters_directed),
                "--status",
                str(status_dir),
            ]
            self._run_proc(jid, direct_cmd, stage="directing")

        # ── 3. Synthesize chapters → mp3 ─────────────────────────────────────
        if not _should_skip("synthesizing"):
            if self._is_cancelled(jid):
                return
            # Clear any stale synth_status.json from a previous run so the UI
            # doesn't show outdated progress while the new run starts up.
            stale = status_dir / "synth_status.json"
            if stale.exists():
                stale.unlink(missing_ok=True)
            self.queue.update(jid, stage="synthesizing")
            # Use directed chapters if available (have .jsonl files), else plain txt
            directed_files = list(chapters_directed.glob("*.jsonl")) if chapters_directed.is_dir() else []
            synth_input = str(chapters_directed) if directed_files else str(chapters_txt)
            synth_cmd = [
                cmd,
                "synthesize",
                "--file",
                synth_input,
                "--out",
                str(chapters_mp3),
                "--out-ext",
                "mp3",
                "--status",
                str(status_dir),
                "--steps",
                str(job["steps"]),
                "--temperature",
                str(job["temperature"]),
                "--emotion",
                str(job["emotion"]),
                "--cfg-rate",
                str(job["cfg_rate"]),
                "--token-target",
                str(job["token_target"]),
            ]
            voice = job.get("voice") or self.default_voice
            if self.voices_dir:
                synth_cmd += ["--voices-dir", self.voices_dir]
                if voice:
                    synth_cmd += ["--voice", voice]
            elif voice:
                synth_cmd += ["--spk-audio-prompt", voice]
            self._run_proc(jid, synth_cmd, stage="synthesizing")

        # ── 4. Package m4b ───────────────────────────────────────────────────
        if not _should_skip("packaging"):
            if self._is_cancelled(jid):
                return
            self.queue.update(jid, stage="packaging")
            self._run_proc(
                jid,
                [
                    cmd,
                    "m4b",
                    "--chapters-dir",
                    str(chapters_mp3),
                    "--out",
                    str(job_dir),
                    "--isbn",
                    isbn,
                ],
                stage="packaging",
            )

        # Find the produced m4b
        m4b_files = list(job_dir.glob("*.m4b"))
        m4b_rel = str(m4b_files[0].relative_to(self.audiobooks_dir)) if m4b_files else None

        self.queue.update(
            jid,
            status=DONE,
            stage=None,
            finished_at=_now(),
            m4b_path=m4b_rel,
        )

    def _run_proc(self, job_id: str, args: list[str], stage: str):
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with self._proc_lock:
            self._current_proc = proc

        log_lines: list[str] = []
        for line in proc.stdout:
            line = line.rstrip()
            log_lines.append(line)
            # Keep only last 200 lines to avoid unbounded memory use
            if len(log_lines) > 200:
                log_lines.pop(0)
            if self._cancel_event.is_set():
                proc.terminate()
                break

        proc.wait()
        with self._proc_lock:
            self._current_proc = None

        if self._cancel_event.is_set():
            self.queue.update(job_id, status=CANCELLED, finished_at=_now(), stage=None)
            return

        if proc.returncode != 0:
            tail = "\n".join(log_lines[-20:])
            raise RuntimeError(f"Step '{stage}' failed (exit {proc.returncode}):\n{tail}")

    def _is_cancelled(self, job_id: str) -> bool:
        if self._cancel_event.is_set():
            self.queue.update(job_id, status=CANCELLED, finished_at=_now(), stage=None)
            return True
        job = self.queue.get_job(job_id)
        if job and job["status"] == CANCELLED:
            return True
        return False

    def _try_read_title(self, job_id: str, epub_path: Path):
        """Best-effort: extract title/author from epub metadata."""
        try:
            import ebooklib
            from ebooklib import epub as _epub

            book = _epub.read_epub(str(epub_path))
            title_meta = book.get_metadata("DC", "title")
            author_meta = book.get_metadata("DC", "creator")
            title = title_meta[0][0] if title_meta else None
            author = author_meta[0][0] if author_meta else None
            self.queue.update(job_id, title=title, author=author)
            # Persist to metadata.json so scans after restart can show the title
            meta_path = epub_path.parent / "metadata.json"
            try:
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                if title:
                    meta["title"] = title
                if author:
                    meta["author"] = author
                tmp = meta_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(meta, indent=2))
                tmp.replace(meta_path)
            except Exception:
                pass
        except Exception:
            pass
