"""Podcast directory watcher, RSS generation, and podcast job helpers."""

from __future__ import annotations

import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from email.utils import formatdate
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from xml.etree.ElementTree import Element, SubElement, tostring

if TYPE_CHECKING:
    from .queue_manager import QueueManager
    from .worker import Worker

from .queue_manager import URGENT_PRIORITY


# ── Config ────────────────────────────────────────────────────────────────────


@dataclass
class PodcastConfig:
    title: str = "Untitled Podcast"
    description: str = ""
    author: str = ""
    email: str = ""
    image_url: str = ""
    language: str = "en"
    category: str = "Technology"
    voice: Optional[str] = None
    voices_dir: Optional[str] = None
    steps: int = 10
    temperature: float = 1.0
    emotion: float = 1.0
    cfg_rate: float = 0.7
    token_target: int = 250
    top_k: int = 30
    gpt_temperature: float = 0.8
    priority: int = URGENT_PRIORITY  # 0 = preempt non-urgent running job
    direct_narration: bool = True    # False = skip classify-emotions, synth from txt

    @classmethod
    def from_file(cls, config_path: Path) -> "PodcastConfig":
        if not config_path.exists():
            return cls()
        try:
            data = json.loads(config_path.read_text())
        except Exception:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


# ── RSS generation ─────────────────────────────────────────────────────────────


def _duration_str(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    secs = int(seconds)
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _mp3_duration(path: Path) -> float:
    """Return duration in seconds using mutagen, or 0.0 on failure."""
    try:
        from mutagen.mp3 import MP3

        return MP3(str(path)).info.length
    except Exception:
        return 0.0


def _episode_title(stem: str) -> str:
    """Convert a filename stem to a human-readable title."""
    return stem.replace("_", " ").replace("-", " ").title()


def generate_rss(podcast_dir: Path, config: PodcastConfig, public_url: str) -> str:
    """Return an RSS 2.0 XML string for the podcast directory."""
    podcast_name = podcast_dir.name
    feed_url = f"{public_url}/podcasts/{podcast_name}/feed"

    rss = Element(
        "rss",
        {
            "version": "2.0",
            "xmlns:itunes": "http://www.itunes.com/dtds/podcast-1.0.dtd",
        },
    )
    channel = SubElement(rss, "channel")

    def _tag(parent, name, text="", **attrs):
        el = SubElement(parent, name, attrs)
        if text:
            el.text = text
        return el

    _tag(channel, "title", config.title)
    _tag(channel, "link", feed_url)
    _tag(channel, "description", config.description)
    _tag(channel, "language", config.language)
    _tag(channel, "itunes:author", config.author)
    if config.image_url:
        _tag(channel, "itunes:image", href=config.image_url)
    if config.category:
        _tag(channel, "itunes:category", text=config.category)
    if config.email:
        owner = SubElement(channel, "itunes:owner")
        _tag(owner, "itunes:email", config.email)
        if config.author:
            _tag(owner, "itunes:name", config.author)

    chapters_mp3 = podcast_dir / "chapters_mp3"
    mp3_files: list[Path] = []
    if chapters_mp3.is_dir():
        mp3_files = sorted(
            (p for p in chapters_mp3.iterdir() if p.suffix.lower() == ".mp3"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    for mp3 in mp3_files:
        stat = mp3.stat()
        audio_url = f"{public_url}/podcasts/{podcast_name}/audio/{mp3.name}"
        duration = _mp3_duration(mp3)

        item = SubElement(channel, "item")
        _tag(item, "title", _episode_title(mp3.stem))
        _tag(
            item,
            "enclosure",
            url=audio_url,
            length=str(stat.st_size),
            type="audio/mpeg",
        )
        _tag(item, "guid", audio_url)
        _tag(item, "pubDate", formatdate(stat.st_mtime, localtime=False))
        _tag(item, "itunes:duration", _duration_str(duration))

    xml_bytes = tostring(rss, encoding="unicode", xml_declaration=False)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes


# ── Watcher ───────────────────────────────────────────────────────────────────


class PodcastWatcher(threading.Thread):
    """Background thread that polls podcast directories for new episodes."""

    POLL_INTERVAL = 30  # seconds

    def __init__(
        self,
        podcast_dirs: List[Path],
        queue: "QueueManager",
        worker: "Worker",
        server_voices_dir: Optional[str],
        public_url: Optional[str],
    ):
        super().__init__(daemon=True, name="indextts-podcast-watcher")
        self.podcast_dirs = [Path(d) for d in podcast_dirs]
        self.queue = queue
        self.worker = worker
        self.server_voices_dir = server_voices_dir
        self.public_url = public_url
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.is_set():
            try:
                self._check_all()
            except Exception:
                pass
            self._stop_event.wait(self.POLL_INTERVAL)

    def _check_all(self):
        for podcast_dir in self.podcast_dirs:
            if podcast_dir.is_dir():
                try:
                    self._check_dir(podcast_dir)
                except Exception as exc:
                    print(
                        f"[podcast watcher] ERROR in _check_dir({podcast_dir.name}): {exc}",
                        file=sys.stderr,
                        flush=True,
                    )

    def _check_dir(self, podcast_dir: Path):
        chapters_txt = podcast_dir / "chapters_txt"
        chapters_directed = podcast_dir / "chapters_directed"
        chapters_mp3 = podcast_dir / "chapters_mp3"

        if not chapters_txt.is_dir():
            return

        txts = {p.stem for p in chapters_txt.glob("*.txt")}
        if not txts:
            return

        # Read config first so we know whether to expect jsonl files or not.
        config = PodcastConfig.from_file(podcast_dir / "config.json")

        jsonl_stems = (
            {p.stem for p in chapters_directed.glob("*.jsonl")}
            if chapters_directed.is_dir()
            else set()
        )
        mp3_stems = (
            {p.stem for p in chapters_mp3.glob("*.mp3")} if chapters_mp3.is_dir() else set()
        )

        # Pending work depends on whether directing is enabled:
        #   direct_narration=True:  txt→jsonl→mp3 pipeline
        #   direct_narration=False: txt→mp3 directly (no jsonl ever created)
        if config.direct_narration:
            pending = (txts - jsonl_stems) | (jsonl_stems - mp3_stems)
        else:
            pending = txts - mp3_stems
        if not pending:
            return

        podcast_name = podcast_dir.name

        # Skip entirely if a podcast job for this dir is already queued or running
        already_active = any(
            j.get("job_type") == "podcast_episode"
            and j.get("podcast_name") == podcast_name
            and j["status"] in ("queued", "running")
            for j in self.queue.all_jobs()
        )
        if already_active:
            return
        effective_voices_dir = config.voices_dir or self.server_voices_dir

        print(
            f"[podcast watcher] {podcast_name}: new work detected — queueing episode"
            f" (pending={len(pending)}, podcast_priority={config.priority})",
            file=sys.stderr,
            flush=True,
        )

        # If this podcast is urgent (priority 0), interrupt any non-urgent running job
        if config.priority == URGENT_PRIORITY:
            active = self.queue.active_job()
            if active:
                active_pri = active.get("priority", 10)
                if active_pri != URGENT_PRIORITY:
                    print(
                        f"[podcast watcher] interrupting running job"
                        f" '{active.get('isbn', active.get('id', '?'))}'"
                        f" (priority={active_pri}) for podcast episode",
                        file=sys.stderr,
                        flush=True,
                    )
                    self.worker.interrupt_current_for_priority_job(active)
                else:
                    print(
                        f"[podcast watcher] running job"
                        f" '{active.get('isbn', active.get('id', '?'))}'"
                        f" is already priority {active_pri} — not interrupting",
                        file=sys.stderr,
                        flush=True,
                    )

        episode = {
            "id": str(uuid.uuid4()),
            "job_type": "podcast_episode",
            "isbn": f"Podcast: {podcast_name}",
            "podcast_dir": str(podcast_dir),
            "podcast_name": podcast_name,
            "voice": config.voice,
            "voices_dir": effective_voices_dir,
            "steps": config.steps,
            "temperature": config.temperature,
            "emotion": config.emotion,
            "cfg_rate": config.cfg_rate,
            "token_target": config.token_target,
            "top_k": config.top_k,
            "gpt_temperature": config.gpt_temperature,
            "priority": config.priority,
            "direct_narration": config.direct_narration,
            "status": "queued",
            "stage": None,
            "created_at": _now(),
            "started_at": None,
            "finished_at": None,
            "error": None,
        }
        self.queue.enqueue_podcast_episode(episode)


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
