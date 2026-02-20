"""Tests for indextts_mlx.web.podcast.

Covers:
  - PodcastConfig: dataclass defaults, from_file (valid, missing, corrupt)
  - generate_rss: XML structure, episode ordering, empty channel
  - PodcastWatcher._check_dir:
      * skips when no chapters_txt dir
      * skips when no txt files
      * skips when all work is already done
      * skips when a podcast episode is already queued/running for this podcast
      * enqueues episode when pending work exists and no active job
      * calls interrupt_current_for_priority_job when a non-urgent job is running
      * does NOT call interrupt when the running job is already priority-0
      * uses txt-only pending logic when direct_narration=False
      * errors in _check_dir are logged to stderr (not silently swallowed)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, call, patch

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_podcast_dir(base: Path, name: str = "my_podcast") -> Path:
    d = base / name
    (d / "chapters_txt").mkdir(parents=True, exist_ok=True)
    return d


def _add_txt(podcast_dir: Path, stem: str, text: str = "Hello.") -> Path:
    p = podcast_dir / "chapters_txt" / f"{stem}.txt"
    p.write_text(text)
    return p


def _add_jsonl(podcast_dir: Path, stem: str) -> Path:
    d = podcast_dir / "chapters_directed"
    d.mkdir(exist_ok=True)
    p = d / f"{stem}.jsonl"
    p.write_text(json.dumps({"text": "Hello.", "emotion": "neutral"}))
    return p


def _add_mp3(podcast_dir: Path, stem: str) -> Path:
    d = podcast_dir / "chapters_mp3"
    d.mkdir(exist_ok=True)
    p = d / f"{stem}.mp3"
    p.write_bytes(b"\xff\xfb" + b"\x00" * 100)  # minimal fake mp3 header
    return p


def _make_queue(tmp_ab_dir: Path):
    from indextts_mlx.web.queue_manager import QueueManager
    return QueueManager(tmp_ab_dir)


def _make_watcher(podcast_dirs, queue, worker, server_voices_dir=None, public_url=None):
    from indextts_mlx.web.podcast import PodcastWatcher
    return PodcastWatcher(
        podcast_dirs=podcast_dirs,
        queue=queue,
        worker=worker,
        server_voices_dir=server_voices_dir,
        public_url=public_url,
    )


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_ab_dir(tmp_path):
    d = tmp_path / "audiobooks"
    d.mkdir()
    return d


@pytest.fixture()
def queue(tmp_ab_dir):
    return _make_queue(tmp_ab_dir)


@pytest.fixture()
def mock_worker():
    return MagicMock()


# ── PodcastConfig ─────────────────────────────────────────────────────────────


class TestPodcastConfig:
    def test_defaults(self):
        from indextts_mlx.web.podcast import PodcastConfig
        from indextts_mlx.web.queue_manager import URGENT_PRIORITY

        cfg = PodcastConfig()
        assert cfg.title == "Untitled Podcast"
        assert cfg.priority == URGENT_PRIORITY
        assert cfg.direct_narration is True
        assert cfg.steps == 10
        assert cfg.voice is None

    def test_from_file_missing_returns_defaults(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig

        cfg = PodcastConfig.from_file(tmp_path / "nonexistent.json")
        assert cfg.title == "Untitled Podcast"

    def test_from_file_corrupt_json_returns_defaults(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig

        bad = tmp_path / "config.json"
        bad.write_text("not valid json {{{{")
        cfg = PodcastConfig.from_file(bad)
        assert cfg.title == "Untitled Podcast"

    def test_from_file_reads_known_fields(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig

        data = {
            "title": "My Show",
            "description": "A great show",
            "voice": "british_female",
            "steps": 15,
            "priority": 5,
            "direct_narration": False,
        }
        (tmp_path / "config.json").write_text(json.dumps(data))
        cfg = PodcastConfig.from_file(tmp_path / "config.json")
        assert cfg.title == "My Show"
        assert cfg.description == "A great show"
        assert cfg.voice == "british_female"
        assert cfg.steps == 15
        assert cfg.priority == 5
        assert cfg.direct_narration is False

    def test_from_file_ignores_unknown_fields(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig

        data = {"title": "Test", "unknown_future_field": 42}
        (tmp_path / "config.json").write_text(json.dumps(data))
        cfg = PodcastConfig.from_file(tmp_path / "config.json")
        assert cfg.title == "Test"


# ── generate_rss ──────────────────────────────────────────────────────────────


class TestGenerateRss:
    def test_returns_xml_string(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig, generate_rss

        d = _make_podcast_dir(tmp_path, "myshow")
        cfg = PodcastConfig(title="My Show")
        xml = generate_rss(d, cfg, "http://localhost:5000")
        assert xml.startswith("<?xml")
        assert "<rss" in xml

    def test_channel_title_in_output(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig, generate_rss

        d = _make_podcast_dir(tmp_path, "myshow")
        cfg = PodcastConfig(title="Daily Update")
        xml = generate_rss(d, cfg, "http://localhost:5000")
        assert "Daily Update" in xml

    def test_empty_mp3_dir_produces_no_items(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig, generate_rss

        d = _make_podcast_dir(tmp_path, "myshow")
        cfg = PodcastConfig()
        xml = generate_rss(d, cfg, "http://localhost:5000")
        assert "<item>" not in xml

    def test_mp3_files_produce_items(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig, generate_rss

        d = _make_podcast_dir(tmp_path, "myshow")
        _add_mp3(d, "ep01")
        _add_mp3(d, "ep02")
        cfg = PodcastConfig()
        xml = generate_rss(d, cfg, "http://localhost:5000")
        assert xml.count("<item>") == 2

    def test_audio_url_contains_public_url_and_filename(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig, generate_rss

        d = _make_podcast_dir(tmp_path, "myshow")
        _add_mp3(d, "episode_01")
        cfg = PodcastConfig()
        xml = generate_rss(d, cfg, "http://myhost:5000")
        assert "http://myhost:5000/podcasts/myshow/audio/episode_01.mp3" in xml

    def test_episode_title_derived_from_stem(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig, generate_rss

        d = _make_podcast_dir(tmp_path, "myshow")
        _add_mp3(d, "the_daily_show")
        cfg = PodcastConfig()
        xml = generate_rss(d, cfg, "http://localhost:5000")
        assert "The Daily Show" in xml

    def test_itunes_author_present_when_set(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig, generate_rss

        d = _make_podcast_dir(tmp_path, "myshow")
        cfg = PodcastConfig(author="Jane Smith")
        xml = generate_rss(d, cfg, "http://localhost:5000")
        assert "Jane Smith" in xml

    def test_no_image_tag_when_image_url_empty(self, tmp_path):
        from indextts_mlx.web.podcast import PodcastConfig, generate_rss

        d = _make_podcast_dir(tmp_path, "myshow")
        cfg = PodcastConfig(image_url="")
        xml = generate_rss(d, cfg, "http://localhost:5000")
        assert "itunes:image" not in xml


# ── PodcastWatcher._check_dir ─────────────────────────────────────────────────


class TestPodcastWatcherCheckDir:
    """Tests for the core _check_dir logic (called synchronously for simplicity)."""

    def _run_check(self, watcher, podcast_dir):
        """Call _check_dir directly without starting the watcher thread."""
        watcher._check_dir(podcast_dir)

    # ── skip conditions ──────────────────────────────────────────────────────

    def test_skips_when_no_chapters_txt_dir(self, tmp_path, queue, mock_worker):
        podcast_dir = tmp_path / "mypodcast"
        podcast_dir.mkdir()
        watcher = _make_watcher([podcast_dir], queue, mock_worker)
        self._run_check(watcher, podcast_dir)
        assert queue.all_jobs() == []

    def test_skips_when_chapters_txt_is_empty(self, tmp_path, queue, mock_worker):
        d = _make_podcast_dir(tmp_path, "mypodcast")
        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)
        assert queue.all_jobs() == []

    def test_skips_when_all_work_done_direct_narration_true(self, tmp_path, queue, mock_worker):
        """If every txt has a jsonl AND every jsonl has an mp3, nothing pending."""
        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")
        _add_jsonl(d, "ep01")
        _add_mp3(d, "ep01")
        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)
        assert queue.all_jobs() == []

    def test_skips_when_all_work_done_direct_narration_false(self, tmp_path, queue, mock_worker):
        """With direct_narration=False, done = every txt has an mp3."""
        d = _make_podcast_dir(tmp_path, "mypodcast")
        (d / "config.json").write_text(json.dumps({"direct_narration": False}))
        _add_txt(d, "ep01")
        _add_mp3(d, "ep01")
        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)
        assert queue.all_jobs() == []

    def test_skips_when_episode_already_queued(self, tmp_path, queue, mock_worker):
        """Second poll: if the episode we queued on the first poll is still queued, don't add more."""
        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")
        watcher = _make_watcher([d], queue, mock_worker)

        # First poll — queues an episode
        self._run_check(watcher, d)
        assert len(queue.all_jobs()) == 1

        # Second poll — episode is still queued; must not add another
        self._run_check(watcher, d)
        assert len(queue.all_jobs()) == 1

    def test_skips_when_episode_running(self, tmp_path, queue, mock_worker):
        """If a podcast episode for this podcast is currently running, don't add more."""
        from indextts_mlx.web.queue_manager import RUNNING

        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")
        watcher = _make_watcher([d], queue, mock_worker)

        # Manually put a running podcast episode in the queue
        episode = {
            "id": "ep-running-1",
            "job_type": "podcast_episode",
            "isbn": "Podcast: mypodcast",
            "podcast_dir": str(d),
            "podcast_name": "mypodcast",
            "priority": 0,
            "status": RUNNING,
            "stage": "synthesizing",
            "created_at": "2024-01-01T00:00:00Z",
            "started_at": "2024-01-01T00:01:00Z",
            "finished_at": None,
            "error": None,
        }
        queue.enqueue_podcast_episode(episode)
        queue.update("ep-running-1", status=RUNNING)

        self._run_check(watcher, d)
        # Still only the one pre-existing episode
        assert len(queue.all_jobs()) == 1

    # ── enqueueing ────────────────────────────────────────────────────────────

    def test_enqueues_episode_when_pending_txt_no_jsonl(self, tmp_path, queue, mock_worker):
        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")
        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)
        jobs = queue.all_jobs()
        assert len(jobs) == 1
        assert jobs[0]["job_type"] == "podcast_episode"
        assert jobs[0]["podcast_name"] == "mypodcast"
        assert jobs[0]["status"] == "queued"

    def test_enqueues_episode_when_jsonl_exists_but_no_mp3(self, tmp_path, queue, mock_worker):
        """directed jsonl without mp3 = pending synthesize work."""
        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")
        _add_jsonl(d, "ep01")
        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)
        assert len(queue.all_jobs()) == 1

    def test_enqueues_episode_direct_narration_false_txt_without_mp3(
        self, tmp_path, queue, mock_worker
    ):
        d = _make_podcast_dir(tmp_path, "mypodcast")
        (d / "config.json").write_text(json.dumps({"direct_narration": False}))
        _add_txt(d, "ep01")
        # No mp3 yet, but also no jsonl (direct_narration=False never creates jsonl)
        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)
        assert len(queue.all_jobs()) == 1
        assert queue.all_jobs()[0]["direct_narration"] is False

    def test_episode_dict_contains_required_fields(self, tmp_path, queue, mock_worker):
        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")
        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)
        job = queue.all_jobs()[0]
        for field in ("id", "job_type", "podcast_dir", "podcast_name",
                      "priority", "status", "created_at"):
            assert field in job, f"missing field: {field}"
        assert job["podcast_dir"] == str(d)

    def test_voices_dir_falls_back_to_server_voices(self, tmp_path, queue, mock_worker):
        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")
        # No config.json → config.voices_dir is None → should fall back to server_voices_dir
        watcher = _make_watcher([d], queue, mock_worker, server_voices_dir="/server/voices")
        self._run_check(watcher, d)
        job = queue.all_jobs()[0]
        assert job["voices_dir"] == "/server/voices"

    def test_voices_dir_from_config_takes_precedence(self, tmp_path, queue, mock_worker):
        d = _make_podcast_dir(tmp_path, "mypodcast")
        (d / "config.json").write_text(json.dumps({"voices_dir": "/podcast/voices"}))
        _add_txt(d, "ep01")
        watcher = _make_watcher([d], queue, mock_worker, server_voices_dir="/server/voices")
        self._run_check(watcher, d)
        assert queue.all_jobs()[0]["voices_dir"] == "/podcast/voices"

    # ── interrupt logic ───────────────────────────────────────────────────────

    def test_interrupt_called_when_non_urgent_job_running(
        self, tmp_path, queue, mock_worker
    ):
        """When a normal-priority (non-0) audiobook job is running and the podcast
        is priority-0, the watcher must call interrupt_current_for_priority_job."""
        from indextts_mlx.web.queue_manager import DEFAULT_PRIORITY, RUNNING

        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")

        # Simulate a running audiobook job
        audiobook = queue.submit(
            isbn="123", epub_url="https://example.com/book.epub", voice=None
        )
        queue.update(audiobook["id"], status=RUNNING)

        watcher = _make_watcher([d], queue, mock_worker)
        # Config defaults: priority=URGENT_PRIORITY (0)
        self._run_check(watcher, d)

        mock_worker.interrupt_current_for_priority_job.assert_called_once()
        interrupted_job = mock_worker.interrupt_current_for_priority_job.call_args[0][0]
        assert interrupted_job["id"] == audiobook["id"]

    def test_interrupt_not_called_when_urgent_job_already_running(
        self, tmp_path, queue, mock_worker
    ):
        """If the running job is already priority-0 (another podcast or urgent job),
        we do NOT call interrupt — it stays running."""
        from indextts_mlx.web.queue_manager import RUNNING, URGENT_PRIORITY

        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")

        # Another podcast episode (different podcast) is running at priority 0
        other_episode = {
            "id": "other-ep",
            "job_type": "podcast_episode",
            "isbn": "Podcast: other_podcast",
            "podcast_dir": str(tmp_path / "other_podcast"),
            "podcast_name": "other_podcast",
            "priority": URGENT_PRIORITY,
            "status": RUNNING,
            "stage": "synthesizing",
            "created_at": "2024-01-01T00:00:00Z",
            "started_at": "2024-01-01T00:01:00Z",
            "finished_at": None,
            "error": None,
        }
        queue.enqueue_podcast_episode(other_episode)
        queue.update("other-ep", status=RUNNING)

        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)

        mock_worker.interrupt_current_for_priority_job.assert_not_called()
        # But the new episode SHOULD still be enqueued
        podcast_jobs = [
            j for j in queue.all_jobs()
            if j.get("podcast_name") == "mypodcast"
        ]
        assert len(podcast_jobs) == 1

    def test_interrupt_not_called_when_no_active_job(self, tmp_path, queue, mock_worker):
        """If the queue is idle (no running job), interrupt must not be called."""
        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")
        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)

        mock_worker.interrupt_current_for_priority_job.assert_not_called()
        assert len(queue.all_jobs()) == 1

    def test_interrupt_not_called_when_podcast_priority_nonzero(
        self, tmp_path, queue, mock_worker
    ):
        """A podcast with non-zero priority behaves like a normal job and never interrupts."""
        from indextts_mlx.web.queue_manager import DEFAULT_PRIORITY, RUNNING

        d = _make_podcast_dir(tmp_path, "mypodcast")
        (d / "config.json").write_text(json.dumps({"priority": DEFAULT_PRIORITY}))
        _add_txt(d, "ep01")

        audiobook = queue.submit(isbn="456", epub_url="https://x.com/b.epub", voice=None)
        queue.update(audiobook["id"], status=RUNNING)

        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)

        mock_worker.interrupt_current_for_priority_job.assert_not_called()

    # ── direct_narration=False pending logic ──────────────────────────────────

    def test_direct_narration_false_no_jsonl_does_not_count_as_pending(
        self, tmp_path, queue, mock_worker
    ):
        """With direct_narration=False, missing jsonl files are NOT pending
        (they're never created). Only txt-without-mp3 counts."""
        d = _make_podcast_dir(tmp_path, "mypodcast")
        (d / "config.json").write_text(json.dumps({"direct_narration": False}))
        _add_txt(d, "ep01")
        _add_mp3(d, "ep01")
        # ep01 is done (has mp3). No pending work.
        watcher = _make_watcher([d], queue, mock_worker)
        self._run_check(watcher, d)
        assert queue.all_jobs() == []

    # ── error handling ────────────────────────────────────────────────────────

    def test_check_dir_error_logged_to_stderr(self, tmp_path, queue, mock_worker, capsys):
        """Exceptions in _check_dir must be printed to stderr, not silently swallowed."""
        d = _make_podcast_dir(tmp_path, "mypodcast")
        _add_txt(d, "ep01")

        watcher = _make_watcher([d], queue, mock_worker)

        # Make the queue raise to simulate an internal error
        queue.all_jobs = MagicMock(side_effect=RuntimeError("simulated DB error"))

        # _check_all catches and prints; call it (not _check_dir) to test the catch path
        watcher._check_all()

        captured = capsys.readouterr()
        assert "ERROR" in captured.err
        assert "mypodcast" in captured.err

    # ── multiple podcasts ─────────────────────────────────────────────────────

    def test_multiple_podcast_dirs_each_get_episode(self, tmp_path, queue, mock_worker):
        d1 = _make_podcast_dir(tmp_path, "show_a")
        d2 = _make_podcast_dir(tmp_path, "show_b")
        _add_txt(d1, "ep01")
        _add_txt(d2, "ep01")

        watcher = _make_watcher([d1, d2], queue, mock_worker)
        watcher._check_all()

        jobs = queue.all_jobs()
        assert len(jobs) == 2
        podcast_names = {j["podcast_name"] for j in jobs}
        assert podcast_names == {"show_a", "show_b"}
