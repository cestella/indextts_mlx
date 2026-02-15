"""Tests for the web UI components: QueueManager, Worker (partial), and Flask app routes."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_ab_dir(tmp_path):
    """A temporary audiobooks directory."""
    d = tmp_path / "audiobooks"
    d.mkdir()
    return d


@pytest.fixture()
def queue(tmp_ab_dir):
    from indextts_mlx.web.queue_manager import QueueManager

    return QueueManager(tmp_ab_dir)


@pytest.fixture()
def app(tmp_ab_dir, queue):
    """Flask test client backed by a real QueueManager and a mock Worker."""
    from indextts_mlx.web.app import create_app

    mock_worker = MagicMock()
    mock_worker.request_cancel.return_value = False
    flask_app = create_app(
        audiobooks_dir=tmp_ab_dir,
        voices_dir=None,
        queue=queue,
        worker=mock_worker,
    )
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    with app.test_client() as c:
        yield c


# ── QueueManager ──────────────────────────────────────────────────────────────


class TestQueueManager:
    def test_empty_queue_on_init(self, queue):
        assert queue.all_jobs() == []
        assert queue.get_next_queued() is None
        assert queue.active_job() is None

    def test_submit_creates_queued_job(self, queue):
        job = queue.submit(isbn="978-0-06-112008-4", epub_url="https://example.com/book.epub", voice="v1")
        assert job["status"] == "queued"
        assert job["isbn"] == "978-0-06-112008-4"
        assert job["dir_name"] == "9780061120084"  # normalized
        assert job["id"]

    def test_submit_normalizes_isbn(self, queue):
        job = queue.submit(isbn="0-7432-7356-5", epub_url="https://x.com/a.epub", voice=None)
        assert job["dir_name"] == "0743273565"

    def test_get_next_queued_returns_first(self, queue):
        queue.submit(isbn="111", epub_url="https://x.com/1.epub", voice=None)
        queue.submit(isbn="222", epub_url="https://x.com/2.epub", voice=None)
        nxt = queue.get_next_queued()
        assert nxt["isbn"] == "111"

    def test_queue_persistence(self, tmp_ab_dir):
        from indextts_mlx.web.queue_manager import QueueManager

        q1 = QueueManager(tmp_ab_dir)
        job = q1.submit(isbn="999", epub_url="https://x.com/3.epub", voice=None)
        jid = job["id"]

        # New instance reads from disk
        q2 = QueueManager(tmp_ab_dir)
        loaded = q2.get_job(jid)
        assert loaded is not None
        assert loaded["isbn"] == "999"
        assert loaded["status"] == "queued"

    def test_running_job_marked_interrupted_on_reload(self, tmp_ab_dir):
        from indextts_mlx.web.queue_manager import QueueManager, RUNNING

        q1 = QueueManager(tmp_ab_dir)
        job = q1.submit(isbn="123", epub_url="https://x.com/b.epub", voice=None)
        q1.update(job["id"], status=RUNNING)

        # Simulate restart
        q2 = QueueManager(tmp_ab_dir)
        reloaded = q2.get_job(job["id"])
        assert reloaded["status"] == "interrupted"
        assert reloaded["error"]  # contains a message

    def test_queued_jobs_survive_reload(self, tmp_ab_dir):
        from indextts_mlx.web.queue_manager import QueueManager

        q1 = QueueManager(tmp_ab_dir)
        job = q1.submit(isbn="456", epub_url="https://x.com/c.epub", voice=None)

        q2 = QueueManager(tmp_ab_dir)
        reloaded = q2.get_job(job["id"])
        assert reloaded["status"] == "queued"  # queued jobs survive restart unchanged

    def test_cancel_queued_job(self, queue):
        job = queue.submit(isbn="aaa", epub_url="https://x.com/d.epub", voice=None)
        prev = queue.cancel(job["id"])
        assert prev == "queued"
        assert queue.get_job(job["id"])["status"] == "cancelled"

    def test_cancel_nonexistent_returns_not_found(self, queue):
        assert queue.cancel("no-such-id") == "not_found"

    def test_update_job(self, queue):
        job = queue.submit(isbn="bbb", epub_url="https://x.com/e.epub", voice=None)
        queue.update(job["id"], stage="extracting", title="My Book")
        updated = queue.get_job(job["id"])
        assert updated["stage"] == "extracting"
        assert updated["title"] == "My Book"

    def test_active_job_returns_running(self, queue):
        from indextts_mlx.web.queue_manager import RUNNING

        assert queue.active_job() is None
        job = queue.submit(isbn="ccc", epub_url="https://x.com/f.epub", voice=None)
        queue.update(job["id"], status=RUNNING)
        active = queue.active_job()
        assert active is not None
        assert active["id"] == job["id"]

    def test_thread_safety(self, queue):
        """Multiple threads submitting simultaneously should not corrupt the queue."""
        errors = []

        def _submit(i):
            try:
                queue.submit(isbn=str(i), epub_url=f"https://x.com/{i}.epub", voice=None)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=_submit, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(queue.all_jobs()) == 20

    def test_queue_file_written_atomically(self, tmp_ab_dir):
        """The queue file should never be empty/partial (tmp-then-rename)."""
        from indextts_mlx.web.queue_manager import QueueManager

        q = QueueManager(tmp_ab_dir)
        queue_file = tmp_ab_dir / ".queue.json"
        q.submit(isbn="x1", epub_url="https://x.com/g.epub", voice=None)
        # File must be valid JSON immediately after submit
        data = json.loads(queue_file.read_text())
        assert "jobs" in data
        assert len(data["jobs"]) == 1


# ── Flask API routes ──────────────────────────────────────────────────────────


class TestFlaskRoutes:
    def test_index_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"IndexTTS" in r.data

    def test_api_queue_empty(self, client):
        r = client.get("/api/queue")
        assert r.status_code == 200
        data = r.get_json()
        assert data["jobs"] == []
        assert data["active"] is None
        assert data["synth_status"] is None

    def test_api_submit_valid(self, client):
        r = client.post(
            "/api/submit",
            json={"isbn": "9780743273565", "epub_url": "https://example.com/book.epub", "voice": "v1"},
        )
        assert r.status_code == 201
        data = r.get_json()
        assert data["job"]["isbn"] == "9780743273565"
        assert data["job"]["status"] == "queued"

    def test_api_submit_missing_isbn(self, client):
        r = client.post("/api/submit", json={"epub_url": "https://x.com/b.epub"})
        assert r.status_code == 400
        assert "isbn" in r.get_json()["error"]

    def test_api_submit_missing_epub_url(self, client):
        r = client.post("/api/submit", json={"isbn": "123"})
        assert r.status_code == 400
        assert "epub_url" in r.get_json()["error"]

    def test_api_submit_empty_isbn(self, client):
        r = client.post("/api/submit", json={"isbn": "  ", "epub_url": "https://x.com/c.epub"})
        assert r.status_code == 400

    def test_api_queue_shows_submitted_job(self, client):
        client.post(
            "/api/submit",
            json={"isbn": "111", "epub_url": "https://x.com/d.epub"},
        )
        r = client.get("/api/queue")
        jobs = r.get_json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["isbn"] == "111"

    def test_api_cancel_queued_job(self, client, queue):
        job = queue.submit(isbn="222", epub_url="https://x.com/e.epub", voice=None)
        r = client.post(f"/api/cancel/{job['id']}")
        assert r.status_code == 200
        assert r.get_json()["ok"] is True
        assert queue.get_job(job["id"])["status"] == "cancelled"

    def test_api_cancel_nonexistent(self, client):
        r = client.post("/api/cancel/no-such-id")
        assert r.status_code == 404

    def test_api_status_known_job(self, client, queue):
        job = queue.submit(isbn="333", epub_url="https://x.com/f.epub", voice=None)
        r = client.get(f"/api/status/{job['id']}")
        assert r.status_code == 200
        data = r.get_json()
        assert data["job"]["id"] == job["id"]
        assert data["synth_status"] is None

    def test_api_status_unknown_job(self, client):
        r = client.get("/api/status/no-such-id")
        assert r.status_code == 404

    def test_api_status_reads_synth_file(self, client, queue, tmp_ab_dir):
        """synth_status is read from .status/synth_status.json when present."""
        from indextts_mlx.web.queue_manager import RUNNING

        job = queue.submit(isbn="444", epub_url="https://x.com/g.epub", voice=None)
        queue.update(job["id"], status=RUNNING)

        # Write a fake synth_status
        status_dir = tmp_ab_dir / job["dir_name"] / ".status"
        status_dir.mkdir(parents=True, exist_ok=True)
        fake = {
            "file_index": 2,
            "total_files": 10,
            "file_name": "chapter_03.txt",
            "job_eta_s": 300.0,
        }
        (status_dir / "synth_status.json").write_text(json.dumps(fake))

        r = client.get(f"/api/status/{job['id']}")
        data = r.get_json()
        assert data["synth_status"]["total_files"] == 10
        assert data["synth_status"]["file_name"] == "chapter_03.txt"

    def test_files_serves_existing_file(self, client, tmp_ab_dir):
        f = tmp_ab_dir / "test.m4b"
        f.write_bytes(b"fake-m4b")
        r = client.get("/files/test.m4b")
        assert r.status_code == 200
        assert r.data == b"fake-m4b"

    def test_files_directory_listing(self, client, tmp_ab_dir):
        d = tmp_ab_dir / "9780743273565"
        d.mkdir()
        (d / "book.m4b").write_bytes(b"x")
        r = client.get("/files/9780743273565")
        assert r.status_code == 200
        data = r.get_json()
        assert any(e["name"] == "book.m4b" for e in data["entries"])

    def test_files_path_traversal_blocked(self, client):
        r = client.get("/files/../../../etc/passwd")
        assert r.status_code in (403, 404)

    def test_files_hidden_files_not_listed(self, client, tmp_ab_dir):
        (tmp_ab_dir / ".queue.json").write_text("{}")
        r = client.get("/files/")
        data = r.get_json()
        names = [e["name"] for e in data["entries"]]
        assert ".queue.json" not in names


# ── ISBN normalization ────────────────────────────────────────────────────────


class TestNormalizeIsbn:
    def test_strips_hyphens(self):
        from indextts_mlx.web.queue_manager import normalize_isbn

        assert normalize_isbn("978-0-06-112008-4") == "9780061120084"

    def test_strips_spaces(self):
        from indextts_mlx.web.queue_manager import normalize_isbn

        assert normalize_isbn("978 0 06 112008 4") == "9780061120084"

    def test_lowercases(self):
        from indextts_mlx.web.queue_manager import normalize_isbn

        assert normalize_isbn("ISBN123ABC") == "isbn123abc"

    def test_already_clean(self):
        from indextts_mlx.web.queue_manager import normalize_isbn

        assert normalize_isbn("9780743273565") == "9780743273565"


# ── --status JSON writer (tts.py helper) ─────────────────────────────────────


class TestWriteSynthStatus:
    def test_writes_valid_json(self, tmp_path):
        from cli.tts import _write_synth_status

        p = tmp_path / "synth_status.json"
        _write_synth_status(
            p,
            file_index=2,
            total_files=10,
            file_name="chapter_03.txt",
            file_done=False,
            chunk_index=5,
            total_chunks=12,
            file_wall_times=[7.0, 7.2, 6.8],
            chunk_wall_times=[6.5, 7.1, 6.9, 7.0, 7.3],
        )
        assert p.exists()
        data = json.loads(p.read_text())
        assert data["file_index"] == 2
        assert data["total_files"] == 10
        assert data["file_name"] == "chapter_03.txt"
        assert data["file_done"] is False
        assert data["chunk_index"] == 5
        assert data["total_chunks"] == 12
        assert data["chunks_remaining"] == 7
        assert data["files_remaining"] == 8  # total - index - (not done)
        assert data["chunk_eta_s"] is not None
        assert data["job_eta_s"] is not None
        assert "updated_at" in data

    def test_eta_none_when_no_timing(self, tmp_path):
        from cli.tts import _write_synth_status

        p = tmp_path / "synth_status.json"
        _write_synth_status(p, 0, 5, "ch01.txt", False, 0, 8, [], [])
        data = json.loads(p.read_text())
        assert data["chunk_eta_s"] is None
        assert data["job_eta_s"] is None

    def test_file_done_reduces_files_remaining(self, tmp_path):
        from cli.tts import _write_synth_status

        p = tmp_path / "synth_status.json"
        _write_synth_status(p, 3, 10, "ch04.txt", True, 8, 8, [7.0] * 3, [7.0] * 8)
        data = json.loads(p.read_text())
        # file_done=True means this file is counted as done → files_remaining = 10 - 3 - 1 = 6
        assert data["files_remaining"] == 6

    def test_atomic_write(self, tmp_path):
        """Verify tmp-then-rename: status file is always fully formed."""
        from cli.tts import _write_synth_status

        p = tmp_path / "synth_status.json"
        for i in range(20):
            _write_synth_status(p, i, 20, f"ch{i:02d}.txt", False, 0, 5, [], [])
        data = json.loads(p.read_text())
        assert data["file_index"] == 19  # last write wins
