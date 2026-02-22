"""Flask web application for the IndexTTS audiobook builder."""

from __future__ import annotations

import json
import mimetypes
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from flask import Flask, abort, jsonify, render_template, request, send_file

if TYPE_CHECKING:
    from .queue_manager import QueueManager
    from .worker import Worker


def create_app(
    audiobooks_dir: Path,
    voices_dir: str | None,
    queue: "QueueManager",
    worker: "Worker",
    public_url: str | None = None,
    dev: bool = False,
    podcast_dirs: list | None = None,
) -> Flask:
    audiobooks_dir = Path(audiobooks_dir)
    podcast_dirs = [Path(d) for d in (podcast_dirs or [])]
    # Map podcast_name → podcast_dir for fast lookups
    _podcast_map: dict[str, Path] = {d.name: d for d in podcast_dirs}

    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB max request
    if dev:
        app.config["TEMPLATES_AUTO_RELOAD"] = True
        app.jinja_env.auto_reload = True

    # ── helpers ───────────────────────────────────────────────────────────────

    def _status_dir(job: dict) -> Path | None:
        if job.get("job_type") == "podcast_episode" and job.get("podcast_dir"):
            return Path(job["podcast_dir"]) / ".status"
        if job.get("dir_name"):
            return audiobooks_dir / job["dir_name"] / ".status"
        return None

    def _synth_status(job: dict) -> dict | None:
        """Read synth_status.json for a running/done job if it exists."""
        sd = _status_dir(job)
        if not sd:
            return None
        status_file = sd / "synth_status.json"
        if status_file.exists():
            try:
                return json.loads(status_file.read_text())
            except Exception:
                return None
        return None

    def _classify_status(job: dict) -> dict | None:
        """Read classify_status.json for a directing job if it exists."""
        sd = _status_dir(job)
        if not sd:
            return None
        status_file = sd / "classify_status.json"
        if status_file.exists():
            try:
                return json.loads(status_file.read_text())
            except Exception:
                return None
        return None

    def _list_voices() -> list[str]:
        if not voices_dir:
            return []
        try:
            from indextts_mlx.voices import list_voices

            return sorted(list_voices(voices_dir))
        except Exception:
            return []

    # ── routes ────────────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html", public_url=public_url or "")

    @app.route("/api/queue")
    def api_queue():
        jobs = queue.all_jobs()
        active = queue.active_job()
        synth = _synth_status(active) if active else None
        cls = _classify_status(active) if active else None
        return jsonify(
            {
                "jobs": jobs,
                "active": active,
                "synth_status": synth,
                "classify_status": cls,
                "voices": _list_voices(),
            }
        )

    @app.route("/api/status/<job_id>")
    def api_status(job_id):
        job = queue.get_job(job_id)
        if not job:
            abort(404)
        synth = _synth_status(job)
        cls = _classify_status(job)
        return jsonify({"job": job, "synth_status": synth, "classify_status": cls})

    @app.route("/api/submit", methods=["POST"])
    def api_submit():
        data = request.get_json(force=True)
        required = ("isbn", "epub_url")
        for field in required:
            if not data.get(field, "").strip():
                return jsonify({"error": f"'{field}' is required"}), 400

        job = queue.submit(
            isbn=data["isbn"].strip(),
            epub_url=data["epub_url"].strip(),
            voice=data.get("voice", "").strip() or None,
            steps=int(data.get("steps", 10)),
            temperature=float(data.get("temperature", 1.0)),
            emotion=float(data.get("emotion", 1.0)),
            cfg_rate=float(data.get("cfg_rate", 0.7)),
            token_target=int(data.get("token_target", 50)),
            direct_narration=bool(data.get("direct_narration", False)),
            priority=int(data.get("priority", 10)),
        )
        return jsonify({"job": job}), 201

    @app.route("/api/library")
    def api_library():
        """Return all .m4b files found under audiobooks_dir."""
        books = []
        for m4b in sorted(audiobooks_dir.rglob("*.m4b")):
            try:
                stat = m4b.stat()  # raises if symlink is dangling
                if not m4b.is_file():
                    continue
            except OSError:
                continue
            rel = m4b.relative_to(audiobooks_dir)
            books.append(
                {
                    "name": m4b.stem,
                    "path": str(rel),
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%dT%H:%M:%S"
                    ),
                }
            )
        return jsonify({"books": books})

    @app.route("/api/scan")
    def api_scan():
        """Return directories detected in audiobooks_dir that can be resumed."""
        return jsonify({"dirs": queue.scan_dirs()})

    @app.route("/api/resume", methods=["POST"])
    def api_resume():
        from indextts_mlx.web.queue_manager import _detect_stage

        data = request.get_json(force=True)
        dir_name = (data.get("dir_name") or "").strip()
        if not dir_name:
            return jsonify({"error": "'dir_name' is required"}), 400
        isbn = (data.get("isbn") or dir_name).strip()

        # Auto-detect start_stage from directory contents if not supplied
        start_stage = (data.get("start_stage") or "").strip() or None
        if not start_stage:
            job_dir = audiobooks_dir / dir_name
            start_stage, _ = _detect_stage(job_dir)
            if start_stage is None:
                return (
                    jsonify(
                        {"error": "Cannot detect resume stage — directory empty or already done"}
                    ),
                    400,
                )

        job = queue.resume(
            dir_name=dir_name,
            isbn=isbn,
            voice=data.get("voice", "").strip() or None,
            start_stage=start_stage,
            steps=int(data.get("steps", 10)),
            temperature=float(data.get("temperature", 1.0)),
            emotion=float(data.get("emotion", 1.0)),
            cfg_rate=float(data.get("cfg_rate", 0.7)),
            token_target=int(data.get("token_target", 50)),
            direct_narration=bool(data.get("direct_narration", False)),
            priority=int(data.get("priority", 10)),
        )
        return jsonify({"job": job}), 201

    @app.route("/api/queue/move/<job_id>/up", methods=["POST"])
    def api_move_up(job_id):
        moved = queue.move_up(job_id)
        if not moved:
            return jsonify({"ok": False, "reason": "not found or already first"}), 400
        return jsonify({"ok": True})

    @app.route("/api/queue/move/<job_id>/down", methods=["POST"])
    def api_move_down(job_id):
        moved = queue.move_down(job_id)
        if not moved:
            return jsonify({"ok": False, "reason": "not found or already last"}), 400
        return jsonify({"ok": True})

    @app.route("/api/cancel/<job_id>", methods=["POST"])
    def api_cancel(job_id):
        job = queue.get_job(job_id)
        if not job:
            abort(404)
        prev_status = queue.cancel(job_id)
        if prev_status == "not_found":
            abort(404)
        # If job was running, also kill the subprocess
        worker.request_cancel(job_id)
        return jsonify({"ok": True, "prev_status": prev_status})

    @app.route("/api/logs/<job_id>")
    def api_logs(job_id):
        job = queue.get_job(job_id)
        if not job:
            abort(404)
        sd = _status_dir(job)
        if not sd:
            return jsonify({"log": ""}), 404
        log_path = sd / "worker.log"
        if not log_path.exists():
            return jsonify({"log": ""}), 404
        try:
            text = log_path.read_text(errors="replace")
        except Exception:
            return jsonify({"log": ""}), 500
        tail = request.args.get("tail", type=int)
        if tail and tail > 0:
            lines = text.splitlines()
            text = "\n".join(lines[-tail:])
        return jsonify({"log": text})

    # ── podcast routes ────────────────────────────────────────────────────────

    @app.route("/podcasts/<podcast_name>/feed")
    def podcast_rss(podcast_name):
        podcast_dir = _podcast_map.get(podcast_name)
        if podcast_dir is None:
            abort(404)
        from indextts_mlx.web.podcast import PodcastConfig, generate_rss

        config = PodcastConfig.from_file(podcast_dir / "config.json")
        base = public_url or request.url_root.rstrip("/")
        xml = generate_rss(podcast_dir, config, base)
        return app.response_class(xml, mimetype="application/rss+xml; charset=utf-8")

    @app.route("/podcasts/<podcast_name>/audio/<path:filename>")
    def podcast_audio(podcast_name, filename):
        from flask import send_from_directory

        podcast_dir = _podcast_map.get(podcast_name)
        if podcast_dir is None:
            abort(404)
        chapters_mp3 = podcast_dir / "chapters_mp3"
        if not chapters_mp3.is_dir():
            abort(404)
        return send_from_directory(str(chapters_mp3), filename)

    @app.route("/files/")
    @app.route("/files/<path:rel_path>")
    def serve_file(rel_path=""):
        """Browse or download files from audiobooks_dir."""
        target = audiobooks_dir / rel_path
        # Security: ensure path stays inside audiobooks_dir
        try:
            target = target.resolve()
            audiobooks_dir.resolve()
            target.relative_to(audiobooks_dir.resolve())
        except ValueError:
            abort(403)

        if target.is_dir():
            # Return a simple JSON listing of downloadable files
            entries = []
            for p in sorted(target.iterdir()):
                if p.name.startswith("."):
                    continue
                entries.append(
                    {
                        "name": p.name,
                        "is_dir": p.is_dir(),
                        "size": p.stat().st_size if p.is_file() else None,
                        "path": str(p.relative_to(audiobooks_dir)),
                    }
                )
            return jsonify({"path": rel_path, "entries": entries})

        if target.is_file():
            mime, _ = mimetypes.guess_type(str(target))
            return send_file(str(target), mimetype=mime or "application/octet-stream")

        abort(404)

    # ── podcast watcher ───────────────────────────────────────────────────────

    if podcast_dirs:
        from indextts_mlx.web.podcast import PodcastWatcher

        watcher = PodcastWatcher(
            podcast_dirs=podcast_dirs,
            queue=queue,
            worker=worker,
            server_voices_dir=voices_dir,
            public_url=public_url,
        )
        watcher.start()

    return app
