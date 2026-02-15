"""Flask web application for the IndexTTS audiobook builder."""

from __future__ import annotations

import json
import mimetypes
import os
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
) -> Flask:
    audiobooks_dir = Path(audiobooks_dir)
    template_dir = Path(__file__).parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB max request

    # ── helpers ───────────────────────────────────────────────────────────────

    def _synth_status(job: dict) -> dict | None:
        """Read synth_status.json for a running/done job if it exists."""
        if not job.get("dir_name"):
            return None
        status_file = audiobooks_dir / job["dir_name"] / ".status" / "synth_status.json"
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
        return render_template("index.html")

    @app.route("/api/queue")
    def api_queue():
        jobs = queue.all_jobs()
        active = queue.active_job()
        synth = _synth_status(active) if active else None
        return jsonify(
            {
                "jobs": jobs,
                "active": active,
                "synth_status": synth,
                "voices": _list_voices(),
            }
        )

    @app.route("/api/status/<job_id>")
    def api_status(job_id):
        job = queue.get_job(job_id)
        if not job:
            abort(404)
        synth = _synth_status(job)
        return jsonify({"job": job, "synth_status": synth})

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
            token_target=int(data.get("token_target", 250)),
        )
        return jsonify({"job": job}), 201

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

    return app
