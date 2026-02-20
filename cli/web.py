#!/usr/bin/env python3
"""CLI: start the IndexTTS audiobook builder web UI."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("web")
@click.option(
    "--port",
    default=5000,
    show_default=True,
    type=int,
    help="Port to listen on.",
)
@click.option(
    "--audiobooks-dir",
    required=True,
    type=click.Path(),
    help="Directory to store downloaded EPUBs, chapter text/audio, and final M4B files.",
)
@click.option(
    "--voices-dir",
    default=None,
    type=click.Path(exists=True),
    help="Directory containing voice reference audio files.",
)
@click.option(
    "--voice",
    "default_voice",
    default=None,
    help="Default voice name (from --voices-dir) to pre-select in the UI.",
)
@click.option(
    "--host",
    default="0.0.0.0",
    show_default=True,
    help="Host to bind to. Use 127.0.0.1 to restrict to localhost.",
)
@click.option(
    "--public-url",
    default=None,
    help=(
        "Public base URL used when generating shareable download links "
        "(e.g. https://myhost.dyndns.org:5000). "
        "If omitted, links use the browser's current origin."
    ),
)
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    help="Disable template caching so HTML changes are picked up on page refresh.",
)
@click.option(
    "--podcast-dir",
    "podcast_dirs",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help=(
        "Directory to watch for podcast episodes.  Each directory must contain a "
        "chapters_txt/ subdirectory and optionally a config.json.  "
        "May be specified multiple times for multiple podcasts."
    ),
)
@click.option(
    "--scheduler-config",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help=(
        "Path to a YAML file defining cron-scheduled shell commands.  "
        "Each job specifies a name, cron schedule, command, optional args list, "
        "and a log_dir where timestamped log files are written."
    ),
)
def web(port, audiobooks_dir, voices_dir, default_voice, host, public_url, dev, podcast_dirs, scheduler_config):
    """Start the IndexTTS audiobook builder web interface.

    The server accepts EPUB URLs and ISBNs, queues download + extraction +
    synthesis + M4B packaging jobs, and serves the finished files.

    \b
    Example:
      indextts web --audiobooks-dir ~/audiobooks --voices-dir ./voices \\
                   --voice british_female --port 5000
    """
    try:
        from flask import Flask
    except ImportError:
        raise click.ClickException(
            "Flask is required for the web UI. Install with: pip install flask"
        )

    from indextts_mlx.web.queue_manager import QueueManager
    from indextts_mlx.web.worker import Worker
    from indextts_mlx.web.app import create_app

    ab_dir = Path(audiobooks_dir).expanduser().resolve()
    ab_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Audiobooks directory: {ab_dir}")
    if voices_dir:
        click.echo(f"Voices directory:     {voices_dir}")
    if default_voice:
        click.echo(f"Default voice:        {default_voice}")
    podcast_dir_paths = [Path(d).expanduser().resolve() for d in podcast_dirs]
    for pd in podcast_dir_paths:
        click.echo(f"Podcast directory:    {pd}")

    queue = QueueManager(ab_dir)
    worker = Worker(
        queue=queue,
        audiobooks_dir=ab_dir,
        voices_dir=voices_dir,
        default_voice=default_voice,
    )
    worker.start()

    app = create_app(
        audiobooks_dir=ab_dir,
        voices_dir=voices_dir,
        queue=queue,
        worker=worker,
        public_url=public_url.rstrip("/") if public_url else None,
        dev=dev,
        podcast_dirs=podcast_dir_paths or None,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    _scheduler = None
    if scheduler_config:
        from indextts_mlx.web.scheduler import load_scheduler_config, start_scheduler

        sched_jobs = load_scheduler_config(scheduler_config)
        if sched_jobs:
            _scheduler = start_scheduler(sched_jobs, worker=worker)
            click.echo(f"Scheduler:            {len(sched_jobs)} job(s) from {scheduler_config}")
        else:
            click.echo("Scheduler:            config loaded but no jobs defined")

    click.echo(f"\nIndexTTS web UI running at http://{host}:{port}/")
    click.echo("Press Ctrl+C to stop.\n")

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    finally:
        worker.stop()
        if _scheduler is not None:
            _scheduler.shutdown(wait=False)


if __name__ == "__main__":
    web()
