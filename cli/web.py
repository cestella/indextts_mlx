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
def web(port, audiobooks_dir, voices_dir, default_voice, host):
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
    )

    click.echo(f"\nIndexTTS web UI running at http://{host}:{port}/")
    click.echo("Press Ctrl+C to stop.\n")

    try:
        app.run(host=host, port=port, debug=False, use_reloader=False)
    finally:
        worker.stop()


if __name__ == "__main__":
    web()
