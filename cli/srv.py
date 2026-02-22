"""CLI commands for the GPU resource management service."""

from __future__ import annotations

import json
import os
import sys

import click


@click.group("srv")
def srv():
    """GPU resource management service."""


@srv.command()
@click.option("--socket", default="/tmp/indextts_srv.sock", help="Unix socket path")
@click.option("--config", default=None, help="Path to models.yaml config")
@click.option("--daemon/--no-daemon", default=False, help="Run as daemon")
def start(socket: str, config: str | None, daemon: bool) -> None:
    """Start the srv service."""
    import uvicorn

    from indextts_mlx.srv.config import SrvConfig, load_models_config
    from indextts_mlx.srv.app import create_app

    srv_config = SrvConfig(socket_path=socket)
    if config:
        srv_config.models_config_path = config

    models_config = load_models_config(srv_config.models_config_path)
    app = create_app(config=srv_config, models_config=models_config)

    # Remove stale socket
    if os.path.exists(socket):
        os.unlink(socket)

    click.echo(f"Starting srv on {socket}")
    uvicorn.run(app, uds=socket, log_level=srv_config.log_level)


def _http_client(socket_path: str = "/tmp/indextts_srv.sock"):
    """Create an httpx client connected to the unix socket."""
    import httpx

    transport = httpx.HTTPTransport(uds=socket_path)
    return httpx.Client(transport=transport, base_url="http://localhost")


@srv.command()
@click.option("--socket", default="/tmp/indextts_srv.sock")
def health(socket: str) -> None:
    """Check service health."""
    try:
        client = _http_client(socket)
        r = client.get("/health")
        r.raise_for_status()
        data = r.json()
        click.echo(json.dumps(data, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@srv.command()
@click.option("--socket", default="/tmp/indextts_srv.sock")
def queue(socket: str) -> None:
    """Show queue status."""
    try:
        client = _http_client(socket)
        r = client.get("/queue")
        r.raise_for_status()
        data = r.json()
        click.echo(json.dumps(data, indent=2))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@srv.command()
@click.argument("model_type")
@click.option("--model", default=None, help="Named model (uses default if omitted)")
@click.option("--app-id", default="cli", help="Application ID for grouping")
@click.option("--priority", default=10, type=int, help="Priority (0 = urgent)")
@click.option("--payload", default="{}", help="JSON payload string")
@click.option("--result-path", default=None, help="Path for result file")
@click.option("--socket", default="/tmp/indextts_srv.sock")
def submit(
    model_type: str,
    model: str | None,
    app_id: str,
    priority: int,
    payload: str,
    result_path: str | None,
    socket: str,
) -> None:
    """Submit a job to the queue."""
    try:
        payload_dict = json.loads(payload)
    except json.JSONDecodeError as e:
        click.echo(f"Invalid JSON payload: {e}", err=True)
        sys.exit(1)

    try:
        client = _http_client(socket)
        body = {
            "model_type": model_type,
            "application_id": app_id,
            "priority": priority,
            "payload": payload_dict,
        }
        if model is not None:
            body["model"] = model
        if result_path is not None:
            body["result_path"] = result_path
        r = client.post("/jobs", json=body)
        r.raise_for_status()
        data = r.json()
        click.echo(f"Job submitted: {data['job_id']}")
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@srv.command()
@click.option("--socket", default="/tmp/indextts_srv.sock")
def stop(socket: str) -> None:
    """Stop the service (sends shutdown signal)."""
    click.echo("Stopping srv... (use Ctrl+C on the server process)")
