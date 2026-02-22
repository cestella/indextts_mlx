"""FastAPI application with background worker for GPU job processing."""

from __future__ import annotations

import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from indextts_mlx.srv.backends import BACKENDS
from indextts_mlx.srv.config import SrvConfig, load_models_config
from indextts_mlx.srv.cpu import router as cpu_router
from indextts_mlx.srv.model_manager import ModelManager
from indextts_mlx.srv.queue import JobQueue


class JobSubmission(BaseModel):
    model_type: str
    model: Optional[str] = None
    application_id: str = "default"
    priority: int = 10
    payload: dict = {}
    result_path: Optional[str] = None


_start_time: float = 0.0


def _worker_loop(
    queue: JobQueue,
    model_manager: ModelManager,
    stop_event: threading.Event,
) -> None:
    """Background worker: picks jobs, loads models, executes."""
    current_app_id: Optional[str] = None
    while not stop_event.is_set():
        job = queue.pick_next(model_manager.current_key, current_app_id)
        if job is None:
            stop_event.wait(0.1)
            continue

        try:
            # Resolve model name before loading
            model_name = model_manager.resolve_model_name(job.model_type, job.model)
            backend = model_manager.ensure_loaded(job.model_type, model_name)
            queue.mark_running(job.id)
            current_app_id = job.application_id
            result = backend.execute(job.payload)
            queue.mark_done(job.id, result)
        except Exception as e:
            queue.mark_failed(job.id, str(e))


def create_app(
    config: Optional[SrvConfig] = None,
    models_config: Optional[dict] = None,
) -> FastAPI:
    """Create the FastAPI app with queue and model manager."""
    if config is None:
        config = SrvConfig()
    if models_config is None:
        models_config = load_models_config(config.models_config_path)

    queue = JobQueue(
        heartbeat_timeout=config.heartbeat_timeout_s,
        max_size=config.max_queue_size,
    )
    model_manager = ModelManager(models_config)

    # Register all known backends
    for name, cls in BACKENDS.items():
        model_manager.register(name, cls)

    stop_event = threading.Event()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _start_time
        _start_time = time.time()
        worker = threading.Thread(
            target=_worker_loop,
            args=(queue, model_manager, stop_event),
            daemon=True,
        )
        worker.start()
        yield
        stop_event.set()
        worker.join(timeout=5.0)
        model_manager.unload()

    app = FastAPI(title="indextts-srv", lifespan=lifespan)
    app.state.queue = queue
    app.state.model_manager = model_manager
    app.state.stop_event = stop_event

    app.include_router(cpu_router)

    @app.post("/jobs")
    async def submit_job(body: JobSubmission) -> dict:
        try:
            # Resolve model name so it's stored explicitly
            model_name = model_manager.resolve_model_name(body.model_type, body.model)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        try:
            job_id = queue.submit(
                model_type=body.model_type,
                model=model_name,
                application_id=body.application_id,
                priority=body.priority,
                payload=body.payload,
                result_path=body.result_path,
            )
        except ValueError as e:
            raise HTTPException(status_code=429, detail=str(e))
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = queue.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.to_dict()

    @app.delete("/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict:
        if not queue.cancel(job_id):
            raise HTTPException(status_code=404, detail="Job not found or not cancellable")
        return {"status": "cancelled"}

    @app.get("/health")
    async def health() -> dict:
        jobs = queue.snapshot()
        return {
            "status": "ok",
            "loaded_model": model_manager.status()["loaded_model"],
            "queue_depth": sum(1 for j in jobs if j["status"] == "queued"),
            "uptime_s": round(time.time() - _start_time, 1),
        }

    @app.get("/queue")
    async def queue_status() -> dict:
        jobs = queue.snapshot()
        active = next((j for j in jobs if j["status"] == "running"), None)
        queued = [j for j in jobs if j["status"] == "queued"]
        recent = [j for j in jobs if j["status"] in ("done", "failed", "cancelled", "expired")]
        return {"active": active, "queued": queued, "recent": recent[-20:]}

    return app
