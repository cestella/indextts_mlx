"""Mock backend for testing — simulates load/execute with sleeps."""

from __future__ import annotations

import time
from pathlib import Path

from indextts_mlx.srv.backends.base import ModelBackend


class MockBackend(ModelBackend):
    model_type = "mock"

    def __init__(self) -> None:
        self._loaded = False

    def load(self, model_params: dict) -> None:
        time.sleep(model_params.get("load_delay", 0.01))
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def execute(self, request: dict) -> dict:
        if not self._loaded:
            raise RuntimeError("MockBackend not loaded")
        time.sleep(request.get("duration", 0.01))
        result_path = request.get("result_path")
        if result_path:
            Path(result_path).write_text("mock result")
        return {"status": "ok", "result_path": result_path}
