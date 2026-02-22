"""Model lifecycle manager — ensures one model loaded at a time with proper cleanup."""

from __future__ import annotations

import gc
from typing import Optional

from indextts_mlx.srv.backends.base import ModelBackend


class ModelManager:
    def __init__(self, models_config: dict) -> None:
        self._current_backend: Optional[ModelBackend] = None
        self._current_key: Optional[tuple[str, str]] = None
        self._backend_classes: dict[str, type[ModelBackend]] = {}
        self._models_config = models_config

    def register(self, model_type: str, backend_cls: type[ModelBackend]) -> None:
        self._backend_classes[model_type] = backend_cls

    def resolve_model_name(self, model_type: str, model_name: Optional[str]) -> str:
        """Resolve model name, using YAML default if None."""
        backends = self._models_config.get("backends", {})
        backend_cfg = backends.get(model_type, {})
        if model_name is None:
            model_name = backend_cfg.get("default")
        if model_name is None:
            raise ValueError(
                f"No model name given and no default for backend '{model_type}'"
            )
        return model_name

    def _get_model_params(self, model_type: str, model_name: str) -> dict:
        """Look up model params from YAML config."""
        backends = self._models_config.get("backends", {})
        backend_cfg = backends.get(model_type, {})
        models = backend_cfg.get("models", {})
        params = models.get(model_name, {})
        return params

    def ensure_loaded(self, model_type: str, model_name: Optional[str] = None) -> ModelBackend:
        """Ensure the given model is loaded, unloading the previous if different."""
        model_name = self.resolve_model_name(model_type, model_name)
        key = (model_type, model_name)

        if self._current_key == key and self._current_backend is not None:
            return self._current_backend

        self.unload()

        if model_type not in self._backend_classes:
            raise ValueError(f"Unknown backend type: {model_type}")

        backend = self._backend_classes[model_type]()
        params = self._get_model_params(model_type, model_name)
        backend.load(params)
        self._current_backend = backend
        self._current_key = key
        return backend

    def unload(self) -> None:
        """Full cleanup: unload backend, gc, clear GPU caches."""
        if self._current_backend is None:
            return
        self._current_backend.unload()
        self._current_backend = None
        self._current_key = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except (ImportError, AttributeError):
            pass
        try:
            import torch
            torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass

    @property
    def current_key(self) -> Optional[tuple[str, str]]:
        return self._current_key

    def status(self) -> dict:
        if self._current_key is not None:
            return {
                "loaded_model": {
                    "type": self._current_key[0],
                    "name": self._current_key[1],
                }
            }
        return {"loaded_model": None}
