"""Abstract base class for model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelBackend(ABC):
    """Base class for GPU model backends.

    Each backend type manages one model at a time. The ModelManager
    calls load/unload/execute in sequence — never concurrently.
    """

    model_type: str = ""

    @abstractmethod
    def load(self, model_params: dict) -> None:
        """Load model weights into GPU memory."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model weights and free GPU memory."""
        ...

    @abstractmethod
    def execute(self, request: dict) -> dict:
        """Run inference. Returns result dict (may include result_path, text, etc.)."""
        ...
