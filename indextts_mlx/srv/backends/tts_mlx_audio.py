"""mlx-audio TTS backend (F5, Kokoro) — stub for future implementation."""

from __future__ import annotations

from indextts_mlx.srv.backends.base import ModelBackend


class MLXAudioTTSBackend(ModelBackend):
    model_type = "tts_mlx_audio"

    def __init__(self) -> None:
        self._loaded = False

    def load(self, model_params: dict) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def execute(self, request: dict) -> dict:
        raise NotImplementedError(
            "tts_mlx_audio backend is not yet implemented. "
            "Planned support for F5-TTS and Kokoro models."
        )
