"""MLX-Whisper transcription backend."""

from __future__ import annotations

from pathlib import Path

from indextts_mlx.srv.backends.base import ModelBackend


class WhisperXBackend(ModelBackend):
    model_type = "whisperx"

    def __init__(self) -> None:
        self._repo: str | None = None

    def load(self, model_params: dict) -> None:
        self._repo = model_params.get("repo", "mlx-community/whisper-large-v3-turbo")

    def unload(self) -> None:
        self._repo = None

    def execute(self, request: dict) -> dict:
        if self._repo is None:
            raise RuntimeError("WhisperXBackend not loaded")

        import mlx_whisper

        audio_path = request["audio_path"]
        language = request.get("language", "en")
        result_path = request.get("result_path")

        result = mlx_whisper.transcribe(
            audio_path, path_or_hf_repo=self._repo, language=language
        )
        text = result["text"]

        if result_path:
            Path(result_path).parent.mkdir(parents=True, exist_ok=True)
            Path(result_path).write_text(text)

        return {"text": text}
