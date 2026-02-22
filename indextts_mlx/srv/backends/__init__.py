"""Backend registry."""

from indextts_mlx.srv.backends.base import ModelBackend
from indextts_mlx.srv.backends.mock import MockBackend
from indextts_mlx.srv.backends.tts_indextts import IndexTTSBackend
from indextts_mlx.srv.backends.llm import LLMBackend
from indextts_mlx.srv.backends.whisperx import WhisperXBackend
from indextts_mlx.srv.backends.translation import TranslationBackend
from indextts_mlx.srv.backends.tts_mlx_audio import MLXAudioTTSBackend
from indextts_mlx.srv.backends.tts_qwen3 import Qwen3TTSBackend

BACKENDS: dict[str, type[ModelBackend]] = {
    "mock": MockBackend,
    "tts_indextts": IndexTTSBackend,
    "llm": LLMBackend,
    "whisperx": WhisperXBackend,
    "translation": TranslationBackend,
    "tts_mlx_audio": MLXAudioTTSBackend,
    "tts_qwen3": Qwen3TTSBackend,
}

__all__ = [
    "ModelBackend",
    "MockBackend",
    "IndexTTSBackend",
    "LLMBackend",
    "WhisperXBackend",
    "TranslationBackend",
    "MLXAudioTTSBackend",
    "Qwen3TTSBackend",
    "BACKENDS",
]
