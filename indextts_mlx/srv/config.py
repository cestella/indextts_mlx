"""Service configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SrvConfig:
    socket_path: str = "/tmp/indextts_srv.sock"
    heartbeat_timeout_s: float = 300.0
    max_queue_size: int = 100
    log_level: str = "info"
    models_config_path: str = "~/.config/indextts_srv/models.yaml"


_DEFAULT_MODELS_CONFIG: dict[str, Any] = {
    "backends": {
        "mock": {
            "default": "default",
            "models": {
                "default": {},
            },
        },
        "tts_indextts": {
            "default": "indextts2",
            "models": {
                "indextts2": {},
            },
        },
        "llm": {
            "default": "qwen2.5-7b",
            "models": {
                "qwen2.5-7b": {
                    "repo": "mlx-community/Qwen2.5-7B-Instruct-4bit",
                },
            },
        },
        "whisperx": {
            "default": "whisper-large-v3-turbo",
            "models": {
                "whisper-large-v3-turbo": {
                    "repo": "mlx-community/whisper-large-v3-turbo",
                },
            },
        },
        "translation": {
            "default": "seamless-m4t-v2",
            "models": {
                "seamless-m4t-v2": {
                    "repo": "facebook/seamless-m4t-v2-large",
                },
            },
        },
        "tts_mlx_audio": {
            "default": "f5-tts",
            "models": {
                "f5-tts": {
                    "repo": "lucasnewman/f5-tts-mlx",
                },
            },
        },
        "tts_qwen3": {
            "default": "base",
            "models": {
                "base": {
                    "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
                },
                "custom-voice": {
                    "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
                },
                "voice-design": {
                    "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
                },
            },
        },
    },
}


def load_models_config(path: str) -> dict[str, Any]:
    """Load and validate the YAML models config. Returns nested dict.

    If the file does not exist, creates it with a default config.
    """
    expanded = Path(os.path.expanduser(path))
    if not expanded.exists():
        expanded.parent.mkdir(parents=True, exist_ok=True)
        expanded.write_text(yaml.dump(_DEFAULT_MODELS_CONFIG, default_flow_style=False))
        return _DEFAULT_MODELS_CONFIG

    with open(expanded) as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict) or "backends" not in config:
        raise ValueError(f"Invalid models config at {expanded}: must have 'backends' key")

    return config
