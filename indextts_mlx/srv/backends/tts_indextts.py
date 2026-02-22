"""IndexTTS-2 backend — real GPU TTS synthesis."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional

from indextts_mlx.srv.backends.base import ModelBackend


class IndexTTSBackend(ModelBackend):
    model_type = "tts_indextts"

    def __init__(self) -> None:
        self._tts = None

    def load(self, model_params: dict) -> None:
        from indextts_mlx import IndexTTS2, WeightsConfig

        kwargs = {}
        if "weights_dir" in model_params:
            kwargs["weights_dir"] = Path(model_params["weights_dir"]).expanduser()
        if "bpe_model" in model_params:
            kwargs["bpe_model"] = Path(model_params["bpe_model"]).expanduser()

        config = WeightsConfig(**kwargs) if kwargs else None
        self._tts = IndexTTS2(config=config)

    def unload(self) -> None:
        self._tts = None

    def execute(self, request: dict) -> dict:
        if self._tts is None:
            raise RuntimeError("IndexTTSBackend not loaded")

        text = request["text"]
        result_path = request.get("result_path")
        if not result_path:
            raise ValueError("result_path is required for TTS jobs")

        # Build synthesize kwargs from request
        synth_kwargs: dict = {}

        # Speaker source
        if "spk_audio_prompt" in request:
            synth_kwargs["spk_audio_prompt"] = request["spk_audio_prompt"]
        if "voices_dir" in request:
            synth_kwargs["voices_dir"] = request["voices_dir"]
        if "voice" in request:
            synth_kwargs["voice"] = request["voice"]

        # Emotion
        for key in ("emotion", "emo_alpha", "emo_vector", "emo_text",
                     "use_emo_text", "emo_audio_prompt"):
            if key in request:
                synth_kwargs[key] = request[key]

        # Determinism
        if "seed" in request:
            synth_kwargs["seed"] = request["seed"]
        if "use_random" in request:
            synth_kwargs["use_random"] = request["use_random"]

        # Quality
        for key in ("cfm_steps", "temperature", "max_codes", "cfg_rate",
                     "gpt_temperature", "top_k"):
            if key in request:
                synth_kwargs[key] = request[key]

        audio = self._tts.synthesize(text, **synth_kwargs)

        # Write output
        result_path = Path(result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(result_path), audio, 22050)

        return {"status": "ok", "result_path": str(result_path), "sample_rate": 22050}
