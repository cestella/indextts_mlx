"""Qwen3-TTS backend — voice cloning, custom voice, and voice design."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from indextts_mlx.srv.backends.base import ModelBackend


class Qwen3TTSBackend(ModelBackend):
    model_type = "tts_qwen3"

    def __init__(self) -> None:
        self._model = None

    def load(self, model_params: dict) -> None:
        from mlx_audio.tts.utils import load_model

        repo = model_params.get("repo", "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
        self._model = load_model(repo)

    def unload(self) -> None:
        self._model = None

    def execute(self, request: dict) -> dict:
        if self._model is None:
            raise RuntimeError("Qwen3TTS model not loaded — call load() first")

        text = request.get("text")
        if not text:
            raise ValueError("'text' is required")

        result_path = request.get("result_path")
        if not result_path:
            raise ValueError("'result_path' is required")

        mode = request.get("mode", "clone")
        language = request.get("language", "english")

        # Common generation kwargs
        gen_kw: dict = {}
        for key in ("temperature", "max_tokens", "top_k", "top_p", "repetition_penalty"):
            if key in request:
                gen_kw[key] = request[key]

        if mode == "voice_design":
            instruct = request.get("instruct")
            if not instruct:
                raise ValueError("'instruct' is required for voice_design mode")
            results = list(self._model.generate_voice_design(
                text=text,
                instruct=instruct,
                language=language,
                **gen_kw,
            ))
        elif mode == "custom_voice":
            speaker = request.get("speaker")
            if not speaker:
                raise ValueError("'speaker' is required for custom_voice mode")
            kw = {}
            if request.get("instruct"):
                kw["instruct"] = request["instruct"]
            results = list(self._model.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                **kw,
                **gen_kw,
            ))
        else:  # clone
            ref_audio = request.get("ref_audio")
            if not ref_audio:
                raise ValueError("'ref_audio' is required for clone mode")
            ref_text = request.get("ref_text", "")
            results = list(self._model.generate(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language,
                **gen_kw,
            ))

        audio = np.concatenate([np.array(r.audio) for r in results])
        sample_rate = self._model.sample_rate
        sf.write(result_path, audio, sample_rate)
        return {"status": "ok", "result_path": result_path, "sample_rate": sample_rate}
