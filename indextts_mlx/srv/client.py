"""Typed client library for the indextts-srv service."""

from __future__ import annotations

import time
from typing import Optional

import httpx


# Maps engine name → (model_type, default model name)
_CLONE_ENGINES: dict[str, tuple[str, str]] = {
    "indextts": ("tts_indextts", "indextts2"),
    "qwen3-tts": ("tts_qwen3", "base"),
}
_CUSTOM_VOICE_ENGINES: dict[str, tuple[str, str]] = {
    "qwen3-tts": ("tts_qwen3", "custom-voice"),
}
_DESIGN_VOICE_ENGINES: dict[str, tuple[str, str]] = {
    "qwen3-tts": ("tts_qwen3", "voice-design"),
}


class SrvClient:
    """Sync client for the indextts-srv Unix-socket API.

    Usage::

        with SrvClient() as client:
            result = client.generate(prompt="Hello")
    """

    def __init__(self, socket_path: str = "/tmp/indextts_srv.sock") -> None:
        self._client = httpx.Client(
            transport=httpx.HTTPTransport(uds=socket_path),
            base_url="http://localhost",
        )

    # -- Context manager --

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SrvClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- Low-level --

    def submit(
        self,
        model_type: str,
        payload: dict,
        model: Optional[str] = None,
        application_id: str = "default",
        priority: int = 10,
        result_path: Optional[str] = None,
    ) -> str:
        """Submit a job and return its job_id."""
        body: dict = {
            "model_type": model_type,
            "application_id": application_id,
            "priority": priority,
            "payload": payload,
        }
        if model is not None:
            body["model"] = model
        if result_path is not None:
            body["result_path"] = result_path
        resp = self._client.post("/jobs", json=body)
        resp.raise_for_status()
        return resp.json()["job_id"]

    def get_job(self, job_id: str) -> dict:
        """Get the current state of a job."""
        resp = self._client.get(f"/jobs/{job_id}")
        resp.raise_for_status()
        return resp.json()

    def wait(
        self,
        job_id: str,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
    ) -> dict:
        """Poll until a job completes. Returns the job dict.

        Raises TimeoutError if the job doesn't finish within *timeout* seconds.
        Raises RuntimeError if the job fails.
        """
        deadline = time.monotonic() + timeout
        while True:
            job = self.get_job(job_id)
            status = job["status"]
            if status == "done":
                return job
            if status in ("failed", "cancelled", "expired"):
                error = job.get("error") or status
                raise RuntimeError(f"Job {job_id} {status}: {error}")
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Job {job_id} did not complete within {timeout}s "
                    f"(last status: {status})"
                )
            time.sleep(poll_interval)

    def cancel(self, job_id: str) -> None:
        """Cancel a queued job."""
        resp = self._client.delete(f"/jobs/{job_id}")
        resp.raise_for_status()

    def health(self) -> dict:
        """Get service health info."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def queue(self) -> dict:
        """Get queue status."""
        resp = self._client.get("/queue")
        resp.raise_for_status()
        return resp.json()

    # -- Convenience: submit + wait --

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
        **kw,
    ) -> dict:
        """Submit an LLM generation job and wait for the result."""
        job_id = self.submit_generate(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            model=model,
            **kw,
        )
        return self.wait(job_id, timeout=timeout, poll_interval=poll_interval)

    def synthesize(
        self,
        text: str,
        result_path: str,
        spk_audio_prompt: Optional[str] = None,
        voice: Optional[str] = None,
        voices_dir: Optional[str] = None,
        seed: int = 0,
        use_random: bool = False,
        emotion: Optional[float] = None,
        cfm_steps: Optional[int] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
        **kw,
    ) -> dict:
        """Submit a TTS synthesis job and wait for the result."""
        job_id = self.submit_synthesize(
            text=text,
            result_path=result_path,
            spk_audio_prompt=spk_audio_prompt,
            voice=voice,
            voices_dir=voices_dir,
            seed=seed,
            use_random=use_random,
            emotion=emotion,
            cfm_steps=cfm_steps,
            model=model,
            **kw,
        )
        return self.wait(job_id, timeout=timeout, poll_interval=poll_interval)

    def transcribe(
        self,
        audio_path: str,
        language: str = "en",
        result_path: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
        **kw,
    ) -> dict:
        """Submit a Whisper transcription job and wait for the result."""
        job_id = self.submit_transcribe(
            audio_path=audio_path,
            language=language,
            result_path=result_path,
            model=model,
            **kw,
        )
        return self.wait(job_id, timeout=timeout, poll_interval=poll_interval)

    def translate(
        self,
        text: str,
        src_lang: str = "ita",
        tgt_lang: str = "eng",
        max_tokens: int = 1024,
        model: Optional[str] = None,
        timeout: float = 300.0,
        poll_interval: float = 0.5,
        **kw,
    ) -> dict:
        """Submit a translation job and wait for the result."""
        job_id = self.submit_translate(
            text=text,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_tokens=max_tokens,
            model=model,
            **kw,
        )
        return self.wait(job_id, timeout=timeout, poll_interval=poll_interval)

    def clone_voice(
        self,
        text: str,
        ref_audio: str,
        result_path: str,
        engine: str = "qwen3-tts",
        timeout: float = 300.0,
        poll_interval: float = 1.0,
        **kw,
    ) -> dict:
        """Clone a voice from reference audio. Submit + wait.

        *engine* selects the TTS backend:
          - ``"indextts"`` — IndexTTS-2.  Engine-specific **kw: seed, use_random,
            emotion, cfm_steps, voice, voices_dir, emo_vector, ...
          - ``"qwen3-tts"`` — Qwen3-TTS Base model.  Engine-specific **kw:
            ref_text, language, temperature, max_tokens, top_k, top_p,
            repetition_penalty, ...
        """
        job_id = self.submit_clone_voice(
            text=text,
            ref_audio=ref_audio,
            result_path=result_path,
            engine=engine,
            **kw,
        )
        return self.wait(job_id, timeout=timeout, poll_interval=poll_interval)

    def custom_voice(
        self,
        text: str,
        speaker: str,
        result_path: str,
        engine: str = "qwen3-tts",
        timeout: float = 300.0,
        poll_interval: float = 1.0,
        **kw,
    ) -> dict:
        """Use a predefined speaker with optional emotion instruct. Submit + wait.

        *engine* selects the TTS backend:
          - ``"qwen3-tts"`` — Qwen3-TTS CustomVoice model.  Engine-specific **kw:
            instruct, language, temperature, max_tokens, top_k, top_p,
            repetition_penalty, ...
        """
        job_id = self.submit_custom_voice(
            text=text,
            speaker=speaker,
            result_path=result_path,
            engine=engine,
            **kw,
        )
        return self.wait(job_id, timeout=timeout, poll_interval=poll_interval)

    def design_voice(
        self,
        text: str,
        instruct: str,
        result_path: str,
        engine: str = "qwen3-tts",
        timeout: float = 300.0,
        poll_interval: float = 1.0,
        **kw,
    ) -> dict:
        """Design a voice from a text description. Submit + wait.

        *engine* selects the TTS backend:
          - ``"qwen3-tts"`` — Qwen3-TTS VoiceDesign model.  Engine-specific **kw:
            language, temperature, max_tokens, top_k, top_p,
            repetition_penalty, ...
        """
        job_id = self.submit_design_voice(
            text=text,
            instruct=instruct,
            result_path=result_path,
            engine=engine,
            **kw,
        )
        return self.wait(job_id, timeout=timeout, poll_interval=poll_interval)

    # -- Async submit (return job_id only) --

    def submit_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        model: Optional[str] = None,
        **kw,
    ) -> str:
        """Submit an LLM generation job; return job_id."""
        payload: dict = {"prompt": prompt, "max_tokens": max_tokens, **kw}
        if system_prompt is not None:
            payload["system_prompt"] = system_prompt
        return self.submit(model_type="llm", payload=payload, model=model)

    def submit_synthesize(
        self,
        text: str,
        result_path: str,
        spk_audio_prompt: Optional[str] = None,
        voice: Optional[str] = None,
        voices_dir: Optional[str] = None,
        seed: int = 0,
        use_random: bool = False,
        emotion: Optional[float] = None,
        cfm_steps: Optional[int] = None,
        model: Optional[str] = None,
        **kw,
    ) -> str:
        """Submit a TTS synthesis job; return job_id."""
        payload: dict = {
            "text": text,
            "result_path": result_path,
            "seed": seed,
            "use_random": use_random,
            **kw,
        }
        if spk_audio_prompt is not None:
            payload["spk_audio_prompt"] = spk_audio_prompt
        if voice is not None:
            payload["voice"] = voice
        if voices_dir is not None:
            payload["voices_dir"] = voices_dir
        if emotion is not None:
            payload["emotion"] = emotion
        if cfm_steps is not None:
            payload["cfm_steps"] = cfm_steps
        return self.submit(
            model_type="tts_indextts",
            payload=payload,
            model=model,
            result_path=result_path,
        )

    def submit_transcribe(
        self,
        audio_path: str,
        language: str = "en",
        result_path: Optional[str] = None,
        model: Optional[str] = None,
        **kw,
    ) -> str:
        """Submit a Whisper transcription job; return job_id."""
        payload: dict = {"audio_path": audio_path, "language": language, **kw}
        return self.submit(
            model_type="whisperx",
            payload=payload,
            model=model,
            result_path=result_path,
        )

    def submit_translate(
        self,
        text: str,
        src_lang: str = "ita",
        tgt_lang: str = "eng",
        max_tokens: int = 1024,
        model: Optional[str] = None,
        **kw,
    ) -> str:
        """Submit a translation job; return job_id."""
        payload: dict = {
            "text": text,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang,
            "max_tokens": max_tokens,
            **kw,
        }
        return self.submit(model_type="translation", payload=payload, model=model)

    def submit_clone_voice(
        self,
        text: str,
        ref_audio: str,
        result_path: str,
        engine: str = "qwen3-tts",
        **kw,
    ) -> str:
        """Submit a voice-cloning job; return job_id.

        See :meth:`clone_voice` for engine-specific **kw documentation.
        """
        if engine not in _CLONE_ENGINES:
            raise ValueError(
                f"Unknown clone engine {engine!r}; "
                f"choose from {sorted(_CLONE_ENGINES)}"
            )
        model_type, default_model = _CLONE_ENGINES[engine]

        if engine == "indextts":
            payload: dict = {
                "text": text,
                "spk_audio_prompt": ref_audio,
                "result_path": result_path,
                **kw,
            }
        else:  # qwen3-tts
            payload = {
                "text": text,
                "mode": "clone",
                "ref_audio": ref_audio,
                "result_path": result_path,
                **kw,
            }

        return self.submit(
            model_type=model_type,
            payload=payload,
            model=default_model,
            result_path=result_path,
        )

    def submit_custom_voice(
        self,
        text: str,
        speaker: str,
        result_path: str,
        engine: str = "qwen3-tts",
        **kw,
    ) -> str:
        """Submit a custom-voice job; return job_id.

        See :meth:`custom_voice` for engine-specific **kw documentation.
        """
        if engine not in _CUSTOM_VOICE_ENGINES:
            raise ValueError(
                f"Unknown custom_voice engine {engine!r}; "
                f"choose from {sorted(_CUSTOM_VOICE_ENGINES)}"
            )
        model_type, default_model = _CUSTOM_VOICE_ENGINES[engine]

        payload: dict = {
            "text": text,
            "mode": "custom_voice",
            "speaker": speaker,
            "result_path": result_path,
            **kw,
        }
        return self.submit(
            model_type=model_type,
            payload=payload,
            model=default_model,
            result_path=result_path,
        )

    def submit_design_voice(
        self,
        text: str,
        instruct: str,
        result_path: str,
        engine: str = "qwen3-tts",
        **kw,
    ) -> str:
        """Submit a voice-design job; return job_id.

        See :meth:`design_voice` for engine-specific **kw documentation.
        """
        if engine not in _DESIGN_VOICE_ENGINES:
            raise ValueError(
                f"Unknown design_voice engine {engine!r}; "
                f"choose from {sorted(_DESIGN_VOICE_ENGINES)}"
            )
        model_type, default_model = _DESIGN_VOICE_ENGINES[engine]

        payload: dict = {
            "text": text,
            "mode": "voice_design",
            "instruct": instruct,
            "result_path": result_path,
            **kw,
        }
        return self.submit(
            model_type=model_type,
            payload=payload,
            model=default_model,
            result_path=result_path,
        )

    # -- CPU (direct, no queue) --

    def normalize(self, text: str, language: str = "en") -> dict:
        """Normalize text to spoken form (CPU endpoint, no queue)."""
        resp = self._client.post(
            "/cpu/normalize", json={"text": text, "language": language}
        )
        resp.raise_for_status()
        return resp.json()

    def segment(
        self,
        text: str,
        language: str = "english",
        strategy: str = "char_count",
        max_chars: int = 300,
    ) -> dict:
        """Segment text into TTS-sized chunks (CPU endpoint, no queue)."""
        resp = self._client.post(
            "/cpu/segment",
            json={
                "text": text,
                "language": language,
                "strategy": strategy,
                "max_chars": max_chars,
            },
        )
        resp.raise_for_status()
        return resp.json()
