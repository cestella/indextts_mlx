"""Integration tests for srv backends — load real models, opt-in via --srv-integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.srv_integration


# ---------------------------------------------------------------------------
# tts_indextts
# ---------------------------------------------------------------------------


class TestIndexTTSIntegration:
    def test_synthesize(self, tmp_path):
        from indextts_mlx.srv.backends.tts_indextts import IndexTTSBackend

        b = IndexTTSBackend()
        b.load({})

        ref_audio = Path.home() / "audiobooks/voices/prunella_scales.wav"
        if not ref_audio.exists():
            pytest.skip(f"Reference audio not found: {ref_audio}")

        out = tmp_path / "out.wav"
        result = b.execute({
            "text": "Hello world.",
            "spk_audio_prompt": str(ref_audio),
            "result_path": str(out),
            "seed": 42,
            "use_random": False,
        })

        assert result["status"] == "ok"
        assert out.exists()
        audio, sr = sf.read(str(out))
        assert sr == 22050
        assert len(audio) > 0
        assert np.isfinite(audio).all()

        b.unload()


# ---------------------------------------------------------------------------
# llm
# ---------------------------------------------------------------------------


class TestLLMIntegration:
    def test_generate(self):
        from indextts_mlx.srv.backends.llm import LLMBackend

        b = LLMBackend()
        b.load({"repo": "mlx-community/Qwen2.5-7B-Instruct-4bit"})

        result = b.execute({
            "prompt": "What is 2+2? Answer with just the number.",
            "max_tokens": 32,
        })

        assert "text" in result
        assert len(result["text"]) > 0

        b.unload()


# ---------------------------------------------------------------------------
# whisperx
# ---------------------------------------------------------------------------


class TestWhisperXIntegration:
    def test_transcribe(self, tmp_path):
        from indextts_mlx.srv.backends.whisperx import WhisperXBackend

        # Create a short silent audio file to transcribe
        audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        audio_path = tmp_path / "silence.wav"
        sf.write(str(audio_path), audio, 16000)

        b = WhisperXBackend()
        b.load({"repo": "mlx-community/whisper-large-v3-turbo"})

        result = b.execute({
            "audio_path": str(audio_path),
            "language": "en",
        })

        assert "text" in result
        assert isinstance(result["text"], str)

        b.unload()


# ---------------------------------------------------------------------------
# translation
# ---------------------------------------------------------------------------


class TestTranslationIntegration:
    def test_translate(self):
        from indextts_mlx.srv.backends.translation import TranslationBackend

        b = TranslationBackend()
        b.load({"repo": "facebook/seamless-m4t-v2-large"})

        result = b.execute({
            "text": "Buongiorno, come stai?",
            "src_lang": "ita",
            "tgt_lang": "eng",
        })

        assert "text" in result
        assert len(result["text"]) > 0

        b.unload()


# ---------------------------------------------------------------------------
# tts_qwen3
# ---------------------------------------------------------------------------


class TestQwen3TTSIntegration:
    def test_voice_design(self, tmp_path):
        from indextts_mlx.srv.backends.tts_qwen3 import Qwen3TTSBackend

        b = Qwen3TTSBackend()
        b.load({"repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"})

        out = tmp_path / "voice_design.wav"
        result = b.execute({
            "text": "La lluvia caia suavemente sobre los tejados.",
            "result_path": str(out),
            "mode": "voice_design",
            "instruct": "A calm Univision news broadcaster",
            "language": "spanish",
        })

        assert result["status"] == "ok"
        assert out.exists()
        audio, sr = sf.read(str(out))
        assert len(audio) > 0
        assert np.isfinite(audio).all()

        b.unload()

    def test_custom_voice(self, tmp_path):
        from indextts_mlx.srv.backends.tts_qwen3 import Qwen3TTSBackend

        b = Qwen3TTSBackend()
        b.load({"repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"})

        out = tmp_path / "custom_voice.wav"
        result = b.execute({
            "text": "I am so excited about this!",
            "result_path": str(out),
            "mode": "custom_voice",
            "speaker": "Vivian",
            "instruct": "Very happy and excited.",
            "language": "english",
        })

        assert result["status"] == "ok"
        assert out.exists()
        audio, sr = sf.read(str(out))
        assert len(audio) > 0
        assert np.isfinite(audio).all()

        b.unload()

    def test_voice_cloning(self, tmp_path):
        from indextts_mlx.srv.backends.tts_qwen3 import Qwen3TTSBackend

        repo_root = Path(__file__).resolve().parent.parent
        ref_audio = repo_root / "voices" / "british_female.wav"
        assert ref_audio.exists(), f"Reference audio not found: {ref_audio}"

        b = Qwen3TTSBackend()
        b.load({"repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"})

        out = tmp_path / "clone.wav"
        result = b.execute({
            "text": "Hello world, this is a test of voice cloning.",
            "result_path": str(out),
            "mode": "clone",
            "ref_audio": str(ref_audio),
            "ref_text": "This is a reference recording.",
            "language": "english",
        })

        assert result["status"] == "ok"
        assert out.exists()
        audio, sr = sf.read(str(out))
        assert len(audio) > 0
        assert np.isfinite(audio).all()

        b.unload()


# ---------------------------------------------------------------------------
# tts_mlx_audio (stub)
# ---------------------------------------------------------------------------


class TestMLXAudioTTSIntegration:
    def test_stub_raises(self):
        from indextts_mlx.srv.backends.tts_mlx_audio import MLXAudioTTSBackend

        b = MLXAudioTTSBackend()
        b.load({})

        with pytest.raises(NotImplementedError):
            b.execute({"text": "hello"})

        b.unload()
