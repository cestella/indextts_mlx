"""Unit tests for srv backends — mock-based, always run (no real models needed)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# tts_indextts
# ---------------------------------------------------------------------------


class TestIndexTTSBackend:
    def _make(self):
        from indextts_mlx.srv.backends.tts_indextts import IndexTTSBackend

        return IndexTTSBackend()

    def test_execute_not_loaded_raises(self):
        b = self._make()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.execute({"text": "hi", "result_path": "/tmp/x.wav"})

    def test_load_unload(self):
        b = self._make()
        # Directly set internal state to simulate load
        mock_tts = MagicMock()
        b._tts = mock_tts
        assert b._tts is not None
        b.unload()
        assert b._tts is None

    def test_execute_returns_result(self, tmp_path):
        b = self._make()
        import numpy as np

        fake_audio = np.zeros(1000, dtype=np.float32)
        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = fake_audio
        b._tts = mock_tts

        out = tmp_path / "out.wav"
        result = b.execute(
            {"text": "hello", "result_path": str(out), "spk_audio_prompt": "speaker.wav"}
        )
        assert result["status"] == "ok"
        assert result["result_path"] == str(out)
        assert result["sample_rate"] == 22050
        mock_tts.synthesize.assert_called_once()
        assert out.exists()


# ---------------------------------------------------------------------------
# llm
# ---------------------------------------------------------------------------


class TestLLMBackend:
    def _make(self):
        from indextts_mlx.srv.backends.llm import LLMBackend

        return LLMBackend()

    def test_execute_not_loaded_raises(self):
        b = self._make()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.execute({"prompt": "hi"})

    def test_load_unload(self):
        b = self._make()
        # Simulate load by setting internals
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        b._model = mock_model
        b._tokenizer = mock_tokenizer
        assert b._model is not None
        assert b._tokenizer is not None
        b.unload()
        assert b._model is None
        assert b._tokenizer is None

    def test_execute_returns_result(self):
        b = self._make()
        b._model = MagicMock()
        b._tokenizer = MagicMock()
        b._tokenizer.apply_chat_template.return_value = "formatted prompt"

        # Create a mock mlx_lm module and inject it
        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.return_value = "Generated response"

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx_lm}):
            # Re-import to pick up the mock
            import importlib
            import indextts_mlx.srv.backends.llm as llm_mod

            importlib.reload(llm_mod)
            b2 = llm_mod.LLMBackend()
            b2._model = b._model
            b2._tokenizer = b._tokenizer
            result = b2.execute({"prompt": "Say hello", "max_tokens": 100})

        assert result == {"text": "Generated response"}
        b._tokenizer.apply_chat_template.assert_called_once()
        mock_mlx_lm.generate.assert_called_once()

    def test_execute_with_system_prompt(self):
        b = self._make()
        b._model = MagicMock()
        b._tokenizer = MagicMock()
        b._tokenizer.apply_chat_template.return_value = "formatted"

        mock_mlx_lm = MagicMock()
        mock_mlx_lm.generate.return_value = "reply"

        with patch.dict(sys.modules, {"mlx_lm": mock_mlx_lm}):
            import importlib
            import indextts_mlx.srv.backends.llm as llm_mod

            importlib.reload(llm_mod)
            b2 = llm_mod.LLMBackend()
            b2._model = b._model
            b2._tokenizer = b._tokenizer
            b2.execute({"prompt": "hi", "system_prompt": "You are helpful"})

        call_args = b._tokenizer.apply_chat_template.call_args
        messages = call_args[0][0]
        assert messages[0] == {"role": "system", "content": "You are helpful"}
        assert messages[1] == {"role": "user", "content": "hi"}


# ---------------------------------------------------------------------------
# whisperx
# ---------------------------------------------------------------------------


class TestWhisperXBackend:
    def _make(self):
        from indextts_mlx.srv.backends.whisperx import WhisperXBackend

        return WhisperXBackend()

    def test_execute_not_loaded_raises(self):
        b = self._make()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.execute({"audio_path": "/tmp/audio.wav"})

    def test_load_unload(self):
        b = self._make()
        b.load({"repo": "mlx-community/whisper-large-v3-turbo"})
        assert b._repo == "mlx-community/whisper-large-v3-turbo"
        b.unload()
        assert b._repo is None

    def test_execute_returns_result(self):
        b = self._make()
        b._repo = "test/whisper"

        mock_mlx_whisper = MagicMock()
        mock_mlx_whisper.transcribe.return_value = {"text": "Hello world"}

        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx_whisper}):
            import importlib
            import indextts_mlx.srv.backends.whisperx as wx_mod

            importlib.reload(wx_mod)
            b2 = wx_mod.WhisperXBackend()
            b2._repo = "test/whisper"
            result = b2.execute({"audio_path": "/tmp/audio.wav", "language": "en"})

        assert result == {"text": "Hello world"}
        mock_mlx_whisper.transcribe.assert_called_once_with(
            "/tmp/audio.wav", path_or_hf_repo="test/whisper", language="en"
        )

    def test_execute_writes_result_path(self, tmp_path):
        b = self._make()
        b._repo = "test/whisper"
        out = tmp_path / "transcript.txt"

        mock_mlx_whisper = MagicMock()
        mock_mlx_whisper.transcribe.return_value = {"text": "Hello world"}

        with patch.dict(sys.modules, {"mlx_whisper": mock_mlx_whisper}):
            import importlib
            import indextts_mlx.srv.backends.whisperx as wx_mod

            importlib.reload(wx_mod)
            b2 = wx_mod.WhisperXBackend()
            b2._repo = "test/whisper"
            result = b2.execute(
                {"audio_path": "/tmp/audio.wav", "result_path": str(out)}
            )

        assert result["text"] == "Hello world"
        assert out.read_text() == "Hello world"


# ---------------------------------------------------------------------------
# translation
# ---------------------------------------------------------------------------


class TestTranslationBackend:
    def _make(self):
        from indextts_mlx.srv.backends.translation import TranslationBackend

        return TranslationBackend()

    def test_execute_not_loaded_raises(self):
        b = self._make()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.execute({"text": "ciao"})

    def test_load_unload(self):
        b = self._make()
        # Simulate load by setting internals
        b._processor = MagicMock()
        b._model = MagicMock()
        assert b._processor is not None
        assert b._model is not None
        b.unload()
        assert b._processor is None
        assert b._model is None

    def test_execute_returns_result(self):
        b = self._make()
        mock_processor = MagicMock()
        mock_model = MagicMock()
        b._processor = mock_processor
        b._model = mock_model

        # Mock processor call: returns dict with input_ids
        mock_inputs = {"input_ids": MagicMock()}
        mock_processor.return_value = mock_inputs

        # Mock model.generate: returns list of tensors
        mock_output = MagicMock()
        mock_output.tolist.return_value = [1, 2, 3, 4]
        mock_model.generate.return_value = [mock_output]

        mock_processor.decode.return_value = "Hello"

        result = b.execute({"text": "Ciao", "src_lang": "ita", "tgt_lang": "eng"})
        assert result == {"text": "Hello"}
        mock_processor.assert_called_once_with(text="Ciao", src_lang="ita", return_tensors="pt")
        mock_processor.decode.assert_called_once()


# ---------------------------------------------------------------------------
# tts_mlx_audio
# ---------------------------------------------------------------------------


class TestQwen3TTSBackend:
    def _make(self):
        from indextts_mlx.srv.backends.tts_qwen3 import Qwen3TTSBackend

        return Qwen3TTSBackend()

    def test_execute_not_loaded_raises(self):
        b = self._make()
        with pytest.raises(RuntimeError, match="not loaded"):
            b.execute({"text": "hi", "result_path": "/tmp/x.wav"})

    def test_execute_requires_text(self, tmp_path):
        b = self._make()
        b._model = MagicMock()
        with pytest.raises(ValueError, match="'text' is required"):
            b.execute({"result_path": str(tmp_path / "x.wav")})

    def test_execute_requires_result_path(self):
        b = self._make()
        b._model = MagicMock()
        with pytest.raises(ValueError, match="'result_path' is required"):
            b.execute({"text": "hello"})

    def test_execute_voice_design_requires_instruct(self):
        b = self._make()
        b._model = MagicMock()
        with pytest.raises(ValueError, match="'instruct' is required"):
            b.execute({"text": "hello", "result_path": "/tmp/x.wav", "mode": "voice_design"})

    def test_execute_custom_voice_requires_speaker(self):
        b = self._make()
        b._model = MagicMock()
        with pytest.raises(ValueError, match="'speaker' is required"):
            b.execute({"text": "hello", "result_path": "/tmp/x.wav", "mode": "custom_voice"})

    def test_execute_clone_requires_ref_audio(self):
        b = self._make()
        b._model = MagicMock()
        with pytest.raises(ValueError, match="'ref_audio' is required"):
            b.execute({"text": "hello", "result_path": "/tmp/x.wav", "mode": "clone"})

    def test_execute_clone_writes_wav(self, tmp_path):
        b = self._make()
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_result = MagicMock()
        mock_result.audio = [0.1, 0.2, 0.3]
        mock_model.generate.return_value = [mock_result]
        b._model = mock_model

        out = tmp_path / "out.wav"
        result = b.execute({
            "text": "hello",
            "result_path": str(out),
            "mode": "clone",
            "ref_audio": "speaker.wav",
            "ref_text": "test",
            "language": "english",
        })
        assert result["status"] == "ok"
        assert result["result_path"] == str(out)
        assert result["sample_rate"] == 24000
        assert out.exists()
        mock_model.generate.assert_called_once_with(
            text="hello",
            ref_audio="speaker.wav",
            ref_text="test",
            language="english",
        )

    def test_execute_voice_design_calls_generate_voice_design(self, tmp_path):
        b = self._make()
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_result = MagicMock()
        mock_result.audio = [0.5]
        mock_model.generate_voice_design.return_value = [mock_result]
        b._model = mock_model

        out = tmp_path / "out.wav"
        result = b.execute({
            "text": "hola",
            "result_path": str(out),
            "mode": "voice_design",
            "instruct": "A calm narrator",
            "language": "spanish",
        })
        assert result["status"] == "ok"
        mock_model.generate_voice_design.assert_called_once_with(
            text="hola",
            instruct="A calm narrator",
            language="spanish",
        )

    def test_execute_custom_voice_calls_generate_custom_voice(self, tmp_path):
        b = self._make()
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_result = MagicMock()
        mock_result.audio = [0.5]
        mock_model.generate_custom_voice.return_value = [mock_result]
        b._model = mock_model

        out = tmp_path / "out.wav"
        result = b.execute({
            "text": "wow",
            "result_path": str(out),
            "mode": "custom_voice",
            "speaker": "Vivian",
            "instruct": "Very excited",
            "language": "english",
        })
        assert result["status"] == "ok"
        mock_model.generate_custom_voice.assert_called_once_with(
            text="wow",
            speaker="Vivian",
            language="english",
            instruct="Very excited",
        )

    def test_unload(self):
        b = self._make()
        b._model = MagicMock()
        b.unload()
        assert b._model is None

    def test_execute_forwards_gen_kwargs(self, tmp_path):
        b = self._make()
        mock_model = MagicMock()
        mock_model.sample_rate = 24000
        mock_result = MagicMock()
        mock_result.audio = [0.1]
        mock_model.generate.return_value = [mock_result]
        b._model = mock_model

        out = tmp_path / "out.wav"
        b.execute({
            "text": "hi",
            "result_path": str(out),
            "mode": "clone",
            "ref_audio": "s.wav",
            "temperature": 0.7,
            "top_k": 50,
        })
        call_kw = mock_model.generate.call_args[1]
        assert call_kw["temperature"] == 0.7
        assert call_kw["top_k"] == 50


class TestMLXAudioTTSBackend:
    def _make(self):
        from indextts_mlx.srv.backends.tts_mlx_audio import MLXAudioTTSBackend

        return MLXAudioTTSBackend()

    def test_load_unload(self):
        b = self._make()
        b.load({})
        assert b._loaded is True
        b.unload()
        assert b._loaded is False

    def test_execute_raises_not_implemented(self):
        b = self._make()
        b.load({})
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            b.execute({"text": "hello"})

    def test_execute_not_loaded_raises(self):
        """Even though it's a stub, execute raises NotImplementedError not RuntimeError."""
        b = self._make()
        b.load({})
        with pytest.raises(NotImplementedError):
            b.execute({})


# ---------------------------------------------------------------------------
# Backend registry
# ---------------------------------------------------------------------------


class TestBackendRegistry:
    def test_all_backends_registered(self):
        from indextts_mlx.srv.backends import BACKENDS

        expected = {"mock", "tts_indextts", "llm", "whisperx", "translation", "tts_mlx_audio", "tts_qwen3"}
        assert set(BACKENDS.keys()) == expected

    def test_all_backends_are_model_backend(self):
        from indextts_mlx.srv.backends import BACKENDS, ModelBackend

        for name, cls in BACKENDS.items():
            assert issubclass(cls, ModelBackend), f"{name} is not a ModelBackend"

    def test_model_type_matches_key(self):
        from indextts_mlx.srv.backends import BACKENDS

        for name, cls in BACKENDS.items():
            assert cls.model_type == name, f"{cls.__name__}.model_type={cls.model_type!r} != {name!r}"
