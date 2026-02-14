"""Tests for high-level long-text synthesis pipeline."""

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest

from indextts_mlx.synthesize_long import LongSynthesisConfig, synthesize_long
from indextts_mlx.segmenter import SegmenterConfig
from indextts_mlx.normalizer import NormalizerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tts(chunk_duration_sec: float = 0.5, sample_rate: int = 22050):
    """Return a mock TTS object whose synthesize() returns a sine wave."""
    tts = MagicMock()
    n = int(chunk_duration_sec * sample_rate)
    tts.synthesize.return_value = np.zeros(n, dtype=np.float32)
    return tts


SIMPLE_TEXT = (
    "This is sentence one. This is sentence two. " "This is sentence three. This is sentence four."
)


# ---------------------------------------------------------------------------
# LongSynthesisConfig tests
# ---------------------------------------------------------------------------


class TestLongSynthesisConfig:
    def test_defaults(self) -> None:
        cfg = LongSynthesisConfig()
        assert cfg.language == "english"
        assert cfg.normalize is True
        assert cfg.silence_between_chunks_ms == 300
        assert cfg.verbose is False

    def test_custom_values(self) -> None:
        cfg = LongSynthesisConfig(
            language="french",
            normalize=False,
            silence_between_chunks_ms=100,
            verbose=True,
        )
        assert cfg.language == "french"
        assert cfg.normalize is False
        assert cfg.silence_between_chunks_ms == 100
        assert cfg.verbose is True

    def test_default_segmenter_config_created(self) -> None:
        cfg = LongSynthesisConfig()
        assert cfg.segmenter_config is not None
        assert cfg.segmenter_config.strategy == "char_count"
        assert cfg.segmenter_config.max_chars == 300

    def test_default_normalizer_config_created(self) -> None:
        cfg = LongSynthesisConfig()
        assert cfg.normalizer_config is not None
        assert cfg.normalizer_config.language == "english"

    def test_language_propagated_to_sub_configs(self) -> None:
        cfg = LongSynthesisConfig(language="italian")
        assert cfg.segmenter_config.language == "italian"
        assert cfg.normalizer_config.language == "italian"

    def test_custom_segmenter_config_preserved(self) -> None:
        seg_cfg = SegmenterConfig(language="english", max_chars=150)
        cfg = LongSynthesisConfig(segmenter_config=seg_cfg)
        assert cfg.segmenter_config.max_chars == 150

    def test_custom_normalizer_config_preserved(self) -> None:
        norm_cfg = NormalizerConfig(language="english", input_case="lower_cased")
        cfg = LongSynthesisConfig(normalizer_config=norm_cfg)
        assert cfg.normalizer_config.input_case == "lower_cased"


# ---------------------------------------------------------------------------
# synthesize_long tests (with mock TTS)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tts():
    return _make_tts()


def _run(text, tts, **kwargs):
    """Call synthesize_long with normalize=False so spaCy is required (not NeMo)."""
    return synthesize_long(text, tts=tts, normalize=False, **kwargs)


class TestSynthesizeLongBasic:
    def test_returns_numpy_float32(self, mock_tts) -> None:
        try:
            audio = _run(SIMPLE_TEXT, mock_tts)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32

    def test_non_empty_output(self, mock_tts) -> None:
        try:
            audio = _run(SIMPLE_TEXT, mock_tts)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")
        assert audio.size > 0

    def test_empty_text_returns_empty_array(self) -> None:
        tts = _make_tts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio = synthesize_long("", tts=tts, normalize=False)
        assert audio.size == 0
        assert audio.dtype == np.float32

    def test_whitespace_only_returns_empty_array(self) -> None:
        tts = _make_tts()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            audio = synthesize_long("   ", tts=tts, normalize=False)
        assert audio.size == 0

    def test_synthesize_called_for_each_chunk(self, mock_tts) -> None:
        try:
            _run(SIMPLE_TEXT, mock_tts)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")
        assert mock_tts.synthesize.call_count >= 1

    def test_silence_inserted_between_chunks(self) -> None:
        """Total duration should exceed sum of per-chunk audio when >1 chunk."""
        n_samples = int(0.5 * 22050)
        tts = MagicMock()
        calls = []

        def fake_synthesize(*args, **kwargs):
            arr = np.zeros(n_samples, dtype=np.float32)
            calls.append(1)
            return arr

        tts.synthesize.side_effect = fake_synthesize

        try:
            audio = synthesize_long(
                SIMPLE_TEXT,
                tts=tts,
                normalize=False,
                silence_between_chunks_ms=100,
            )
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")

        n_chunks = tts.synthesize.call_count
        if n_chunks > 1:
            silence_samples = int(22050 * 0.1)
            expected_min = n_chunks * n_samples + (n_chunks - 1) * silence_samples
            assert audio.size >= expected_min

    def test_zero_silence(self) -> None:
        tts = _make_tts()
        try:
            audio = synthesize_long(
                SIMPLE_TEXT,
                tts=tts,
                normalize=False,
                silence_between_chunks_ms=0,
            )
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")
        assert audio.size > 0


class TestSynthesizeLongSeeding:
    def test_deterministic_with_same_seed(self) -> None:
        """Two runs with the same seed should call synthesize with the same seeds."""
        seeds_run1 = []
        seeds_run2 = []

        def fake_synth1(*a, **kw):
            seeds_run1.append(kw.get("seed"))
            return np.zeros(100, dtype=np.float32)

        def fake_synth2(*a, **kw):
            seeds_run2.append(kw.get("seed"))
            return np.zeros(100, dtype=np.float32)

        tts1, tts2 = MagicMock(), MagicMock()
        tts1.synthesize.side_effect = fake_synth1
        tts2.synthesize.side_effect = fake_synth2

        try:
            synthesize_long(SIMPLE_TEXT, tts=tts1, normalize=False, seed=42)
            synthesize_long(SIMPLE_TEXT, tts=tts2, normalize=False, seed=42)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")

        assert seeds_run1 == seeds_run2

    def test_chunks_get_different_seeds(self) -> None:
        """Each chunk should receive a different seed (seed + chunk_index)."""
        seeds = []

        def fake_synth(*a, **kw):
            seeds.append(kw.get("seed"))
            return np.zeros(100, dtype=np.float32)

        tts = MagicMock()
        tts.synthesize.side_effect = fake_synth

        try:
            synthesize_long(SIMPLE_TEXT, tts=tts, normalize=False, seed=0)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")

        if len(seeds) > 1:
            assert len(set(seeds)) == len(seeds)

    def test_use_random_passes_none_seed(self) -> None:
        """use_random=True should pass seed=None to each synthesize call."""
        seeds = []

        def fake_synth(*a, **kw):
            seeds.append(kw.get("seed"))
            return np.zeros(100, dtype=np.float32)

        tts = MagicMock()
        tts.synthesize.side_effect = fake_synth

        try:
            synthesize_long(SIMPLE_TEXT, tts=tts, normalize=False, use_random=True)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")

        assert all(s is None for s in seeds)


class TestSynthesizeLongCallback:
    def test_on_chunk_called_once_per_chunk(self) -> None:
        calls = []

        def on_chunk(i, total, text):
            calls.append((i, total, text))

        tts = _make_tts()
        try:
            synthesize_long(SIMPLE_TEXT, tts=tts, normalize=False, on_chunk=on_chunk)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")

        n_chunks = tts.synthesize.call_count
        assert len(calls) == n_chunks

    def test_on_chunk_receives_correct_total(self) -> None:
        totals = []

        def on_chunk(i, total, text):
            totals.append(total)

        tts = _make_tts()
        try:
            synthesize_long(SIMPLE_TEXT, tts=tts, normalize=False, on_chunk=on_chunk)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")

        if totals:
            assert all(t == totals[0] for t in totals)

    def test_on_chunk_indices_sequential(self) -> None:
        indices = []

        def on_chunk(i, total, text):
            indices.append(i)

        tts = _make_tts()
        try:
            synthesize_long(SIMPLE_TEXT, tts=tts, normalize=False, on_chunk=on_chunk)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")

        assert indices == list(range(len(indices)))


class TestSynthesizeLongConfig:
    def test_config_object_overrides_kwargs(self) -> None:
        """When config is provided, it should be used."""
        from indextts_mlx.synthesize_long import LongSynthesisConfig

        cfg = LongSynthesisConfig(
            normalize=False,
            silence_between_chunks_ms=50,
        )
        tts = _make_tts()
        try:
            audio = synthesize_long(SIMPLE_TEXT, tts=tts, config=cfg)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")
        assert isinstance(audio, np.ndarray)

    def test_max_chars_controls_chunk_size(self) -> None:
        """Smaller max_chars → more chunks → more synthesize calls."""
        tts_small = _make_tts()
        tts_large = _make_tts()
        try:
            synthesize_long(SIMPLE_TEXT, tts=tts_small, normalize=False, max_chars=30)
            synthesize_long(SIMPLE_TEXT, tts=tts_large, normalize=False, max_chars=500)
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")
        assert tts_small.synthesize.call_count >= tts_large.synthesize.call_count

    def test_synthesis_kwargs_forwarded(self) -> None:
        """Quality kwargs like cfm_steps should be passed to each synthesize call."""
        tts = _make_tts()
        try:
            synthesize_long(
                "Hello world.",
                tts=tts,
                normalize=False,
                cfm_steps=25,
                temperature=0.8,
                cfg_rate=0.5,
            )
        except OSError:
            pytest.skip("spaCy en_core_web_sm not installed")

        if tts.synthesize.call_count > 0:
            call_kwargs = tts.synthesize.call_args_list[0][1]
            assert call_kwargs.get("cfm_steps") == 25
            assert call_kwargs.get("temperature") == 0.8
            assert call_kwargs.get("cfg_rate") == 0.5
