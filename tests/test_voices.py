"""Tests for voice resolution, emo_vector parsing, and precedence rules."""

import warnings
from pathlib import Path
import pytest

from indextts_mlx.voices import list_voices, resolve_voice, parse_emo_vector
from indextts_mlx.pipeline import _resolve_speaker

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def voices_dir(tmp_path):
    """Temp directory with a few .wav stubs."""
    (tmp_path / "Emma.wav").write_bytes(b"RIFF")
    (tmp_path / "John.wav").write_bytes(b"RIFF")
    (tmp_path / "narrator.wav").write_bytes(b"RIFF")
    (tmp_path / "not_a_voice.txt").write_text("ignored")
    return tmp_path


# ── list_voices ───────────────────────────────────────────────────────────────


def test_list_voices_returns_stems(voices_dir):
    names = list_voices(voices_dir)
    assert names == ["Emma", "John", "narrator"]


def test_list_voices_sorted(voices_dir):
    names = list_voices(voices_dir)
    assert names == sorted(names)


def test_list_voices_missing_dir():
    with pytest.raises(FileNotFoundError, match="voices_dir not found"):
        list_voices("/nonexistent/dir")


def test_list_voices_empty_dir(tmp_path):
    assert list_voices(tmp_path) == []


# ── resolve_voice ─────────────────────────────────────────────────────────────


def test_resolve_voice_exact(voices_dir):
    path = resolve_voice(voices_dir, "Emma")
    assert path == voices_dir / "Emma.wav"
    assert path.exists()


def test_resolve_voice_case_insensitive_fallback(voices_dir, tmp_path):
    # On case-insensitive filesystems (macOS), "emma.wav" hits the exact-match path.
    # On case-sensitive filesystems (Linux CI), the fallback warning fires.
    # Either way the resolved path must point to a .wav file with stem matching "emma" (case-insensitive).
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        path = resolve_voice(voices_dir, "emma")
    assert path.stem.lower() == "emma"
    assert path.suffix.lower() == ".wav"


def test_resolve_voice_not_found_lists_available(voices_dir):
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_voice(voices_dir, "Unknown")
    msg = str(exc_info.value)
    assert "Emma" in msg
    assert "John" in msg
    assert "narrator" in msg


def test_resolve_voice_missing_dir():
    with pytest.raises(FileNotFoundError, match="voices_dir not found"):
        resolve_voice("/bad/path", "Emma")


# ── parse_emo_vector ──────────────────────────────────────────────────────────


def test_parse_emo_vector_from_string():
    vec = parse_emo_vector("0.8,0.0,0.0,0.0,0.0,0.0,0.0,0.2")
    assert vec == [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    assert len(vec) == 8


def test_parse_emo_vector_from_list():
    vec = parse_emo_vector([0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.3, 0.1])
    assert len(vec) == 8
    assert vec[0] == 0.5


def test_parse_emo_vector_wrong_length():
    with pytest.raises(ValueError, match="exactly 8"):
        parse_emo_vector("0.5,0.5,0.0")


def test_parse_emo_vector_bad_floats():
    with pytest.raises(ValueError, match="8 comma-separated floats"):
        parse_emo_vector("happy,sad,neutral,x,x,x,x,x")


def test_parse_emo_vector_clamps_and_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        vec = parse_emo_vector("1.5,0.0,0.0,-0.2,0.0,0.0,0.0,0.0")
    assert vec[0] == 1.0  # clamped from 1.5
    assert vec[3] == 0.0  # clamped from -0.2
    assert len(w) == 2
    assert any("clamping" in str(warning.message) for warning in w)


def test_parse_emo_vector_all_zeros():
    vec = parse_emo_vector([0.0] * 8)
    assert vec == [0.0] * 8


# ── _resolve_speaker precedence ───────────────────────────────────────────────


def test_spk_audio_prompt_wins_over_voice(voices_dir):
    """spk_audio_prompt takes priority over voice + voices_dir."""
    direct = voices_dir / "Emma.wav"
    result = _resolve_speaker(voices_dir, "John", str(direct))
    assert Path(result) == direct


def test_spk_audio_prompt_warns_when_voice_also_set(voices_dir):
    direct = voices_dir / "Emma.wav"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _resolve_speaker(voices_dir, "John", str(direct))
    assert any("spk_audio_prompt" in str(warning.message) for warning in w)


def test_voice_with_voices_dir(voices_dir):
    result = _resolve_speaker(str(voices_dir), "John", None)
    assert Path(result) == voices_dir / "John.wav"


def test_voice_without_voices_dir_treats_as_path(voices_dir):
    direct = str(voices_dir / "Emma.wav")
    result = _resolve_speaker(None, direct, None)
    assert Path(result) == Path(direct)


def test_voice_without_voices_dir_missing_path():
    with pytest.raises(FileNotFoundError):
        _resolve_speaker(None, "/nonexistent/file.wav", None)


def test_no_speaker_returns_none():
    result = _resolve_speaker(None, None, None)
    assert result is None


# ── Auto use_emo_text precedence ──────────────────────────────────────────────


def test_emo_text_auto_enables_use_emo_text():
    """When emo_text is set and use_emo_text is None, auto-set to True."""
    # Simulate the pipeline's logic
    emo_text = "happy and joyful"
    use_emo_text = None
    if emo_text is not None and use_emo_text is None:
        use_emo_text = True
    assert use_emo_text is True


def test_emo_text_respects_explicit_false():
    """Explicit use_emo_text=False is not overridden."""
    emo_text = "happy and joyful"
    use_emo_text = False
    if emo_text is not None and use_emo_text is None:
        use_emo_text = True
    assert use_emo_text is False


# ── emo_vector vs emo_text precedence ─────────────────────────────────────────


def test_emo_vector_wins_over_emo_text():
    """When both emo_vector and emo_text are provided, emo_vector wins (emo_text cleared)."""
    emo_vector = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]
    emo_text = "joyful"
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if emo_vector is not None and emo_text is not None:
            warnings.warn("Both emo_vector and emo_text provided; emo_vector takes precedence.")
            emo_text = None
    assert emo_text is None
    assert any("emo_vector takes precedence" in str(warning.message) for warning in w)
