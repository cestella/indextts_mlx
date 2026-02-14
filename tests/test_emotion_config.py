"""Unit tests for indextts_mlx.emotion_config."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from indextts_mlx.emotion_config import (
    EmotionConfig,
    EmotionDrift,
    EmotionPreset,
    EmotionResolver,
    load_emotion_config,
    resolve_emotion_config_path,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_CONFIG = {
    "version": 1,
    "vector_order": ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"],
    "emotions": {
        "neutral": {
            "base": {"emo_vector": [0, 0, 0, 0, 0, 0, 0, 0.3], "emo_alpha": 0.1},
            "drift": {"vector_sigma": [0.02, 0, 0, 0, 0, 0, 0.01, 0.04], "alpha_sigma": 0.02, "smoothing": 0.85},
        },
        "melancholic": {
            "base": {"emo_vector": [0, 0.05, 0.15, 0.05, 0, 0.45, 0, 0.55], "emo_alpha": 0.3},
            "drift": {"vector_sigma": [0, 0.01, 0.03, 0.01, 0, 0.04, 0, 0.03], "alpha_sigma": 0.02, "smoothing": 0.85},
        },
        "joyful": {
            "base": {"emo_vector": [0.42, 0.05, 0, 0, 0, 0.22, 0, 0.58], "emo_alpha": 0.3},
        },
    },
}


def _write_config(tmp_path: Path, data: dict | None = None) -> Path:
    p = tmp_path / "emotions.json"
    p.write_text(json.dumps(data or _MINIMAL_CONFIG), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# load_emotion_config
# ---------------------------------------------------------------------------


class TestLoadEmotionConfig:
    def test_loads_version(self, tmp_path):
        p = _write_config(tmp_path)
        cfg = load_emotion_config(p)
        assert cfg.version == 1

    def test_loads_vector_order(self, tmp_path):
        p = _write_config(tmp_path)
        cfg = load_emotion_config(p)
        assert len(cfg.vector_order) == 8
        assert cfg.vector_order[0] == "happy"

    def test_loads_emotion_labels(self, tmp_path):
        p = _write_config(tmp_path)
        cfg = load_emotion_config(p)
        assert "neutral" in cfg.emotions
        assert "melancholic" in cfg.emotions

    def test_base_emo_vector_correct(self, tmp_path):
        p = _write_config(tmp_path)
        cfg = load_emotion_config(p)
        assert cfg.emotions["neutral"].base.emo_vector == [0, 0, 0, 0, 0, 0, 0, 0.3]

    def test_base_emo_alpha_correct(self, tmp_path):
        p = _write_config(tmp_path)
        cfg = load_emotion_config(p)
        assert cfg.emotions["neutral"].base.emo_alpha == pytest.approx(0.1)

    def test_drift_loaded(self, tmp_path):
        p = _write_config(tmp_path)
        cfg = load_emotion_config(p)
        drift = cfg.emotions["neutral"].drift
        assert drift is not None
        assert drift.smoothing == pytest.approx(0.85)
        assert len(drift.vector_sigma) == 8

    def test_drift_optional(self, tmp_path):
        p = _write_config(tmp_path)
        cfg = load_emotion_config(p)
        # joyful in minimal config has no drift
        assert cfg.emotions["joyful"].drift is None

    def test_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_emotion_config(tmp_path / "nonexistent.json")

    def test_raises_on_missing_version(self, tmp_path):
        bad = dict(_MINIMAL_CONFIG)
        del bad["version"]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="version"):
            load_emotion_config(p)

    def test_raises_on_bad_vector_order(self, tmp_path):
        bad = dict(_MINIMAL_CONFIG)
        bad["vector_order"] = ["only", "six", "items", "here", "not", "eight"]
        p = tmp_path / "bad.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="vector_order"):
            load_emotion_config(p)


# ---------------------------------------------------------------------------
# resolve_emotion_config_path
# ---------------------------------------------------------------------------


class TestResolveEmotionConfigPath:
    def test_explicit_wins(self, tmp_path):
        explicit = tmp_path / "custom.json"
        explicit.touch()
        result = resolve_emotion_config_path(explicit, tmp_path)
        assert result == explicit

    def test_auto_discover_from_voices_dir(self, tmp_path):
        (tmp_path / "emotions.json").touch()
        result = resolve_emotion_config_path(None, tmp_path)
        assert result == tmp_path / "emotions.json"

    def test_returns_none_when_nothing_found(self, tmp_path):
        result = resolve_emotion_config_path(None, tmp_path)
        assert result is None

    def test_returns_none_when_no_voices_dir(self):
        result = resolve_emotion_config_path(None, None)
        assert result is None


# ---------------------------------------------------------------------------
# EmotionResolver — no drift
# ---------------------------------------------------------------------------


class TestEmotionResolverNoDrift:
    @pytest.fixture
    def resolver(self, tmp_path):
        p = _write_config(tmp_path)
        return EmotionResolver.from_path(p, enable_drift=False)

    def test_neutral_default_when_none_label(self, resolver):
        vec, alpha = resolver.resolve(None)
        assert vec == [0, 0, 0, 0, 0, 0, 0, 0.3]
        assert alpha == pytest.approx(0.1)

    def test_neutral_default_when_unknown_label(self, resolver):
        vec, alpha = resolver.resolve("unknown_label")
        assert vec == [0, 0, 0, 0, 0, 0, 0, 0.3]
        assert alpha == pytest.approx(0.1)

    def test_melancholic_base_returned(self, resolver):
        vec, alpha = resolver.resolve("melancholic")
        assert vec == [0, 0.05, 0.15, 0.05, 0, 0.45, 0, 0.55]
        assert alpha == pytest.approx(0.3)

    def test_override_vector_takes_precedence(self, resolver):
        override = [0.1] * 8
        vec, alpha = resolver.resolve("melancholic", override_vector=override)
        assert vec == override
        # alpha still comes from preset since not overridden
        assert alpha == pytest.approx(0.3)

    def test_override_alpha_takes_precedence(self, resolver):
        vec, alpha = resolver.resolve("melancholic", override_alpha=0.99)
        assert alpha == pytest.approx(0.99)
        assert vec == [0, 0.05, 0.15, 0.05, 0, 0.45, 0, 0.55]

    def test_both_overrides_bypass_preset(self, resolver):
        override_v = [0.5] * 8
        vec, alpha = resolver.resolve("melancholic", override_vector=override_v, override_alpha=0.77)
        assert vec == override_v
        assert alpha == pytest.approx(0.77)

    def test_no_drift_returns_identical_base_each_call(self, resolver):
        results = [resolver.resolve("neutral") for _ in range(10)]
        vecs = [r[0] for r in results]
        alphas = [r[1] for r in results]
        assert all(v == vecs[0] for v in vecs)
        assert all(a == pytest.approx(alphas[0]) for a in alphas)


# ---------------------------------------------------------------------------
# EmotionResolver — with drift
# ---------------------------------------------------------------------------


class TestEmotionResolverDrift:
    @pytest.fixture
    def resolver(self, tmp_path):
        p = _write_config(tmp_path)
        return EmotionResolver.from_path(p, enable_drift=True, seed=42)

    def test_drift_varies_across_calls(self, resolver):
        results = [resolver.resolve("neutral") for _ in range(20)]
        alphas = [r[1] for r in results]
        # After EMA settling, values should vary slightly
        assert not all(a == pytest.approx(alphas[0], abs=1e-9) for a in alphas)

    def test_drift_stays_bounded(self, tmp_path):
        # Run many segments and verify output never exceeds base ± 2*sigma
        p = _write_config(tmp_path)
        resolver = EmotionResolver.from_path(p, enable_drift=True, seed=0)
        cfg = load_emotion_config(p)
        base_v = np.array(cfg.emotions["neutral"].base.emo_vector)
        base_a = cfg.emotions["neutral"].base.emo_alpha
        sigma_v = np.array(cfg.emotions["neutral"].drift.vector_sigma)
        sigma_a = cfg.emotions["neutral"].drift.alpha_sigma

        for _ in range(500):
            vec, alpha = resolver.resolve("neutral")
            arr = np.array(vec)
            # Each dimension must be within base ± 2*sigma (and >= 0)
            assert np.all(arr >= 0.0), f"Vector went negative: {arr}"
            assert np.all(arr <= base_v + 2 * sigma_v + 1e-9), (
                f"Vector exceeded upper bound: {arr}"
            )
            assert alpha >= 0.0
            assert alpha <= base_a + 2 * sigma_a + 1e-9

    def test_drift_is_reproducible_with_same_seed(self, tmp_path):
        p = _write_config(tmp_path)
        r1 = EmotionResolver.from_path(p, enable_drift=True, seed=7)
        r2 = EmotionResolver.from_path(p, enable_drift=True, seed=7)
        for _ in range(10):
            v1, a1 = r1.resolve("neutral")
            v2, a2 = r2.resolve("neutral")
            assert v1 == pytest.approx(v2)
            assert a1 == pytest.approx(a2)

    def test_drift_independent_per_label(self, tmp_path):
        p = _write_config(tmp_path)
        resolver = EmotionResolver.from_path(p, enable_drift=True, seed=99)
        # Neutral and melancholic should drift independently
        neutral_alphas = [resolver.resolve("neutral")[1] for _ in range(5)]
        mel_alphas = [resolver.resolve("melancholic")[1] for _ in range(5)]
        # They should not be identical sequences
        assert neutral_alphas != mel_alphas

    def test_override_bypasses_drift(self, tmp_path):
        p = _write_config(tmp_path)
        resolver = EmotionResolver.from_path(p, enable_drift=True, seed=1)
        override_v = [0.9] * 8
        for _ in range(10):
            vec, _ = resolver.resolve("neutral", override_vector=override_v)
            assert vec == override_v

    def test_no_drift_when_preset_has_no_drift_block(self, tmp_path):
        # joyful has no drift block — should return base unchanged even when enable_drift=True
        p = _write_config(tmp_path)
        resolver = EmotionResolver.from_path(p, enable_drift=True, seed=42)
        cfg = load_emotion_config(p)
        base_v = cfg.emotions["joyful"].base.emo_vector
        base_a = cfg.emotions["joyful"].base.emo_alpha
        results = [resolver.resolve("joyful") for _ in range(10)]
        for vec, alpha in results:
            assert vec == base_v
            assert alpha == pytest.approx(base_a)


# ---------------------------------------------------------------------------
# EmotionResolver.from_voices_dir
# ---------------------------------------------------------------------------


class TestEmotionResolverFromVoicesDir:
    def test_returns_resolver_when_emotions_json_present(self, tmp_path):
        _write_config(tmp_path)
        resolver = EmotionResolver.from_voices_dir(voices_dir=tmp_path)
        assert resolver is not None

    def test_returns_none_when_no_config_found(self, tmp_path):
        resolver = EmotionResolver.from_voices_dir(voices_dir=tmp_path)
        assert resolver is None

    def test_explicit_path_overrides_voices_dir(self, tmp_path):
        explicit = tmp_path / "custom.json"
        explicit.write_text(json.dumps(_MINIMAL_CONFIG))
        resolver = EmotionResolver.from_voices_dir(
            voices_dir=None, explicit_path=explicit
        )
        assert resolver is not None
        vec, _ = resolver.resolve("joyful")
        assert vec[0] == pytest.approx(0.42)

    def test_warns_and_returns_none_on_bad_config(self, tmp_path):
        bad = tmp_path / "emotions.json"
        bad.write_text('{"version": 1}')  # missing vector_order and emotions
        with pytest.warns(UserWarning, match="Could not load"):
            resolver = EmotionResolver.from_voices_dir(voices_dir=tmp_path)
        assert resolver is None
