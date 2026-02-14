"""Emotion preset configuration with optional bounded drift.

Loads an emotion_config JSON (schemas/emotion_config.schema.json) and
resolves per-segment (emo_vector, emo_alpha) from a label string.

When drift is enabled, each emotion label maintains its own EMA state
so consecutive segments of the same emotion drift slowly and smoothly
within a bounded band around the base vector.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class EmotionBase:
    emo_vector: List[float]  # 8 floats
    emo_alpha: float  # 0..1


@dataclass
class EmotionDrift:
    vector_sigma: List[float]  # per-dimension sigma (8 floats)
    alpha_sigma: float
    smoothing: float  # EMA factor for the existing smoothed value (0..1)


@dataclass
class EmotionPreset:
    base: EmotionBase
    drift: Optional[EmotionDrift] = None


@dataclass
class EmotionConfig:
    version: int
    vector_order: List[str]
    emotions: Dict[str, EmotionPreset]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_emotion_config(path: Union[str, Path]) -> EmotionConfig:
    """Load and parse an emotion config JSON file.

    Raises:
        FileNotFoundError: if path does not exist.
        ValueError: if the JSON is missing required fields.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Emotion config not found: {p}")

    with p.open(encoding="utf-8") as f:
        raw = json.load(f)

    version = raw.get("version")
    if version is None:
        raise ValueError("emotion config missing 'version'")
    vector_order = raw.get("vector_order")
    if not vector_order or len(vector_order) != 8:
        raise ValueError("emotion config 'vector_order' must be a list of 8 strings")
    emotions_raw = raw.get("emotions")
    if not emotions_raw:
        raise ValueError("emotion config missing 'emotions'")

    emotions: Dict[str, EmotionPreset] = {}
    for label, entry in emotions_raw.items():
        base_raw = entry.get("base")
        if not base_raw:
            raise ValueError(f"emotion '{label}' missing 'base'")
        base = EmotionBase(
            emo_vector=[float(x) for x in base_raw["emo_vector"]],
            emo_alpha=float(base_raw["emo_alpha"]),
        )
        drift = None
        drift_raw = entry.get("drift")
        if drift_raw:
            drift = EmotionDrift(
                vector_sigma=[float(x) for x in drift_raw["vector_sigma"]],
                alpha_sigma=float(drift_raw["alpha_sigma"]),
                smoothing=float(drift_raw["smoothing"]),
            )
        emotions[label] = EmotionPreset(base=base, drift=drift)

    return EmotionConfig(version=version, vector_order=vector_order, emotions=emotions)


def resolve_emotion_config_path(
    explicit: Optional[Union[str, Path]],
    voices_dir: Optional[Union[str, Path]],
) -> Optional[Path]:
    """Return emotion config path: explicit > voices_dir/emotions.json > None."""
    if explicit is not None:
        return Path(explicit)
    if voices_dir is not None:
        candidate = Path(voices_dir) / "emotions.json"
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Per-label drift state
# ---------------------------------------------------------------------------


@dataclass
class _DriftState:
    vector: np.ndarray  # smoothed current vector (float32, shape (8,))
    alpha: float  # smoothed current alpha


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class EmotionResolver:
    """Resolves a string emotion label to (emo_vector, emo_alpha).

    When ``enable_drift=True``, each label maintains an EMA-smoothed state
    that drifts within ±2*sigma of the base, adding subtle variation across
    consecutive segments of the same emotion.

    Per-segment explicit overrides (emo_vector or emo_alpha) always take
    precedence over the preset, bypassing drift entirely for that field.

    Args:
        config: Loaded EmotionConfig.
        enable_drift: Whether to apply per-segment drift.
        seed: Optional RNG seed for reproducible drift.
    """

    def __init__(
        self,
        config: EmotionConfig,
        enable_drift: bool = False,
        seed: Optional[int] = None,
    ):
        self.config = config
        self.enable_drift = enable_drift
        self._rng = np.random.default_rng(seed)
        self._states: Dict[str, _DriftState] = {}

    # ── public ───────────────────────────────────────────────────────────────

    def resolve(
        self,
        label: Optional[str],
        override_vector: Optional[List[float]] = None,
        override_alpha: Optional[float] = None,
    ) -> tuple[List[float], float]:
        """Return (emo_vector, emo_alpha) for the given label.

        Args:
            label: Emotion label string (e.g. "melancholic"). Defaults to
                   "neutral" if None or not found in config.
            override_vector: Explicit emo_vector from the segment record.
                             Takes precedence over the preset.
            override_alpha: Explicit emo_alpha from the segment record.
                            Takes precedence over the preset.

        Returns:
            Tuple of (emo_vector: List[float], emo_alpha: float).
        """
        # Resolve preset (fall back to neutral, then to first available)
        effective_label = label if (label and label in self.config.emotions) else "neutral"
        if effective_label not in self.config.emotions:
            effective_label = next(iter(self.config.emotions))

        preset = self.config.emotions[effective_label]
        base_v = np.array(preset.base.emo_vector, dtype=np.float64)
        base_a = preset.base.emo_alpha

        if self.enable_drift and preset.drift is not None:
            vec, alpha = self._apply_drift(effective_label, preset, base_v, base_a)
        else:
            vec = base_v.tolist()
            alpha = base_a

        # Per-segment explicit overrides always win
        final_vector = override_vector if override_vector is not None else vec
        final_alpha = override_alpha if override_alpha is not None else alpha

        return final_vector, final_alpha

    # ── drift internals ──────────────────────────────────────────────────────

    def _get_or_init_state(self, label: str, base_v: np.ndarray, base_a: float) -> _DriftState:
        if label not in self._states:
            self._states[label] = _DriftState(
                vector=base_v.copy(),
                alpha=base_a,
            )
        return self._states[label]

    def _apply_drift(
        self,
        label: str,
        preset: EmotionPreset,
        base_v: np.ndarray,
        base_a: float,
    ) -> tuple[List[float], float]:
        drift = preset.drift
        sigma_v = np.array(drift.vector_sigma, dtype=np.float64)
        sigma_a = drift.alpha_sigma
        s = drift.smoothing

        state = self._get_or_init_state(label, base_v, base_a)

        # Sample noise, clamp to ±2*sigma
        noise_v = self._rng.normal(0.0, np.where(sigma_v > 0, sigma_v, 1e-9))
        noise_v = np.clip(noise_v, -2.0 * sigma_v, 2.0 * sigma_v)

        noise_a = self._rng.normal(0.0, sigma_a) if sigma_a > 0 else 0.0
        noise_a = float(np.clip(noise_a, -2.0 * sigma_a, 2.0 * sigma_a))

        # Target = base + noise, clamped >= 0
        target_v = np.clip(base_v + noise_v, 0.0, None)
        target_a = max(0.0, base_a + noise_a)

        # EMA smooth
        state.vector = s * state.vector + (1.0 - s) * target_v
        state.alpha = s * state.alpha + (1.0 - s) * target_a

        return state.vector.tolist(), float(state.alpha)

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_path(
        cls,
        path: Union[str, Path],
        enable_drift: bool = False,
        seed: Optional[int] = None,
    ) -> "EmotionResolver":
        """Load config from file and return a ready resolver."""
        config = load_emotion_config(path)
        return cls(config, enable_drift=enable_drift, seed=seed)

    @classmethod
    def from_voices_dir(
        cls,
        voices_dir: Optional[Union[str, Path]],
        explicit_path: Optional[Union[str, Path]] = None,
        enable_drift: bool = False,
        seed: Optional[int] = None,
    ) -> Optional["EmotionResolver"]:
        """Attempt to build a resolver, returning None if no config is found."""
        resolved = resolve_emotion_config_path(explicit_path, voices_dir)
        if resolved is None:
            return None
        try:
            return cls.from_path(resolved, enable_drift=enable_drift, seed=seed)
        except (FileNotFoundError, ValueError) as e:
            warnings.warn(f"Could not load emotion config from {resolved}: {e}")
            return None
