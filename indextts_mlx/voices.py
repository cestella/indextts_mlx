"""Voice directory resolution helpers."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional


def list_voices(voices_dir: str | Path) -> List[str]:
    """Return sorted list of voice names (stems) available in voices_dir.

    A voice is any .wav file directly under voices_dir.
    """
    d = Path(voices_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"voices_dir not found: {d}")
    return sorted(p.stem for p in d.iterdir() if p.suffix.lower() == ".wav")


def resolve_voice(voices_dir: str | Path, voice: str) -> Path:
    """Resolve voice name to a .wav path under voices_dir.

    Case-sensitive match first; falls back to case-insensitive if exactly one
    case-insensitive match exists.

    Raises FileNotFoundError with a helpful listing if not found.
    """
    d = Path(voices_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"voices_dir not found: {d}")

    # Case-sensitive
    exact = d / f"{voice}.wav"
    if exact.exists():
        return exact

    # Case-insensitive fallback
    lower = voice.lower()
    matches = [p for p in d.iterdir() if p.suffix.lower() == ".wav" and p.stem.lower() == lower]
    if len(matches) == 1:
        warnings.warn(
            f"Voice '{voice}' matched case-insensitively to '{matches[0].stem}'. "
            "Use the exact filename stem to silence this warning.",
            stacklevel=3,
        )
        return matches[0]
    if len(matches) > 1:
        stems = [p.stem for p in matches]
        raise FileNotFoundError(
            f"Voice '{voice}' is ambiguous (case-insensitive): {stems}. "
            "Use the exact filename stem."
        )

    available = list_voices(d)
    avail_str = ", ".join(available) if available else "(none)"
    raise FileNotFoundError(f"Voice '{voice}' not found in {d}. Available voices: {avail_str}")


def parse_emo_vector(raw: str | list) -> List[float]:
    """Parse and validate an emotion vector.

    Accepts a comma-separated string or a list of 8 floats.
    Order: [happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]

    Clamps out-of-range values to [0, 1] with a warning.
    """
    EMO_NAMES = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]

    if isinstance(raw, str):
        parts = [s.strip() for s in raw.split(",")]
        try:
            values = [float(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"emo_vector must be 8 comma-separated floats: {e}") from e
    else:
        values = [float(v) for v in raw]

    if len(values) != 8:
        raise ValueError(f"emo_vector must have exactly 8 values, got {len(values)}")

    clamped = []
    for i, v in enumerate(values):
        if v < 0.0 or v > 1.0:
            warnings.warn(
                f"emo_vector[{i}] ({EMO_NAMES[i]}={v:.4f}) is outside [0, 1]; clamping.",
                stacklevel=3,
            )
            v = max(0.0, min(1.0, v))
        clamped.append(v)

    return clamped
