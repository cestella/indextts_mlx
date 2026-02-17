"""Voice directory resolution helpers."""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import List, Optional


def list_voices(voices_dir: str | Path) -> List[str]:
    """Return sorted list of voice names available in voices_dir.

    Includes both plain voices (.wav files) and meta voices (directories
    containing emotion-labelled subdirectories of .wav clips).
    """
    d = Path(voices_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"voices_dir not found: {d}")
    names = []
    for p in d.iterdir():
        if p.suffix.lower() == ".wav":
            names.append(p.stem)
        elif p.is_dir():
            names.append(p.name)
    return sorted(names)


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


def is_meta_voice(voices_dir: str | Path, voice: str) -> bool:
    """Return True if voice resolves to a directory (a meta voice).

    A meta voice is a directory (or symlink to a directory) whose subdirectories
    are named after emotion labels and contain .wav reference clips.
    """
    return (Path(voices_dir) / voice).is_dir()


def resolve_meta_voice(
    voices_dir: str | Path,
    voice: str,
    emotion_label: Optional[str] = None,
    fallback: str = "neutral",
) -> Path:
    """Pick a random .wav from the emotion subdirectory of a meta voice.

    Looks in ``voices_dir/voice/emotion_label/*.wav``.  If the requested
    emotion subdirectory is missing or empty, falls back to ``fallback``
    (default ``"neutral"``).

    Raises FileNotFoundError if no .wav files can be found in either the
    requested emotion directory or the fallback.
    """
    voice_dir = Path(voices_dir) / voice
    label = emotion_label or fallback

    def _wavs(lbl: str) -> list[Path]:
        sub = voice_dir / lbl
        if sub.is_dir():
            return [p for p in sub.iterdir() if p.suffix.lower() == ".wav"]
        return []

    wavs = _wavs(label)
    if not wavs:
        if label != fallback:
            warnings.warn(
                f"Meta voice '{voice}': no .wav files in '{label}/'; "
                f"falling back to '{fallback}'.",
                stacklevel=2,
            )
            wavs = _wavs(fallback)
    if not wavs:
        raise FileNotFoundError(
            f"Meta voice '{voice}': no .wav files found in '{label}/' or "
            f"'{fallback}/' under {voice_dir}"
        )
    return random.choice(wavs)


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
