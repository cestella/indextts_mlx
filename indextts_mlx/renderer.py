"""JSONL chapter renderer for audiobook production.

Reads a JSONL file (one JSON object per line, schema: schemas/segment.schema.json),
synthesizes each segment, inserts silence pauses, concatenates, and writes a WAV.
"""

from __future__ import annotations

import json
import hashlib
import warnings
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .pipeline import IndexTTS2, OUTPUT_SAMPLE_RATE
from .normalizer import Normalizer, NormalizerConfig
from .synthesize_long import synthesize_long
from .emotion_config import EmotionResolver
from .voices import parse_emo_vector

# Sentinel so we can distinguish "not passed" from None
_UNSET = object()

# Silence padding value
_SILENCE = np.float32(0.0)


def _silence_samples(ms: int, sample_rate: int = OUTPUT_SAMPLE_RATE) -> np.ndarray:
    """Return a float32 array of silence for the given duration."""
    n = int(sample_rate * ms / 1000)
    return np.zeros(n, dtype=np.float32)


def _merge(a: Optional[Any], b: Optional[Any]) -> Optional[Any]:
    """Return b if it is not None, else a (segment overrides default)."""
    return b if b is not None else a


def _validate_segment(record: Dict, lineno: int) -> None:
    """Minimal validation — raises ValueError with line number on failure."""
    if "text" not in record or not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError(f"Line {lineno}: 'text' field is required and must be a non-empty string")
    ev = record.get("emo_vector")
    if ev is not None:
        if not isinstance(ev, list) or len(ev) != 8:
            raise ValueError(f"Line {lineno}: 'emo_vector' must be a list of 8 floats")
    emo_alpha = record.get("emo_alpha")
    if emo_alpha is not None and not (0.0 <= float(emo_alpha) <= 1.0):
        raise ValueError(f"Line {lineno}: 'emo_alpha' must be between 0.0 and 1.0")
    pause_before = record.get("pause_before_ms", 0)
    pause_after = record.get("pause_after_ms", 0)
    if not (0 <= int(pause_before) <= 60000):
        raise ValueError(f"Line {lineno}: 'pause_before_ms' must be 0..60000")
    if not (0 <= int(pause_after) <= 60000):
        raise ValueError(f"Line {lineno}: 'pause_after_ms' must be 0..60000")


def _segment_cache_key(
    model_version: str,
    text: str,
    spk_source: str,
    emotion: float,
    emo_alpha: float,
    emo_vector: Optional[List[float]],
    emo_text: Optional[str],
    seed: Optional[int],
    use_random: bool,
    cfm_steps: int,
    temperature: float,
    cfg_rate: float,
    gpt_temperature: float,
    top_k: int,
    sample_rate: int,
) -> str:
    """Return a stable hex digest for the synthesis parameters (for caching)."""
    parts = [
        model_version,
        text,
        spk_source,
        str(emotion),
        str(emo_alpha),
        str(emo_vector),
        str(emo_text),
        str(seed),
        str(use_random),
        str(cfm_steps),
        str(temperature),
        str(cfg_rate),
        str(gpt_temperature),
        str(top_k),
        str(sample_rate),
    ]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def render_segments_jsonl(
    jsonl_path: Union[str, Path],
    output_path: Union[str, Path],
    tts: Optional[IndexTTS2] = None,
    *,
    # Defaults (overridable per-segment)
    voices_dir: Optional[Union[str, Path]] = None,
    voice: Optional[str] = None,
    spk_audio_prompt: Optional[Union[str, Path]] = None,
    emotion: float = 1.0,
    emo_alpha: float = 0.0,
    emo_vector: Optional[List[float]] = None,
    emo_text: Optional[str] = None,
    use_emo_text: Optional[bool] = None,
    emo_audio_prompt: Optional[Union[str, Path]] = None,
    seed: Optional[int] = None,
    use_random: bool = False,
    cfm_steps: int = 10,
    temperature: float = 1.0,
    max_codes: int = 1500,
    cfg_rate: float = 0.7,
    gpt_temperature: float = 0.8,
    top_k: int = 30,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
    # Long-text pipeline controls (applied per segment)
    normalize: bool = True,
    language: str = "english",
    silence_between_chunks_ms: int = 300,
    crossfade_ms: int = 10,
    # Cache directory (None = no caching)
    cache_dir: Optional[Union[str, Path]] = None,
    # Model version string (for cache key)
    model_version: str = "indextts2-mlx-v1",
    # Emotion preset config (path to emotions.json, or auto-detected from voices_dir)
    emotion_config: Optional[Union[str, Path]] = None,
    # Enable per-segment drift (bounded Gaussian noise + EMA smoothing)
    enable_drift: bool = False,
    # Optional audio file appended after the last segment (e.g. a chapter-end chime)
    end_chime: Optional[Union[str, Path]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Render a JSONL chapter file to a single WAV.

    Each line is a JSON object matching schemas/segment.schema.json.
    Per-segment fields override the global defaults supplied here.

    Args:
        jsonl_path: Path to input JSONL file.
        output_path: Path to write the concatenated WAV.
        tts: Optional pre-loaded IndexTTS2 instance. Created if not provided.
        voices_dir: Default voices directory.
        voice: Default voice name.
        spk_audio_prompt: Default speaker prompt path.
        ... (all other synthesize() params as defaults) ...
        cache_dir: Directory for per-segment audio cache (.npy files).
        model_version: String embedded in cache keys.
        verbose: Print per-segment progress.

    Returns:
        Concatenated float32 audio array at sample_rate Hz.
    """
    jsonl_path = Path(jsonl_path)
    output_path = Path(output_path)
    cache_dir = Path(cache_dir) if cache_dir else None

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Build the emotion resolver (auto-discovers emotions.json from voices_dir if not explicit)
    resolver = EmotionResolver.from_voices_dir(
        voices_dir=voices_dir,
        explicit_path=emotion_config,
        enable_drift=enable_drift,
    )

    # Load JSONL
    records: List[Dict] = []
    with open(jsonl_path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {lineno}: invalid JSON: {e}") from e
            _validate_segment(record, lineno)
            records.append(record)

    if not records:
        raise ValueError(f"No valid segments found in {jsonl_path}")

    if tts is None:
        tts = IndexTTS2()

    # Build the normalizer once so NeMo is not re-initialised per segment.
    shared_normalizer = Normalizer(NormalizerConfig(language=language)) if normalize else None

    chunks: List[np.ndarray] = []
    total_segments = len(records)

    for idx, record in enumerate(records):
        seg_id = record.get("segment_id", idx)
        text = record["text"]

        # ── Per-segment overrides ─────────────────────────────────────────────
        seg_voices_dir = _merge(voices_dir, record.get("voices_dir"))
        seg_voice = _merge(voice, record.get("voice"))
        seg_spk = _merge(spk_audio_prompt, record.get("spk_audio_prompt"))
        _rec_emotion = record.get("emotion", None)
        # emotion field may be a string label (from classifier) — ignore it here;
        # it is resolved to emo_vector/emo_alpha below via emotions_map.
        _rec_emotion_float = _rec_emotion if isinstance(_rec_emotion, (int, float)) else None
        seg_emotion = float(_merge(emotion, _rec_emotion_float) or emotion)
        seg_emo_alpha = float(_merge(emo_alpha, record.get("emo_alpha", None)) or 0.0)
        seg_emo_vector = _merge(emo_vector, record.get("emo_vector"))
        seg_emo_text = _merge(emo_text, record.get("emo_text"))
        seg_use_emo_text = _merge(use_emo_text, record.get("use_emo_text"))
        seg_emo_audio = _merge(emo_audio_prompt, record.get("emo_audio_prompt"))
        seg_seed = _merge(seed, record.get("seed"))
        seg_use_random = bool(_merge(use_random, record.get("use_random")))
        seg_cfm_steps = int(_merge(cfm_steps, record.get("cfm_steps", None)) or cfm_steps)
        seg_temperature = float(_merge(temperature, record.get("temperature", None)) or temperature)
        seg_cfg_rate = float(_merge(cfg_rate, record.get("cfg_rate", None)) or cfg_rate)
        seg_gpt_temperature = float(
            _merge(gpt_temperature, record.get("gpt_temperature", None)) or gpt_temperature
        )
        seg_top_k = int(_merge(top_k, record.get("top_k", None)) or top_k)
        seg_max_codes = int(_merge(max_codes, record.get("max_codes", None)) or max_codes)
        seg_sample_rate = int(_merge(sample_rate, record.get("sample_rate", None)) or sample_rate)

        pause_before = int(record.get("pause_before_ms", 0))
        pause_after = int(record.get("pause_after_ms", 0))

        # ── Emotion label resolution ──────────────────────────────────────────
        # If a resolver is available, use it to map the emotion label to
        # (emo_vector, emo_alpha). Per-segment explicit fields take precedence
        # over the preset; global CLI defaults do not override the preset.
        emotion_label = _rec_emotion if isinstance(_rec_emotion, str) else None
        if resolver is not None:
            seg_emo_vector, seg_emo_alpha = resolver.resolve(
                label=emotion_label,
                override_vector=record.get("emo_vector"),
                override_alpha=record.get("emo_alpha"),
            )

        # Resolve effective spk source string for cache key
        if seg_spk is not None:
            spk_source_key = str(seg_spk)
        elif seg_voice is not None:
            spk_source_key = f"voice:{seg_voices_dir}/{seg_voice}"
        else:
            spk_source_key = "default"

        # ── Cache lookup ──────────────────────────────────────────────────────
        cache_key = None
        cached_audio = None
        if cache_dir:
            cache_key = _segment_cache_key(
                model_version,
                text,
                spk_source_key,
                seg_emotion,
                seg_emo_alpha,
                seg_emo_vector,
                seg_emo_text,
                seg_seed,
                seg_use_random,
                seg_cfm_steps,
                seg_temperature,
                seg_cfg_rate,
                seg_gpt_temperature,
                seg_top_k,
                seg_sample_rate,
            )
            cache_file = cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                cached_audio = np.load(str(cache_file))
                if verbose:
                    print(
                        f"  [{idx+1}/{total_segments}] seg={seg_id!r} (cached) {len(cached_audio)/seg_sample_rate:.2f}s"
                    )

        if cached_audio is None:
            emotion_label = _rec_emotion if isinstance(_rec_emotion, str) else None
            if verbose:
                preview = text[:60] + ("..." if len(text) > 60 else "")
                emotion_tag = f" [{emotion_label}]" if emotion_label else ""
                if enable_drift and emotion_label and seg_emo_vector is not None:
                    vec_str = "[" + ", ".join(f"{v:.3f}" for v in seg_emo_vector) + "]"
                    drift_tag = f" α={seg_emo_alpha:.3f} v={vec_str}"
                else:
                    drift_tag = ""
                print(
                    f"  [{idx+1}/{total_segments}] seg={seg_id!r}{emotion_tag}{drift_tag}"
                    f" synthesizing: {preview!r}"
                )

            def _on_chunk(i, total, chunk_text):
                if verbose:
                    preview = chunk_text[:50] + ("..." if len(chunk_text) > 50 else "")
                    print(f"    chunk {i+1}/{total}: {preview!r}")

            def _on_chunk_done(i, total, stats):
                if verbose:
                    print(
                        f"           audio: {stats['audio_duration_s']:.2f}s"
                        f" | wall: {stats['wall_time_s']:.1f}s"
                        f" | {stats['realtime_factor']:.1f}x realtime"
                    )

            seg_audio = synthesize_long(
                text,
                tts=tts,
                voices_dir=seg_voices_dir,
                voice=seg_voice,
                spk_audio_prompt=seg_spk,
                emotion=seg_emotion,
                emo_alpha=seg_emo_alpha,
                emo_vector=seg_emo_vector,
                emo_text=seg_emo_text,
                use_emo_text=seg_use_emo_text,
                emo_audio_prompt=seg_emo_audio,
                seed=seg_seed,
                use_random=seg_use_random,
                cfm_steps=seg_cfm_steps,
                temperature=seg_temperature,
                max_codes=seg_max_codes,
                cfg_rate=seg_cfg_rate,
                gpt_temperature=seg_gpt_temperature,
                top_k=seg_top_k,
                normalize=normalize,
                language=language,
                silence_between_chunks_ms=silence_between_chunks_ms,
                crossfade_ms=crossfade_ms,
                normalizer=shared_normalizer,
                on_chunk=_on_chunk,
                on_chunk_done=_on_chunk_done,
                verbose=False,
            )

            # Resample if needed
            if seg_sample_rate != OUTPUT_SAMPLE_RATE:
                import librosa

                seg_audio = librosa.resample(
                    seg_audio, orig_sr=OUTPUT_SAMPLE_RATE, target_sr=seg_sample_rate
                ).astype(np.float32)

            if cache_dir and cache_key:
                np.save(str(cache_dir / f"{cache_key}.npy"), seg_audio)

            cached_audio = seg_audio

        # ── Assemble ──────────────────────────────────────────────────────────
        if pause_before > 0:
            chunks.append(_silence_samples(pause_before, seg_sample_rate))
        chunks.append(cached_audio)
        if pause_after > 0:
            chunks.append(_silence_samples(pause_after, seg_sample_rate))

    full_audio = np.concatenate(chunks)

    if end_chime is not None:
        chime_audio, chime_sr = sf.read(str(end_chime), dtype="float32")
        if chime_audio.ndim > 1:
            chime_audio = chime_audio.mean(axis=1)
        if chime_sr != sample_rate:
            import librosa
            chime_audio = librosa.resample(
                chime_audio, orig_sr=chime_sr, target_sr=sample_rate
            ).astype(np.float32)
        full_audio = np.concatenate([full_audio, chime_audio])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), full_audio, sample_rate)

    if verbose:
        total_dur = len(full_audio) / sample_rate
        print(f"\nChapter audio: {total_dur:.1f}s → {output_path}")

    return full_audio
