"""High-level long-text synthesis pipeline.

Takes arbitrary-length text, chunks it into sentences, synthesizes each
chunk, and stitches the audio together.

Pipeline:
    text → [NeMo normalizer] → [spaCy segmenter] → [IndexTTS2 synthesize]
         → [numpy concat with silence] → final audio array

Example::

    from indextts_mlx import IndexTTS2
    from indextts_mlx.synthesize_long import synthesize_long, LongSynthesisConfig

    tts = IndexTTS2()
    audio = synthesize_long(
        "It was the best of times, it was the worst of times...",
        tts=tts,
        spk_audio_prompt="speaker.wav",
    )
    import soundfile as sf
    sf.write("output.wav", audio, 22050)
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

import numpy as np

from .normalizer import Normalizer, NormalizerConfig
from .segmenter import Segmenter, SegmenterConfig

OUTPUT_SAMPLE_RATE = 22050


@dataclass
class LongSynthesisConfig:
    """Configuration for long-text synthesis.

    Args:
        language: Language for normalization and segmentation
                  (english/en, french/fr, spanish/es, italian/it, german/de).
        normalize: Whether to run NeMo text normalization before synthesis.
                   Requires nemo_text_processing + pynini. If unavailable,
                   normalization is silently skipped.
        silence_between_chunks_ms: Milliseconds of silence inserted between
                                   synthesized chunks. Default 300 ms. Ignored
                                   when crossfade_ms > 0.
        crossfade_ms: Milliseconds of crossfade overlap between chunks. When
                      non-zero, silence_between_chunks_ms is ignored and chunks
                      are blended with a linear crossfade. Default 0 (disabled).
        segmenter_config: Override the segmenter config. If None, uses
                          char_count strategy with max_chars=300.
        normalizer_config: Override the normalizer config. If None, uses
                           language default with 'cased' input.
        verbose: Print progress (chunk count, index, text preview).
    """

    language: str = "english"
    normalize: bool = True
    silence_between_chunks_ms: int = 300
    crossfade_ms: int = 10
    segmenter_config: Optional[SegmenterConfig] = None
    normalizer_config: Optional[NormalizerConfig] = None
    verbose: bool = False

    def __post_init__(self) -> None:
        # Build defaults if not provided
        if self.segmenter_config is None:
            self.segmenter_config = SegmenterConfig(
                language=self.language,
                strategy="char_count",
                max_chars=300,
            )
        if self.normalizer_config is None:
            self.normalizer_config = NormalizerConfig(language=self.language)


def synthesize_long(
    text: str,
    *,
    tts,  # IndexTTS2 instance
    spk_audio_prompt: Optional[Union[str, Path]] = None,
    voices_dir: Optional[Union[str, Path]] = None,
    voice: Optional[str] = None,
    reference_audio=None,  # backward-compat alias
    # Emotion
    emotion: float = 1.0,
    emo_alpha: float = 0.0,
    emo_vector: Optional[List[float]] = None,
    emo_text: Optional[str] = None,
    use_emo_text: Optional[bool] = None,
    emo_audio_prompt: Optional[Union[str, Path]] = None,
    # Determinism
    seed: Optional[int] = None,
    use_random: bool = False,
    # Quality
    cfm_steps: int = 10,
    temperature: float = 1.0,
    max_codes: int = 1500,
    cfg_rate: float = 0.7,
    gpt_temperature: float = 0.8,
    top_k: int = 30,
    # Long-text config
    config: Optional[LongSynthesisConfig] = None,
    language: str = "english",
    normalize: bool = True,
    silence_between_chunks_ms: int = 300,
    crossfade_ms: int = 10,
    max_chars: int = 300,
    verbose: bool = False,
    # Optional pre-built Normalizer to reuse (avoids re-initialising NeMo on every call)
    normalizer: Optional["Normalizer"] = None,
    # Optional EmotionResolver + label: when supplied, resolver.resolve() is called
    # per chunk so drift advances sentence-by-sentence within the segment.
    # Ignored when emo_vector or emo_alpha are explicitly set (overrides take precedence).
    emotion_resolver=None,  # Optional[EmotionResolver]
    emotion_label: Optional[str] = None,
    # Optional progress callback: f(chunk_index, total_chunks, chunk_text)
    # Called BEFORE synthesis of each chunk.
    on_chunk: Optional[Callable[[int, int, str], None]] = None,
    # Optional post-chunk callback: f(chunk_index, total_chunks, stats_dict)
    # stats_dict keys: chunk_text, audio_duration_s, wall_time_s, realtime_factor, tokens
    on_chunk_done: Optional[Callable[[int, int, dict], None]] = None,
) -> np.ndarray:
    """Synthesize arbitrarily long text by chunking, synthesizing, and stitching.

    Args:
        text: Input text of any length.
        tts: An IndexTTS2 instance (loaded once outside this call).
        spk_audio_prompt: Reference speaker audio path.
        voices_dir / voice: Alternative speaker source (see IndexTTS2.synthesize).
        reference_audio: Backward-compat alias for spk_audio_prompt.
        emotion / emo_*: Emotion controls forwarded to each synthesize() call.
        seed / use_random: Determinism controls. When use_random=False (default),
                           each chunk gets seed + chunk_index so that chunks are
                           individually deterministic but not identical.
        cfm_steps / temperature / ...: Quality controls forwarded to synthesize().
        config: Full LongSynthesisConfig (overrides language/normalize/etc. if set).
        language: Language for normalization and segmentation.
        normalize: Run NeMo normalization before synthesis (requires nemo_text_processing).
        silence_between_chunks_ms: Gap between chunks in the final audio.
        max_chars: Max characters per chunk (char_count strategy).
        verbose: Print progress.
        on_chunk: Optional callback invoked before each chunk is synthesized.

    Returns:
        float32 numpy array at 22050 Hz containing the full audio.
    """
    if config is None:
        config = LongSynthesisConfig(
            language=language,
            normalize=normalize,
            silence_between_chunks_ms=silence_between_chunks_ms,
            crossfade_ms=crossfade_ms,
            segmenter_config=SegmenterConfig(
                language=language,
                strategy="char_count",
                max_chars=max_chars,
            ),
            verbose=verbose,
        )

    # ── 1. Normalize ─────────────────────────────────────────────────────────
    if config.normalize:
        if normalizer is None:
            normalizer = Normalizer(config.normalizer_config)
        text = normalizer.normalize(text)

    # ── 2. Segment ───────────────────────────────────────────────────────────
    segmenter = Segmenter(config.segmenter_config)
    chunks = segmenter.segment(text)

    if not chunks:
        warnings.warn("synthesize_long: no text chunks produced — returning empty audio.")
        return np.zeros(0, dtype=np.float32)

    if config.verbose:
        print(f"synthesize_long: {len(chunks)} chunk(s) from {len(text)} chars")

    # ── 3. Synthesize each chunk ──────────────────────────────────────────────
    silence_samples = int(OUTPUT_SAMPLE_RATE * config.silence_between_chunks_ms / 1000)
    crossfade_samples = int(OUTPUT_SAMPLE_RATE * config.crossfade_ms / 1000)

    audio_parts: List[np.ndarray] = []

    for i, chunk in enumerate(chunks):
        if config.verbose:
            preview = chunk[:60] + ("..." if len(chunk) > 60 else "")
            print(f"  [{i+1}/{len(chunks)}] {preview!r}")

        if on_chunk is not None:
            on_chunk(i, len(chunks), chunk)

        # Per-chunk seed: offset by chunk index so each chunk is individually
        # reproducible but not identical to its neighbours
        chunk_seed = None
        if not use_random:
            base = seed if seed is not None else 0
            chunk_seed = base + i

        # Per-chunk drift: advance the resolver state for each sentence so
        # variation is smooth within the segment, not just between segments.
        if emotion_resolver is not None:
            chunk_emo_vector, chunk_emo_alpha = emotion_resolver.resolve(emotion_label)
        else:
            chunk_emo_vector, chunk_emo_alpha = emo_vector, emo_alpha

        t0 = time.perf_counter()
        audio = tts.synthesize(
            text=chunk,
            spk_audio_prompt=spk_audio_prompt,
            voices_dir=voices_dir,
            voice=voice,
            reference_audio=reference_audio,
            emotion=emotion,
            emo_alpha=chunk_emo_alpha,
            emo_vector=chunk_emo_vector,
            emo_text=emo_text,
            use_emo_text=use_emo_text,
            emo_audio_prompt=emo_audio_prompt,
            seed=chunk_seed,
            use_random=use_random,
            cfm_steps=cfm_steps,
            temperature=temperature,
            max_codes=max_codes,
            cfg_rate=cfg_rate,
            gpt_temperature=gpt_temperature,
            top_k=top_k,
        )
        wall_time = time.perf_counter() - t0

        audio_duration = len(audio) / OUTPUT_SAMPLE_RATE
        realtime_factor = audio_duration / wall_time if wall_time > 0 else 0.0

        if on_chunk_done is not None:
            on_chunk_done(
                i,
                len(chunks),
                {
                    "chunk_text": chunk,
                    "audio_duration_s": audio_duration,
                    "wall_time_s": wall_time,
                    "realtime_factor": realtime_factor,
                },
            )

        audio_parts.append(audio)

    # ── 4. Stitch ─────────────────────────────────────────────────────────────
    if len(audio_parts) == 1:
        return audio_parts[0]

    if crossfade_samples > 0:
        return _crossfade_concat(audio_parts, crossfade_samples)
    else:
        silence = np.zeros(silence_samples, dtype=np.float32)
        pieces: List[np.ndarray] = []
        for i, part in enumerate(audio_parts):
            pieces.append(part)
            if i < len(audio_parts) - 1:
                pieces.append(silence)
        return np.concatenate(pieces, axis=0)


def _crossfade_concat(parts: List[np.ndarray], crossfade_samples: int) -> np.ndarray:
    """Concatenate audio arrays with a linear crossfade overlap between each pair."""
    result = parts[0]
    for nxt in parts[1:]:
        cf = min(crossfade_samples, len(result), len(nxt))
        if cf <= 0:
            result = np.concatenate([result, nxt])
            continue
        # Linear fade-out / fade-in ramps
        fade_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
        fade_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
        overlap = result[-cf:] * fade_out + nxt[:cf] * fade_in
        result = np.concatenate([result[:-cf], overlap, nxt[cf:]])
    return result
