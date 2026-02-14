#!/usr/bin/env python3
"""Command-line interface for IndexTTS-2 MLX."""

import sys
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import click

from indextts_mlx import IndexTTS2, WeightsConfig
from indextts_mlx.voices import list_voices, parse_emo_vector, resolve_voice
from indextts_mlx.renderer import render_segments_jsonl
from indextts_mlx.synthesize_long import synthesize_long, LongSynthesisConfig
from indextts_mlx.segmenter import SegmenterConfig

# The GPT has max_text_tokens=602; 250 tokens ~2-3 sentences is a comfortable
# chunk size that produces natural-sounding audio without overloading the model.
_DEFAULT_TOKEN_TARGET = 250


# ── helpers ───────────────────────────────────────────────────────────────────


def _build_config(weights_dir, bpe_model):
    kwargs = {}
    if weights_dir:
        kwargs["weights_dir"] = Path(weights_dir)
    if bpe_model:
        kwargs["bpe_model"] = Path(bpe_model)
    return WeightsConfig(**kwargs)


def _effective_emo_alpha(emo_alpha, emo_vector, emo_text, emo_audio_prompt):
    """Auto-set emo_alpha=1.0 when emo_audio_prompt is provided and the user left it at 0."""
    if emo_audio_prompt is not None and emo_alpha == 0.0:
        return 1.0
    return emo_alpha


# ── shared options ─────────────────────────────────────────────────────────────

_SHARED_WEIGHTS_OPTS = [
    click.option(
        "--weights-dir", default=None, type=click.Path(), help="Override weights directory."
    ),
    click.option(
        "--bpe-model", default=None, type=click.Path(exists=True), help="Override BPE model path."
    ),
]

_SHARED_SPEAKER_OPTS = [
    click.option(
        "--voices-dir",
        default=None,
        type=click.Path(),
        help="Directory of voice .wav files. Voice names are file stems.",
    ),
    click.option("--voice", default=None, help="Voice name resolved to voices_dir/{voice}.wav."),
    click.option(
        "--spk-audio-prompt",
        default=None,
        type=click.Path(exists=True),
        help="Reference speaker audio file (overrides --voice/--voices-dir).",
    ),
]

_SHARED_EMO_OPTS = [
    click.option(
        "--emo-alpha",
        default=0.0,
        show_default=True,
        type=float,
        help="Emotion blend strength (0..1). Non-zero only when an emo source is provided.",
    ),
    click.option(
        "--emo-vector",
        default=None,
        help="8 comma-separated floats: happy,angry,sad,afraid,disgusted,melancholic,surprised,calm",
    ),
    click.option(
        "--emo-text",
        default=None,
        help="Text description of desired emotion (auto-enables --use-emo-text).",
    ),
    click.option(
        "--use-emo-text/--no-use-emo-text",
        default=None,
        help="Enable/disable emo_text conditioning (default: auto).",
    ),
    click.option(
        "--emo-audio-prompt",
        default=None,
        type=click.Path(exists=True),
        help="Path to emotion reference audio.",
    ),
]

_SHARED_DETERMINISM_OPTS = [
    click.option(
        "--seed", default=None, type=int, help="Random seed. Default (use_random=False): seed=0."
    ),
    click.option(
        "--use-random/--no-use-random",
        default=False,
        show_default=True,
        help="Enable random sampling (non-deterministic). Off by default for audiobooks.",
    ),
]

_SHARED_QUALITY_OPTS = [
    click.option(
        "--emotion",
        default=1.0,
        show_default=True,
        type=float,
        help="Internal emotion vector scale (0=neutral, 1=default, 2=expressive).",
    ),
    click.option(
        "--steps",
        default=10,
        show_default=True,
        type=int,
        help="CFM diffusion steps (10=fast, 25=quality).",
    ),
    click.option(
        "--temperature",
        default=1.0,
        show_default=True,
        type=float,
        help="CFM sampling temperature.",
    ),
    click.option(
        "--cfg-rate",
        default=0.7,
        show_default=True,
        type=float,
        help="Classifier-free guidance rate.",
    ),
    click.option(
        "--max-codes",
        default=1500,
        show_default=True,
        type=int,
        help="Maximum GPT tokens to generate.",
    ),
    click.option(
        "--gpt-temperature",
        default=0.8,
        show_default=True,
        type=float,
        help="GPT sampling temperature (0.8 matches original IndexTTS-2).",
    ),
    click.option(
        "--top-k", default=30, show_default=True, type=int, help="Top-k for GPT token sampling."
    ),
]


def add_options(options):
    """Decorator factory: attach a list of click.option decorators."""

    def decorator(f):
        for opt in reversed(options):
            f = opt(f)
        return f

    return decorator


# ── main command ──────────────────────────────────────────────────────────────


@click.command("synthesize")
@click.option("--text", default=None, help="Text to synthesize.")
@click.option(
    "--file",
    "text_file",
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    help="Text file (.txt), JSONL segments file (.jsonl), or directory of input files.",
)
# Long-text pipeline controls
@click.option(
    "--normalize/--no-normalize",
    default=True,
    show_default=True,
    help="Run NeMo text normalization before synthesis (requires nemo_text_processing).",
)
@click.option(
    "--language",
    default="english",
    show_default=True,
    help="Language for normalization and segmentation.",
)
@click.option(
    "--token-target",
    default=_DEFAULT_TOKEN_TARGET,
    show_default=True,
    type=int,
    help="Target BPE tokens per synthesis chunk. Chunks always break on sentence "
    "boundaries — sentences are never split mid-way. Packs sentences until "
    "this limit is reached, then starts a new chunk (GPT hard max ~600).",
)
@click.option(
    "--silence-ms",
    default=300,
    show_default=True,
    type=int,
    help="Milliseconds of silence inserted between synthesized chunks. "
    "Ignored when --crossfade-ms > 0.",
)
@click.option(
    "--crossfade-ms",
    default=10,
    show_default=True,
    type=int,
    help="Milliseconds of linear crossfade overlap between chunks. "
    "When non-zero, replaces silence with a smooth blend.",
)
# Speaker
@add_options(_SHARED_SPEAKER_OPTS)
# Legacy --voice alias (kept for backward compat)
@click.option(
    "--voice-file",
    default=None,
    type=click.Path(exists=True),
    hidden=True,
    help="[Deprecated] Use --spk-audio-prompt instead.",
)
# Output
@click.option(
    "--out",
    default="output.wav",
    show_default=True,
    help="Output file or directory. In directory mode (--file is a directory), "
    "this must be a directory; one file per input is written there.",
)
@click.option(
    "--out-ext",
    default="mp3",
    show_default=True,
    type=click.Choice(["wav", "mp3", "pcm"], case_sensitive=False),
    help="Output format for directory batch mode.",
)
@click.option(
    "--audio-format",
    default=None,
    type=click.Choice(["wav", "mp3", "pcm"], case_sensitive=False),
    help="Override output format (default: infer from --out extension).",
)
@click.option(
    "--sample-rate", default=22050, show_default=True, type=int, help="Output sample rate (Hz)."
)
@click.option("--play", is_flag=True, help="Play output via afplay after synthesis (macOS).")
# Emotion
@add_options(_SHARED_EMO_OPTS)
# Determinism
@add_options(_SHARED_DETERMINISM_OPTS)
# Quality
@add_options(_SHARED_QUALITY_OPTS)
# Weights
@add_options(_SHARED_WEIGHTS_OPTS)
@click.option(
    "--cache-dir",
    default=None,
    type=click.Path(),
    help="Segment audio cache directory (JSONL mode).",
)
@click.option(
    "--emotion-config",
    default=None,
    type=click.Path(),
    help="Path to emotions.json preset config (JSONL mode). Auto-detected from --voices-dir if not set.",
)
@click.option(
    "--enable-drift",
    is_flag=True,
    default=False,
    help="Apply bounded per-segment drift to emotion vectors (JSONL mode).",
)
@click.option(
    "--end-chime",
    default=None,
    type=click.Path(exists=True),
    help="Audio file appended to the end of the chapter (JSONL mode). Resampled if needed.",
)
# Utility
@click.option(
    "--list-voices",
    "do_list_voices",
    is_flag=True,
    help="List available voice names in --voices-dir and exit.",
)
@click.option("-v", "--verbose", is_flag=True, help="Print effective settings summary.")
def synthesize(
    text,
    text_file,
    normalize,
    language,
    token_target,
    silence_ms,
    crossfade_ms,
    voice,
    voice_file,
    spk_audio_prompt,
    voices_dir,
    out,
    out_ext,
    audio_format,
    sample_rate,
    play,
    emotion,
    emo_alpha,
    emo_vector,
    emo_text,
    use_emo_text,
    emo_audio_prompt,
    seed,
    use_random,
    steps,
    temperature,
    cfg_rate,
    max_codes,
    gpt_temperature,
    top_k,
    weights_dir,
    bpe_model,
    cache_dir,
    emotion_config,
    enable_drift,
    end_chime,
    do_list_voices,
    verbose,
):
    """Synthesize speech with IndexTTS-2 (MLX).

    Provide input via --text or --file. Text is always chunked and run through
    the long-text pipeline (normalize → segment → synthesize → stitch).

    \b
    Input (pick one):
      --text "..."               inline text
      --file PATH                read text from a UTF-8 file (.txt or .jsonl)
      --file DIR --out DIR       batch-process all .txt/.jsonl files in a directory

    \b
    Speaker source (pick one; spk-audio-prompt wins):
      --spk-audio-prompt PATH    direct audio file
      --voice NAME --voices-dir PATH  name resolved to voices_dir/NAME.wav

    \b
    Emotion controls:
      --emotion FLOAT            internal emo_vec scale (0=neutral, 2=expressive)
      --emo-vector "0,0.5,..."   8-float emotion blend vector
      --emo-alpha FLOAT          blend strength (0..1)
      --emo-text "..."           text description; auto-enables --use-emo-text

    \b
    Determinism (audiobook defaults):
      --no-use-random            deterministic output (default, seed=0)
      --seed INT                 explicit seed
      --use-random               non-deterministic sampling

    \b
    Examples:
      indextts synthesize --text "Hello world" --spk-audio-prompt speaker.wav

      indextts synthesize --file chapter01.txt --voices-dir ~/voices --voice Emma --out ch01.wav

      indextts synthesize --text "What a day!" --spk-audio-prompt speaker.wav \\
          --emo-vector "0.8,0,0,0,0,0,0,0.2" --emo-alpha 0.5

      indextts synthesize --file chapter01.jsonl \\
          --voices-dir ~/voices --out chapter01.wav

      indextts synthesize --list-voices --voices-dir ~/voices

      indextts synthesize --file ~/chapters --out ~/audio --out-ext mp3 \\
          --voices-dir ~/voices --voice Emma
    """
    # ── --list-voices utility ─────────────────────────────────────────────────
    if do_list_voices:
        if not voices_dir:
            raise click.UsageError("--list-voices requires --voices-dir")
        names = list_voices(voices_dir)
        if names:
            click.echo("\n".join(names))
        else:
            click.echo(f"No voices found in {voices_dir}", err=True)
        return

    # ── Backward-compat: --voice-file ─────────────────────────────────────────
    if voice_file and not spk_audio_prompt:
        spk_audio_prompt = voice_file

    # ── Parse emo_vector ──────────────────────────────────────────────────────
    parsed_emo_vector = None
    if emo_vector:
        try:
            parsed_emo_vector = parse_emo_vector(emo_vector)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="--emo-vector")

    # ── Auto-set emo_alpha when emo_audio_prompt given without explicit alpha ──
    emo_alpha = _effective_emo_alpha(emo_alpha, parsed_emo_vector, emo_text, emo_audio_prompt)

    # ── Verbose settings summary ──────────────────────────────────────────────
    if verbose:
        click.echo("Effective settings:")
        click.echo(f"  speaker: {spk_audio_prompt or (f'{voice} in {voices_dir}') or 'none'}")
        click.echo(f"  emotion scale: {emotion}  |  emo_alpha: {emo_alpha}")
        if parsed_emo_vector:
            click.echo(f"  emo_vector: {parsed_emo_vector}")
        if emo_text:
            click.echo(f"  emo_text: {emo_text!r}  |  use_emo_text: {use_emo_text}")
        click.echo(f"  seed: {seed}  |  use_random: {use_random}")
        click.echo(
            f"  normalize: {normalize}  |  language: {language}  |  token_target: {token_target}"
        )
        click.echo(
            f"  cfm_steps: {steps}  |  sample_rate: {sample_rate}  |  format: {audio_format}"
        )

    config = _build_config(weights_dir, bpe_model)

    # ── Directory batch mode ──────────────────────────────────────────────────
    if text_file and Path(text_file).is_dir():
        in_dir = Path(text_file)
        out_dir = Path(out)
        if out_dir.suffix:
            raise click.UsageError("--out must be a directory path when --file is a directory.")
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect all processable files (.txt and .jsonl), sorted for predictable order
        candidates = sorted(
            p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in (".txt", ".jsonl")
        )
        if not candidates:
            raise click.ClickException(f"No .txt or .jsonl files found in {in_dir}")

        # Determine which files still need processing
        ext = out_ext.lstrip(".")
        pending = []
        skipped = []
        for inp in candidates:
            out_file = out_dir / (inp.stem + "." + ext)
            if out_file.exists():
                skipped.append(inp)
            else:
                pending.append(inp)

        click.echo(f"Batch mode: {len(candidates)} file(s) in {in_dir} → {out_dir}")
        if skipped:
            click.echo(f"  Skipping {len(skipped)} already-processed file(s):")
            for p in skipped:
                click.echo(f"    {p.name} → {out_dir / (p.stem + '.' + ext)} (exists)")
        if not pending:
            click.echo("  Nothing to do.")
            return
        click.echo(f"  Processing {len(pending)} file(s):")
        for p in pending:
            click.echo(f"    {p.name}")

        click.echo(f"Loading models from {config.weights_dir}...")
        tts = IndexTTS2(config=config)

        for idx, inp in enumerate(pending):
            out_file = out_dir / (inp.stem + "." + ext)
            click.echo(f"\n[{idx+1}/{len(pending)}] {inp.name} → {out_file.name}")

            if inp.suffix.lower() == ".jsonl":
                # render_segments_jsonl always writes WAV; use a temp path then convert
                wav_out = out_file if ext == "wav" else out_file.with_suffix(".wav.tmp")
                render_segments_jsonl(
                    jsonl_path=inp,
                    output_path=wav_out,
                    tts=tts,
                    voices_dir=voices_dir,
                    voice=voice,
                    spk_audio_prompt=spk_audio_prompt,
                    emotion=emotion,
                    emo_alpha=emo_alpha,
                    emo_vector=parsed_emo_vector,
                    emo_text=emo_text,
                    use_emo_text=use_emo_text,
                    emo_audio_prompt=emo_audio_prompt,
                    seed=seed,
                    use_random=use_random,
                    cfm_steps=steps,
                    temperature=temperature,
                    max_codes=max_codes,
                    cfg_rate=cfg_rate,
                    gpt_temperature=gpt_temperature,
                    top_k=top_k,
                    sample_rate=sample_rate,
                    normalize=normalize,
                    language=language,
                    token_target=token_target,
                    silence_between_chunks_ms=silence_ms,
                    crossfade_ms=crossfade_ms,
                    cache_dir=cache_dir,
                    emotion_config=emotion_config,
                    enable_drift=enable_drift,
                    end_chime=end_chime,
                    verbose=True,
                )
                if ext != "wav":
                    wav_audio, wav_sr = sf.read(str(wav_out), dtype="float32")
                    wav_out.unlink()
                    if ext == "mp3":
                        _write_mp3(wav_audio, wav_sr, out_file)
                    elif ext == "pcm":
                        out_file.write_bytes(
                            (np.clip(wav_audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                        )
            else:
                # Plain text → synthesize_long
                input_text = inp.read_text(encoding="utf-8")
                seg_config = SegmenterConfig(
                    language=language,
                    strategy="token_count",
                    token_target=token_target,
                    bpe_model_path=str(config.bpe_model),
                )
                long_config = LongSynthesisConfig(
                    language=language,
                    normalize=normalize,
                    silence_between_chunks_ms=silence_ms,
                    crossfade_ms=crossfade_ms,
                    segmenter_config=seg_config,
                    verbose=False,
                )

                def _on_chunk(i, total, chunk_text):
                    preview = chunk_text[:60].replace("\n", " ")
                    click.echo(f"  [{i+1}/{total}] {preview!r}")

                def _on_chunk_done(i, total, stats):
                    click.echo(
                        f"         audio: {stats['audio_duration_s']:.2f}s | "
                        f"wall: {stats['wall_time_s']:.1f}s | "
                        f"{stats['realtime_factor']:.1f}x realtime"
                    )

                audio = synthesize_long(
                    input_text,
                    tts=tts,
                    spk_audio_prompt=spk_audio_prompt,
                    voices_dir=voices_dir,
                    voice=voice,
                    emotion=emotion,
                    emo_alpha=emo_alpha,
                    emo_vector=parsed_emo_vector,
                    emo_text=emo_text,
                    use_emo_text=use_emo_text,
                    emo_audio_prompt=emo_audio_prompt,
                    seed=seed,
                    use_random=use_random,
                    cfm_steps=steps,
                    temperature=temperature,
                    max_codes=max_codes,
                    cfg_rate=cfg_rate,
                    gpt_temperature=gpt_temperature,
                    top_k=top_k,
                    config=long_config,
                    on_chunk=_on_chunk,
                    on_chunk_done=_on_chunk_done,
                )
                if sample_rate != 22050:
                    import librosa as _librosa

                    audio = _librosa.resample(audio, orig_sr=22050, target_sr=sample_rate).astype(
                        np.float32
                    )
                if ext == "mp3":
                    _write_mp3(audio, sample_rate, out_file)
                elif ext == "pcm":
                    out_file.write_bytes(
                        (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    )
                else:
                    sf.write(str(out_file), audio, sample_rate)
                click.echo(f"  Saved {len(audio)/sample_rate:.2f}s → {out_file}")

        click.echo(f"\nBatch complete: {len(pending)} file(s) written to {out_dir}")
        return

    # ── JSONL chapter mode (auto-detected from .jsonl extension) ─────────────
    if text_file and Path(text_file).suffix.lower() == ".jsonl":
        chapter_out = Path(out)
        click.echo(f"Loading models from {config.weights_dir}...")
        tts = IndexTTS2(config=config)
        click.echo(f"Rendering chapter from {text_file} → {chapter_out}")
        render_segments_jsonl(
            jsonl_path=text_file,
            output_path=chapter_out,
            tts=tts,
            voices_dir=voices_dir,
            voice=voice,
            spk_audio_prompt=spk_audio_prompt,
            emotion=emotion,
            emo_alpha=emo_alpha,
            emo_vector=parsed_emo_vector,
            emo_text=emo_text,
            use_emo_text=use_emo_text,
            emo_audio_prompt=emo_audio_prompt,
            seed=seed,
            use_random=use_random,
            cfm_steps=steps,
            temperature=temperature,
            max_codes=max_codes,
            cfg_rate=cfg_rate,
            gpt_temperature=gpt_temperature,
            top_k=top_k,
            sample_rate=sample_rate,
            normalize=normalize,
            language=language,
            token_target=token_target,
            silence_between_chunks_ms=silence_ms,
            crossfade_ms=crossfade_ms,
            cache_dir=cache_dir,
            emotion_config=emotion_config,
            enable_drift=enable_drift,
            end_chime=end_chime,
            verbose=True,
        )
        if play:
            subprocess.run(["afplay", str(chapter_out)], check=False)
        return

    # ── Text input mode ───────────────────────────────────────────────────────
    if text and text_file:
        raise click.UsageError("Provide --text or --file, not both.")
    if not text and not text_file:
        raise click.UsageError("Provide --text or --file (plain text or .jsonl).")

    if text_file:
        input_text = Path(text_file).read_text(encoding="utf-8")
    else:
        input_text = text

    click.echo(f"Loading models from {config.weights_dir}...")
    tts = IndexTTS2(config=config)

    preview = input_text[:80].replace("\n", " ")
    click.echo(f"Synthesizing: {preview}{'...' if len(input_text) > 80 else ''}")
    spk_label = spk_audio_prompt or (f"{voice} ({voices_dir})" if voice else "none")
    click.echo(f"  Speaker: {spk_label}  |  Emotion: {emotion}  |  Steps: {steps}")
    click.echo(
        f"  Normalize: {normalize}  |  Language: {language}  |  Token target: {token_target}"
    )

    seg_config = SegmenterConfig(
        language=language,
        strategy="token_count",
        token_target=token_target,
        bpe_model_path=str(config.bpe_model),
    )
    long_config = LongSynthesisConfig(
        language=language,
        normalize=normalize,
        silence_between_chunks_ms=silence_ms,
        crossfade_ms=crossfade_ms,
        segmenter_config=seg_config,
        verbose=verbose,
    )

    chunk_start_times: list = []

    def _on_chunk(i, total, chunk_text):
        preview = chunk_text[:60].replace("\n", " ")
        click.echo(f"  [{i+1}/{total}] {preview!r}")
        chunk_start_times.append(i)  # just a marker; timing is in on_chunk_done

    def _on_chunk_done(i, total, stats):
        audio_dur = stats["audio_duration_s"]
        wall_time = stats["wall_time_s"]
        rtf = stats["realtime_factor"]
        click.echo(
            f"         audio: {audio_dur:.2f}s | "
            f"wall: {wall_time:.1f}s | "
            f"{rtf:.1f}x realtime"
        )

    audio = synthesize_long(
        input_text,
        tts=tts,
        spk_audio_prompt=spk_audio_prompt,
        voices_dir=voices_dir,
        voice=voice,
        emotion=emotion,
        emo_alpha=emo_alpha,
        emo_vector=parsed_emo_vector,
        emo_text=emo_text,
        use_emo_text=use_emo_text,
        emo_audio_prompt=emo_audio_prompt,
        seed=seed,
        use_random=use_random,
        cfm_steps=steps,
        temperature=temperature,
        max_codes=max_codes,
        cfg_rate=cfg_rate,
        gpt_temperature=gpt_temperature,
        top_k=top_k,
        config=long_config,
        on_chunk=_on_chunk,
        on_chunk_done=_on_chunk_done,
    )

    # Resample if requested
    if sample_rate != 22050:
        import librosa

        audio = librosa.resample(audio, orig_sr=22050, target_sr=sample_rate).astype(np.float32)

    duration = len(audio) / sample_rate
    out_path = Path(out)

    # Resolve format: explicit flag > file extension > wav
    ext = out_path.suffix.lstrip(".").lower()
    fmt = (audio_format or ext or "wav").lower()

    if fmt == "pcm":
        pcm_path = out_path.with_suffix(".pcm")
        raw = (audio * 32767).astype(np.int16).tobytes()
        pcm_path.write_bytes(raw)
        click.echo(f"Saved {duration:.2f}s of PCM audio to {pcm_path}")
        out_path = pcm_path
    elif fmt == "mp3":
        mp3_path = out_path.with_suffix(".mp3")
        _write_mp3(audio, sample_rate, mp3_path)
        click.echo(f"Saved {duration:.2f}s of MP3 audio to {mp3_path}")
        out_path = mp3_path
    else:
        wav_path = out_path.with_suffix(".wav")
        sf.write(str(wav_path), audio, sample_rate)
        click.echo(f"Saved {duration:.2f}s of audio to {wav_path}")
        out_path = wav_path

    if play:
        subprocess.run(["afplay", str(out_path)], check=False)


def _write_mp3(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    """Write float32 audio to an MP3 file via pydub (requires ffmpeg)."""
    try:
        from pydub import AudioSegment
    except ImportError:
        raise click.ClickException(
            "pydub is required for MP3 output. Install with: pip install pydub\n"
            "(also requires ffmpeg: brew install ffmpeg)"
        )
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    seg = AudioSegment(
        pcm.tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1,
    )
    seg.export(str(path), format="mp3", bitrate="192k")


if __name__ == "__main__":
    synthesize()
