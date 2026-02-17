#!/usr/bin/env python3
"""Command-line interface for IndexTTS-2 MLX."""

import sys
import time
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


def _maybe_resolve_meta_voice(voices_dir, voice, spk_audio_prompt, emotion_label=None):
    """If voice is a meta voice directory, resolve to a concrete .wav path.

    Returns (voice, spk_audio_prompt) — voice is cleared to None when resolved.
    For plain-text synthesis (no per-segment emotion), pass emotion_label=None
    to default to 'neutral'.
    """
    if voice is not None and voices_dir is not None and spk_audio_prompt is None:
        from indextts_mlx.voices import is_meta_voice, resolve_meta_voice
        if is_meta_voice(voices_dir, voice):
            resolved = resolve_meta_voice(
                voices_dir, voice, emotion_label=emotion_label, fallback="neutral"
            )
            return None, str(resolved)
    return voice, spk_audio_prompt


def _effective_emo_alpha(emo_alpha, emo_vector, emo_text, emo_audio_prompt):
    """Auto-set emo_alpha=1.0 when emo_audio_prompt is provided and the user left it at 0."""
    if emo_audio_prompt is not None and emo_alpha == 0.0:
        return 1.0
    return emo_alpha


def _fmt_eta(seconds: float) -> str:
    """Format a seconds value as a human-readable ETA string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _write_synth_status(
    status_path: Path,
    file_index: int,
    total_files: int,
    file_name: str,
    file_done: bool,
    chunk_index: int,
    total_chunks: int,
    file_wall_times: list,
    chunk_wall_times: list,
    chunk_audio_times: list | None = None,
) -> None:
    """Atomically write a JSON status snapshot for the web UI to poll."""
    import json as _json

    mean_chunk = sum(chunk_wall_times) / len(chunk_wall_times) if chunk_wall_times else None
    mean_file = sum(file_wall_times) / len(file_wall_times) if file_wall_times else None
    chunks_left = total_chunks - chunk_index
    files_left = total_files - file_index - (1 if file_done else 0)
    chunk_eta = mean_chunk * chunks_left if mean_chunk else None
    # Job ETA requires at least one completed file to estimate per-file cost.
    # Until then it's unknown — don't conflate it with chunk ETA.
    job_eta = mean_file * files_left if (mean_file and files_left > 0) else None
    if job_eta is not None and chunk_eta is not None:
        job_eta += chunk_eta  # add remaining time for the current in-progress file
    # Real-time factor: audio seconds produced per wall second (higher = faster than real-time)
    rtf = None
    if chunk_audio_times and chunk_wall_times:
        total_audio = sum(chunk_audio_times)
        total_wall = sum(chunk_wall_times)
        if total_wall > 0:
            rtf = round(total_audio / total_wall, 2)
    data = {
        "file_index": file_index,
        "total_files": total_files,
        "file_name": file_name,
        "file_done": file_done,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "chunks_remaining": chunks_left,
        "files_remaining": files_left,
        "chunk_eta_s": round(chunk_eta, 1) if chunk_eta is not None else None,
        "job_eta_s": round(job_eta, 1) if job_eta is not None else None,
        "avg_wall_s_per_chunk": round(mean_chunk, 2) if mean_chunk else None,
        "rtf": rtf,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(data))
    tmp.replace(status_path)


def _make_verbose_callbacks():
    """Return (on_chunk, on_chunk_done) callbacks that print per-chunk progress
    with an ETA derived from the rolling mean wall-time per chunk."""
    wall_times: list[float] = []

    def on_chunk(i, total, chunk_text):
        remaining = total - i
        if wall_times:
            mean_wall = sum(wall_times) / len(wall_times)
            eta = _fmt_eta(mean_wall * remaining)
            eta_str = f"  ETA ~{eta} ({remaining} chunk(s) left)"
        else:
            eta_str = f"  ({remaining} chunk(s) remaining)"
        preview = chunk_text[:60].replace("\n", " ")
        click.echo(f"  [{i+1}/{total}] {preview!r}{eta_str}")

    def on_chunk_done(i, total, stats):
        wall_times.append(stats["wall_time_s"])
        mean_wall = sum(wall_times) / len(wall_times)
        remaining = total - (i + 1)
        eta_str = (
            f"  ETA ~{_fmt_eta(mean_wall * remaining)} ({remaining} left)"
            if remaining > 0
            else "  done"
        )
        click.echo(
            f"         audio: {stats['audio_duration_s']:.2f}s | "
            f"wall: {stats['wall_time_s']:.1f}s | "
            f"{stats['realtime_factor']:.1f}x realtime |{eta_str}"
        )

    return on_chunk, on_chunk_done


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
@click.option(
    "--status",
    "status_dir",
    default=None,
    type=click.Path(),
    help="Directory to write synth_status.json after every chunk (batch mode).",
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
    status_dir,
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

        try:
            from tqdm import tqdm as _tqdm

            _batch_iter = pending if verbose else _tqdm(pending, desc="batch", unit="file")
        except ImportError:
            _batch_iter = pending

        _file_wall_times: list[float] = []
        _status_path = Path(status_dir) / "synth_status.json" if status_dir else None
        if _status_path:
            _status_path.parent.mkdir(parents=True, exist_ok=True)

        for _file_idx, inp in enumerate(_batch_iter):
            out_file = out_dir / (inp.stem + "." + ext)
            _files_remaining = len(pending) - _file_idx
            if verbose and _file_wall_times:
                _mean_file = sum(_file_wall_times) / len(_file_wall_times)
                _file_eta = _fmt_eta(_mean_file * _files_remaining)
                click.echo(
                    f"\n[{_file_idx+1}/{len(pending)}] {inp.name} → {out_file.name}"
                    f"  (ETA ~{_file_eta} for {_files_remaining} file(s))"
                )
            else:
                click.echo(f"\n[{_file_idx+1}/{len(pending)}] {inp.name} → {out_file.name}")
            _file_t0 = time.monotonic()

            if inp.suffix.lower() == ".jsonl":
                # render_segments_jsonl always writes WAV; use a temp path then convert
                wav_out = out_file if ext == "wav" else out_file.with_suffix(".wav.tmp")

                _chunk_wall_times: list[float] = []
                _chunk_audio_times: list[float] = []
                _cur_file_idx = _file_idx
                _cur_inp_name = inp.name

                def _jsonl_on_chunk_done(i, total, stats,
                                         _fi=_cur_file_idx, _fn=_cur_inp_name):
                    _chunk_wall_times.append(stats["wall_time_s"])
                    _chunk_audio_times.append(stats["audio_duration_s"])
                    if _status_path:
                        _write_synth_status(
                            _status_path,
                            _fi,
                            len(pending),
                            _fn,
                            False,
                            i + 1,
                            total,
                            _file_wall_times,
                            _chunk_wall_times,
                            _chunk_audio_times,
                        )

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
                    on_chunk_done=_jsonl_on_chunk_done,
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
                _file_wall_times.append(time.monotonic() - _file_t0)
                click.echo(
                    f"  Saved → {out_file}"
                    + (f"  (wall: {_fmt_eta(_file_wall_times[-1])})" if verbose else "")
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

                _chunk_wall_times: list[float] = []
                _chunk_audio_times: list[float] = []

                if verbose:
                    _on_chunk, _on_chunk_done_base = _make_verbose_callbacks()
                    _chunk_bar = None
                else:
                    _chunk_bar = None

                    def _on_chunk(i, total, chunk_text):
                        nonlocal _chunk_bar
                        try:
                            from tqdm import tqdm as _tqdm

                            if _chunk_bar is None:
                                _chunk_bar = _tqdm(
                                    total=total, desc="  chunks", unit="chunk", leave=False
                                )
                            _chunk_bar.set_postfix_str(chunk_text[:40].replace("\n", " "))
                        except ImportError:
                            preview = chunk_text[:60].replace("\n", " ")
                            click.echo(f"  [{i+1}/{total}] {preview!r}")

                    def _on_chunk_done_base(i, total, stats):
                        if _chunk_bar is not None:
                            _chunk_bar.update(1)
                            _chunk_bar.set_postfix_str(
                                f"{stats['audio_duration_s']:.1f}s | {stats['realtime_factor']:.1f}x"
                            )
                        else:
                            click.echo(
                                f"         audio: {stats['audio_duration_s']:.2f}s | "
                                f"wall: {stats['wall_time_s']:.1f}s | "
                                f"{stats['realtime_factor']:.1f}x realtime"
                            )

                # Capture loop vars for closure
                _cur_file_idx = _file_idx
                _cur_inp_name = inp.name

                def _on_chunk_done(i, total, stats, _fi=_cur_file_idx, _fn=_cur_inp_name):
                    _chunk_wall_times.append(stats["wall_time_s"])
                    _chunk_audio_times.append(stats["audio_duration_s"])
                    _on_chunk_done_base(i, total, stats)
                    if _status_path:
                        _write_synth_status(
                            _status_path,
                            _fi,
                            len(pending),
                            _fn,
                            False,
                            i + 1,
                            total,
                            _file_wall_times,
                            _chunk_wall_times,
                            _chunk_audio_times,
                        )

                _voice, _spk = _maybe_resolve_meta_voice(voices_dir, voice, spk_audio_prompt)
                audio = synthesize_long(
                    input_text,
                    tts=tts,
                    spk_audio_prompt=_spk,
                    voices_dir=voices_dir,
                    voice=_voice,
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
                if _chunk_bar is not None:
                    _chunk_bar.close()
                if sample_rate != 22050:
                    import librosa as _librosa

                    audio = _librosa.resample(audio, orig_sr=22050, target_sr=sample_rate).astype(
                        np.float32
                    )
                if end_chime is not None:
                    chime_audio, chime_sr = sf.read(str(end_chime), dtype="float32")
                    if chime_audio.ndim > 1:
                        chime_audio = chime_audio.mean(axis=1)
                    chime_audio = chime_audio.ravel()
                    if chime_sr != sample_rate:
                        import librosa as _librosa

                        chime_audio = (
                            _librosa.resample(chime_audio, orig_sr=chime_sr, target_sr=sample_rate)
                            .astype(np.float32)
                            .ravel()
                        )
                    audio = np.concatenate([audio, chime_audio])
                if ext == "mp3":
                    _write_mp3(audio, sample_rate, out_file)
                elif ext == "pcm":
                    out_file.write_bytes(
                        (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    )
                else:
                    sf.write(str(out_file), audio, sample_rate)
                _file_wall_times.append(time.monotonic() - _file_t0)
                if _status_path:
                    _write_synth_status(
                        _status_path,
                        _file_idx,
                        len(pending),
                        inp.name,
                        True,
                        len(_chunk_wall_times),
                        len(_chunk_wall_times),
                        _file_wall_times,
                        _chunk_wall_times,
                        _chunk_audio_times,
                    )
                click.echo(
                    f"  Saved {len(audio)/sample_rate:.2f}s → {out_file}"
                    + (f"  (wall: {_fmt_eta(_file_wall_times[-1])})" if verbose else "")
                )

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
        verbose=False,  # callbacks below handle all progress output
    )

    _on_chunk, _on_chunk_done = _make_verbose_callbacks()

    voice, spk_audio_prompt = _maybe_resolve_meta_voice(voices_dir, voice, spk_audio_prompt)
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

    # Append end-chime if requested
    if end_chime is not None:
        chime_audio, chime_sr = sf.read(str(end_chime), dtype="float32")
        if chime_audio.ndim > 1:
            chime_audio = chime_audio.mean(axis=1)
        chime_audio = chime_audio.ravel()
        if chime_sr != sample_rate:
            import librosa

            chime_audio = (
                librosa.resample(chime_audio, orig_sr=chime_sr, target_sr=sample_rate)
                .astype(np.float32)
                .ravel()
            )
        audio = np.concatenate([audio, chime_audio])

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
