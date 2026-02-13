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


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_config(weights_dir, bpe_model):
    kwargs = {}
    if weights_dir:
        kwargs["weights_dir"] = Path(weights_dir)
    if bpe_model:
        kwargs["bpe_model"] = Path(bpe_model)
    return WeightsConfig(**kwargs)


def _effective_emo_alpha(emo_alpha, emo_vector, emo_text, emo_audio_prompt):
    """If no emo source is set, emo_alpha is irrelevant but harmless at 0.0."""
    return emo_alpha


# ── shared options ─────────────────────────────────────────────────────────────

_SHARED_WEIGHTS_OPTS = [
    click.option("--weights-dir", default=None, type=click.Path(),
                 help="Override weights directory."),
    click.option("--bpe-model", default=None, type=click.Path(exists=True),
                 help="Override BPE model path."),
]

_SHARED_SPEAKER_OPTS = [
    click.option("--voices-dir", default=None, type=click.Path(),
                 help="Directory of voice .wav files. Voice names are file stems."),
    click.option("--voice", default=None,
                 help="Voice name resolved to voices_dir/{voice}.wav."),
    click.option("--spk-audio-prompt", default=None, type=click.Path(exists=True),
                 help="Reference speaker audio file (overrides --voice/--voices-dir)."),
]

_SHARED_EMO_OPTS = [
    click.option("--emo-alpha", default=0.0, show_default=True, type=float,
                 help="Emotion blend strength (0..1). Non-zero only when an emo source is provided."),
    click.option("--emo-vector", default=None,
                 help='8 comma-separated floats: happy,angry,sad,afraid,disgusted,melancholic,surprised,calm'),
    click.option("--emo-text", default=None,
                 help="Text description of desired emotion (auto-enables --use-emo-text)."),
    click.option("--use-emo-text/--no-use-emo-text", default=None,
                 help="Enable/disable emo_text conditioning (default: auto)."),
    click.option("--emo-audio-prompt", default=None, type=click.Path(exists=True),
                 help="Path to emotion reference audio."),
]

_SHARED_DETERMINISM_OPTS = [
    click.option("--seed", default=None, type=int,
                 help="Random seed. Default (use_random=False): seed=0."),
    click.option("--use-random/--no-use-random", default=False, show_default=True,
                 help="Enable random sampling (non-deterministic). Off by default for audiobooks."),
]

_SHARED_QUALITY_OPTS = [
    click.option("--emotion", default=1.0, show_default=True, type=float,
                 help="Internal emotion vector scale (0=neutral, 1=default, 2=expressive)."),
    click.option("--steps", default=10, show_default=True, type=int,
                 help="CFM diffusion steps (10=fast, 25=quality)."),
    click.option("--temperature", default=1.0, show_default=True, type=float,
                 help="CFM sampling temperature."),
    click.option("--cfg-rate", default=0.7, show_default=True, type=float,
                 help="Classifier-free guidance rate."),
    click.option("--max-codes", default=1500, show_default=True, type=int,
                 help="Maximum GPT tokens to generate."),
    click.option("--gpt-temperature", default=0.8, show_default=True, type=float,
                 help="GPT sampling temperature (0.8 matches original IndexTTS-2)."),
    click.option("--top-k", default=30, show_default=True, type=int,
                 help="Top-k for GPT token sampling."),
]


def add_options(options):
    """Decorator factory: attach a list of click.option decorators."""
    def decorator(f):
        for opt in reversed(options):
            f = opt(f)
        return f
    return decorator


# ── main command ──────────────────────────────────────────────────────────────

@click.command()
# Text (positional, optional when --segments-jsonl is used)
@click.argument("text", required=False, default=None)
# Speaker
@add_options(_SHARED_SPEAKER_OPTS)
# Legacy --voice alias (kept for backward compat; was required path arg)
@click.option("--voice-file", default=None, type=click.Path(exists=True), hidden=True,
              help="[Deprecated] Use --spk-audio-prompt instead.")
# Output
@click.option("--out", default="output.wav", show_default=True,
              help="Output WAV file.")
@click.option("--audio-format", default="wav", show_default=True,
              type=click.Choice(["wav", "pcm"], case_sensitive=False),
              help="Output audio format.")
@click.option("--sample-rate", default=22050, show_default=True, type=int,
              help="Output sample rate (Hz).")
@click.option("--play", is_flag=True,
              help="Play output via afplay after synthesis (macOS).")
@click.option("--return-timestamps", is_flag=True,
              help="Print segment timestamps to stdout (placeholder for future support).")
# Emotion
@add_options(_SHARED_EMO_OPTS)
# Determinism
@add_options(_SHARED_DETERMINISM_OPTS)
# Quality
@add_options(_SHARED_QUALITY_OPTS)
# Weights
@add_options(_SHARED_WEIGHTS_OPTS)
# JSONL chapter mode
@click.option("--segments-jsonl", default=None, type=click.Path(exists=True),
              help="JSONL file for chapter rendering (one segment JSON per line).")
@click.option("--output", default=None, type=click.Path(),
              help="Chapter output WAV path (JSONL mode; defaults to --out value).")
@click.option("--cache-dir", default=None, type=click.Path(),
              help="Segment audio cache directory (JSONL mode).")
# Utility
@click.option("--list-voices", "do_list_voices", is_flag=True,
              help="List available voice names in --voices-dir and exit.")
@click.option("-v", "--verbose", is_flag=True,
              help="Print effective settings summary.")
def main(
    text, voice, voice_file, spk_audio_prompt, voices_dir,
    out, audio_format, sample_rate, play, return_timestamps,
    emotion, emo_alpha, emo_vector, emo_text, use_emo_text, emo_audio_prompt,
    seed, use_random,
    steps, temperature, cfg_rate, max_codes, gpt_temperature, top_k,
    weights_dir, bpe_model,
    segments_jsonl, output, cache_dir,
    do_list_voices, verbose,
):
    """Synthesize speech with IndexTTS-2 (MLX).

    TEXT is the text to synthesize. Omit when using --segments-jsonl.

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
      indextts-tts "Hello world" --spk-audio-prompt speaker.wav

      indextts-tts "Hello world" --voices-dir ~/voices --voice Emma

      indextts-tts "What a day!" --spk-audio-prompt speaker.wav \\
          --emo-vector "0.8,0,0,0,0,0,0,0.2" --emo-alpha 0.5

      indextts-tts --segments-jsonl chapter01.jsonl \\
          --voices-dir ~/voices --out chapter01.wav

      indextts-tts --list-voices --voices-dir ~/voices
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

    # ── Backward-compat: --voice-file  ───────────────────────────────────────
    if voice_file and not spk_audio_prompt:
        spk_audio_prompt = voice_file

    # ── Parse emo_vector ──────────────────────────────────────────────────────
    parsed_emo_vector = None
    if emo_vector:
        try:
            parsed_emo_vector = parse_emo_vector(emo_vector)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="--emo-vector")

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
        click.echo(f"  cfm_steps: {steps}  |  sample_rate: {sample_rate}  |  format: {audio_format}")

    config = _build_config(weights_dir, bpe_model)

    # ── JSONL chapter mode ────────────────────────────────────────────────────
    if segments_jsonl:
        chapter_out = Path(output or out)
        click.echo(f"Loading models from {config.weights_dir}...")
        tts = IndexTTS2(config=config)
        click.echo(f"Rendering chapter from {segments_jsonl} → {chapter_out}")
        render_segments_jsonl(
            jsonl_path=segments_jsonl,
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
            cache_dir=cache_dir,
            verbose=True,
        )
        if play:
            subprocess.run(["afplay", str(chapter_out)], check=False)
        return

    # ── Single-text mode ──────────────────────────────────────────────────────
    if not text:
        raise click.UsageError("TEXT argument is required (or use --segments-jsonl for chapter mode).")

    click.echo(f"Loading models from {config.weights_dir}...")
    tts = IndexTTS2(config=config)

    click.echo(f"Synthesizing: {text[:60]}{'...' if len(text) > 60 else ''}")
    if verbose or True:  # always show speaker line
        spk_label = spk_audio_prompt or (f"{voice} ({voices_dir})" if voice else "none")
        click.echo(f"  Speaker: {spk_label}  |  Emotion: {emotion}  |  Steps: {steps}")

    audio = tts.synthesize(
        text=text,
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
        cfg_rate=cfg_rate,
        max_codes=max_codes,
        gpt_temperature=gpt_temperature,
        top_k=top_k,
    )

    # Resample if requested
    if sample_rate != 22050:
        import librosa
        audio = librosa.resample(audio, orig_sr=22050, target_sr=sample_rate).astype(np.float32)

    duration = len(audio) / sample_rate
    out_path = Path(out)

    if audio_format == "pcm":
        raw = (audio * 32767).astype(np.int16).tobytes()
        out_path.with_suffix(".pcm").write_bytes(raw)
        click.echo(f"Saved {duration:.2f}s of PCM audio to {out_path.with_suffix('.pcm')}")
    else:
        sf.write(str(out_path), audio, sample_rate)
        click.echo(f"Saved {duration:.2f}s of audio to {out_path}")

    if return_timestamps:
        click.echo(f"Timestamps: 0.000 - {duration:.3f}s (full segment)")

    if play:
        subprocess.run(["afplay", str(out_path)], check=False)


if __name__ == "__main__":
    main()
