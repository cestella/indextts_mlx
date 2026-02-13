#\!/usr/bin/env python3
"""Command-line interface for IndexTTS-2 MLX."""
import sys
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import click

from indextts_mlx import IndexTTS2, WeightsConfig


@click.command()
@click.argument("text")
@click.option("--voice", required=True, type=click.Path(exists=True), help="Reference audio file")
@click.option("--out", default="output.wav", show_default=True, help="Output WAV file path")
@click.option("--emotion", default=1.0, show_default=True, type=float,
              help="Emotion intensity: 0.0=neutral, 1.0=default, 2.0=expressive")
@click.option("--steps", default=10, show_default=True, type=int,
              help="CFM diffusion steps (10=fast, 25=higher quality)")
@click.option("--temperature", default=1.0, show_default=True, type=float,
              help="CFM sampling temperature")
@click.option("--cfg-rate", default=0.7, show_default=True, type=float,
              help="Classifier-free guidance rate")
@click.option("--max-codes", default=1500, show_default=True, type=int,
              help="Maximum GPT tokens to generate")
@click.option("--gpt-temperature", default=0.8, show_default=True, type=float,
              help="GPT sampling temperature (0.8 matches original IndexTTS-2)")
@click.option("--top-k", default=200, show_default=True, type=int,
              help="Top-k for GPT token sampling")
@click.option("--weights-dir", default=None, type=click.Path(),
              help="Override weights directory")
@click.option("--bpe-model", default=None, type=click.Path(exists=True),
              help="Override BPE model path")
@click.option("--play", is_flag=True, help="Play audio after generation (macOS)")
def main(text, voice, out, emotion, steps, temperature, cfg_rate, max_codes,
         gpt_temperature, top_k, weights_dir, bpe_model, play):
    """Synthesize speech with IndexTTS-2 (MLX).

    TEXT is the text to synthesize. Requires --voice reference audio.

    Examples:

      indextts-tts "Hello world" --voice speaker.wav

      indextts-tts "Long sentence here." --voice speaker.wav --out result.wav --steps 25

      indextts-tts "Dramatic reading." --voice speaker.wav --emotion 1.8 --play
    """
    config_kwargs = {}
    if weights_dir:
        config_kwargs["weights_dir"] = Path(weights_dir)
    if bpe_model:
        config_kwargs["bpe_model"] = Path(bpe_model)
    config = WeightsConfig(**config_kwargs)

    click.echo(f"Loading models from {config.weights_dir}...")
    tts = IndexTTS2(config=config)

    click.echo(f"Synthesizing: {text[:60]}{'...'if len(text) > 60 else ''}")
    click.echo(f"  Voice: {voice}  |  Emotion: {emotion}  |  Steps: {steps}")

    audio = tts.synthesize(
        text=text,
        reference_audio=voice,
        emotion=emotion,
        cfm_steps=steps,
        temperature=temperature,
        cfg_rate=cfg_rate,
        max_codes=max_codes,
        gpt_temperature=gpt_temperature,
        top_k=top_k,
    )

    duration = len(audio) / 22050
    sf.write(out, audio, 22050)
    click.echo(f"Saved {duration:.2f}s of audio to {out}")

    if play:
        subprocess.run(["afplay", out], check=False)


if __name__ == "__main__":
    main()
