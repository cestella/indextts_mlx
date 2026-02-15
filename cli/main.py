#!/usr/bin/env python3
"""IndexTTS-2 MLX — unified command-line interface.

Usage:

    indextts synthesize --text "Hello world" --spk-audio-prompt speaker.wav
    indextts classify-emotions chapter01.txt chapter01.jsonl
    indextts m4b --isbn 9780743273565 --chapters-dir ~/audio --out ~/audio
    indextts download-weights --out-dir ~/weights
"""

import click

from cli.tts import synthesize
from cli.classify_emotions import classify_emotions
from cli.extract import extract
from cli.m4b import m4b
from cli.download_weights import download_weights
from cli.web import web


@click.group()
def main():
    """IndexTTS-2 MLX text-to-speech toolkit."""


main.add_command(synthesize)
main.add_command(classify_emotions)
main.add_command(extract)
main.add_command(m4b)
main.add_command(download_weights)
main.add_command(web)


if __name__ == "__main__":
    main()
