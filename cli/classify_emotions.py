"""CLI: classify emotions for each sentence in a chapter text file.

Usage::

    indextts-classify-emotions INPUT.txt OUTPUT.jsonl [options]
"""

import sys
from pathlib import Path

import click


@click.command("classify-emotions")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option(
    "--model",
    default=None,
    show_default=True,
    help="MLX-LM model repo ID or local path (default: ClassifierConfig.model).",
)
@click.option(
    "--chapter-id",
    default=None,
    help="Optional chapter identifier written to every JSONL record.",
)
@click.option(
    "--language",
    default="english",
    show_default=True,
    help="Language for sentence segmentation (english/en, french/fr, …).",
)
@click.option(
    "--context-window",
    default=5,
    show_default=True,
    type=int,
    help="Number of surrounding sentences to include as paragraph context.",
)
@click.option(
    "--hysteresis-min-run",
    default=2,
    show_default=True,
    type=int,
    help=(
        "Minimum consecutive non-neutral predictions required to keep an emotion. "
        "Isolated spikes shorter than this are collapsed back to neutral."
    ),
)
@click.option(
    "--max-retries",
    default=3,
    show_default=True,
    type=int,
    help="Retries per sentence if the model returns an invalid token.",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress per-sentence progress output.",
)
def classify_emotions(
    input_file: str,
    output_file: str,
    model: str,
    chapter_id,
    language: str,
    context_window: int,
    hysteresis_min_run: int,
    max_retries: int,
    quiet: bool,
) -> None:
    """Classify emotions for each sentence in INPUT_FILE and write to OUTPUT_FILE.

    \b
    INPUT_FILE   Plain-text chapter file (UTF-8).
    OUTPUT_FILE  Destination JSONL file; one record per line.

    \b
    Example:
      indextts classify-emotions chapter01.txt chapter01.jsonl
    """
    try:
        from indextts_mlx.emotion_classifier import ClassifierConfig, EmotionClassifier
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # Coerce chapter_id to int if it looks like one
    if chapter_id is not None:
        try:
            chapter_id = int(chapter_id)
        except ValueError:
            pass  # keep as string

    config = ClassifierConfig(
        max_retries=max_retries,
        context_window=context_window,
        language=language,
        hysteresis_min_run=hysteresis_min_run,
    )
    if model is not None:
        config.model = model

    clf = EmotionClassifier(config)

    click.echo(f"Input:  {input_file}")
    click.echo(f"Output: {output_file}")
    click.echo(f"Model:  {config.model}")

    records = clf.classify_chapter(
        input_file,
        chapter_id=chapter_id,
        verbose=not quiet,
    )

    EmotionClassifier.write_jsonl(records, output_file)


if __name__ == "__main__":
    classify_emotions()
