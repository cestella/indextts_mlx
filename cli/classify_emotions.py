"""CLI: classify emotions for each sentence in a chapter text file.

Usage::

    indextts classify-emotions INPUT.txt OUTPUT.jsonl [options]
    indextts classify-emotions INPUT_DIR/ OUTPUT_DIR/ [options]
"""

import sys
from pathlib import Path

import click


@click.command("classify-emotions")
@click.argument("input_path", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
@click.option(
    "--model",
    default=None,
    show_default=True,
    help="MLX-LM model repo ID or local path (default: ClassifierConfig.model).",
)
@click.option(
    "--chapter-id",
    default=None,
    help="Optional chapter identifier written to every JSONL record (single-file mode only).",
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
    input_path: str,
    output_path: str,
    model: str,
    chapter_id,
    language: str,
    context_window: int,
    hysteresis_min_run: int,
    max_retries: int,
    quiet: bool,
) -> None:
    """Classify emotions for each sentence in INPUT_PATH and write to OUTPUT_PATH.

    \b
    INPUT_PATH   Plain-text chapter file (UTF-8) or a directory of .txt files.
    OUTPUT_PATH  Destination JSONL file, or a directory when INPUT_PATH is a directory.

    \b
    Examples:
      indextts classify-emotions chapter01.txt chapter01.jsonl
      indextts classify-emotions chapters/ classified/
    """
    try:
        from indextts_mlx.emotion_classifier import ClassifierConfig, EmotionClassifier
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    config = ClassifierConfig(
        max_retries=max_retries,
        context_window=context_window,
        language=language,
        hysteresis_min_run=hysteresis_min_run,
    )
    if model is not None:
        config.model = model

    in_p = Path(input_path)
    out_p = Path(output_path)

    # ── Directory batch mode ──────────────────────────────────────────────────
    if in_p.is_dir():
        if out_p.suffix:
            raise click.UsageError(
                "OUTPUT_PATH must be a directory (no file extension) when INPUT_PATH is a directory."
            )
        out_p.mkdir(parents=True, exist_ok=True)

        candidates = sorted(p for p in in_p.iterdir() if p.is_file() and p.suffix.lower() == ".txt")
        if not candidates:
            raise click.ClickException(f"No .txt files found in {in_p}")

        pending = []
        skipped = []
        for inp in candidates:
            out_file = out_p / (inp.stem + ".jsonl")
            if out_file.exists():
                skipped.append(inp)
            else:
                pending.append(inp)

        click.echo(f"Batch mode: {len(candidates)} file(s) in {in_p} → {out_p}")
        click.echo(f"  Model: {config.model}")
        if skipped:
            click.echo(f"  Skipping {len(skipped)} already-processed file(s).")
        if not pending:
            click.echo("  Nothing to do.")
            return
        click.echo(f"  Processing {len(pending)} file(s).")

        # Load model once for the whole batch
        clf = EmotionClassifier(config)

        try:
            from tqdm import tqdm as _tqdm

            _iter = _tqdm(pending, desc="classify", unit="file")
        except ImportError:
            _iter = pending

        for inp in _iter:
            out_file = out_p / (inp.stem + ".jsonl")
            if not quiet:
                click.echo(f"\n  {inp.name} → {out_file.name}")
            records = clf.classify_chapter(
                str(inp),
                chapter_id=None,
                verbose=(not quiet),
            )
            EmotionClassifier.write_jsonl(records, str(out_file))

        click.echo(f"\nBatch complete: {len(pending)} file(s) written to {out_p}")
        return

    # ── Single-file mode ──────────────────────────────────────────────────────
    if out_p.is_dir():
        raise click.UsageError(
            "OUTPUT_PATH is a directory but INPUT_PATH is a file. Provide a .jsonl output path."
        )

    # Coerce chapter_id to int if it looks like one
    if chapter_id is not None:
        try:
            chapter_id = int(chapter_id)
        except ValueError:
            pass  # keep as string

    clf = EmotionClassifier(config)

    click.echo(f"Input:  {input_path}")
    click.echo(f"Output: {output_path}")
    click.echo(f"Model:  {config.model}")

    records = clf.classify_chapter(
        input_path,
        chapter_id=chapter_id,
        verbose=not quiet,
    )

    EmotionClassifier.write_jsonl(records, output_path)


if __name__ == "__main__":
    classify_emotions()
