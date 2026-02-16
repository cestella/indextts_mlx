"""CLI: classify emotions + pauses for each sentence in a chapter text file.

Usage::

    indextts classify-emotions INPUT.txt OUTPUT.jsonl [options]
    indextts classify-emotions INPUT_DIR/ OUTPUT_DIR/ [options]
"""

import json
import sys
import time
from pathlib import Path

import click


def _write_classify_status(
    status_path: Path,
    file_index: int,
    total_files: int,
    file_name: str,
    file_done: bool,
    sentence_index: int,
    total_sentences: int,
    file_sentence_times: list,
    sentence_times: list,
) -> None:
    """Atomically write a JSON status snapshot for the web UI to poll."""
    mean_sent = sum(sentence_times) / len(sentence_times) if sentence_times else None
    mean_file = sum(file_sentence_times) / len(file_sentence_times) if file_sentence_times else None
    sents_left = total_sentences - sentence_index
    files_left = total_files - file_index - (1 if file_done else 0)
    sent_eta = mean_sent * sents_left if mean_sent else None
    # File-level ETA requires at least one completed file
    file_eta = mean_file * files_left if (mean_file and files_left > 0) else None
    if file_eta is not None and sent_eta is not None:
        file_eta += sent_eta

    payload = {
        "file_index": file_index,
        "total_files": total_files,
        "file_name": file_name,
        "sentence_index": sentence_index,
        "total_sentences": total_sentences,
        "sent_eta_s": round(sent_eta, 1) if sent_eta is not None else None,
        "file_eta_s": round(file_eta, 1) if file_eta is not None else None,
    }
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(status_path)


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
    "--no-boundary-detection",
    is_flag=True,
    default=False,
    help="Disable semantic boundary detection for pause label upgrading.",
)
@click.option(
    "--boundary-threshold",
    default=0.85,
    show_default=True,
    type=float,
    help="Cosine similarity threshold for boundary detection (lower = fewer boundaries).",
)
@click.option(
    "--status",
    "status_dir",
    default=None,
    type=click.Path(),
    help="Directory to write classify_status.json for web UI polling.",
)
@click.option(
    "--quiet",
    is_flag=True,
    default=False,
    help="Suppress per-sentence progress output.",
)
@click.option(
    "--head",
    default=None,
    type=int,
    help="Only classify the first N sentences (useful for quick testing).",
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
    no_boundary_detection: bool,
    boundary_threshold: float,
    status_dir: str,
    quiet: bool,
    head: int | None,
) -> None:
    """Classify emotions + pauses for each sentence in INPUT_PATH and write to OUTPUT_PATH.

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
        use_boundary_detection=not no_boundary_detection,
        boundary_threshold=boundary_threshold,
    )
    if model is not None:
        config.model = model

    in_p = Path(input_path)
    out_p = Path(output_path)
    status_path = Path(status_dir) / "classify_status.json" if status_dir else None
    if status_path:
        status_path.parent.mkdir(parents=True, exist_ok=True)

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

        clf = EmotionClassifier(config)

        try:
            from tqdm import tqdm as _tqdm
            _iter = _tqdm(pending, desc="classify", unit="file")
        except ImportError:
            _iter = pending

        file_sentence_times: list[float] = []

        for file_idx, inp in enumerate(pending):
            out_file = out_p / (inp.stem + ".jsonl")
            if not quiet:
                click.echo(f"\n  {inp.name} → {out_file.name}")

            # Count sentences first for status reporting
            _sent_times: list[float] = []
            _sent_start = [0.0]
            _total_sents = [0]

            def _on_sentence(sent_idx: int, total: int) -> None:
                now = time.monotonic()
                elapsed = now - _sent_start[0]
                _sent_start[0] = now
                _sent_times.append(elapsed)
                _total_sents[0] = total
                if status_path:
                    _write_classify_status(
                        status_path,
                        file_index=file_idx,
                        total_files=len(pending),
                        file_name=inp.name,
                        file_done=False,
                        sentence_index=sent_idx + 1,
                        total_sentences=total,
                        file_sentence_times=file_sentence_times,
                        sentence_times=_sent_times,
                    )

            _sent_start[0] = time.monotonic()
            records = clf.classify_chapter(
                str(inp),
                chapter_id=inp.stem,
                verbose=(not quiet),
                on_sentence=_on_sentence,
                head=head,
            )
            EmotionClassifier.write_jsonl(records, str(out_file))

            # Record total wall time for this file (sum of per-sentence times)
            if _sent_times:
                file_sentence_times.append(sum(_sent_times))

            if status_path:
                _write_classify_status(
                    status_path,
                    file_index=file_idx + 1,
                    total_files=len(pending),
                    file_name=inp.name,
                    file_done=True,
                    sentence_index=_total_sents[0],
                    total_sentences=_total_sents[0],
                    file_sentence_times=file_sentence_times,
                    sentence_times=_sent_times,
                )

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

    _sent_times: list[float] = []
    _sent_start = [time.monotonic()]
    _total_sents = [0]

    def _on_sentence(sent_idx: int, total: int) -> None:
        now = time.monotonic()
        elapsed = now - _sent_start[0]
        _sent_start[0] = now
        _sent_times.append(elapsed)
        _total_sents[0] = total
        if status_path:
            _write_classify_status(
                status_path,
                file_index=0,
                total_files=1,
                file_name=in_p.name,
                file_done=False,
                sentence_index=sent_idx + 1,
                total_sentences=total,
                file_sentence_times=[],
                sentence_times=_sent_times,
            )

    records = clf.classify_chapter(
        input_path,
        chapter_id=chapter_id,
        verbose=not quiet,
        on_sentence=_on_sentence,
        head=head,
    )

    EmotionClassifier.write_jsonl(records, output_path)

    if status_path:
        _write_classify_status(
            status_path,
            file_index=1,
            total_files=1,
            file_name=in_p.name,
            file_done=True,
            sentence_index=_total_sents[0],
            total_sentences=_total_sents[0],
            file_sentence_times=[sum(_sent_times)] if _sent_times else [],
            sentence_times=_sent_times,
        )


if __name__ == "__main__":
    classify_emotions()
