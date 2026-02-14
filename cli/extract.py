#!/usr/bin/env python3
"""CLI: extract chapters from an EPUB file as plain-text files."""

from __future__ import annotations

import re
from pathlib import Path

import click


@click.command("extract")
@click.argument("epub_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_dir", type=click.Path(file_okay=False))
@click.option(
    "--toc/--no-toc",
    "use_toc",
    default=True,
    show_default=True,
    help="Use the book's Table of Contents to detect chapters (default). "
    "Falls back to spine order if TOC yields fewer than 3 chapters.",
)
@click.option(
    "--min-words",
    default=100,
    show_default=True,
    type=int,
    help="Minimum word count for a spine item to be kept as a chapter.",
)
@click.option(
    "--sentence-per-line/--no-sentence-per-line",
    default=True,
    show_default=True,
    help="Run spaCy sentence segmentation and write one sentence per line.",
)
@click.option(
    "--language",
    default="english",
    show_default=True,
    help="Language for spaCy sentence segmentation (english, french, spanish, …).",
)
@click.option(
    "--combined/--no-combined",
    default=False,
    show_default=True,
    help="Also write an all_chapters.txt that concatenates every chapter.",
)
@click.option("-v", "--verbose", is_flag=True, help="Print per-chapter details.")
def extract(epub_file, output_dir, use_toc, min_words, sentence_per_line, language, combined, verbose):
    """Extract chapters from EPUB_FILE and write them as .txt files into OUTPUT_DIR.

    Each chapter is written as chapter_NN_<title>.txt.  Footnotes, tables,
    code blocks, and images are stripped automatically.

    \b
    Examples:
      indextts extract book.epub ~/chapters/
      indextts extract book.epub ~/chapters/ --no-toc --min-words 200
      indextts extract book.epub ~/chapters/ --combined -v
    """
    try:
        from indextts_mlx.epub_extractor import EPUBParser
    except ImportError as e:
        raise click.ClickException(
            f"Missing dependency: {e}\n"
            "Install with: pip install ebooklib beautifulsoup4 lxml"
        )

    epub_path = Path(epub_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"Parsing {epub_path.name} ...")
    parser = EPUBParser(epub_path)

    meta = parser.get_metadata()
    if meta.get("title"):
        click.echo(f"  Title:  {meta['title']}")
    if meta.get("author"):
        click.echo(f"  Author: {meta['author']}")

    chapters = parser.extract_chapters(use_toc=use_toc, min_words=min_words)

    click.echo(f"  Extraction method: {parser.extraction_method}")
    click.echo(f"  Chapters found: {len(chapters)}")

    if not chapters:
        raise click.ClickException("No chapters extracted. Try --no-toc or a lower --min-words.")

    # Load spaCy once if sentence-per-line is requested
    nlp = None
    if sentence_per_line:
        try:
            from indextts_mlx.segmenter import Segmenter, SegmenterConfig
        except ImportError as e:
            raise click.ClickException(f"Missing dependency: {e}")
        seg_config = SegmenterConfig(language=language, use_pysbd=False)
        segmenter = Segmenter(seg_config)
        nlp = segmenter.nlp  # trigger load + error early
        click.echo(f"  spaCy model: {seg_config.resolved_spacy_model}")

    try:
        import ftfy as _ftfy
        _fix_text = _ftfy.fix_text
    except ImportError:
        _fix_text = lambda t: t  # noqa: E731

    def _to_sentences(text: str) -> str:
        """Return text with one sentence per line."""
        # Fix Unicode encoding issues first.
        text = _fix_text(text)
        # Process each paragraph independently (preserving heading/paragraph breaks
        # as natural sentence boundaries), but flatten single newlines within each
        # paragraph so spaCy gets clean prose for resegmentation.
        paragraphs = re.split(r"\n{2,}", text)
        sentences: list[str] = []
        for para in paragraphs:
            flat = re.sub(r"\s*\n\s*", " ", para).strip()
            flat = re.sub(r"  +", " ", flat)
            if not flat:
                continue
            # A paragraph that is itself a bare heading (single spaCy sentence,
            # no terminal punctuation) gets a period appended so TTS treats it
            # as a complete utterance rather than a dangling fragment.
            is_bare_heading = not re.search(r"[.!?…]$", flat)
            doc = nlp(flat)
            sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            if is_bare_heading and len(sents) == 1:
                sents[0] = sents[0] + "."
            sentences.extend(sents)
        return "\n".join(sentences)

    written = []
    for ch in chapters:
        safe_title = re.sub(r"[^\w\s-]", "", ch.title)
        safe_title = re.sub(r"[-\s]+", "-", safe_title).strip("-")
        filename = f"chapter_{ch.number:02d}_{safe_title[:50]}.txt"
        filepath = out_dir / filename
        content = _to_sentences(ch.content) if sentence_per_line else ch.content
        filepath.write_text(content, encoding="utf-8")
        written.append(filepath)
        if verbose:
            click.echo(f"  [{ch.number:02d}] {ch.title!r}  ({ch.word_count:,} words) → {filename}")

    if combined:
        combined_path = out_dir / "all_chapters.txt"
        with open(combined_path, "w", encoding="utf-8") as f:
            for ch, path in zip(chapters, written):
                f.write(f"\n\n{'=' * 50}\n")
                f.write(f"CHAPTER {ch.number}: {ch.title}\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(path.read_text(encoding="utf-8"))
                f.write("\n")
        click.echo(f"  Combined → {combined_path.name}")

    total_words = sum(ch.word_count for ch in chapters)
    click.echo(f"Wrote {len(written)} file(s) to {out_dir}  ({total_words:,} words total)")


if __name__ == "__main__":
    extract()
