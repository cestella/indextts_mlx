#!/usr/bin/env python3
"""CLI to package a directory of audio chapters into a .m4b audiobook.

Looks up book metadata (title, author, year, cover art, description) via ISBN
and invokes m4b-tool to produce the final file.

Requires:
    pip install isbnlib
    brew install m4b-tool
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option(
    "--chapters-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Directory of audio chapter files (mp3, m4a, wav, …).",
)
@click.option(
    "--out",
    required=True,
    type=click.Path(),
    help="Output directory where the .m4b file will be written.",
)
@click.option(
    "--isbn",
    required=True,
    help="ISBN-10 or ISBN-13 used to fetch book metadata and cover art.",
)
@click.option(
    "--bitrate",
    default="64k",
    show_default=True,
    help="Audio bitrate passed to m4b-tool (e.g. 64k, 128k).",
)
@click.option(
    "--jobs",
    default="4",
    show_default=True,
    help="Number of parallel encoding jobs for m4b-tool.",
)
@click.option(
    "--use-filenames-as-chapters/--no-use-filenames-as-chapters",
    default=True,
    show_default=True,
    help="Pass --use-filenames-as-chapters to m4b-tool.",
)
@click.option(
    "--m4b-arg",
    "extra_args",
    multiple=True,
    metavar="KEY=VALUE",
    help="Extra arguments forwarded to m4b-tool as --KEY VALUE. "
    "Use KEY= (empty value) for flag-only args. Can be repeated.",
)
@click.option("-v", "--verbose", is_flag=True, help="Print m4b-tool output.")
def main(
    chapters_dir,
    out,
    isbn,
    bitrate,
    jobs,
    use_filenames_as_chapters,
    extra_args,
    verbose,
):
    """Package audio chapters into a .m4b audiobook with ISBN metadata.

    \b
    Example:
      indextts-m4b --chapters-dir ~/audio/chapters \\
                   --out ~/audio \\
                   --isbn 9780743273565
    """
    from indextts_mlx.m4b_creator import M4bCreator, M4bCreatorConfig

    # Build m4b_tool_args dict
    m4b_tool_args: dict[str, str] = {
        "audio-bitrate": bitrate,
        "jobs": jobs,
    }
    if use_filenames_as_chapters:
        m4b_tool_args["use-filenames-as-chapters"] = ""

    # Parse any extra --m4b-arg KEY=VALUE pairs
    for arg in extra_args:
        if "=" in arg:
            k, _, v = arg.partition("=")
            m4b_tool_args[k.strip()] = v.strip()
        else:
            m4b_tool_args[arg.strip()] = ""

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = M4bCreatorConfig(m4b_tool_args=m4b_tool_args)
    creator = M4bCreator(config)

    try:
        output_file = creator.create_m4b(
            isbn=isbn,
            chapters_dir=Path(chapters_dir),
            output_parent_dir=out_dir,
            verbose=verbose,
        )
    except ValueError as e:
        raise click.ClickException(str(e))
    except RuntimeError as e:
        raise click.ClickException(str(e))

    click.echo(f"\nM4B written to: {output_file}")


if __name__ == "__main__":
    main()
