"""M4B audiobook creation using m4b-tool and ISBN metadata."""

import json
import re
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class M4bCreatorConfig:
    """Configuration for M4B audiobook creation.

    Args:
        m4b_tool_args: Dictionary of additional arguments to pass to m4b-tool.
            Keys are argument names (without '--' prefix).
            Values are argument values, or empty string for flags without values.

            Example:
                {
                    "audio-bitrate": "64k",
                    "use-filenames-as-chapters": "",  # Flag with no value
                    "jobs": "4"
                }
    """

    m4b_tool_args: dict[str, str] = field(default_factory=dict)

    def to_json(self, path: Path) -> None:
        """Serialize config to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"m4b_tool_args": self.m4b_tool_args}, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "M4bCreatorConfig":
        """Load config from JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(m4b_tool_args=data.get("m4b_tool_args", {}))


class M4bCreator:
    """Create M4B audiobooks from audio chapters using m4b-tool."""

    def __init__(self, config: M4bCreatorConfig):
        self.config = config

    def create_m4b(
        self,
        isbn: str,
        chapters_dir: Path,
        output_parent_dir: Path,
        verbose: bool = False,
    ) -> Path:
        """Create M4B audiobook from chapters using ISBN metadata.

        Args:
            isbn: ISBN-10 or ISBN-13 for metadata lookup
            chapters_dir: Directory containing audio chapter files
            output_parent_dir: Parent directory where .m4b file will be created
            verbose: Whether to print verbose output

        Returns:
            Path to created .m4b file

        Raises:
            ValueError: If ISBN metadata cannot be fetched
            RuntimeError: If m4b-tool execution fails
        """
        print(f"Fetching metadata for ISBN: {isbn}")

        metadata = self._fetch_metadata(isbn)

        print("\nBook Metadata:")
        print(f"  Title: {metadata['title']}")
        print(f"  Author(s): {metadata['authors']}")
        print(f"  Year: {metadata['year']}")
        if metadata["publisher"]:
            print(f"  Publisher: {metadata['publisher']}")
        if metadata["language"]:
            print(f"  Language: {metadata['language']}")
        if metadata["description"]:
            desc_preview = (
                metadata["description"][:100] + "..."
                if len(metadata["description"]) > 100
                else metadata["description"]
            )
            print(f"  Description: {desc_preview}")

        filename = self._sanitize_filename(metadata["title"])
        output_file = output_parent_dir / f"{filename}.m4b"

        cover_path = None
        if metadata.get("cover_url"):
            cover_path = self._download_cover(metadata["cover_url"], verbose)
            if cover_path:
                print("  Cover art: Downloaded")

        try:
            cmd = self._build_command(metadata, chapters_dir, output_file, cover_path)

            print("\nExecuting m4b-tool command:")
            print(f"  {' '.join(cmd)}")
            print()

            self._execute_command(cmd, verbose)

        finally:
            if cover_path and cover_path.exists():
                cover_path.unlink()

        return output_file

    def _fetch_metadata(self, isbn: str) -> dict:
        """Fetch book metadata from ISBN using isbnlib."""
        try:
            import isbnlib  # type: ignore[import-untyped]
        except ImportError as err:
            raise ImportError(
                "isbnlib is not installed. Install with: pip install isbnlib"
            ) from err

        isbn_clean = isbnlib.canonical(isbn)
        if not isbn_clean:
            raise ValueError(f"Invalid ISBN: {isbn}")

        # Try multiple services — Google Books first, then Open Library, then Wikipedia
        meta = None
        last_err = None
        for service in ("goob", "openl", "wiki"):
            try:
                meta = isbnlib.meta(isbn_clean, service=service)
                if meta:
                    break
            except Exception as exc:
                print(f"  {service} failed: {exc}")
                last_err = exc
        if not meta:
            raise RuntimeError(
                f"All ISBN metadata services failed for {isbn}: {last_err}"
            ) from last_err

        description = ""
        try:
            desc = isbnlib.desc(isbn_clean)
            if desc:
                description = desc
        except Exception:
            pass

        cover_url = ""
        try:
            cover = isbnlib.cover(isbn_clean)
            if cover:
                cover_url = cover.get("thumbnail", cover.get("smallThumbnail", ""))
        except Exception:
            pass

        return {
            "title": meta.get("Title", "Unknown Title"),
            "authors": ", ".join(meta.get("Authors", ["Unknown Author"])),
            "year": meta.get("Year", ""),
            "publisher": meta.get("Publisher", ""),
            "language": meta.get("Language", ""),
            "description": description,
            "cover_url": cover_url,
        }

    def _download_cover(self, cover_url: str, verbose: bool = False) -> Path | None:
        """Download cover image to a temporary file."""
        if not cover_url:
            return None
        try:
            if verbose:
                print(f"Downloading cover art from: {cover_url}")
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, mode="wb") as tmp:
                temp_path = Path(tmp.name)
            urllib.request.urlretrieve(cover_url, temp_path)
            if verbose:
                print(f"  Cover downloaded: {temp_path.stat().st_size / 1024:.1f} KB")
            return temp_path
        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to download cover: {e}")
            return None

    def _sanitize_filename(self, title: str) -> str:
        """Sanitize title for use as a filename."""
        sanitized = re.sub(r"[^\w\s-]", "", title)
        sanitized = re.sub(r"[-\s]+", "_", sanitized)
        return sanitized.lower().strip("_")

    def _build_command(
        self,
        metadata: dict,
        chapters_dir: Path,
        output_file: Path,
        cover_path: Path | None = None,
    ) -> list:
        """Build m4b-tool merge command."""
        cmd = ["m4b-tool", "merge", str(chapters_dir)]
        cmd.extend(["--output-file", str(output_file)])
        cmd.extend(["--name", metadata["title"]])
        cmd.extend(["--artist", metadata["authors"]])
        cmd.extend(["--albumartist", metadata["authors"]])
        cmd.extend(["--album", metadata["title"]])
        if metadata["year"]:
            cmd.extend(["--year", str(metadata["year"])])
        if metadata["description"]:
            cmd.extend(["--description", metadata["description"]])
            cmd.extend(["--longdesc", metadata["description"]])
        if metadata["publisher"]:
            cmd.extend(["--comment", f"Publisher: {metadata['publisher']}"])
        if cover_path and cover_path.exists():
            cmd.extend(["--cover", str(cover_path)])
        for key, value in self.config.m4b_tool_args.items():
            if value == "":
                cmd.append(f"--{key}")
            else:
                cmd.extend([f"--{key}", value])
        return cmd

    def _execute_command(self, cmd: list, verbose: bool = False) -> None:
        """Execute m4b-tool command with exponential-backoff retry for up to 10 minutes."""
        import time as _time

        deadline = _time.monotonic() + 600  # 10-minute window
        delay = 5.0
        attempt = 0

        while True:
            attempt += 1
            result = subprocess.run(cmd, capture_output=not verbose, text=True)
            if result.returncode == 0:
                return

            err = result.stderr if result.stderr else ""
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"m4b-tool failed after {attempt} attempt(s) "
                    f"(exit {result.returncode}):\n{err}"
                )

            wait = min(delay, remaining)
            print(
                f"m4b-tool failed (exit {result.returncode}), attempt {attempt}. "
                f"Retrying in {wait:.0f}s ({remaining:.0f}s remaining)..."
            )
            if err:
                print(f"  stderr: {err[:200]}")
            _time.sleep(wait)
            delay = min(delay * 2, 120)  # cap at 2-minute intervals

    def __repr__(self) -> str:
        return f"M4bCreator(args={self.config.m4b_tool_args})"
