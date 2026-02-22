import re
from pathlib import Path
from typing import Optional

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub


class EPUBChapter:
    """Represents a chapter extracted from an EPUB file."""

    def __init__(self, number: int, title: str, content: str, toc_item=None):
        self.number = number
        self.title = title
        self.content = content
        self.word_count = len(content.split())
        self.toc_item = toc_item  # Original TOC item from epub

    def __repr__(self):
        return f"EPUBChapter({self.number}: {self.title}, {self.word_count} words)"


class EPUBParser:
    """Parser for EPUB files that extracts chapters using the book's native structure."""

    # -------- heuristics / patterns --------
    NOTE_LIKE_CLASS_RE = re.compile(r"\b(footnote|endnote|noteref|fnref|fn|note)\b", re.I)
    NOTE_LIKE_ID_RE = re.compile(r"\b(footnote|endnote|noteref|fnref|fn|notes)\b", re.I)

    # Most common “note target” href patterns
    NOTE_HREF_RE = re.compile(
        r"(?i)("
        r"#(fn|footnote|endnote|note)\d*\b|"
        r"(footnote|endnote|notes?)\.x?html?#|"
        r"(footnote|endnote|notes?)\b"
        r")"
    )

    # Titles to skip when using spine fallback
    SKIP_TITLES_RE = re.compile(
        r"(?i)"
        r"(\babout the author\b|\bnote on the author\b|\bby the same author\b"
        r"|\bcopyright\b|\backnowledg|\bcontents\b|\btable of contents\b|\btoc\b"
        r"|\bindex\b|\bbibliograph|\bglossary\b|\bnotes\b|\bendnotes\b|\balso by\b)"
    )

    # Stray artifacts that often appear after DOM cleanup
    STRAY_BRACKETED_NOTE_RE = re.compile(r"\[\s*\d+\s*\]")
    STRAY_NUMERIC_LINE_RE = re.compile(r"(?m)^\s*\d+\s*$")

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.book = epub.read_epub(str(filepath))
        self.chapters: list[EPUBChapter] = []
        self.extraction_method = None  # Track which method was used

    # ---------- DOM cleanup helpers ----------

    def _soup(self, html_content: str) -> BeautifulSoup:
        """
        Prefer an HTML/XHTML-capable parser; 'xml' is often too strict for real-world EPUB XHTML.
        """
        import warnings
        from bs4 import XMLParsedAsHTMLWarning

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            return BeautifulSoup(html_content, "lxml")

    def _strip_notes(self, soup: BeautifulSoup) -> None:
        """
        Remove footnote/endnote markers and their content blocks.
        This is layered:
          A) standards-based epub:type / ARIA roles
          B) common container patterns by class/id
          C) href-based heuristics
        """
        # A) Standards + ARIA roles for refs
        for a in soup.find_all("a", attrs={"epub:type": "noteref"}):
            a.decompose()
        for a in soup.find_all("a", attrs={"role": "doc-noteref"}):
            a.decompose()

        # A) Standards + ARIA roles for note bodies
        for el in soup.find_all(attrs={"epub:type": "footnote"}):
            el.decompose()
        for el in soup.find_all(attrs={"epub:type": "endnote"}):
            el.decompose()
        for el in soup.find_all(attrs={"role": "doc-footnote"}):
            el.decompose()
        for el in soup.find_all(attrs={"role": "doc-endnote"}):
            el.decompose()

        # Remove backlink anchors commonly inside notes
        for a in soup.find_all("a", attrs={"role": "doc-backlink"}):
            a.decompose()
        for a in soup.find_all("a", attrs={"epub:type": "referrer"}):
            a.decompose()

        # B) Common containers by id
        for el in soup.find_all(id=self.NOTE_LIKE_ID_RE):
            # Only delete containers likely to be a note apparatus, not an inline span.
            if el.name in ("aside", "section", "div", "ol", "ul"):
                el.decompose()

        # B) Common containers by class
        for el in soup.find_all(class_=self.NOTE_LIKE_CLASS_RE):
            if el.name in ("aside", "section", "div", "ol", "ul", "li"):
                el.decompose()

        # B) Very common "footnotes list" patterns
        for el in soup.select(
            "ol.footnotes, section.footnotes, div.footnotes, "
            "div.footnote, section.footnote, aside.footnote, "
            "div.endnotes, section.endnotes, ol.endnotes, "
            "#footnotes, #endnotes, #notes"
        ):
            el.decompose()

        # C) Href heuristic: remove anchors pointing to note-like targets
        # Also remove tiny anchor text that is typical for footnote markers: "1", "*", "†"
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            if self.NOTE_HREF_RE.search(href):
                a.decompose()
                continue
            txt = a.get_text(strip=True)
            if txt and (txt.isdigit() or txt in {"*", "†", "‡"}):
                # only delete if it looks like a reference, not a normal numeric link
                if a.get("class") and any(
                    self.NOTE_LIKE_CLASS_RE.search(c) for c in a.get("class", [])
                ):
                    a.decompose()

    def _strip_non_narration_elements(self, soup: BeautifulSoup) -> None:
        """
        Remove things you generally don't want narrated for audiobooks.
        """
        # Remove obvious non-text
        for el in soup(["script", "style", "head", "meta", "noscript"]):
            el.decompose()

        # Drop tables and common table wrappers
        for el in soup(["table", "thead", "tbody", "tfoot", "tr", "th", "td"]):
            el.decompose()

        # Drop code/pre
        for el in soup(["pre", "code", "kbd", "samp"]):
            el.decompose()

        # Drop images/figures (captions often become weird)
        for el in soup(["img", "svg", "figure", "figcaption", "canvas"]):
            el.decompose()

        # Drop math-like tags that sometimes appear
        for el in soup(["math"]):
            el.decompose()

        # Links: keep visible text, drop URL
        for a in soup.find_all("a"):
            a.unwrap()  # keep inner text only

    def _normalize_text(self, text: str) -> str:
        """
        Normalize whitespace and remove common leftover artifacts.
        """
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove some common stray note markers that survive as text
        text = self.STRAY_BRACKETED_NOTE_RE.sub("", text)
        text = self.STRAY_NUMERIC_LINE_RE.sub("", text)

        # Strip trailing spaces per line and collapse consecutive empty lines
        # into a single empty line (preserving paragraph boundaries from <p> tags).
        lines = [ln.strip() for ln in text.split("\n")]
        collapsed: list[str] = []
        for ln in lines:
            if ln == "":
                if collapsed and collapsed[-1] != "":
                    collapsed.append("")
            else:
                collapsed.append(ln)
        # Drop leading/trailing empty lines
        while collapsed and collapsed[-1] == "":
            collapsed.pop()
        while collapsed and collapsed[0] == "":
            collapsed.pop(0)
        lines = collapsed

        # Drop lines that are only punctuation/whitespace (orphaned quote chars etc.)
        # but preserve empty lines (paragraph boundary markers).
        lines = [ln for ln in lines if ln == "" or re.search(r"\w", ln)]

        # Re-join lines where the previous line ends with an open quote and the
        # next line starts with the quoted content (epub line-break inside a span).
        merged: list[str] = []
        for ln in lines:
            if merged and re.search(r'["\u201c\u2018]\s*$', merged[-1]):
                merged[-1] = merged[-1].rstrip() + ln
            else:
                merged.append(ln)
        lines = merged

        # Ensure heading-like lines are separated from surrounding text by blank
        # lines so they become their own paragraph when later split on \n\n.
        # A heading must be: short (≤ 80 chars), no terminal sentence punctuation,
        # AND look like a heading — either ALL CAPS, Title Case with no lowercase
        # run longer than 3 chars, or starts with "Chapter"/"Part"/"Section"/etc.
        HEADING_WORD_RE = re.compile(
            r"^(chapter|part|section|epilogue|prologue|introduction|conclusion|"
            r"preface|foreword|afterword|appendix)\b",
            re.I,
        )

        def _is_heading(ln: str) -> bool:
            if len(ln) > 80:
                return False
            if re.search(r"[.!?,;:…\"\u201d]$", ln):
                return False
            if HEADING_WORD_RE.match(ln):
                return True
            # All-caps (with possible spaces/digits/roman numerals): ≥ 4 chars
            if ln.isupper() and len(ln) >= 4:
                return True
            return False

        result: list[str] = []
        for i, ln in enumerate(lines):
            prev_blank = (not result) or (result[-1] == "")
            # Also treat a short unpunctuated line as a heading if it immediately
            # follows a blank line (i.e. it's a subtitle after a chapter heading).
            is_subtitle = (
                prev_blank
                and len(ln) <= 60
                and not re.search(r"[.!?,;:…\"\u201d]$", ln)
                and re.search(r"[A-Z]", ln)  # at least one capital
            )
            if (_is_heading(ln) or is_subtitle) and not prev_blank:
                result.append("")
            result.append(ln)
            next_is_text = (i + 1 < len(lines)) and lines[i + 1] != ""
            if (_is_heading(ln) or is_subtitle) and next_is_text:
                result.append("")

        text = "\n".join(result)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return text

    # ---------- core extraction ----------

    def extract_text_from_html(self, html_content: str) -> str:
        """Extract clean, audiobook-friendly text from HTML content."""
        soup = self._soup(html_content)

        self._strip_non_narration_elements(soup)
        self._strip_notes(soup)

        # Insert an extra newline after block-level elements so that
        # consecutive <p> tags produce \n\n (paragraph boundary) rather
        # than a single \n when get_text("\n") joins all text nodes.
        for tag in soup.find_all(["p", "blockquote"]):
            tag.insert_after("\n")

        text = soup.get_text("\n")

        return self._normalize_text(text)

    def get_toc_chapters(self, min_words: int = 100) -> list[EPUBChapter]:
        """Extract chapters based on the EPUB's Table of Contents."""
        chapters: list[EPUBChapter] = []
        chapter_num = 1

        toc = self.book.toc

        def process_toc_item(item, chapter_num: int):
            """Recursively process TOC items."""
            if isinstance(item, tuple):
                section, children = item
                chapters_from_section: list[EPUBChapter] = []
                for child in children:
                    child_chapters, chapter_num = process_toc_item(child, chapter_num)
                    chapters_from_section.extend(child_chapters)
                return chapters_from_section, chapter_num
            else:
                title = item.title if hasattr(item, "title") else str(item)
                href = item.href if hasattr(item, "href") else None
                if href:
                    # Skip known front/back matter by title
                    if title and self.SKIP_TITLES_RE.search(title):
                        return [], chapter_num
                    content = self.get_content_by_href(href)
                    if content and len(content.split()) >= min_words:
                        chapter = EPUBChapter(
                            number=chapter_num,
                            title=title,
                            content=content,
                            toc_item=item,
                        )
                        return [chapter], chapter_num + 1
                return [], chapter_num

        for item in toc:
            item_chapters, chapter_num = process_toc_item(item, chapter_num)
            chapters.extend(item_chapters)

        return chapters

    def get_content_by_href(self, href: str) -> Optional[str]:
        """Get content for a specific href from the EPUB."""
        if "#" in href:
            href = href.split("#")[0]

        for item in self.book.get_items():
            if (
                item.get_name() == href or item.get_id() == href
            ) and item.get_type() == ebooklib.ITEM_DOCUMENT:
                html_content = item.get_content().decode("utf-8", errors="ignore")
                return self.extract_text_from_html(html_content)

        return None

    def get_spine_chapters(self, min_words: int = 100) -> list[EPUBChapter]:
        """
        Extract chapters based on the EPUB's spine (reading order).
        Fallback method if TOC is not available or incomplete.
        """
        chapters: list[EPUBChapter] = []
        chapter_num = 1

        for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            item_name = item.get_name()
            html_content = item.get_content().decode("utf-8", errors="ignore")
            text_content = self.extract_text_from_html(html_content)

            # Skip empty/very short items (often nav, cover, etc.)
            if len(text_content.split()) < min_words:
                continue

            # Try to extract a title from HTML (more robust parser)
            soup = self._soup(html_content)
            title = None
            for tag in ["h1", "h2", "h3", "title"]:
                title_elem = soup.find(tag)
                if title_elem:
                    candidate = title_elem.get_text().strip()
                    if candidate:
                        title = candidate
                        break

            # Fallback to filename if no title
            if not title:
                title = f"Section {chapter_num}"
                if item_name:
                    name_parts = (Path(item_name).stem.replace("_", " ").replace("-", " ")).strip()
                    if name_parts and not name_parts.isdigit():
                        title = name_parts.title()

            # Skip obvious back matter in spine mode (optional but helpful)
            if title and self.SKIP_TITLES_RE.search(title):
                # If you prefer to keep About the Author etc., remove this block.
                continue

            chapter = EPUBChapter(
                number=chapter_num,
                title=title,
                content=text_content,
            )
            chapters.append(chapter)
            chapter_num += 1

        return chapters

    def extract_chapters(self, use_toc: bool = True, min_words: int = 100) -> list[EPUBChapter]:
        """
        Extract all chapters from the EPUB.

        Args:
            use_toc: If True, try to use TOC first, fallback to spine if needed

        Returns:
            List of EPUBChapter objects and sets self.extraction_method
        """
        if use_toc:
            chapters = self.get_toc_chapters(min_words=min_words)
            if len(chapters) < 3:
                chapters = self.get_spine_chapters(min_words=min_words)
                self.extraction_method = "SPINE (fallback)"
            else:
                self.extraction_method = "TOC"
        else:
            chapters = self.get_spine_chapters(min_words=min_words)
            self.extraction_method = "SPINE (direct)"

        self.chapters = chapters
        return chapters

    def get_metadata(self) -> dict:
        """Extract book metadata."""
        metadata = {}

        title = self.book.get_metadata("DC", "title")
        if title:
            metadata["title"] = title[0][0] if title else "Unknown"

        creator = self.book.get_metadata("DC", "creator")
        if creator:
            metadata["author"] = creator[0][0] if creator else "Unknown"

        language = self.book.get_metadata("DC", "language")
        if language:
            metadata["language"] = language[0][0] if language else "Unknown"

        publisher = self.book.get_metadata("DC", "publisher")
        if publisher:
            metadata["publisher"] = publisher[0][0] if publisher else "Unknown"

        date = self.book.get_metadata("DC", "date")
        if date:
            metadata["date"] = date[0][0] if date else "Unknown"

        identifier = self.book.get_metadata("DC", "identifier")
        if identifier:
            for ident in identifier:
                if "isbn" in str(ident).lower():
                    metadata["isbn"] = ident[0]
                    break

        return metadata

    def get_chapter_summary_info(self) -> list[dict]:
        """Get summary information about detected chapters."""
        return [
            {"number": ch.number, "title": ch.title, "word_count": ch.word_count}
            for ch in self.chapters
        ]

    def save_chapters_as_text(self, output_dir: Path):
        """Save each chapter as a separate text file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        for chapter in self.chapters:
            safe_title = re.sub(r"[^\w\s-]", "", chapter.title)
            safe_title = re.sub(r"[-\s]+", "-", safe_title)
            filename = f"chapter_{chapter.number:02d}_{safe_title[:50]}.txt"
            filepath = output_dir / filename
            filepath.write_text(chapter.content, encoding="utf-8")

        combined_path = output_dir / "all_chapters.txt"
        with open(combined_path, "w", encoding="utf-8") as f:
            for chapter in self.chapters:
                f.write(f"\n\n{'=' * 50}\n")
                f.write(f"CHAPTER {chapter.number}: {chapter.title}\n")
                f.write(f"{'=' * 50}\n\n")
                f.write(chapter.content)
                f.write("\n")
