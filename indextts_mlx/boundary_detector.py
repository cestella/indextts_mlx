"""Semantic boundary detector for chapter text.

Detects paragraph/passage boundaries between sentences using a two-tier approach:

  Tier 1 — Explicit paragraph breaks (``\\n\\n`` in the original text) are always
            treated as boundaries regardless of semantic similarity.

  Tier 2 — Semantic windows: for each candidate sentence boundary we compare
            the mean embedding of the ``window`` sentences to the left vs. the
            ``window`` sentences to the right using cosine similarity.  A boundary
            is inserted where cosine similarity drops below ``threshold``.

The result is a set of sentence indices that are *followed* by a boundary.
Sentence i being in the result means: insert a boundary *after* sentence i.
"""

from __future__ import annotations

from typing import List, Set


def detect_boundaries(
    sentences: List[str],
    original_text: str,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    window: int = 3,
    threshold: float = 0.85,
) -> Set[int]:
    """Return the set of sentence indices after which a major boundary occurs.

    A sentence index ``i`` in the returned set means that there is a significant
    topic/passage boundary between sentence ``i`` and sentence ``i+1``.

    Args:
        sentences: List of sentence strings in document order.
        original_text: The original unsegmented text, used to detect ``\\n\\n``
                       paragraph breaks.
        model_name: Sentence-transformers model for embeddings.
        window: Number of sentences on each side of a candidate boundary used
                to compute the local mean embedding.
        threshold: Cosine similarity below which a boundary is inserted.
                   Lower values → fewer boundaries; higher → more.

    Returns:
        Set of 0-based sentence indices followed by a boundary.
    """
    if len(sentences) < 2:
        return set()

    boundaries: Set[int] = set()

    # ── Tier 1: explicit paragraph breaks ────────────────────────────────────
    # Locate each sentence in the original text to detect double-newlines before
    # the next sentence.  We do a simple linear scan: find the end of sentence[i]
    # in the text, then check if there's a \n\n before the start of sentence[i+1].
    search_start = 0
    sent_positions: List[int] = []  # char start of each sentence in original_text
    for sent in sentences:
        pos = original_text.find(sent, search_start)
        if pos == -1:
            sent_positions.append(-1)
        else:
            sent_positions.append(pos)
            search_start = pos + len(sent)

    for i in range(len(sentences) - 1):
        pos_i = sent_positions[i]
        pos_next = sent_positions[i + 1]
        if pos_i == -1 or pos_next == -1:
            continue
        # Text between end of sentence[i] and start of sentence[i+1]
        between = original_text[pos_i + len(sentences[i]) : pos_next]
        if "\n\n" in between:
            boundaries.add(i)

    # ── Tier 2: cosine similarity window ─────────────────────────────────────
    # Only run if sentence-transformers is available.
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        # No sentence-transformers — only explicit paragraph breaks are used.
        return boundaries

    n = len(sentences)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
    # Normalize once
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    emb = embeddings / norms  # (n, d)

    for i in range(n - 1):
        # Already a boundary from tier 1 — skip computation
        if i in boundaries:
            continue
        # Left window: sentences max(0, i-window+1) .. i  (inclusive)
        left_lo = max(0, i - window + 1)
        left_mean = emb[left_lo : i + 1].mean(axis=0)
        # Right window: sentences i+1 .. min(n-1, i+window)
        right_hi = min(n, i + 1 + window)
        right_mean = emb[i + 1 : right_hi].mean(axis=0)
        # Cosine similarity of the two window means (already unit-normalised per
        # sentence but the mean is not unit-length, so normalise again)
        lnorm = np.linalg.norm(left_mean)
        rnorm = np.linalg.norm(right_mean)
        if lnorm == 0 or rnorm == 0:
            continue
        cos_sim = float(np.dot(left_mean / lnorm, right_mean / rnorm))
        if cos_sim < threshold:
            boundaries.add(i)

    return boundaries
