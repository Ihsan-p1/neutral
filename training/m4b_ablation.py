"""Ablation helpers for M4b-1, in numpy only so the tests run in serving CI.

NLTK chunking happens in `run_m4b.py`; this module gets the chunks already
built, so it never imports nltk.
"""

from collections.abc import Sequence

import numpy as np

# A token or entity phrase, and its NLTK entity label (None for a plain token).
Chunk = tuple[str, str | None]

ENTITY_WORDS = {
    "PERSON": "someone",
    "ORGANIZATION": "an organization",
    "GPE": "a place",
    "LOCATION": "a place",
    "GSP": "a place",
}
OTHER_ENTITY = "something"


def mask_chunks(chunks: Sequence[Chunk]) -> tuple[str, str, int]:
    """(original, masked, entity count), both sides space-joined from the same chunks.

    Joining both sides the same way means the only difference the model sees
    is the replacement, not NLTK's tokenisation.
    """
    original, masked, count = [], [], 0
    for text, label in chunks:
        original.append(text)
        if label is None:
            masked.append(text)
        else:
            masked.append(ENTITY_WORDS.get(label, OTHER_ENTITY))
            count += 1
    return " ".join(original), " ".join(masked), count


def word_count(text: str) -> int:
    return len(text.split())


def length_quartiles(lengths: np.ndarray) -> np.ndarray:
    """Quartile 0..3 of each length, edges from these lengths themselves."""
    lengths = np.asarray(lengths, dtype=np.float64)
    edges = np.quantile(lengths, [0.25, 0.5, 0.75])
    return np.searchsorted(edges, lengths, side="right")


def entity_verdict(point: float, lo: float) -> str:
    """Verdict on F1(original) - F1(masked), fixed before the run (spec 7.1)."""
    if point > 0 and lo > 0:
        return "relies on entity names"
    return "no measurable reliance on entity names"


def length_verdict(lo: float) -> str:
    """Verdict on F1(length-only) - F1(constant), fixed before the run (spec 7.2)."""
    if lo > 0:
        return "length alone carries signal"
    return "length alone carries no measurable signal"
