from dataclasses import dataclass

import pysbd


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int


_segmenter = pysbd.Segmenter(language="en", clean=False, char_span=True)


def split_sentences(text: str) -> list[Sentence]:
    """Split an article into sentences, keeping offsets into the source text.

    Offsets are preserved so the UI can highlight sentences in the exact text
    the user submitted, without re-matching strings.
    """
    if not text or not text.strip():
        return []

    out: list[Sentence] = []
    for span in _segmenter.segment(text):
        raw = span.sent
        stripped = raw.strip()
        if not stripped:
            continue
        # Shift the offset past any leading whitespace so it still maps to the
        # original text after stripping.
        lead = len(raw) - len(raw.lstrip())
        start = span.start + lead
        out.append(Sentence(text=stripped, start=start, end=start + len(stripped)))
    return out
