"""Outlet units, their AllSides ratings, and the Baly articles scored for PRD 8.3.

Torch-free so the tests run in serving CI. The selection rules here are fixed by
the spec (sections 5 and 6) before any article is scored.
"""

import csv
import hashlib
import json
import pathlib
import random
from collections import Counter
from dataclasses import dataclass

from neutral.segment import split_sentences
from training.basil import normalise
from training.split import group_key_from_url

RATINGS_CSV = pathlib.Path("training/outlet_ratings.csv")
BALY_ROOT = pathlib.Path("data/baly/repo/data/jsons")
BALY_SIZE = 37554
PER_UNIT = 50
MIN_UNITS = 12
MIN_SENTENCES = 5
MAX_CHARS = 50_000
MIN_OVERLAP_CHARS = 40
UNRATED = ("Mixed", "Not Rated")

_LEAN = {"Left": 2, "Lean Left": 1, "Center": 0, "Lean Right": 1, "Right": 2}
_SIDE = {
    "Left": "left",
    "Lean Left": "left",
    "Center": "center",
    "Lean Right": "right",
    "Right": "right",
}


def lean(rating: str | None) -> int | None:
    """Distance from centre: 0, 1 or 2. Mixed, Not Rated and missing give None."""
    return _LEAN.get(rating) if rating else None


def side(rating: str | None) -> str | None:
    return _SIDE.get(rating) if rating else None


@dataclass(frozen=True)
class Unit:
    name: str
    kind: str
    sources: tuple[str, ...]
    rating_2020: str | None
    snapshot_2020: str | None
    rating_2026: str | None
    babe_outlet: str | None


@dataclass(frozen=True)
class Article:
    id: str
    source: str
    url: str
    text: str
    label: str


def load_units(path: pathlib.Path) -> list[Unit]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        Unit(
            name=row["unit"],
            kind=row["kind"],
            sources=tuple(row["baly_sources"].split("|")),
            rating_2020=row["rating_2020"] or None,
            snapshot_2020=row["snapshot_2020"] or None,
            rating_2026=row["rating_2026"] or None,
            babe_outlet=row["babe_outlet"] or None,
        )
        for row in rows
    ]


def load_baly(root: pathlib.Path) -> list[Article]:
    articles = []
    for path in root.glob("*.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        articles.append(
            Article(
                id=raw["ID"],
                source=raw["source"],
                url=raw.get("url") or "",
                text=raw.get("content_original") or "",
                label=raw["bias_text"],
            )
        )
    return articles


def text_digest(text: str) -> str:
    """Short SHA-256 of an article's text, so a cached score is reused only for that text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_score_cache(path: pathlib.Path, fingerprint: str) -> tuple[dict[str, dict], int]:
    """Cached score records for one model, by article id; a later line wins.

    A run killed mid-write leaves a line cut short. Such lines are skipped and counted
    rather than blocking the resume.
    """
    if not path.exists():
        return {}, 0
    cached, skipped = {}, 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if record["fingerprint"] == fingerprint:
            cached[record["id"]] = record
    return cached, skipped


def babe_index(sentences: list[str]) -> list[str]:
    """Normalised BABE sentences long enough not to match an article by chance."""
    kept = {normalise(s) for s in sentences}
    return sorted(s for s in kept if len(s) >= MIN_OVERLAP_CHARS)


def _classify(article: Article, babe: list[str], babe_urls: set[str]) -> list[str]:
    """Counter keys for one article; the last key decides whether it is kept."""
    if not article.text.strip():
        return ["no_text"]
    if len(article.text) > MAX_CHARS:
        return ["too_long"]
    same_url = group_key_from_url(article.url) in babe_urls
    text = normalise(article.text)
    if any(sentence in text for sentence in babe):
        return ["babe_url_and_text" if same_url else "babe_text_only"]
    if len(split_sentences(article.text)) < MIN_SENTENCES:
        return ["too_short"]
    return ["babe_url_only", "kept"] if same_url else ["kept"]


def pick_articles(
    articles: list[Article],
    unit: Unit,
    babe: list[str],
    babe_urls: set[str],
    per_unit: int = PER_UNIT,
    seed: int = 0,
) -> tuple[list[Article], Counter]:
    """Walk the unit's articles in a seeded random order and keep the first `per_unit`
    that pass every filter.

    Walking a random permutation and stopping at `per_unit` gives a uniform sample of
    the passing articles while splitting only the articles it looks at.
    """
    mine = sorted((a for a in articles if a.source in unit.sources), key=lambda a: a.id)
    random.Random(f"{seed}:{unit.name}").shuffle(mine)
    kept: list[Article] = []
    counts: Counter = Counter()
    for article in mine:
        if len(kept) == per_unit:
            break
        keys = _classify(article, babe, babe_urls)
        counts.update(keys)
        if keys[-1] == "kept":
            kept.append(article)
    return kept, counts
