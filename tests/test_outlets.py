import json

from training.outlets import (
    MAX_CHARS,
    Article,
    Unit,
    babe_index,
    lean,
    load_baly,
    load_score_cache,
    load_units,
    pick_articles,
    side,
    text_digest,
)

BABE_SENTENCE = "The senator gave a long and very rambling speech on the floor today."
UNIT = Unit("Fox News", "news", ("Fox News",), "Right", "20200601000000", "Right", None)


def _text(n: int) -> str:
    return " ".join(f"Sentence number {i} is here." for i in range(n))


def _article(i: int, text: str | None = None, url: str = "") -> Article:
    return Article(
        f"id{i:02d}", "Fox News", url or f"https://fox.com/a{i}", text or _text(6), "right"
    )


def test_lean_and_side_mapping():
    ratings = ["Left", "Lean Left", "Center", "Lean Right", "Right"]
    assert [lean(r) for r in ratings] == [2, 1, 0, 1, 2]
    assert lean("Mixed") is None
    assert lean("Not Rated") is None
    assert lean(None) is None
    assert side("Lean Left") == "left"
    assert side("Center") == "center"
    assert side("Right") == "right"
    assert side("Mixed") is None


def test_load_units_splits_sources_and_blanks(tmp_path):
    path = tmp_path / "ratings.csv"
    path.write_text(
        "unit,kind,baly_sources,allsides_path,rating_2020,snapshot_2020,rating_2026,babe_outlet\n"
        "Fox News,news,Fox Online News|Fox News,/news-source/fox-news-media-bias,"
        "Right,20200601000000,Right,\n"
        "Salon,news,Salon,,,,Left,\n",
        encoding="utf-8",
    )
    fox, salon = load_units(path)
    assert fox.sources == ("Fox Online News", "Fox News")
    assert fox.rating_2020 == "Right"
    assert fox.babe_outlet is None
    assert salon.rating_2020 is None
    assert salon.snapshot_2020 is None
    assert salon.rating_2026 == "Left"


def test_load_baly_reads_fields(tmp_path):
    full = {"ID": "a1", "source": "Vox", "url": "https://vox.com/x",
            "content_original": "Body text.", "bias_text": "left"}
    empty = {"ID": "a2", "source": "Vox", "url": "https://vox.com/y", "bias_text": "left"}
    (tmp_path / "a1.json").write_text(json.dumps(full), encoding="utf-8")
    (tmp_path / "a2.json").write_text(json.dumps(empty), encoding="utf-8")
    articles = sorted(load_baly(tmp_path), key=lambda a: a.id)
    assert articles[0] == Article("a1", "Vox", "https://vox.com/x", "Body text.", "left")
    assert articles[1].text == ""


def test_babe_index_normalises_and_drops_short():
    long = "A “quoted” sentence that is long enough to count here."
    assert babe_index(["Short one.", long, long]) == [
        'A "quoted" sentence that is long enough to count here.'
    ]


def test_pick_drops_babe_text_even_with_other_url():
    article = _article(1, text=_text(6) + " " + BABE_SENTENCE)
    kept, counts = pick_articles([article], UNIT, babe_index([BABE_SENTENCE]), set())
    assert kept == []
    assert counts["babe_text_only"] == 1


def test_pick_drops_same_url_and_text():
    article = _article(1, text=BABE_SENTENCE + " " + _text(6), url="https://fox.com/a1?x=1")
    kept, counts = pick_articles(
        [article], UNIT, babe_index([BABE_SENTENCE]), {"https://fox.com/a1"}
    )
    assert kept == []
    assert counts["babe_url_and_text"] == 1


def test_pick_keeps_same_url_without_shared_text():
    article = _article(1, url="https://fox.com/a1/")
    kept, counts = pick_articles(
        [article], UNIT, babe_index([BABE_SENTENCE]), {"https://fox.com/a1"}
    )
    assert kept == [article]
    assert counts["babe_url_only"] == 1
    assert counts["kept"] == 1


def test_pick_skips_too_long():
    article = _article(1, text="x" * (MAX_CHARS + 1))
    kept, counts = pick_articles([article], UNIT, [], set())
    assert kept == []
    assert counts["too_long"] == 1


def test_pick_counts_no_text():
    kept, counts = pick_articles([_article(1, text="   ")], UNIT, [], set())
    assert kept == []
    assert counts["no_text"] == 1


def test_pick_skips_too_short():
    kept, counts = pick_articles([_article(1, text=_text(4))], UNIT, [], set())
    assert kept == []
    assert counts["too_short"] == 1


def test_pick_ignores_other_sources():
    other = Article("x", "Vox", "https://vox.com/x", _text(6), "left")
    kept, counts = pick_articles([other], UNIT, [], set())
    assert kept == []
    assert sum(counts.values()) == 0


def test_pick_is_seeded_and_stops_at_per_unit():
    articles = [_article(i) for i in range(10)]
    first, counts = pick_articles(articles, UNIT, [], set(), per_unit=3)
    again, _ = pick_articles(list(reversed(articles)), UNIT, [], set(), per_unit=3)
    assert [a.id for a in first] == [a.id for a in again]
    assert len(first) == 3
    assert counts["kept"] == 3


def test_pick_returns_short_list_when_too_few():
    articles = [_article(0), _article(1), _article(2, _text(4)), _article(3, _text(4))]
    kept, counts = pick_articles(articles, UNIT, [], set(), per_unit=3)
    assert len(kept) == 2
    assert counts["too_short"] == 2


def test_text_digest_changes_with_text():
    assert text_digest("a b") == text_digest("a b")
    assert text_digest("a b") != text_digest("a c")
    assert len(text_digest("a b")) == 16


def test_load_score_cache_keeps_this_model_and_skips_a_cut_line(tmp_path):
    path = tmp_path / "scores.jsonl"
    rows = [
        {"id": "a", "fingerprint": "m1", "digest": "d", "n_sentences": 6, "n_biased": 1},
        {"id": "b", "fingerprint": "m2", "digest": "d", "n_sentences": 6, "n_biased": 2},
        {"id": "a", "fingerprint": "m1", "digest": "e", "n_sentences": 7, "n_biased": 3},
    ]
    text = "".join(json.dumps(r) + "\n" for r in rows) + '{"id": "c", "fingerp'
    path.write_text(text, encoding="utf-8")
    cached, skipped = load_score_cache(path, "m1")
    assert list(cached) == ["a"]
    assert cached["a"]["digest"] == "e"
    assert skipped == 1


def test_load_score_cache_missing_file(tmp_path):
    assert load_score_cache(tmp_path / "none.jsonl", "m1") == ({}, 0)
