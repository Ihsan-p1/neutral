"""PRD 8.3 runner: does the served article_score rise with how partisan AllSides
rated an outlet in 2019-2020?

Run in .venv-train with the serving package importable:
    PYTHONPATH=src .venv-train/Scripts/python.exe -u -m training.run_outlets [--self-check]

Spec: docs/superpowers/specs/2026-10-24-prd83-outlet-validation-design.md. Every
rule used here (50 articles per unit, median, bootstrap over units, the verdict)
was fixed before scoring. No article text is printed or written: the cache keeps
ids and counts only.
"""

import argparse
import json
import pathlib
import subprocess
import time
from collections import Counter

import numpy as np

from neutral.classifier import OnnxClassifier
from neutral.config import Settings
from neutral.pipeline import analyze
from neutral.segment import split_sentences
from training.baseline_tfidf import fit_predict_proba
from training.data import load_babe
from training.outlet_stats import (
    bootstrap_spearman,
    median_gap,
    paired_spearman_diff,
    verdict,
)
from training.outlets import (
    BALY_ROOT,
    BALY_SIZE,
    MIN_UNITS,
    PER_UNIT,
    RATINGS_CSV,
    Article,
    Unit,
    babe_index,
    lean,
    load_baly,
    load_units,
    pick_articles,
    side,
)
from training.run_m8 import (
    MODEL_DIR,
    _model_fingerprint,
    _model_version,
    assert_onnx_reproduces,
    deberta_scores,
    load_datasets,
)

RUNS = pathlib.Path("runs")
SCORE_CACHE = RUNS / "outlets_scores.jsonl"
REPORT_PATH = RUNS / "outlets_report.txt"
THRESHOLD = 0.5


def _head() -> str:
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return out.stdout.strip()


def self_check() -> None:
    units = load_units(RATINGS_CSV)
    sources = {a.source for a in load_baly(BALY_ROOT)}
    missing = [s for u in units for s in u.sources if s not in sources]
    if missing:
        raise RuntimeError(f"Baly has no source named {missing}")
    for unit in units:
        if not unit.rating_2026:
            raise RuntimeError(f"{unit.name} has no 2026 rating")
        for rating in (unit.rating_2020, unit.rating_2026):
            if rating and lean(rating) is None:
                raise RuntimeError(f"{unit.name}: unknown rating {rating!r}")
    rated = sum(1 for u in units if lean(u.rating_2020) is not None)
    print(f"units: {len(units)}, with a 2020 lean: {rated}")
    print("self-check ok")


def _guard() -> float:
    fold0 = load_datasets()["babe_fold0"]
    probs, _ = deberta_scores({"babe_fold0": fold0})["babe_fold0"]
    return assert_onnx_reproduces(probs, fold0.labels)


def _load_cache(fingerprint: str) -> dict[str, dict]:
    if not SCORE_CACHE.exists():
        return {}
    cached = {}
    for line in SCORE_CACHE.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["fingerprint"] == fingerprint:
            cached[record["id"]] = record
    return cached


def score_deberta(articles: list[Article]) -> dict[str, float]:
    """article_score per article id, through the same analyze() the API calls."""
    fingerprint = _model_fingerprint()
    cached = _load_cache(fingerprint)
    todo = [a for a in articles if a.id not in cached]
    if todo:
        clf = OnnxClassifier(str(MODEL_DIR), temperature=1.0, version=_model_version())
        settings = Settings()
        RUNS.mkdir(exist_ok=True)
        start = time.perf_counter()
        with SCORE_CACHE.open("a", encoding="utf-8") as f:
            for i, article in enumerate(todo, 1):
                result = analyze(article.text, clf, settings, return_sentences=False)
                record = {
                    "id": article.id,
                    "fingerprint": fingerprint,
                    "n_sentences": result.n_sentences,
                    "n_biased": result.n_biased,
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                cached[article.id] = record
                if i % 100 == 0:
                    rate = (time.perf_counter() - start) / i
                    print(f"DeBERTa scoring    : {i}/{len(todo)}, {rate:.2f} s per article")
    return {a.id: cached[a.id]["n_biased"] / cached[a.id]["n_sentences"] for a in articles}


def score_tfidf(articles: list[Article], babe) -> dict[str, float]:
    """article_score from TF-IDF on the same sentences analyze() scores."""
    owners, sentences = [], []
    limit = Settings().max_sentences
    for article in articles:
        for sentence in split_sentences(article.text)[:limit]:
            owners.append(article.id)
            sentences.append(sentence.text)
    probs = fit_predict_proba([r.text for r in babe], [r.label for r in babe], sentences)
    biased, total = Counter(), Counter()
    for owner, prob in zip(owners, probs, strict=True):
        total[owner] += 1
        biased[owner] += prob >= THRESHOLD
    return {a.id: biased[a.id] / total[a.id] for a in articles}


def _interval(label: str, result: tuple[float, float, float, int]) -> str:
    point, lo, hi, skipped = result
    return f"{label:44} {point:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (skipped draws {skipped})"


def _rho_block(title: str, units: list[Unit], deb: dict, tfidf: dict, rating_of) -> list[str]:
    rows = [(lean(rating_of(u)), deb[u.name], tfidf[u.name]) for u in units]
    rows = [r for r in rows if r[0] is not None]
    if len(rows) < 3:
        return [f"{title}: {len(rows)} units, too few"]
    x, d, t = zip(*rows, strict=True)
    rho = bootstrap_spearman(x, d)
    return [
        f"{title} ({len(rows)} units)",
        _interval("  Spearman |lean| vs DeBERTa median", rho),
        f"  verdict: {verdict(*rho[:3])}",
        _interval("  Spearman |lean| vs TF-IDF median", bootstrap_spearman(x, t)),
        _interval("  DeBERTa - TF-IDF (paired)", paired_spearman_diff(x, d, t)),
    ]


def report(units, picks, counts, deb_unit, tfidf_unit, excluded, guard_f1) -> str:
    lines = [
        "PRD 8.3: article_score against AllSides outlet ratings",
        f"HEAD {_head()}  model {_model_fingerprint()[:60]}",
        f"reproduction guard F1 macro {guard_f1:.4f}  per unit {PER_UNIT}  threshold {THRESHOLD}",
        "",
        f"{'unit':28} {'kind':8} {'2020':11} {'2026':11} lean kept  BABE(url+text/text/url)"
        "  DeBERTa TF-IDF",
    ]
    for u in units:
        c = counts[u.name]
        overlap = f"{c['babe_url_and_text']}/{c['babe_text_only']}/{c['babe_url_only']}"
        lean_text = "-" if lean(u.rating_2020) is None else str(lean(u.rating_2020))
        lines.append(
            f"{u.name:28} {u.kind:8} {u.rating_2020 or '-':11} {u.rating_2026 or '-':11} "
            f"{lean_text:4} {len(picks[u.name]):4}  {overlap:23}  "
            f"{deb_unit[u.name]:.3f}   {tfidf_unit[u.name]:.3f}"
        )
    lines += ["", "Excluded: " + ("; ".join(excluded) or "none"), ""]
    total = sum(counts.values(), Counter())
    totals = ", ".join(f"{k} {v}" for k, v in sorted(total.items()))
    lines.append(f"Article filter totals: {totals}")
    lines.append("")
    lines += _rho_block("PRIMARY: 2020 ratings", units, deb_unit, tfidf_unit,
                        lambda u: u.rating_2020)
    rated = [u for u in units if lean(u.rating_2020) is not None]
    partisan = [lean(u.rating_2020) > 0 for u in rated]
    for name, scores in (("DeBERTa", deb_unit), ("TF-IDF", tfidf_unit)):
        gap = median_gap([scores[u.name] for u in rated], partisan)
        lines.append(_interval(f"Centre vs partisan, {name} (partisan - centre)", gap))
    lines.append("")
    lines += _rho_block("SENSITIVITY: 2026 ratings", units, deb_unit, tfidf_unit,
                        lambda u: u.rating_2026)
    lines += _rho_block("SENSITIVITY: news units only, 2020",
                        [u for u in units if u.kind == "news"], deb_unit, tfidf_unit,
                        lambda u: u.rating_2020)
    lines += _rho_block("SENSITIVITY: outlets not in BABE, 2020",
                        [u for u in units if not u.babe_outlet], deb_unit, tfidf_unit,
                        lambda u: u.rating_2020)
    agree, disagree = 0, []
    for u in rated:
        majority = Counter(a.label for a in picks[u.name]).most_common(1)[0][0]
        if majority == side(u.rating_2020):
            agree += 1
        else:
            disagree.append(f"{u.name} (Baly {majority}, 2020 {u.rating_2020})")
    lines += ["", f"Baly label vs 2020 rating side: {agree}/{len(rated)} agree"]
    lines += [f"  {d}" for d in disagree]
    return "\n".join(lines) + "\n"


def main() -> None:
    guard_f1 = _guard()
    babe, _ = load_babe()
    articles = load_baly(BALY_ROOT)
    if len(articles) != BALY_SIZE:
        raise RuntimeError(f"Baly has {len(articles)} articles, expected {BALY_SIZE}")
    babe_text = babe_index([r.text for r in babe])
    babe_urls = {r.group_key for r in babe}
    picks, counts, excluded, kept_units = {}, {}, [], []
    for unit in load_units(RATINGS_CSV):
        kept, count = pick_articles(articles, unit, babe_text, babe_urls)
        if len(kept) < PER_UNIT:
            excluded.append(f"{unit.name} ({len(kept)} qualifying articles)")
            continue
        picks[unit.name], counts[unit.name] = kept, count
        kept_units.append(unit)
    rated = [u for u in kept_units if lean(u.rating_2020) is not None]
    excluded += [f"{u.name} (no 2019-2020 rating)" for u in kept_units if u not in rated]
    if len(rated) < MIN_UNITS:
        raise RuntimeError(f"only {len(rated)} rated units qualify, PRD 8.3 needs {MIN_UNITS}")
    chosen = [a for u in kept_units for a in picks[u.name]]
    print(f"units: {len(kept_units)} ({len(rated)} with a 2020 lean), articles: {len(chosen)}")
    deb = score_deberta(chosen)
    tfidf = score_tfidf(chosen, babe)
    deb_unit = {u.name: float(np.median([deb[a.id] for a in picks[u.name]])) for u in kept_units}
    tfidf_unit = {u.name: float(np.median([tfidf[a.id] for a in picks[u.name]]))
                  for u in kept_units}
    text = report(kept_units, picks, counts, deb_unit, tfidf_unit, excluded, guard_f1)
    RUNS.mkdir(exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    if parser.parse_args().self_check:
        self_check()
    else:
        main()
