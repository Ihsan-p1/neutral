"""Entry point for M8: the conformal router DeBERTa -> Qwen3.

Run from the repo root, with `src` on the path so the serving classifier loads:

    PYTHONPATH=src .venv-train/Scripts/python.exe -m training.run_m8 --pilot
    PYTHONPATH=src .venv-train/Scripts/python.exe -m training.run_m8 --warm
    PYTHONPATH=src .venv-train/Scripts/python.exe -m training.run_m8 --report

DeBERTa is scored through `neutral.classifier.OnnxClassifier` on CPU, one
sentence per call, so its latency is what the API sees. Nothing printed here
contains sentence text: counts, rates and timings only.
"""

import argparse
import dataclasses
import hashlib
import json
import pathlib
import platform
import random
import subprocess
import time

import numpy as np

from neutral.classifier import OnnxClassifier
from training import m8_report
from training.basil import load_basil
from training.data import load_babe
from training.llm_judge import (
    MODEL_TAG,
    PROMPT_VERSION,
    JudgeCache,
    cache_key,
    check_ollama,
    judge,
    judge_pending,
)
from training.m8_stats import f1_macro, percentile
from training.run_m4a import M2_FOLD0_F1, drop_babe_overlap, split_dev_test
from training.split import group_split, train_val_test_folds

MODEL_DIR = pathlib.Path("models/current")
RUNS = pathlib.Path("runs")
DEBERTA_CACHE = RUNS / "m8_deberta.npz"
JUDGE_CACHE = RUNS / "m8_judge.jsonl"
REPRODUCTION_TOLERANCE = 0.01
PILOT_SIZE = 50
N_FOLDS = 5
SAMPLE_FILE = RUNS / "m8_basil_sample.json"
SAMPLE_HOURS_LIMIT = 4.0
SAMPLE_SIZE = 1500
REPORT_PATH = RUNS / "m8_report.txt"
TRACES_PATH = RUNS / "m8_traces.jsonl"


@dataclasses.dataclass(frozen=True)
class Dataset:
    name: str
    texts: list[str]
    labels: np.ndarray
    groups: list[str]


def load_datasets() -> dict[str, Dataset]:
    """BABE test fold 0 and val fold 1 as in M2; BASIL dev/test as in M4a."""
    babe, _ = load_babe()
    folds = group_split([row.group_key for row in babe], N_FOLDS, seed=0)
    _, val_fold, test_fold = train_val_test_folds(0, N_FOLDS)

    def babe_fold(name: str, fold: int) -> Dataset:
        rows = [row for row, f in zip(babe, folds, strict=True) if f == fold]
        return Dataset(
            name,
            [row.text for row in rows],
            np.array([row.label for row in rows]),
            [row.group_key for row in rows],
        )

    basil, _ = load_basil()
    basil, _ = drop_babe_overlap(basil, [row.text for row in babe])
    dev, test = split_dev_test(basil)

    def basil_part(name: str, indices: list[int]) -> Dataset:
        rows = [basil[i] for i in indices]
        return Dataset(
            name,
            [row.text for row in rows],
            np.array([row.lex for row in rows]),
            [row.article_key for row in rows],
        )

    return {
        "babe_fold0": babe_fold("babe_fold0", test_fold),
        "babe_fold1": babe_fold("babe_fold1", val_fold),
        "basil_dev": basil_part("basil_dev", dev),
        "basil_test": basil_part("basil_test", test),
    }


def _model_version() -> str:
    meta = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    return str(meta["version"])


def score_deberta(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """P(biased) and seconds per sentence, one sentence per call after a warm-up."""
    classifier = OnnxClassifier(str(MODEL_DIR), temperature=1.0, version=_model_version())
    classifier.predict_proba(texts[:1])
    probs = np.empty(len(texts))
    latencies = np.empty(len(texts))
    for i, text in enumerate(texts):
        start = time.perf_counter()
        probs[i] = classifier.predict_proba([text])[0]
        latencies[i] = time.perf_counter() - start
    return probs, latencies


def _texts_digest(texts: list[str]) -> str:
    return hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()


def deberta_scores(datasets: dict[str, Dataset]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Cached DeBERTa scores per dataset; recomputed when the texts change."""
    stored = dict(np.load(DEBERTA_CACHE, allow_pickle=False)) if DEBERTA_CACHE.exists() else {}
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    changed = False
    for name, ds in datasets.items():
        digest = _texts_digest(ds.texts)
        if str(stored.get(f"{name}/digest", "")) != digest:
            print(f"DeBERTa scoring    : {name}, {len(ds.texts)} sentences")
            probs, latencies = score_deberta(ds.texts)
            stored[f"{name}/probs"] = probs
            stored[f"{name}/latency"] = latencies
            stored[f"{name}/digest"] = np.array(digest)
            changed = True
        result[name] = (stored[f"{name}/probs"], stored[f"{name}/latency"])
    if changed:
        RUNS.mkdir(exist_ok=True)
        np.savez(DEBERTA_CACHE, **stored)
    return result


def assert_onnx_reproduces(probs: np.ndarray, labels: np.ndarray) -> float:
    """The served ONNX model must reproduce M2's fold-0 F1 before anything else."""
    measured = f1_macro(labels, (probs >= 0.5).astype(int))
    if abs(measured - M2_FOLD0_F1) > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"ONNX gives F1 macro {measured:.4f} on fold 0, M2 recorded {M2_FOLD0_F1}"
        )
    print(f"reproduction guard : F1 macro {measured:.4f} on {len(labels)} rows")
    return measured


def prepare() -> tuple[dict[str, Dataset], dict[str, tuple[np.ndarray, np.ndarray]]]:
    datasets = load_datasets()
    for ds in datasets.values():
        print(f"{ds.name:18} : {len(ds.texts)} sentences, {int(ds.labels.sum())} biased")
    scores = deberta_scores(datasets)
    fold0 = datasets["babe_fold0"]
    assert_onnx_reproduces(scores["babe_fold0"][0], fold0.labels)
    return datasets, scores


def pilot_indices(ds: Dataset) -> list[int]:
    """The PILOT_SIZE sentences with the smallest cache keys: fixed, no RNG."""
    order = sorted(range(len(ds.texts)), key=lambda i: cache_key(ds.texts[i]))
    return order[:PILOT_SIZE]


def run_pilot(datasets: dict[str, Dataset]) -> None:
    check_ollama()
    ds = datasets["babe_fold1"]
    cache = JudgeCache(JUDGE_CACHE)
    labels, gold, latencies, failures = [], [], [], 0
    for i in pilot_indices(ds):
        result = cache.get(ds.texts[i])
        if result is None:
            result = judge(ds.texts[i])
            cache.put(ds.texts[i], result)
        latencies.append(result.latency_s)
        if result.label is None:
            failures += 1
            continue
        labels.append(result.label)
        gold.append(int(ds.labels[i]))
    p50 = percentile(latencies, 50)
    total = sum(len(datasets[n].texts) for n in ("babe_fold0", "babe_fold1", "basil_test"))
    accuracy = float(np.mean(np.array(labels) == np.array(gold))) if labels else 0.0
    print(f"model / prompt     : {MODEL_TAG} / {PROMPT_VERSION}")
    print(f"pilot latency      : p50 {p50:.2f} s, p95 {percentile(latencies, 95):.2f} s")
    print(f"pilot quality      : accuracy {accuracy:.3f}, "
          f"F1 macro {f1_macro(np.array(gold), np.array(labels)):.4f}")
    print(f"pilot failures     : {failures} of {PILOT_SIZE}")
    print(f"projected hours    : {total * p50 / 3600:.1f} for {total} sentences at p50")


def basil_test_sample(ds: Dataset, projected_hours: float) -> list[int]:
    """All rows when the run fits the budget, else every positive plus sampled negatives."""
    every = list(range(len(ds.texts)))
    if projected_hours <= SAMPLE_HOURS_LIMIT:
        return every
    positives = [i for i in every if ds.labels[i] == 1]
    negatives = [i for i in every if ds.labels[i] == 0]
    drawn = random.Random(0).sample(negatives, SAMPLE_SIZE - len(positives))
    return sorted(positives + drawn)


def _projected_hours(datasets: dict[str, Dataset], cache: JudgeCache) -> float:
    pilot = datasets["babe_fold1"]
    latencies = [cache.get(pilot.texts[i]).latency_s for i in pilot_indices(pilot)
                 if cache.get(pilot.texts[i]) is not None]
    if not latencies:
        raise RuntimeError("no pilot answers in the cache; run --pilot first")
    total = sum(len(datasets[n].texts) for n in ("babe_fold0", "babe_fold1", "basil_test"))
    return total * percentile(latencies, 50) / 3600


def run_warm(datasets: dict[str, Dataset]) -> None:
    """Judge every sentence M8 needs that is not cached yet. Safe to interrupt."""
    check_ollama()
    cache = JudgeCache(JUDGE_CACHE)
    hours = _projected_hours(datasets, cache)
    basil = datasets["basil_test"]
    sample = basil_test_sample(basil, hours)
    SAMPLE_FILE.write_text(json.dumps({
        "indices": sample,
        "n_positive": int(basil.labels[sample].sum()),
        "prior": float(basil.labels[sample].mean()),
        "full": len(sample) == len(basil.texts),
        "projected_hours": hours,
    }), encoding="utf-8")
    todo = [*datasets["babe_fold0"].texts, *datasets["babe_fold1"].texts,
            *(basil.texts[i] for i in sample)]
    pending = [text for text in todo if cache.get(text) is None]
    print(f"warm               : {len(todo)} needed, {len(pending)} not cached, "
          f"BASIL sample {len(sample)} of {len(basil.texts)}")
    failures = judge_pending(pending, cache)
    print(f"warm done          : judged {len(pending)}, label None {failures}")


def _environment_lines() -> list[str]:
    try:
        ollama = subprocess.run(["ollama", "--version"], capture_output=True, text=True,
                                timeout=30).stdout.strip().splitlines()[-1]
    except (OSError, subprocess.SubprocessError, IndexError):
        ollama = "unknown"
    return [
        f"cpu / platform     : {platform.processor()} / {platform.platform()}",
        f"ollama             : {ollama}",
        f"LLM / prompt       : {MODEL_TAG} / {PROMPT_VERSION}",
        f"DeBERTa            : {_model_version()} (ONNX, CPU, temperature 1.0)",
    ]


def run_report(datasets: dict[str, Dataset], scores: dict) -> None:
    """Every table from cached scores and answers; never calls a model."""
    cache = JudgeCache(JUDGE_CACHE)
    sample = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    fold0, fold1 = datasets["babe_fold0"], datasets["babe_fold1"]
    p0, lat0 = scores["babe_fold0"]
    llm0 = m8_report.llm_arrays(fold0.texts, cache, "babe_fold0")
    llm1 = m8_report.llm_arrays(fold1.texts, cache, "babe_fold1")
    tau = m8_report.tuned_tau(scores["babe_fold1"][0], fold1.labels, llm1["labels"])
    babe_lines, cost, raw_traces = m8_report.babe_block(fold0, p0, lat0, llm0, tau)

    full = datasets["basil_test"]
    idx = sample["indices"]
    basil = Dataset("basil_test", [full.texts[i] for i in idx], full.labels[idx],
                    [full.groups[i] for i in idx])
    llm_basil = m8_report.llm_arrays(basil.texts, cache, "basil_test")
    basil_lines = m8_report.basil_block(
        p0, fold0.labels, basil, scores["basil_test"][0][idx],
        datasets["basil_dev"], scores["basil_dev"][0], llm_basil, sample, tau)

    lines = [*_environment_lines(), "", *babe_lines, "", *basil_lines, "",
             *m8_report.cost_block(cost)]
    report = "\n".join(lines)
    print(report)
    REPORT_PATH.write_text(report + "\n", encoding="utf-8")
    m8_report.write_traces(TRACES_PATH, fold0, lat0, cache, raw_traces)
    print(f"\nwritten            : {REPORT_PATH}, {TRACES_PATH}")


def _self_check() -> None:
    labels = np.array([1] * 20 + [0] * 2000)
    ds = Dataset("synthetic", ["x"] * len(labels), labels, ["g"] * len(labels))
    sample = basil_test_sample(ds, 5.0)
    assert len(sample) == SAMPLE_SIZE
    assert set(range(20)) <= set(sample)
    assert sample == basil_test_sample(ds, 5.0)
    assert basil_test_sample(ds, 3.0) == list(range(len(labels)))
    print("self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--warm", action="store_true")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    datasets, scores = prepare()
    if args.pilot:
        run_pilot(datasets)
    if args.warm:
        run_warm(datasets)
    if args.report:
        run_report(datasets, scores)


if __name__ == "__main__":
    main()
