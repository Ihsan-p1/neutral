"""Entry point for M4b-1: temperature scaling, ECE on BABE and BASIL, ablations.

Run from the repo root, with `src` on the path so the serving classifier loads:

    PYTHONPATH=src .venv-train/Scripts/python.exe -m training.run_m4b --self-check
    PYTHONPATH=src .venv-train/Scripts/python.exe -u -m training.run_m4b

Nothing printed or written here contains sentence text: counts, rates and
digests only.
"""

import argparse
import datetime
import json
import pathlib
import subprocess
import tempfile

import numpy as np

from neutral.classifier import OnnxClassifier
from training.calibration import (
    CALIBRATION_PATH,
    N_BOOT,
    apply_calibration,
    brier,
    calibration_verdict,
    ece,
    file_sha256,
    fit_temperature,
    group_bootstrap,
    logit_diff,
    nll,
    prob_biased,
    reliability_bins,
)
from training.m4b_ablation import (
    Chunk,
    entity_verdict,
    length_quartiles,
    length_verdict,
    mask_chunks,
    word_count,
)
from training.m8_stats import bootstrap_f1, f1_macro
from training.nela import ensure_nltk_resources
from training.outlet_stats import spearman
from training.run_m4a import M2_FOLD0_F1
from training.run_m8 import (
    MODEL_DIR,
    REPRODUCTION_TOLERANCE,
    Dataset,
    _texts_digest,
    load_datasets,
)

RUNS = pathlib.Path("runs")
LOGITS_CACHE = RUNS / "m4b_logits.npz"
REPORT_PATH = RUNS / "m4b_report.txt"
PLOT_PATH = pathlib.Path("docs/img/m4b_reliability.png")
CHECKPOINT = pathlib.Path("models/deberta-babe/model.safetensors")
# Only the files that decide the logits. metadata.json is left out on purpose:
# this runner rewrites it, and that must not force a 12-minute rescore.
FINGERPRINT_FILES = ("model.onnx", "model.onnx.data", "tokenizer.json")
NOTE = (
    "temperature fitted on BABE fold 1 (n=883) by NLL; ECE on BASIL test is in the "
    "README; babe_fold0_f1_macro belongs to a single checkpoint, not the 25-run mean "
    "of 0.8477 +/- 0.0277 that M2 reports."
)

Interval = tuple[float, float, float]


def _onnx_fingerprint() -> str:
    parts = []
    for name in FINGERPRINT_FILES:
        stat = (MODEL_DIR / name).stat()
        parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    return "|".join(parts)


def _classifier() -> OnnxClassifier:
    return OnnxClassifier(str(MODEL_DIR), temperature=1.0, version="m4b-logits")


def onnx_logits(clf: OnnxClassifier, texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Raw (n, 2) logits from the served ONNX session; predict_proba hides them."""
    out = np.empty((len(texts), 2))
    for start in range(0, len(texts), batch_size):
        encoded = clf.tokenizer.encode_batch(texts[start : start + batch_size])
        feeds = {
            "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
        }
        out[start : start + len(encoded)] = clf.session.run(None, feeds)[0]
    return out


def cached_logits(datasets: dict[str, Dataset]) -> dict[str, np.ndarray]:
    """Logits per dataset, rescored only when the texts or the ONNX files change."""
    stored = dict(np.load(LOGITS_CACHE, allow_pickle=False)) if LOGITS_CACHE.exists() else {}
    fingerprint = _onnx_fingerprint()
    clf = None
    changed = False
    for name, ds in datasets.items():
        digest = _texts_digest(ds.texts, fingerprint)
        if str(stored.get(f"{name}/digest", "")) != digest:
            print(f"scoring            : {name}, {len(ds.texts)} sentences", flush=True)
            clf = clf or _classifier()
            stored[f"{name}/logits"] = onnx_logits(clf, ds.texts)
            stored[f"{name}/digest"] = np.array(digest)
            changed = True
    if changed:
        RUNS.mkdir(exist_ok=True)
        np.savez(LOGITS_CACHE, **stored)
    return {name: stored[f"{name}/logits"] for name in datasets}


def assert_reproduces(diff: np.ndarray, labels: np.ndarray) -> float:
    """The served ONNX logits must reproduce M2's fold-0 F1 before anything else."""
    measured = f1_macro(labels, (diff >= 0).astype(int))
    if abs(measured - M2_FOLD0_F1) > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"ONNX gives F1 macro {measured:.4f} on fold 0, M2 recorded {M2_FOLD0_F1}"
        )
    return measured


def _fmt(interval: Interval) -> str:
    point, lo, hi = interval
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def _metric_ci(metric, p: np.ndarray, y: np.ndarray, groups, n_boot: int) -> Interval:
    return group_bootstrap(lambda idx: metric(p[idx], y[idx]), groups, n_boot)


def _set_stats(
    name: str, diff: np.ndarray, y: np.ndarray, groups: list[str], temperature: float, n_boot: int
) -> dict:
    before, after = prob_biased(diff, 1.0), prob_biased(diff, temperature)
    if not np.array_equal(before >= 0.5, after >= 0.5):
        raise RuntimeError(f"temperature changed a label in {name}")
    s = {
        "ece_before": _metric_ci(ece, before, y, groups, n_boot),
        "ece_after": _metric_ci(ece, after, y, groups, n_boot),
        "ece_diff": group_bootstrap(
            lambda idx: ece(after[idx], y[idx]) - ece(before[idx], y[idx]), groups, n_boot
        ),
        "brier_before": _metric_ci(brier, before, y, groups, n_boot),
        "brier_after": _metric_ci(brier, after, y, groups, n_boot),
        "nll_before": group_bootstrap(lambda idx: nll(diff[idx], y[idx], 1.0), groups, n_boot),
        "nll_after": group_bootstrap(
            lambda idx: nll(diff[idx], y[idx], temperature), groups, n_boot
        ),
        "panels": [
            (f"{name}, T = 1", reliability_bins(before, y)),
            (f"{name}, T = {temperature:.2f}", reliability_bins(after, y)),
        ],
    }
    s["verdict"] = calibration_verdict(s["ece_diff"][1], s["ece_diff"][2])
    return s


def calibration_block(
    fit_diff: np.ndarray,
    fit_labels: np.ndarray,
    sets: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    n_boot: int = N_BOOT,
) -> tuple[float, list[str], dict]:
    """Fit T on one set, then ECE, Brier and NLL before and after on each report set."""
    temperature = fit_temperature(fit_diff, fit_labels)
    lines = [f"temperature        : {temperature:.4f} (NLL on fit set, n={len(fit_labels)})"]
    stats: dict[str, dict] = {}
    for name, (diff, y, groups) in sets.items():
        s = stats[name] = _set_stats(name, diff, y, groups, temperature, n_boot)
        t = f"T={temperature:.2f}"
        lines += [
            "",
            f"{name} (n={len(y)}, {len(set(groups))} articles, {int(np.sum(y))} biased)",
            f"  ECE   T=1 {_fmt(s['ece_before'])}   {t} {_fmt(s['ece_after'])}",
            f"  ECE after - before {_fmt(s['ece_diff'])} -> {s['verdict']}",
            f"  Brier T=1 {_fmt(s['brier_before'])}   {t} {_fmt(s['brier_after'])}",
            f"  NLL   T=1 {_fmt(s['nll_before'])}   {t} {_fmt(s['nll_after'])}",
        ]
    return temperature, lines, stats


def plot_reliability(
    panels: list[tuple[str, list[tuple[float, float, int]]]], path: pathlib.Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (len(panels) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(10, 4.5 * rows), squeeze=False)
    for ax, (title, bins) in zip(axes.flat, panels, strict=False):
        ax.plot([0.5, 1.0], [0.5, 1.0], linestyle="--", color="grey", linewidth=1)
        ax.plot([b[0] for b in bins], [b[1] for b in bins], marker="o")
        for conf, acc, count in bins:
            ax.annotate(str(count), (conf, acc), textcoords="offset points", xytext=(0, 6),
                        ha="center", fontsize=7)
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("confidence (top label)")
        ax.set_ylabel("accuracy")
        ax.set_title(title)
    for ax in list(axes.flat)[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


CAVEAT = (
    "Both ablations measure sensitivity, not proof of a shortcut: the model never saw "
    "modified input in training."
)


def nltk_chunks(text: str) -> list[Chunk]:
    import nltk

    tree = nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(text)))
    chunks: list[Chunk] = []
    for node in tree:
        if isinstance(node, nltk.Tree):
            chunks.append((" ".join(word for word, _ in node.leaves()), node.label()))
        else:
            chunks.append((node[0], None))
    return chunks


def _entity_lines(
    fold0: Dataset, fold0_diff: np.ndarray, score_diff, n_boot: int
) -> tuple[list[str], dict]:
    keep, originals, masked, entities = [], [], [], 0
    for i, text in enumerate(fold0.texts):
        original, masked_text, count = mask_chunks(nltk_chunks(text))
        if count:
            keep.append(i)
            originals.append(original)
            masked.append(masked_text)
            entities += count
    if not keep:
        return ["entities: none found by NLTK in babe_fold0; no verdict"], {"flip_rate": None}
    y = fold0.labels[keep]
    groups = [fold0.groups[i] for i in keep]
    d_orig, d_mask = score_diff(originals), score_diff(masked)
    p_orig, p_mask = (d_orig >= 0).astype(int), (d_mask >= 0).astype(int)
    drop = group_bootstrap(
        lambda idx: f1_macro(y[idx], p_orig[idx]) - f1_macro(y[idx], p_mask[idx]), groups, n_boot
    )
    flip_rate = float(np.mean(p_orig != p_mask))
    shift = float(np.mean(prob_biased(d_mask, 1.0) - prob_biased(d_orig, 1.0)))
    raw_f1 = f1_macro(y, (fold0_diff[keep] >= 0).astype(int))
    lines = [
        f"entities: {len(keep)} of {len(fold0.texts)} sentences, "
        f"{entities} entities (NLTK ne_chunk)",
        f"  F1 raw text {raw_f1:.4f}, re-joined {f1_macro(y, p_orig):.4f}, "
        f"masked {f1_macro(y, p_mask):.4f}",
        f"  F1 re-joined - masked {_fmt(drop)} -> {entity_verdict(drop[0], drop[1])}",
        f"  label flip rate {flip_rate:.4f}, mean change in P(biased) {shift:+.4f}",
    ]
    return lines, {"flip_rate": flip_rate, "drop": drop}


def _length_lines(
    fold0: Dataset, fold1: Dataset, fold0_diff: np.ndarray, n_boot: int
) -> tuple[list[str], dict]:
    from sklearn.linear_model import LogisticRegression

    len0 = np.array([word_count(t) for t in fold0.texts], dtype=np.float64)
    len1 = np.array([word_count(t) for t in fold1.texts], dtype=np.float64)
    y0 = fold0.labels
    model = LogisticRegression().fit(len1.reshape(-1, 1), fold1.labels)
    p_len = model.predict(len0.reshape(-1, 1))
    majority = int(np.mean(fold1.labels) >= 0.5)
    p_const = np.full(len(y0), majority)
    gain = group_bootstrap(
        lambda idx: f1_macro(y0[idx], p_len[idx]) - f1_macro(y0[idx], p_const[idx]),
        fold0.groups,
        n_boot,
    )
    p_deberta = (fold0_diff >= 0).astype(int)
    lines = [
        f"length: F1 length-only LR {f1_macro(y0, p_len):.4f}, constant "
        f"(class {majority}) {f1_macro(y0, p_const):.4f}, DeBERTa {f1_macro(y0, p_deberta):.4f}",
        f"  F1 length-only - constant {_fmt(gain)} -> {length_verdict(gain[1])}",
    ]
    quartile = length_quartiles(len0)
    for q in range(4):
        rows = np.flatnonzero(quartile == q)
        if not len(rows):
            continue
        groups = [fold0.groups[i] for i in rows]
        lo, hi = bootstrap_f1(y0[rows], p_deberta[rows], groups, n_boot=n_boot)
        lines.append(
            f"  quartile {q + 1}: {int(len0[rows].min())}-{int(len0[rows].max())} words, "
            f"n={len(rows)}, DeBERTa F1 {f1_macro(y0[rows], p_deberta[rows]):.4f} "
            f"[{lo:.4f}, {hi:.4f}]"
        )
    lines.append(
        f"  Spearman length vs P(biased) {spearman(len0, prob_biased(fold0_diff, 1.0)):+.4f}, "
        f"length vs gold {spearman(len0, y0):+.4f}"
    )
    return lines, {"gain": gain}


def ablation_block(
    fold0: Dataset, fold1: Dataset, fold0_diff: np.ndarray, score_diff, n_boot: int = N_BOOT
) -> tuple[list[str], dict]:
    """Entity and length sensitivity on BABE fold 0; `score_diff` maps texts to logit diffs."""
    entity_lines, entity_stats = _entity_lines(fold0, fold0_diff, score_diff, n_boot)
    length_lines, length_stats = _length_lines(fold0, fold1, fold0_diff, n_boot)
    lines = ["ablation on babe_fold0", *entity_lines, *length_lines, CAVEAT]
    return lines, {"entities": entity_stats, "length": length_stats}


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def write_calibration(temperature: float, n: int, ece_after: Interval) -> dict:
    point, lo, hi = ece_after
    calibration = {
        "temperature": temperature,
        "fitted_on": "babe_fold1",
        "n": n,
        "model_version": "deberta-babe-f0s3",
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "ece_babe_after": {"point": point, "lo": lo, "hi": hi},
        "note": NOTE,
        "date": datetime.date.today().isoformat(),
        "commit": _git_head(),
    }
    CALIBRATION_PATH.write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    return calibration


def update_metadata(calibration: dict) -> None:
    """Rewrite models/current/metadata.json through the same function the export uses."""
    path = MODEL_DIR / "metadata.json"
    existing = json.loads(path.read_text(encoding="utf-8"))
    meta, applied = apply_calibration(existing, calibration, file_sha256(CHECKPOINT))
    if not applied:
        raise RuntimeError("calibration did not apply to models/current; checkpoint hash differs")
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def run() -> None:
    datasets = load_datasets()
    for ds in datasets.values():
        print(f"{ds.name:18} : {len(ds.texts)} sentences, {int(ds.labels.sum())} biased")
    logits = cached_logits(datasets)
    diffs = {name: logit_diff(value) for name, value in logits.items()}
    fold0, fold1 = datasets["babe_fold0"], datasets["babe_fold1"]
    basil_dev, basil_test = datasets["basil_dev"], datasets["basil_test"]
    guard = assert_reproduces(diffs["babe_fold0"], fold0.labels)
    print(f"reproduction guard : F1 macro {guard:.4f} on {len(fold0.labels)} rows", flush=True)

    sets = {
        "babe_fold0": (diffs["babe_fold0"], fold0.labels, fold0.groups),
        "basil_test": (diffs["basil_test"], basil_test.labels, basil_test.groups),
    }
    temperature, cal_lines, stats = calibration_block(diffs["babe_fold1"], fold1.labels, sets)
    t_basil = fit_temperature(diffs["basil_dev"], basil_dev.labels)
    p_basil = prob_biased(diffs["basil_test"], t_basil)
    ece_basil = _metric_ci(ece, p_basil, basil_test.labels, basil_test.groups, N_BOOT)
    ensure_nltk_resources()
    clf = _classifier()
    abl_lines, _ = ablation_block(
        fold0, fold1, diffs["babe_fold0"], lambda texts: logit_diff(onnx_logits(clf, texts))
    )

    report = [
        "M4b-1: temperature scaling and calibration",
        f"date               : {datetime.date.today().isoformat()}, HEAD {_git_head()}",
        f"reproduction guard : F1 macro {guard:.4f} on {len(fold0.labels)} rows (babe_fold0)",
        *[f"{ds.name:18} : {len(ds.texts)} sentences" for ds in datasets.values()],
        "",
        *cal_lines,
        "",
        f"comparison, not served: T fitted on basil_dev (n={len(basil_dev.labels)}) = "
        f"{t_basil:.4f}; basil_test ECE {_fmt(ece_basil)}",
        "",
        "",
        *abl_lines,
        "",
        "ECE: top-label confidence, 15 equal-mass bins, edges recomputed per draw; "
        f"CIs: {N_BOOT} article-level bootstrap draws, seed 0",
    ]
    plot_reliability(stats["babe_fold0"]["panels"] + stats["basil_test"]["panels"], PLOT_PATH)
    RUNS.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    calibration = write_calibration(
        temperature, len(fold1.labels), stats["babe_fold0"]["ece_after"]
    )
    update_metadata(calibration)
    print("\n".join(report))
    print(f"\nwritten            : {REPORT_PATH}, {PLOT_PATH}, {CALIBRATION_PATH}, metadata.json")


def _self_check() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    true = rng.normal(0, 2, n)
    labels = (rng.random(n) < 1 / (1 + np.exp(-true))).astype(int)
    groups = [f"g{i % 200}" for i in range(n)]
    diff = true * 3.0
    half = n // 2
    sets = {"synthetic": (diff[half:], labels[half:], groups[half:])}
    temperature, lines, stats = calibration_block(diff[:half], labels[:half], sets, n_boot=200)
    assert abs(temperature - 3.0) / 3.0 < 0.1, temperature
    after, before = stats["synthetic"]["ece_after"][0], stats["synthetic"]["ece_before"][0]
    assert after < before, (after, before)
    assert stats["synthetic"]["verdict"] == "lowers"
    assert lines and all(isinstance(line, str) for line in lines)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "plot.png"
        plot_reliability(stats["synthetic"]["panels"], path)
        assert path.stat().st_size > 0
    _self_check_ablation()
    print("self-check ok")


def _self_check_ablation() -> None:
    ensure_nltk_resources()
    _, _, count = mask_chunks(nltk_chunks("Barack Obama visited Paris with Angela Merkel."))
    assert count >= 1, count
    _, _, none = mask_chunks(nltk_chunks("the rain fell all day ."))
    assert none == 0, none

    def synthetic(name: str, n: int) -> Dataset:
        texts = [
            "Senator Jane Doe slammed the reckless plan in Ohio ." if i % 2
            else "Officials in Texas said the report would be released on Tuesday ."
            for i in range(n)
        ]
        labels = np.array([i % 2 for i in range(n)])
        return Dataset(name, texts, labels, [f"g{i % 10}" for i in range(n)])

    def score_diff(texts: list[str]) -> np.ndarray:
        return np.array([2.0 if "slammed" in text else -2.0 for text in texts])

    fold0, fold1 = synthetic("f0", 40), synthetic("f1", 40)
    lines, stats = ablation_block(fold0, fold1, score_diff(fold0.texts), score_diff, n_boot=100)
    text = "\n".join(lines)
    assert "no measurable reliance on entity names" in text, text
    assert "length alone" in text, text
    assert stats["entities"]["flip_rate"] == 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        _self_check()
        return
    run()


if __name__ == "__main__":
    main()
