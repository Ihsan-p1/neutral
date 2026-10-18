"""Tables for the M8 report: BABE over 100 splits, BASIL, cost, alpha sweep.

Kept apart from `run_m8.py`, which loads data and talks to Ollama; this module
only reads arrays and the judge cache, so it never calls a model. Labels are
0 = neutral and 1 = biased; a failed Qwen3 answer is -1.
"""

import json
import pathlib

import numpy as np

from training.conformal import (
    calibration_split,
    coverage,
    coverage_by_class,
    forward_mask,
    marginal_qhat,
    mondrian_qhats,
    mondrian_sets,
    prediction_sets,
)
from training.llm_judge import JudgeCache, route, trace_record
from training.m8_stats import (
    equivalence_verdict,
    f1_macro,
    percentile,
    pooled_paired_bootstrap,
)

ALPHA = 0.05
ALPHAS = (0.05, 0.10, 0.15, 0.20)
N_SPLITS = 100
FIXED_TAU = 0.9
TAU_GRID = [value / 100 for value in range(50, 100)]
M4A_BASIL_F1 = (0.5656, 0.5854)  # as-is and dev-recalibrated threshold, M4a spec 11


def router_predictions(
    deberta: np.ndarray, llm: np.ndarray, forward: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Qwen3's label where forwarded and answered, DeBERTa's otherwise."""
    use_llm = forward & (llm >= 0)
    return np.where(use_llm, llm, deberta), forward & (llm < 0)


def answer_labels(sets: list[frozenset[int]], deberta: np.ndarray) -> np.ndarray:
    """The set's label where it has exactly one, DeBERTa's argmax otherwise.

    With per-class q-hats a one-label set can disagree with the argmax; spec 4.2
    says the router answers with the label in the set.
    """
    return np.array([next(iter(s)) if len(s) == 1 else int(d)
                     for s, d in zip(sets, deberta, strict=True)])


def tau_sets(p_biased: np.ndarray, tau: float) -> list[frozenset[int]]:
    """Sets implied by "forward when max(p) < tau": the marginal rule with q-hat = tau."""
    return prediction_sets(p_biased, tau)


def mode_metrics(gold: np.ndarray, pred: np.ndarray, forward: np.ndarray | None) -> dict:
    metrics = {"f1": f1_macro(gold, pred), "forwarded": None, "singleton_error": None}
    if forward is not None:
        kept = ~forward
        metrics["forwarded"] = float(forward.mean())
        metrics["singleton_error"] = float((pred[kept] != gold[kept]).mean()) if kept.any() else 0.0
    return metrics


def llm_arrays(texts: list[str], cache: JudgeCache, name: str) -> dict[str, np.ndarray]:
    labels, latency, prompt, completion = [], [], [], []
    for text in texts:
        result = cache.get(text)
        if result is None:
            raise RuntimeError(f"Qwen3 cache miss in {name}; run --warm first")
        labels.append(-1 if result.label is None else result.label)
        latency.append(result.latency_s)
        prompt.append(result.prompt_tokens)
        completion.append(result.completion_tokens)
    return {
        "labels": np.array(labels),
        "latency": np.array(latency),
        "tokens": np.array(prompt) + np.array(completion),
    }


def tuned_tau(p: np.ndarray, gold: np.ndarray, llm: np.ndarray) -> float:
    """The max-p threshold with the best router F1 on fold 1."""
    deberta = (p >= 0.5).astype(int)
    confidence = np.maximum(p, 1 - p)
    best = max(
        TAU_GRID,
        key=lambda tau: f1_macro(gold, router_predictions(deberta, llm, confidence < tau)[0]),
    )
    return float(best)


def _fmt(values: list[float | None], digits: int = 4) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "-"
    return f"{np.mean(present):.{digits}f} ± {np.std(present):.{digits}f}"


def _routers(p_cal, y_cal, p_te, tau, alpha=ALPHA) -> dict[str, tuple[np.ndarray, list]]:
    """Forward mask and prediction sets per router; tau routers carry their implied sets."""
    routers = {
        "router marginal": prediction_sets(p_te, marginal_qhat(p_cal, y_cal, alpha)),
        "router Mondrian": mondrian_sets(p_te, mondrian_qhats(p_cal, y_cal, alpha)),
        f"router tau {FIXED_TAU}": tau_sets(p_te, FIXED_TAU),
        f"router tau {tau:.2f} (fold 1)": tau_sets(p_te, tau),
    }
    return {name: (forward_mask(sets), sets) for name, sets in routers.items()}


def _row_values(gold, pred, forward, sets, fallback) -> dict:
    """One split's numbers for one mode; None where the mode has no such number."""
    values = mode_metrics(gold, pred, forward)
    by_class = coverage_by_class(sets, gold) if sets is not None else {}
    values["coverage"] = coverage(sets, gold) if sets is not None else None
    values["cov0"] = by_class.get(0, (None,))[0]
    values["cov1"] = by_class.get(1, (None,))[0]
    values["fallbacks"] = float(fallback.sum()) if fallback is not None else None
    return values


def babe_block(fold0, p0, lat0, llm0, tau) -> tuple[list[str], dict, list[dict]]:
    rows: dict[str, dict[str, list]] = {}
    vs_qwen, vs_deberta = "router marginal vs Qwen3 only", "router marginal vs DeBERTa only"
    pairs: dict[str, list] = {vs_qwen: [], vs_deberta: []}
    cost: dict[str, list[float]] = {"DeBERTa only": [], "Qwen3 only": [], "router marginal": []}
    tokens: list[float] = []
    sweep = {a: {"forwarded": [], "f1": [], "coverage": []} for a in ALPHAS}
    traces: list[dict] = []
    for seed in range(N_SPLITS):
        cal, te = (np.array(side) for side in calibration_split(fold0.groups, seed))
        gold, deb = fold0.labels[te], (p0[te] >= 0.5).astype(int)
        llm = llm0["labels"][te]
        qwen, qwen_fallback = router_predictions(deb, llm, np.ones(len(te), dtype=bool))
        # Each mode: (prediction, forward mask, prediction sets, fallback mask).
        modes = {"DeBERTa only": (deb, None, None, None),
                 "Qwen3 only": (qwen, None, None, qwen_fallback)}
        for name, (forward, sets) in _routers(p0[cal], fold0.labels[cal], p0[te], tau).items():
            pred, fallback = router_predictions(answer_labels(sets, deb), llm, forward)
            modes[name] = (pred, forward, sets, fallback)
        for name, (pred, forward, sets, fallback) in modes.items():
            row = rows.setdefault(name, {})
            for key, value in _row_values(gold, pred, forward, sets, fallback).items():
                row.setdefault(key, []).append(value)
        groups = [fold0.groups[i] for i in te]
        router_pred, router_forward, router_sets, _ = modes["router marginal"]
        pairs[vs_qwen].append((gold, router_pred, qwen, groups))
        pairs[vs_deberta].append((gold, router_pred, deb, groups))
        llm_latency = llm0["latency"][te]
        cost["DeBERTa only"].extend(lat0[te])
        cost["Qwen3 only"].extend(llm_latency)
        cost["router marginal"].extend(lat0[te] + np.where(router_forward, llm_latency, 0.0))
        tokens.append(float((llm0["tokens"][te] * router_forward).mean() * 1000))
        for alpha in ALPHAS:
            sets = prediction_sets(p0[te], marginal_qhat(p0[cal], fold0.labels[cal], alpha))
            forward = forward_mask(sets)
            sweep[alpha]["forwarded"].append(float(forward.mean()))
            sweep[alpha]["f1"].append(f1_macro(gold, router_predictions(deb, llm, forward)[0]))
            sweep[alpha]["coverage"].append(coverage(sets, gold))
        if seed == 0:
            traces = [{"index": int(i), "set_size": len(router_sets[j])}
                      for j, i in enumerate(te)]

    lines = [f"BABE fold 0, {N_SPLITS} calibration/test splits by article, alpha {ALPHA}",
             f"{'mode':28} {'F1 macro':>17} {'coverage':>17} {'cov neutral':>17} "
             f"{'cov biased':>17} {'forwarded':>17} {'singleton err':>17} {'fallbacks':>9}"]
    for name, row in rows.items():
        lines.append(f"{name:28} {_fmt(row['f1']):>17} {_fmt(row['coverage']):>17} "
                     f"{_fmt(row['cov0']):>17} {_fmt(row['cov1']):>17} "
                     f"{_fmt(row['forwarded']):>17} {_fmt(row['singleton_error']):>17} "
                     f"{_fmt(row['fallbacks'], 1):>9}")
    lines.append("")
    for name, splits in pairs.items():
        mean, lo, hi = pooled_paired_bootstrap(splits)
        lines.append(f"{name:32}: F1 diff {mean:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}] "
                     f"-> {equivalence_verdict(lo, hi)} (margin 0.03)")
    lines.append("")
    lines.append("alpha sweep, router marginal")
    for alpha, values in sweep.items():
        lines.append(f"  alpha {alpha:.2f}: forwarded {_fmt(values['forwarded'])}, "
                     f"F1 {_fmt(values['f1'])}, coverage {_fmt(values['coverage'])}")
    cost_summary = {"latency": cost, "router_tokens_per_1000": float(np.mean(tokens)),
                    "qwen_tokens_per_1000": float(llm0["tokens"].mean() * 1000)}
    return lines, cost_summary, traces


def basil_block(p0, y0, basil, p_te, dev, p_dev, llm, sample_info, tau) -> list[str]:
    gold, deb = basil.labels, (p_te >= 0.5).astype(int)
    qwen, _ = router_predictions(deb, llm["labels"], np.ones(len(gold), dtype=bool))
    entries = {"DeBERTa only (argmax)": (deb, None, None), "Qwen3 only": (qwen, None, None)}
    calibrations = {"BABE q-hat": (p0, y0), "BASIL dev q-hat": (p_dev, dev.labels)}
    routers = {f"router tau {FIXED_TAU}": tau_sets(p_te, FIXED_TAU),
               f"router tau {tau:.2f} (fold 1)": tau_sets(p_te, tau)}
    for label, (p_cal, y_cal) in calibrations.items():
        routers[f"router marginal, {label}"] = prediction_sets(
            p_te, marginal_qhat(p_cal, y_cal, ALPHA))
        routers[f"router Mondrian, {label}"] = mondrian_sets(
            p_te, mondrian_qhats(p_cal, y_cal, ALPHA))
    for name, sets in routers.items():
        forward = forward_mask(sets)
        pred, _ = router_predictions(answer_labels(sets, deb), llm["labels"], forward)
        entries[name] = (pred, forward, sets)
    lines = [f"BASIL test, {len(gold)} sentences ({'full' if sample_info['full'] else 'sample'}), "
             f"biased prior {gold.mean():.3f}; calibration rows: BABE fold 0 {len(y0)}, "
             f"BASIL dev {len(dev.labels)} ({int(dev.labels.sum())} biased)",
             f"M4a reference, DeBERTa on full BASIL test: as-is {M4A_BASIL_F1[0]}, "
             f"dev-recalibrated threshold {M4A_BASIL_F1[1]}",
             f"{'mode':34} {'F1 macro':>8} {'coverage':>8} {'cov neu':>8} {'cov bias':>8} "
             f"{'forwarded':>9} {'single err':>10}"]
    for name, (pred, forward, sets) in entries.items():
        metrics = mode_metrics(gold, pred, forward)
        by_class = coverage_by_class(sets, gold) if sets is not None else {}

        def cell(value, width=8):
            return f"{value:>{width}.4f}" if value is not None else f"{'-':>{width}}"
        lines.append(f"{name:34} {cell(metrics['f1'])} "
                     f"{cell(coverage(sets, gold) if sets is not None else None)} "
                     f"{cell(by_class.get(0, (None,))[0])} {cell(by_class.get(1, (None,))[0])} "
                     f"{cell(metrics['forwarded'], 9)} {cell(metrics['singleton_error'], 10)}")
    return lines


def cost_block(summary: dict) -> list[str]:
    lines = ["cost per sentence, BABE test halves pooled over all splits"]
    for name, values in summary["latency"].items():
        lines.append(f"  {name:16}: p50 {percentile(values, 50) * 1000:7.1f} ms, "
                     f"p95 {percentile(values, 95) * 1000:7.1f} ms, "
                     f"{np.mean(values) * 1000:7.1f} s per 1,000 sentences")
    lines.append(f"  Qwen3 tokens per 1,000 sentences: Qwen3 only "
                 f"{summary['qwen_tokens_per_1000']:.0f}, router marginal "
                 f"{summary['router_tokens_per_1000']:.0f}")
    return lines


def write_traces(path: pathlib.Path, fold0, lat0, cache: JudgeCache, raw: list[dict]) -> None:
    """One text-free record per routed sentence of split 0."""
    with path.open("w", encoding="utf-8") as handle:
        for entry in raw:
            text = fold0.texts[entry["index"]]
            llm = cache.get(text) if entry["set_size"] != 1 else None
            _, source = route(entry["set_size"], 0, llm)
            latency = float(lat0[entry["index"]])
            handle.write(json.dumps(trace_record(text, entry["set_size"], source, latency, llm)))
            handle.write("\n")
