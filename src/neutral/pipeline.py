from neutral.classifier import Classifier
from neutral.config import Settings
from neutral.detect import TextTooLargeError, is_english
from neutral.evaluation import EVALUATIONS, select_evaluation
from neutral.schemas import AnalyzeResponse, SentenceResult, Warning_
from neutral.segment import split_sentences


def analyze(
    text: str,
    clf: Classifier,
    settings: Settings,
    return_sentences: bool = True,
) -> AnalyzeResponse:
    """Run the full pipeline: validate, segment, classify, aggregate."""
    if len(text) > settings.max_chars:
        raise TextTooLargeError(
            f"text of {len(text)} characters exceeds the limit of {settings.max_chars}"
        )

    evaluation = select_evaluation(
        EVALUATIONS, clf.version, settings.model_backend, settings.decision_threshold
    )

    warnings: list[Warning_] = []
    if text.strip() and not is_english(text):
        warnings.append("non_english")

    sentences = split_sentences(text)
    if len(sentences) > settings.max_sentences:
        sentences = sentences[: settings.max_sentences]
        warnings.append("truncated")

    if not sentences:
        return AnalyzeResponse(
            article_score=0.0,
            n_sentences=0,
            n_biased=0,
            confidence_mean=0.0,
            sentences=[],
            model_version=clf.version,
            warnings=warnings,
            evaluation=evaluation,
        )

    if len(sentences) < settings.min_sentences_warn:
        warnings.append("too_short")

    probs = clf.predict_proba([s.text for s in sentences])

    results = [
        SentenceResult(
            text=s.text,
            label="biased" if p >= settings.decision_threshold else "neutral",
            prob=round(p, 4),
            start=s.start,
            end=s.end,
        )
        for s, p in zip(sentences, probs, strict=True)
    ]

    n_biased = sum(1 for r in results if r.label == "biased")
    # The article score is a proportion, not a mean probability, so short and
    # long articles can be compared directly.
    article_score = n_biased / len(results)
    # Confidence is distance from the undecided point, not the mean biased
    # probability: 0.5 is where the model is least certain, not least biased.
    confidence_mean = sum(abs(p - 0.5) * 2 for p in probs) / len(probs)

    return AnalyzeResponse(
        article_score=round(article_score, 4),
        n_sentences=len(results),
        n_biased=n_biased,
        confidence_mean=round(confidence_mean, 4),
        sentences=results if return_sentences else [],
        model_version=clf.version,
        warnings=warnings,
        evaluation=evaluation,
    )
