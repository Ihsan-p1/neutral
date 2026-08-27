from collections.abc import Callable

import gradio as gr

from neutral.classifier import Classifier
from neutral.config import Settings
from neutral.detect import TextTooLargeError
from neutral.pipeline import analyze

# The UI is in English: the model only handles English news, and the demo is
# linked from an English-language portfolio.
_WARNING_TEXT = {
    "non_english": (
        "This text does not look like English. The model was trained on English news only."
    ),
    "too_short": "This article is very short, so the proportion is not reliable.",
    "truncated": "The article was truncated because it exceeded the sentence limit.",
}

AnalyzeFn = Callable[[str], tuple[str, list[tuple[str, str | None]]]]


def make_analyze_fn(settings: Settings, get_classifier: Callable[[], Classifier]) -> AnalyzeFn:
    """Build the Gradio callback.

    The classifier is injected as a getter so the UI shares one loaded model
    with the API rather than loading a second ONNX session of its own.
    """

    def analyze_for_ui(text: str) -> tuple[str, list[tuple[str, str | None]]]:
        try:
            result = analyze(text, get_classifier(), settings)
        except TextTooLargeError:
            return f"Text is too long. The limit is {settings.max_chars:,} characters.", []

        if result.n_sentences == 0:
            return "0 sentences detected. Paste article text to analyse it.", []

        lines = [
            f"**{result.n_biased} of {result.n_sentences} sentences** flagged as biased "
            f"({result.article_score:.0%}).",
            f"Mean confidence {result.confidence_mean:.0%} · model `{result.model_version}`",
        ]
        lines += [f"> {_WARNING_TEXT[w]}" for w in result.warnings if w in _WARNING_TEXT]

        highlights = [
            (s.text + " ", s.label if s.label == "biased" else None) for s in result.sentences
        ]
        return "\n\n".join(lines), highlights

    return analyze_for_ui


def build_ui(settings: Settings, get_classifier: Callable[[], Classifier]) -> gr.Blocks:
    analyze_for_ui = make_analyze_fn(settings, get_classifier)

    # analytics_enabled=False is load-bearing: Gradio otherwise calls
    # api.gradio.app during startup, which breaks the "no network call at
    # boot" constraint and slows cold start on every platform.
    with gr.Blocks(title="Media Neutrality Classifier", analytics_enabled=False) as demo:
        gr.Markdown(
            "# Media Neutrality Classifier\n"
            "Paste an English news article. The model flags sentences that show "
            "lexical bias — loaded word choice, not factual content.\n\n"
            "The article score is the **proportion of flagged sentences**, so short and "
            "long articles compare directly. The sentence count is always shown, because "
            "a proportion over few sentences is unreliable."
        )
        inp = gr.Textbox(lines=12, label="Article text", placeholder="Paste text here...")
        btn = gr.Button("Analyse", variant="primary")
        summary = gr.Markdown()
        highlighted = gr.HighlightedText(
            label="Per sentence", color_map={"biased": "red"}, show_legend=True
        )
        btn.click(analyze_for_ui, inputs=inp, outputs=[summary, highlighted])
    return demo
