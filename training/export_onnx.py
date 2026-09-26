"""Export the M2 checkpoint to fp32 ONNX so `OnnxClassifier` can serve it.

Run: .venv-train/Scripts/python.exe -m training.export_onnx

This module imports torch, transformers and onnx, so it only runs under
.venv-train/. Neither `onnx` nor `onnxscript` appears in any requirements.txt:
the serving image may load onnxruntime and nothing else, and CI fails the build
if PyTorch reaches it.

Output goes to `models/current/`, which is gitignored along with `*.onnx`. The
artifact never enters history; only this script does.

Two checks run before anything is written permanently. Both close a failure that
raises no error on its own:

1. Tokenizer parity. Serving calls `tokenizers.Tokenizer.from_file()` directly
   while training goes through `AutoTokenizer`. If the two post-processors
   disagree about [CLS]/[SEP], token ids shift and the scores shift with them.
2. Logit parity. The ONNX graph is compared against a PyTorch forward pass over
   the same sentences. An export with the wrong axes or opset still produces
   numbers that look plausible.
"""

import json
import pathlib
import shutil
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from training.train import MAX_LENGTH

# The torch.onnx progress reporter prints emoji. A Windows console defaults to
# cp1252 and raises UnicodeEncodeError partway through an export that had in
# fact already succeeded.
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

MODEL_DIR = "models/deberta-babe"
OUT_DIR = "models/current"

# Opset 17 is the oldest that still carries every DeBERTa-v2 operator without
# decomposition, and the pinned onnxruntime 1.20.1 loads it.
OPSET = 17

# Largest logit difference still attributable to floating point error. fp32 to
# fp32 with no quantization should land far below this.
LOGIT_TOLERANCE = 1e-3

# Parity probe sentences. Lengths are deliberately mixed so padding and
# truncation are exercised too, not just the uniform path.
PROBE_SENTENCES = [
    "The senator voted against the bill.",
    "The reckless senator once again betrayed the public trust with a vote that "
    "no honest person could defend, and the consequences will be felt for years.",
    "Rain.",
    "Officials said the report would be released on Tuesday.",
    "A shameful, cowardly decision.",
]

# The calibrated temperature is an M4b output, and M4b has not run. 1.0 means a
# softmax with no scaling: the probabilities are valid as a ranking but are not
# calibrated. This value comes from M4b, it is not guessed here.
TEMPERATURE = 1.0

# Fold 0, seed 3, from models/deberta-babe/m2_selection.json. One checkpoint,
# not the 25-run mean M2 reports.
VERSION = "deberta-babe-f0s3"


def assert_tokenizers_agree(model_dir: str) -> None:
    """Compare the training tokenizer's ids against the serving tokenizer's.

    `OnnxClassifier` sets truncation to 128 and enables padding through the raw
    tokenizers library. This mirrors that setup exactly, then checks it against
    the `AutoTokenizer` used during training.
    """
    from tokenizers import Tokenizer

    auto = AutoTokenizer.from_pretrained(model_dir)
    raw = Tokenizer.from_file(str(pathlib.Path(model_dir) / "tokenizer.json"))
    raw.enable_truncation(max_length=MAX_LENGTH)
    raw.enable_padding()

    expected = auto(
        PROBE_SENTENCES, truncation=True, max_length=MAX_LENGTH, padding=True
    )
    actual = raw.encode_batch(PROBE_SENTENCES)

    for i, sentence in enumerate(PROBE_SENTENCES):
        if expected["input_ids"][i] != actual[i].ids:
            raise AssertionError(
                "the serving and training tokenizers disagree on sentence "
                f"{i}: {sentence!r}\n"
                f"  AutoTokenizer : {expected['input_ids'][i]}\n"
                f"  tokenizers    : {actual[i].ids}"
            )
        if expected["attention_mask"][i] != actual[i].attention_mask:
            raise AssertionError(f"attention_mask differs on sentence {i}")

    print(f"tokenizer parity : pass, {len(PROBE_SENTENCES)} sentences")


def export(model_dir: str, out_path: pathlib.Path) -> None:
    """Write an fp32 ONNX graph taking the two inputs `OnnxClassifier` feeds.

    `token_type_ids` is not exported. This checkpoint's config.json sets
    `type_vocab_size: 0`, so it has no segment embedding and that input is never
    read. Exporting it would force serving to feed a tensor that is ignored.

    The batch and sequence axes are dynamic. Without that the graph locks to the
    shape of the probe sentences, and other batches fail at runtime rather than
    at export.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoded = tokenizer(
        PROBE_SENTENCES,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    args = (encoded["input_ids"], encoded["attention_mask"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        args,
        str(out_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=OPSET,
        do_constant_folding=True,
    )
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"graph written    : {out_path} ({size_mb:.1f} MB, opset {OPSET})")


def assert_logits_match(model_dir: str, onnx_path: pathlib.Path) -> float:
    """Run the same sentences through PyTorch and ONNX; return the max delta."""
    import onnxruntime as ort

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, dtype=torch.float32
    )
    model.eval()

    encoded = tokenizer(
        PROBE_SENTENCES,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        torch_logits = model(
            input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
        ).logits.numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_logits = session.run(
        None,
        {
            "input_ids": encoded["input_ids"].numpy().astype(np.int64),
            "attention_mask": encoded["attention_mask"].numpy().astype(np.int64),
        },
    )[0]

    delta = float(np.abs(torch_logits - onnx_logits).max())
    if delta > LOGIT_TOLERANCE:
        raise AssertionError(
            f"ONNX logits drift {delta:.2e} from PyTorch, "
            f"limit is {LOGIT_TOLERANCE:.0e}"
        )
    print(f"logit parity     : pass, max delta {delta:.2e}")
    return delta


def write_sidecars(model_dir: pathlib.Path, out_dir: pathlib.Path) -> None:
    """Copy the tokenizer and write the metadata `load_classifier` reads."""
    shutil.copy2(model_dir / "tokenizer.json", out_dir / "tokenizer.json")
    metadata = {
        "version": VERSION,
        "temperature": TEMPERATURE,
        "calibrated": False,
        "max_length": MAX_LENGTH,
        "opset": OPSET,
        "quantized": False,
        "source_checkpoint": str(model_dir),
        "babe_fold0_f1_macro": 0.8765,
        "note": (
            "temperature 1.0 means uncalibrated; calibration is M4b. "
            "babe_fold0_f1_macro belongs to a single checkpoint, not the "
            "25-run mean of 0.8477 +/- 0.0277 that M2 reports."
        ),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"sidecars written : tokenizer.json, metadata.json -> {out_dir}")


def main() -> None:
    model_dir = pathlib.Path(MODEL_DIR)
    out_dir = pathlib.Path(OUT_DIR)
    onnx_path = out_dir / "model.onnx"

    assert_tokenizers_agree(MODEL_DIR)
    export(MODEL_DIR, onnx_path)
    assert_logits_match(MODEL_DIR, onnx_path)
    write_sidecars(model_dir, out_dir)

    print()
    print(
        f"done. Serve with NEUTRAL_MODEL_BACKEND=onnx and NEUTRAL_MODEL_DIR={OUT_DIR}"
    )


if __name__ == "__main__":
    main()
