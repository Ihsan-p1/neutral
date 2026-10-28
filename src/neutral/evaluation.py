import math
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from neutral.schemas import Evaluation

EVALUATION_PATH = Path(__file__).with_name("evaluation.json")

_ADAPTER = TypeAdapter(dict[str, Evaluation])


def load_evaluations(path: Path) -> dict[str, Evaluation]:
    """Read the measured scores per model version.

    Any problem with the file yields an empty map, so serving keeps running
    and responses carry no numbers rather than wrong ones. One bad entry
    empties the whole map: a partly valid file is not trusted.
    """
    try:
        return _ADAPTER.validate_json(path.read_bytes())
    except (OSError, ValidationError):
        return {}


EVALUATIONS = load_evaluations(EVALUATION_PATH)


def select_evaluation(
    evaluations: dict[str, Evaluation], version: str, backend: str, threshold: float
) -> Evaluation | None:
    """Return the scores only if they belong to what is being served.

    The numbers were measured on one ONNX checkpoint at one threshold. The
    stub can report the same version string, and another threshold changes
    every label, so both get None.
    """
    ev = evaluations.get(version)
    if ev is None or backend != "onnx":
        return None
    if not math.isclose(threshold, ev.threshold, abs_tol=1e-9):
        return None
    # A copy, so a caller editing one response cannot change the next.
    return ev.model_copy(deep=True)
