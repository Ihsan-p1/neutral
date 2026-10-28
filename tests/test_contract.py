import json
from pathlib import Path

from neutral.schemas import AnalyzeResponse

SNAPSHOT = Path(__file__).parent / "contract" / "analyze_response.schema.json"


def test_analyze_response_schema_matches_snapshot():
    current = AnalyzeResponse.model_json_schema()
    assert current == json.loads(SNAPSHOT.read_text(encoding="utf-8")), (
        "AnalyzeResponse changed shape. Sentra depends on it: update the snapshot "
        "in the same commit only if the change is intended."
    )
