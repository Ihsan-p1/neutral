"""Fill the 2019-2020 AllSides rating of each unit from Wayback Machine snapshots.

Run: PYTHONPATH=src .venv-train/Scripts/python.exe -m training.fetch_ratings [CSV]
(rerun to retry rows still blank after a network error)

AllSides ratings change over time (USA Today was Center in June 2020 and Lean
Left in October 2026), so the Baly articles (mostly 2012-2020) are compared
with the rating of that period. AllSides Media Bias Ratings are licensed under
CC BY-NC 4.0 and used here for research with attribution.
"""

import csv
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import datetime

from training.outlets import RATINGS_CSV

START = "20190101000000"
END = "20201231235959"
NEAR = "20200601000000"
PAUSE_S = 1.0
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) neutral-prd83"
_RATING = re.compile(
    r"Media Bias Rating: (Lean Left|Lean Right|Left|Center|Right|Mixed|Not Rated)"
)


def parse_rating(html: str) -> str | None:
    match = _RATING.search(html)
    return match.group(1) if match else None


def _seconds(timestamp: str) -> float:
    return datetime.strptime(timestamp[:14], "%Y%m%d%H%M%S").timestamp()


def pick_snapshot(
    rows: list[tuple[str, str]], start: str = START, end: str = END, near: str = NEAR
) -> tuple[str, str] | None:
    """The (timestamp, url) row inside [start, end] closest to `near`, or None."""
    inside = [row for row in rows if start <= row[0][:14] <= end]
    if not inside:
        return None
    target = _seconds(near)
    return min(inside, key=lambda row: abs(_seconds(row[0]) - target))


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def _cdx_url(path: str) -> str:
    query = urllib.parse.urlencode(
        {
            "url": f"allsides.com{path}",
            "from": START[:4],
            "to": END[:4],
            "output": "json",
            "filter": "statuscode:200",
            "fl": "timestamp,original",
        }
    )
    return f"https://web.archive.org/cdx/search/cdx?{query}"


def rating_for(path: str, get: Callable[[str], str] = _get) -> tuple[str, str] | None:
    """(timestamp, rating) of the in-window snapshot, None when there is none.

    A network error returns ("error", "") so one failed outlet does not lose the
    rest of the run; a rerun retries only the rows still blank.
    """
    try:
        body = get(_cdx_url(path)).strip()
        rows = json.loads(body)[1:] if body else []
        picked = pick_snapshot([(ts, original) for ts, original in rows])
        if picked is None:
            return None
        timestamp, original = picked
        rating = parse_rating(get(f"https://web.archive.org/web/{timestamp}/{original}"))
    except (OSError, ValueError):
        return ("error", "")
    return (timestamp, rating) if rating else None


def main(path: pathlib.Path) -> None:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    for row in rows:
        if row["rating_2020"] or not row["allsides_path"]:
            continue
        found = rating_for(row["allsides_path"])
        time.sleep(PAUSE_S)
        if found is None or found[0] == "error":
            print(f"{row['unit']:32} {'error' if found else 'none'}")
            continue
        row["snapshot_2020"], row["rating_2020"] = found
        print(f"{row['unit']:32} {found[0]} {found[1]}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else RATINGS_CSV)
