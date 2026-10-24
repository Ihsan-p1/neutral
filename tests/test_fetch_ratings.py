from training.fetch_ratings import parse_rating, pick_snapshot, rating_for


def test_parse_rating_finds_first():
    html = '<img alt="Political News Media Bias Rating: Lean Left"> ... Media Bias Rating: Right'
    assert parse_rating(html) == "Lean Left"


def test_parse_rating_none_when_absent():
    assert parse_rating("<html>no rating</html>") is None


def test_pick_snapshot_closest_in_window():
    rows = [
        ("20190514012536", "u1"),
        ("20200604083709", "u2"),
        ("20201130194646", "u3"),
    ]
    assert pick_snapshot(rows) == ("20200604083709", "u2")


def test_pick_snapshot_ignores_out_of_window():
    assert pick_snapshot([("20251030141728", "u"), ("20181231000000", "v")]) is None


def test_rating_for_reads_cdx_then_page():
    pages = {
        "cdx": '[["timestamp","original"],["20200604083709","https://www.allsides.com/x"]]',
        "web": '<img alt="Media Bias Rating: Center">',
    }

    def fake_get(url):
        return pages["cdx"] if "/cdx/" in url else pages["web"]

    assert rating_for("/news-source/x", fake_get) == ("20200604083709", "Center")


def test_rating_for_no_snapshot():
    assert rating_for("/news-source/x", lambda url: '[["timestamp","original"]]') is None


def test_rating_for_survives_network_error():
    def broken(url):
        raise OSError("HTTP Error 429")

    assert rating_for("/news-source/x", broken) == ("error", "")
