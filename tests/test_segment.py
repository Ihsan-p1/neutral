from neutral.segment import split_sentences


def test_splits_simple_sentences():
    out = split_sentences("The bill passed. The senator objected.")
    assert len(out) == 2
    assert out[0].text == "The bill passed."


def test_abbreviations_do_not_split():
    # This is the whole reason for using pysbd rather than splitting on periods.
    text = "The U.S. economy grew. Mr. Smith of Acme Inc. disagreed."
    out = split_sentences(text)
    assert len(out) == 2
    assert "U.S." in out[0].text
    assert "Inc." in out[1].text


def test_offsets_map_back_to_source():
    text = "The bill passed. The senator objected."
    for s in split_sentences(text):
        assert text[s.start : s.end] == s.text


def test_offsets_survive_multiline_text():
    text = "The bill passed.\n\n  The senator objected.  \nCritics called it reckless."
    for s in split_sentences(text):
        assert text[s.start : s.end] == s.text


def test_empty_and_whitespace():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []
