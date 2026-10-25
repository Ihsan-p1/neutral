import numpy as np

from training.m4b_ablation import (
    entity_verdict,
    length_quartiles,
    length_verdict,
    mask_chunks,
    word_count,
)


def test_mask_chunks_replaces_only_entities():
    chunks = [("Senator", None), ("Jane Doe", "PERSON"), ("slammed", None),
              ("the", None), ("EPA", "ORGANIZATION"), ("in", None), ("Ohio", "GPE"),
              ("at", None), ("Camp David", "FACILITY")]
    original, masked, n = mask_chunks(chunks)
    assert original == "Senator Jane Doe slammed the EPA in Ohio at Camp David"
    assert masked == "Senator someone slammed the an organization in a place at something"
    assert n == 4


def test_mask_chunks_without_entities_is_identity():
    original, masked, n = mask_chunks([("Rain", None), (".", None)])
    assert original == masked == "Rain ." and n == 0


def test_word_count():
    assert word_count("  The  senator voted. ") == 3


def test_length_quartiles():
    q = length_quartiles(np.array([1, 2, 3, 4, 5, 6, 7, 8]))
    assert list(q) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_verdicts():
    assert entity_verdict(0.04, 0.01) == "relies on entity names"
    assert entity_verdict(0.04, -0.01) == "no measurable reliance on entity names"
    assert entity_verdict(-0.02, -0.05) == "no measurable reliance on entity names"
    assert length_verdict(0.02) == "length alone carries signal"
    assert length_verdict(-0.01) == "length alone carries no measurable signal"
