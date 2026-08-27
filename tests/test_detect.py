from neutral.detect import is_english


def test_detects_english():
    assert is_english(
        "The senator delivered a scathing rebuke of the proposed budget on Tuesday."
    )


def test_detects_non_english():
    assert not is_english(
        "Senator itu menyampaikan kritik keras terhadap rancangan anggaran pada hari Selasa."
    )


def test_short_text_defaults_to_english():
    # Short text cannot be detected reliably. Do not flag a user wrongly;
    # treat it as in scope and let it through.
    assert is_english("Budget cuts.")


def test_empty_text_is_in_scope():
    assert is_english("")
