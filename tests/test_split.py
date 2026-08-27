import ast
import pathlib

import pytest

from training.split import fold_of, group_key_from_url, group_split


def test_same_group_always_lands_in_one_fold():
    keys = ["a/1", "a/1", "a/1", "b/2", "b/2"]
    folds = group_split(keys, n_folds=5)
    assert folds[0] == folds[1] == folds[2]
    assert folds[3] == folds[4]


def test_no_group_appears_in_two_folds():
    # PRD 5.2 menyebut ini pemeriksaan terpenting di seluruh bagian data:
    # kalimat dari satu artikel tidak boleh berada di train dan test sekaligus.
    keys = [f"article/{i // 7}" for i in range(700)]
    folds = group_split(keys, n_folds=5)
    seen: dict[str, int] = {}
    for key, fold in zip(keys, folds, strict=True):
        if key in seen:
            assert seen[key] == fold, f"grup {key} ada di fold {seen[key]} dan {fold}"
        else:
            seen[key] = fold


def test_split_is_deterministic():
    keys = [f"article/{i}" for i in range(50)]
    assert group_split(keys, n_folds=5) == group_split(keys, n_folds=5)


def test_seed_changes_assignment():
    keys = [f"article/{i}" for i in range(200)]
    assert group_split(keys, n_folds=5, seed=0) != group_split(keys, n_folds=5, seed=1)


def test_every_fold_is_used_when_groups_are_plentiful():
    keys = [f"article/{i}" for i in range(500)]
    assert set(group_split(keys, n_folds=5)) == {0, 1, 2, 3, 4}


def test_folds_are_roughly_balanced():
    # Hash tidak menjamin fold sama besar. Yang perlu dijamin hanya bahwa tidak
    # ada fold yang nyaris kosong, karena F1 per fold akan tidak stabil.
    keys = [f"article/{i}" for i in range(1000)]
    folds = group_split(keys, n_folds=5)
    sizes = [folds.count(f) for f in range(5)]
    assert min(sizes) > 1000 / 5 * 0.7, f"fold terlalu timpang: {sizes}"


def test_rejects_fewer_than_two_folds():
    with pytest.raises(ValueError):
        fold_of("a", n_folds=1)


def test_group_key_drops_query_and_fragment():
    a = group_key_from_url("https://example.com/news/story?utm_source=x#top")
    b = group_key_from_url("https://example.com/news/story")
    assert a == b


def test_group_key_drops_trailing_slash_and_lowercases_host():
    a = group_key_from_url("https://Example.COM/news/story/")
    b = group_key_from_url("https://example.com/news/story")
    assert a == b


def test_group_key_keeps_distinct_articles_distinct():
    # Redirect feedproxy membawa id artikel di path, jadi keduanya tidak boleh
    # menyatu menjadi satu grup. Dua URL ini diambil dari BABE.
    a = group_key_from_url("http://feedproxy.google.com/~r/breitbart/~3/F5vueG8YCQQ/")
    b = group_key_from_url("http://feedproxy.google.com/~r/breitbart/~3/ksXbL6hU-Ko/")
    assert a != b


def test_split_module_imports_only_stdlib():
    # split.py harus tetap dapat diimpor di job CI serving, yang hanya memasang
    # requirements-dev.txt. Bila ia menumbuhkan impor datasets/pandas/sklearn/
    # torch/numpy, test ini gagal di sini, bukan gagal diam-diam di CI.
    source = (
        pathlib.Path(__file__).resolve().parents[1] / "training" / "split.py"
    ).read_text(encoding="utf-8")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"hashlib", "urllib"}, f"impor tidak diizinkan: {imported}"
