from __future__ import annotations

import pytest

from g2p_ko import G2P, InputValidationError
from g2p_ko.routing import is_han
from tests._internal_pipeline import convert_pipeline


def test_g2p_runs_normalizer_and_pronunciation_in_one_call() -> None:
    assert G2P()("사과 3개와 국물") == "사과 세 개와 궁물"


def test_g2p_applies_user_lexicon_in_the_same_call() -> None:
    assert G2P(lexicon={"NAVER": "네이버"})("NAVER 뉴스") == "네이버 뉴스"


def test_g2p_exposes_only_the_string_call_path() -> None:
    g2p = G2P()

    assert not hasattr(g2p, "convert")
    assert not hasattr(g2p, "normalize")


def test_internal_diagnostic_keeps_narrow_output_span() -> None:
    result = convert_pipeline("가🙂나")
    diagnostic = next(item for item in result.diagnostics if item.code == "unsupported_emoji")

    assert (diagnostic.source_spans[0].start, diagnostic.source_spans[0].end) == (1, 2)


def test_boolean_integer_limit_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="max_length"):
        G2P(max_length=True)


def test_english_is_always_read_as_letter_names() -> None:
    result = convert_pipeline("ABCD")

    assert result.normalized_text == "에이비씨디"
    assert result.surface_pronunciation == "에이비씨디"
    assert [item.rule_id for item in result.rewrites] == ["english.letter_names.v1"]


def test_unsupported_script_is_preserved_and_incomplete() -> None:
    result = convert_pipeline("жця")

    assert result.normalized_text == "жця"
    assert result.surface_pronunciation == "жця"
    assert not result.complete
    assert [(item.code, item.rule_id) for item in result.diagnostics] == [
        ("unsupported_script", "routing.unsupported_script"),
    ]


def test_foreign_script_and_han_are_reported_separately() -> None:
    sample = "жця世界"
    split = next(index for index, character in enumerate(sample) if is_han(character))
    result = convert_pipeline(sample)

    assert result.normalized_text == sample
    assert result.surface_pronunciation == sample
    assert not result.complete
    assert {
        (item.code, item.source_spans[0].start, item.source_spans[0].end)
        for item in result.diagnostics
    } == {
        ("unsupported_script", 0, split),
        ("unconverted_han", split, len(sample)),
    }


def test_japanese_kana_is_unsupported() -> None:
    result = convert_pipeline("가テスト나")

    assert result.normalized_text == "가テスト나"
    assert result.surface_pronunciation == "가テスト나"
    assert not result.complete
    assert [item.code for item in result.diagnostics] == ["unsupported_script"]


@pytest.mark.parametrize("source", ["123", "٤٢", "²Ⅳ", "四五"])
def test_unicode_numbers_are_unchanged_complete_passthrough(source: str) -> None:
    result = convert_pipeline(source)

    assert result.normalized_text == source
    assert result.surface_pronunciation == source
    assert result.diagnostics == ()
    assert result.complete


@pytest.mark.parametrize("source", ["테스트3", "가3", "3가"])
def test_hangul_numeric_structure_is_complete_passthrough(source: str) -> None:
    result = convert_pipeline(source)

    assert result.normalized_text == source
    assert result.surface_pronunciation == source
    assert result.diagnostics == ()
    assert result.complete


def test_protected_unsupported_script_is_preserved_and_reported() -> None:
    source = "https://жця.example.com"
    result = convert_pipeline(source)

    assert not result.complete
    assert any(
        item.code == "unsupported_script" and item.source_spans[0].surface == "жця"
        for item in result.diagnostics
    )
