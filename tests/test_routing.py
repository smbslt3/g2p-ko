from __future__ import annotations

from g2p_ko import Handler
from g2p_ko.routing import route_candidates
import pytest


def test_latin_candidates_are_english_only() -> None:
    source = "Hello"
    routed = route_candidates(source, ())

    assert [(item.start, item.end, item.handler) for item in routed.candidates] == [(0, len(source), Handler.ENGLISH)]
    assert routed.numeric_ranges == ()
    assert routed.opaque_candidates == ()


def test_ambiguous_han_stays_separate_candidate() -> None:
    source = "東京"
    routed = route_candidates(source, ())

    assert [(item.start, item.end, item.handler) for item in routed.candidates] == [(0, len(source), Handler.KOREAN)]
    assert routed.numeric_ranges == ()
    assert routed.opaque_candidates == ()


def test_numeric_latin_structure_is_marked_opaque() -> None:
    source = "GPT-4o"
    routed = route_candidates(source, ())

    assert routed.numeric_ranges == ((0, len(source)),)
    assert [(item.start, item.end, item.handler) for item in routed.opaque_candidates] == [(0, len(source), Handler.ENGLISH)]
    assert routed.candidates == ()


def test_hangul_numeric_structure_is_silent_numeric_range() -> None:
    source = "테스트3"
    routed = route_candidates(source, ())

    assert routed.numeric_ranges == ((0, 4),)
    assert routed.opaque_candidates == ()
    assert routed.candidates == ()


@pytest.mark.parametrize("source", ["жця"])
def test_unsupported_script_is_routed_for_preserve(source: str) -> None:
    routed = route_candidates(source, ())

    assert [(item.start, item.end, item.handler, item.reason) for item in routed.candidates] == [
        (0, len(source), Handler.KOREAN, "unsupported_script"),
    ]
    assert routed.opaque_candidates == ()
    assert routed.numeric_ranges == ()


@pytest.mark.parametrize("source", ["жця世界"])
def test_foreign_script_mixed_with_han_routes_only_han_candidate(source: str) -> None:
    routed = route_candidates(source, ())
    split = len(source) - 2

    assert routed.numeric_ranges == ()
    assert [(item.start, item.end, item.handler) for item in routed.opaque_candidates] == []
    assert [(item.start, item.end, item.handler, item.reason) for item in routed.candidates] == [
        (0, split, Handler.KOREAN, "unsupported_script"),
        (split, len(source), Handler.KOREAN, "ambiguous_han"),
    ]


def test_kana_is_routed_as_unsupported_script() -> None:
    routed = route_candidates("テスト", ())
    assert [(item.reason, item.start, item.end) for item in routed.candidates] == [
        ("unsupported_script", 0, 3),
    ]
    assert routed.opaque_candidates == ()


@pytest.mark.parametrize("source", ["123", "٤٢", "²Ⅳ", "四五"])
def test_unicode_numbers_are_silent_numeric_ranges(source: str) -> None:
    routed = route_candidates(source, ())

    assert routed.numeric_ranges == ((0, len(source)),)
    assert routed.candidates == ()
    assert routed.opaque_candidates == ()


def test_numeric_han_inside_a_han_word_keeps_han_routing() -> None:
    source = "四季"
    routed = route_candidates(source, ())

    assert routed.numeric_ranges == ()
    assert [(item.start, item.end, item.reason) for item in routed.candidates] == [
        (0, len(source), "ambiguous_han"),
    ]
