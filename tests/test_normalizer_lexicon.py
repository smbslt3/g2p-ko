from __future__ import annotations

from time import perf_counter
import unicodedata

import pytest

from g2p_ko import InputValidationError
from g2p_ko.normalizer.lexicon import CompiledLexicon, LexiconMatch


def _identity(text: str) -> str:
    return text


def test_input_mapping_is_copied_and_values_use_base_normalizer() -> None:
    source = {"X": "5개", "Y": "X"}
    calls: list[str] = []

    def base_normalize(text: str) -> str:
        calls.append(text)
        return text.replace("5개", "다섯 개")

    lexicon = CompiledLexicon(source, base_normalize=base_normalize)
    source["X"] = "변경됨"

    assert calls == ["5개", "X"]
    assert dict(lexicon) == {"X": "다섯 개", "Y": "X"}
    assert lexicon["X"] == "다섯 개"
    assert len(lexicon) == 2


@pytest.mark.parametrize(
    "entries",
    [
        None,
        [],
        {1: "값"},
        {"키": 1},
        {"": "값"},
        {"키": ""},
    ],
)
def test_mapping_and_entry_shapes_are_validated(entries: object) -> None:
    with pytest.raises(InputValidationError):
        CompiledLexicon(entries, base_normalize=_identity)  # type: ignore[arg-type]


def test_base_normalizer_must_be_callable() -> None:
    with pytest.raises(InputValidationError, match="base_normalize"):
        CompiledLexicon(
            {"키": "값"},
            base_normalize=None,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "entries",
    [
        {"키\ud800": "값"},
        {"키": "값\udfff"},
        {"키\x00": "값"},
        {"키": "값\n"},
        {"키": "값\u200b"},
    ],
)
def test_surrogate_and_control_characters_are_rejected(entries: dict[str, str]) -> None:
    with pytest.raises(InputValidationError):
        CompiledLexicon(entries, base_normalize=_identity)


def test_nfc_equivalent_keys_are_rejected_as_duplicates() -> None:
    with pytest.raises(InputValidationError, match="NFC.*중복"):
        CompiledLexicon(
            {"가": "첫째", "\u1100\u1161": "둘째"},
            base_normalize=_identity,
        )


@pytest.mark.parametrize(
    ("entries", "limits"),
    [
        ({"긴키": "값"}, {"max_key_length": 1}),
        ({"키": "긴값"}, {"max_value_length": 1}),
        ({"A": "가", "B": "나"}, {"max_entries": 1}),
        ({"A": "가", "B": "나"}, {"max_total_size": 3}),
    ],
)
def test_resource_limits_are_enforced(
    entries: dict[str, str],
    limits: dict[str, int],
) -> None:
    with pytest.raises(InputValidationError):
        CompiledLexicon(entries, base_normalize=_identity, **limits)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("max_key_length", 0),
        ("max_value_length", -1),
        ("max_entries", True),
        ("max_total_size", 1.5),
    ],
)
def test_resource_limits_must_be_positive_integers(option: str, value: object) -> None:
    with pytest.raises(InputValidationError):
        CompiledLexicon(
            {"A": "가"},
            base_normalize=_identity,
            **{option: value},
        )


@pytest.mark.parametrize("result", ["", 1, "값\x00"])
def test_base_normalizer_output_is_validated(result: object) -> None:
    def base_normalize(_: str) -> str:
        return result  # type: ignore[return-value]

    with pytest.raises(InputValidationError):
        CompiledLexicon({"키": "값"}, base_normalize=base_normalize)


def test_base_normalizer_expansion_cannot_bypass_value_or_total_limits() -> None:
    with pytest.raises(InputValidationError):
        CompiledLexicon(
            {"A": "가"},
            base_normalize=lambda _: "아주 긴 값",
            max_value_length=3,
        )
    with pytest.raises(InputValidationError):
        CompiledLexicon(
            {"A": "가", "B": "나"},
            base_normalize=lambda _: "긴값",
            max_total_size=5,
        )


def test_values_are_not_interpreted_as_references_to_other_keys() -> None:
    lexicon = CompiledLexicon(
        {
            "A": "AB",
            "AB": "B",
            "B": "종단",
            "문장": "A와 B",
        },
        base_normalize=_identity,
    )

    assert lexicon["A"] == "AB"
    assert lexicon["AB"] == "B"
    assert lexicon["문장"] == "A와 B"


def test_leftmost_longest_iterator_returns_non_overlapping_matches() -> None:
    lexicon = CompiledLexicon(
        {"A": "에이", "AI": "에이아이", "API": "에이피아이"},
        base_normalize=_identity,
    )
    text = "API와 AI, A!"

    assert list(lexicon.iter_matches(text)) == [
        LexiconMatch(0, 3, "API", "에이피아이"),
        LexiconMatch(5, 7, "AI", "에이아이"),
        LexiconMatch(9, 10, "A", "에이"),
    ]


def test_ascii_and_hangul_boundaries_match_the_existing_contract() -> None:
    lexicon = CompiledLexicon(
        {"AI": "에이아이", "서울": "서울특별시"},
        base_normalize=_identity,
    )
    text = "XAI AI2 AI는 신서울 서울역 서울 출발"

    assert list(lexicon.iter_matches(text)) == [
        LexiconMatch(8, 10, "AI", "에이아이"),
        LexiconMatch(20, 22, "서울", "서울특별시"),
    ]


def test_match_iterator_validates_text_and_range() -> None:
    lexicon = CompiledLexicon({"A": "에이"}, base_normalize=_identity)

    with pytest.raises(InputValidationError):
        list(lexicon.iter_matches(None))  # type: ignore[arg-type]
    with pytest.raises(InputValidationError):
        list(lexicon.iter_matches("A", start=-1))
    with pytest.raises(InputValidationError):
        list(lexicon.iter_matches("A", start=1, end=0))


def test_irrelevant_text_scan_stays_fast_with_five_thousand_keys() -> None:
    entries = {f"KEY{index:04d}": f"값{index}" for index in range(5_000)}
    lexicon = CompiledLexicon(entries, base_normalize=_identity)
    text = "무관한문장" * 2_000

    started = perf_counter()
    matches = list(lexicon.iter_matches(text))
    elapsed = perf_counter() - started

    assert matches == []
    assert len(text) == 10_000
    assert elapsed < 1.0


def test_keys_and_callback_results_are_stored_in_nfc() -> None:
    lexicon = CompiledLexicon(
        {"\u1100\u1161": "\u1102\u1161"},
        base_normalize=lambda text: unicodedata.normalize("NFD", text),
    )

    assert dict(lexicon) == {"가": "나"}
