"""TTS 규칙집의 과학 표기와 공개 API 경계 계약을 추가로 검증한다."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from g2p_ko import ConversionPolicyError, InputValidationError, RewriteStage
from g2p_ko.normalizer import KoreanTTSNormalizer


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2026-08-07", "이천이십육 년 팔 월 칠 일"),
        ("10^-5", "십의 마이너스 오승"),
        ("3g/mL", "삼 그램 퍼 밀리리터"),
        ("4.19 혁명", "사일구 혁명"),
        ("10.26 사건", "십이육 사건"),
    ],
)
def test_tts_rulebook_science_and_calendar_forms(source: str, expected: str) -> None:
    assert KoreanTTSNormalizer()(source) == expected


def test_unrelated_history_date_and_event_name_are_not_lexicalized() -> None:
    result = KoreanTTSNormalizer().convert("7.4 민주화운동")

    assert result.normalized_text == "칠 점 사 민주화운동"
    assert all(item.rule_id != "normalizer.historical_date.v1" for item in result.rewrites)


@pytest.mark.parametrize("source", ["H2O", "NaOH", ".json", "USB-C", "GPT3"])
def test_benchmark_derived_narrow_conversions_are_preserved(source: str) -> None:
    """벤치마크 유래의 좁은 읽기 규칙은 제거하고 원문을 보존한다."""

    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert not result.complete


def test_invalid_calendar_date_and_malformed_number_are_preserved() -> None:
    normalizer = KoreanTTSNormalizer()

    invalid_date = normalizer.convert("2026-02-30")
    malformed_number = normalizer.convert("12,34")

    assert invalid_date.normalized_text == "2026-02-30"
    assert malformed_number.normalized_text == "12,34"
    assert {item.code for item in invalid_date.diagnostics} == {"ambiguous_numeric"}
    assert {item.code for item in malformed_number.diagnostics} == {"ambiguous_numeric"}


def test_decimal_slash_expression_is_preserved_instead_of_crashing() -> None:
    result = KoreanTTSNormalizer().convert("1.5/2.5")

    assert result.normalized_text == "1.5/2.5"
    assert {item.code for item in result.diagnostics} == {"ambiguous_numeric"}


def test_oversized_chemical_count_is_preserved_instead_of_crashing() -> None:
    source = "Na" + "9" * 33 + "Cl"
    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert not result.complete


@pytest.mark.parametrize("source", ["₩0012", "₩ 0012", "0012%", "0012 %", "01 개"])
def test_unresolved_numeric_token_preserves_its_exact_surface(source: str) -> None:
    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert {item.code for item in result.diagnostics} == {"ambiguous_numeric"}
    assert not result.rewrites


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ㄳ", "기역시옷"),
        ("ㄵ", "니은지읒"),
        ("ㄶ", "니은히읗"),
        ("ㄺ", "리을기역"),
        ("ㄻ", "리을미음"),
        ("ㄼ", "리을비읍"),
        ("ㄽ", "리을시옷"),
        ("ㄾ", "리을티읕"),
        ("ㄿ", "리을피읖"),
        ("ㅀ", "리을히읗"),
        ("ㅄ", "비읍시옷"),
    ],
)
def test_compound_consonant_jamo_has_a_canonical_name(source: str, expected: str) -> None:
    assert KoreanTTSNormalizer()(source) == expected


@pytest.mark.parametrize(
    ("source", "code"),
    [
        ("B2B", "ambiguous_identifier"),
        ("Wi-Fi", "unconverted_latin"),
        ("01개", "ambiguous_numeric"),
    ],
)
def test_ambiguous_mixed_tokens_are_not_partially_normalized(source: str, code: str) -> None:
    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert {item.code for item in result.diagnostics} == {code}


@pytest.mark.parametrize("source", ["-8GF", "37T5L", "4m8g"])
def test_unknown_alphanumeric_identifier_is_atomic_and_idempotent(source: str) -> None:
    normalizer = KoreanTTSNormalizer()
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert normalizer(result.normalized_text) == source
    assert {item.code for item in result.diagnostics} == {"ambiguous_identifier"}


@pytest.mark.parametrize("source", [".4GPU", "PIN 1234", "주문번호 12345", "좌석 12"])
def test_contextual_or_punctuated_numeric_identifier_is_not_read_as_quantity(
    source: str,
) -> None:
    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.diagnostics


def test_user_lexicon_is_copied_and_runs_before_unknown_latin() -> None:
    lexicon = {"KRAFTON": "크래프톤"}
    normalizer = KoreanTTSNormalizer(lexicon=lexicon)
    lexicon["KRAFTON"] = "변경됨"

    result = normalizer.convert("KRAFTON은 회사다.")

    assert result.normalized_text == "크래프톤은 회사다."
    assert result.complete
    assert result.diagnostics == ()


def test_hangul_lexicon_key_does_not_replace_inside_a_longer_word() -> None:
    normalizer = KoreanTTSNormalizer(lexicon={"서울": "서울특별시"})

    assert normalizer("서울역에서 서울 출발") == "서울역에서 서울특별시 출발"


def test_lexicon_value_is_verbalized_once_and_remains_idempotent() -> None:
    normalizer = KoreanTTSNormalizer(lexicon={"X": "5개"})

    once = normalizer("X")
    assert once == "다섯 개"
    assert normalizer(once) == once


def test_ambiguous_policy_can_suppress_or_raise_diagnostic() -> None:
    preserved = KoreanTTSNormalizer(on_ambiguous="preserve").convert("Minnie")

    assert preserved.normalized_text == "Minnie"
    assert preserved.diagnostics == ()
    assert not preserved.complete
    with pytest.raises(ConversionPolicyError, match="unconverted_latin"):
        KoreanTTSNormalizer(on_ambiguous="error").normalize("Minnie")


def test_result_serialization_exposes_verbalization_provenance() -> None:
    result = KoreanTTSNormalizer().convert("총 5개")
    payload = result.to_dict()

    assert payload["normalized_text"] == "총 다섯 개"
    assert "text" not in payload
    assert payload["rewrites"][0]["stage"] == RewriteStage.VERBALIZATION.value


def test_nfd_input_rewrite_keeps_original_numeric_coordinates() -> None:
    source = "\u1100\u1161 5개"
    result = KoreanTTSNormalizer().convert(source)
    numeric = next(item for item in result.rewrites if item.rule_id.startswith("normalizer.number_"))

    assert result.normalized_text == "가 다섯 개"
    assert numeric.source_spans[0].start == 3
    assert numeric.source_spans[0].end == 5
    assert numeric.source_spans[0].surface == "5개"


def test_batch_apis_are_lazy_and_do_not_materialize_input() -> None:
    consumed: list[str] = []

    def source() -> Iterator[str]:
        for item in ("1개", "$5"):
            consumed.append(item)
            yield item

    normalizer = KoreanTTSNormalizer()
    normalized = normalizer.normalize_many(source())

    assert consumed == []
    assert next(normalized) == "한 개"
    assert consumed == ["1개"]
    assert next(normalized) == "오 달러"
    assert [result.normalized_text for result in normalizer.convert_many(iter(("1개", "$5")))] == [
        "한 개",
        "오 달러",
    ]


@pytest.mark.parametrize(
    ("option", "value"),
    [("on_ambiguous", "ignore"), ("max_length", True)],
)
def test_public_options_reject_unsupported_values(option: str, value: object) -> None:
    with pytest.raises(InputValidationError):
        KoreanTTSNormalizer(**{option: value})


@pytest.mark.parametrize("source", ["APPLE", "SEOUL", "CoCo", "BaNaNa"])
def test_unknown_uppercase_word_or_name_is_not_guessed_as_an_acronym_or_formula(
    source: str,
) -> None:
    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.diagnostics


@pytest.mark.parametrize("source", ["3시스템", "3개발", "3점프"])
def test_korean_unit_prefix_does_not_partially_convert_a_longer_word(source: str) -> None:
    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert {item.code for item in result.diagnostics} == {"ambiguous_numeric"}


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("제2장", "제 이 장"),
        ("1번째", "첫 번째"),
        ("1~3", "일에서 삼"),
    ],
)
def test_ordinal_context_overrides_generic_number_ambiguity(
    source: str,
    expected: str,
) -> None:
    assert KoreanTTSNormalizer()(source) == expected


@pytest.mark.parametrize("source", ["10000000000원", "10000000000개"])
def test_eleven_plus_digit_numbers_are_preserved_even_with_units(source: str) -> None:
    """11자리 이상 숫자는 단위가 붙어도 식별번호일 수 있어 보존한다."""

    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert not result.complete


@pytest.mark.parametrize(
    "source",
    ["5$", "3×4", "10±2kg", "2^3개", "3kg/s", "3+4", "-1/2"],
)
def test_unsupported_semantic_structure_is_preserved_atomically(source: str) -> None:
    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert any(
        span.start == 0 and span.end == len(source)
        for item in result.diagnostics
        for span in item.source_spans
    )


@pytest.mark.parametrize("source", ["$", "%", "±", "×", "²"])
def test_standalone_semantic_symbol_is_not_reported_as_complete(source: str) -> None:
    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert {item.code for item in result.diagnostics} == {"unsupported_symbol"}


@pytest.mark.parametrize("source", [".5", "-.5", "1e+3", "３개"])
def test_unsupported_numeric_surface_is_preserved_instead_of_partially_read(
    source: str,
) -> None:
    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert {item.code for item in result.diagnostics} == {"ambiguous_numeric"}


def test_oversized_historical_date_is_preserved_instead_of_raising() -> None:
    source = "9" * 33 + ".1절"
    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert not result.complete


@pytest.mark.parametrize(
    "source",
    ["171866458525는", "314159265358번을", "12345678901개"],
)
def test_long_digit_identifier_guard_survives_attached_particles_and_units(
    source: str,
) -> None:
    """조사나 단위가 붙어도 11자리 이상 숫자는 수량으로 읽지 않는다."""

    result = KoreanTTSNormalizer().convert(source)

    assert result.normalized_text == source
    assert not result.complete
