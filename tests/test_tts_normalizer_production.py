"""실제 TTS 투입을 막는 회귀를 우선 고정한 프로덕션 명세다."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

import pytest

from g2p_ko import InputValidationError
from g2p_ko.normalizer import KoreanTTSNormalizer


@pytest.fixture
def normalizer() -> KoreanTTSNormalizer:
    return KoreanTTSNormalizer()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("전압은 3.3V다.", "전압은 삼 점 삼 볼트다."),
        ("샘플링은 44.1kHz다.", "샘플링은 사십사 점 일 킬로헤르츠다."),
        ("속도는 100km/h다.", "속도는 시속 백 킬로미터다."),
        ("용량은 16GB다.", "용량은 십육 기가바이트다."),
        ("면적은 3m²다.", "면적은 삼 제곱미터다."),
        ("체적은 2㎥다.", "체적은 이 세제곱미터다."),
        ("전력은 100kW다.", "전력은 백 킬로와트다."),
        ("소음은 42dB입니다.", "소음은 사십이 데시벨입니다."),
    ],
)
def test_common_tts_measurements_are_read_as_one_token(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "3.3XYZ",
        "12%abc",
        "1/2abc",
        "010-1234-5678abc",
        "3개abc",
        "₩5달러abc",
        "2026-08-07입니다abc",
    ],
)
def test_unknown_numeric_suffix_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()
    assert any(
        span.start == 0 and span.end == len(source)
        for diagnostic in result.diagnostics
        for span in diagnostic.source_spans
    )


@pytest.mark.parametrize("source", ["12345678901.5", "12345678901,5"])
def test_long_numeric_cluster_is_never_partially_converted(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()
    assert len(result.diagnostics) == 1


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("오전 9:30", "오전 아홉 시 삼십 분"),
        ("오후 3:05:07", "오후 세 시 오 분 칠 초"),
        ("12:30", "열두 시 삼십 분"),
        ("13:05", "십삼 시 오 분"),
        ("09시 05분", "아홉 시 오 분"),
        ("24시 00분", "이십사 시 영 분"),
    ],
)
def test_clock_time_is_validated_and_verbalized_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == expected
    assert result.complete


@pytest.mark.parametrize("source", ["25:00", "12:60", "24:01", "오후 13:00", "3:1"])
def test_invalid_or_ratio_like_clock_is_preserved(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert len(result.diagnostics) == 1


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("오전 3시~오후 4시", "오전 세 시에서 오후 네 시"),
        ("3시 30분~5시 10분", "세 시 삼십 분에서 다섯 시 십 분"),
        ("13:00~14:30", "십삼 시 영 분에서 십사 시 삼십 분"),
        ("1시〜2시까지", "한 시에서 두 시까지"),
    ],
)
def test_complete_time_range_is_verbalized_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == expected
    assert result.complete
    assert [item.rule_id for item in result.rewrites] == ["normalizer.clock_range.v1"]


@pytest.mark.parametrize(
    "source",
    ["25:00~26:00", "1시부터~2시", "1시~2시간", "1~2시"],
)
def test_ambiguous_time_like_range_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()
    assert len(result.diagnostics) == 1


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2026. 8. 7.", "이천이십육 년 팔 월 칠 일."),
        ("2026. 8. 7.입니다", "이천이십육 년 팔 월 칠 일입니다"),
        ("2024년 09월 01일", "이천이십사 년 구 월 일 일"),
        ("2024 년 2 월 29 일", "이천이십사 년 이 월 이십구 일"),
    ],
)
def test_spaced_and_korean_calendar_dates_use_one_validated_rule(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize("source", ["2024년 2월 30일", "2023. 2. 29.", "2024 년 13 월 1 일"])
def test_invalid_calendar_date_is_preserved_as_one_token(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()
    assert len(result.diagnostics) == 1


@pytest.mark.parametrize("source", ["13월", "2024년 13월"])
def test_invalid_month_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()
    assert len(result.diagnostics) == 1


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("0507-1234-5678", "공 오 공 칠 일 이 삼 사 오 육 칠 팔"),
        ("1588-1234", "일 오 팔 팔 일 이 삼 사"),
        ("1588 1234", "일 오 팔 팔 일 이 삼 사"),
        ("010 1234 5678", "공 일 공 일 이 삼 사 오 육 칠 팔"),
        ("010.1234.5678", "공 일 공 일 이 삼 사 오 육 칠 팔"),
        ("+82-10-1234-5678", "플러스 팔 이 일 공 일 이 삼 사 오 육 칠 팔"),
    ],
)
def test_common_korean_phone_forms_are_read_digit_by_digit(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("₩500원", "오백 원"),
        ("$5달러", "오 달러"),
        ("USD 5", "오 달러"),
        ("5 USD", "오 달러"),
        ("3.5만원", "삼만오천 원"),
        ("1~2만원", "일에서 이만 원"),
    ],
)
def test_currency_forms_do_not_duplicate_or_lose_scale(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("KRW 1000원", "천 원"),
        ("USD 5달러", "오 달러"),
        ("5 EUR유로", "오 유로"),
    ],
)
def test_iso_currency_may_repeat_only_the_matching_spoken_unit(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    "source",
    ["USD 5원", "$5 USD", "USD $5", "₩1000 KRW", "KRW ₩1000", "₩3.5만원"],
)
def test_mixed_or_conflicting_currency_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()


def test_signed_iso_currency_is_read_as_one_token(
    normalizer: KoreanTTSNormalizer,
) -> None:
    assert normalizer("-1000 KRW") == "마이너스 천 원"


def test_sign_before_currency_symbol_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
) -> None:
    result = normalizer.convert("-$5")

    assert result.normalized_text == "-$5"
    assert not result.complete
    assert result.rewrites == ()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("3~5세", "삼에서 오 세"),
        ("10~20%", "십에서 이십 퍼센트"),
        ("-3~-1℃", "마이너스 삼에서 마이너스 일 도씨"),
        ("1∼3개", "하나에서 세 개"),
    ],
)
def test_range_reads_trailing_unit_once(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("3개였다", "세 개였다"),
        ("3개라고", "세 개라고"),
        ("3개에는", "세 개에는"),
        ("2명이었다", "두 명이었다"),
        ("5원이라고", "오 원이라고"),
        ("3개월간", "삼 개월간"),
        ("5개당", "다섯 개당"),
        ("사과 3알", "사과 세 알"),
        ("꽃 2송이", "꽃 두 송이"),
    ],
)
def test_common_counter_suffixes_remain_readable(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("3천원", "삼천 원"),
        ("3백kg", "삼백 킬로그램"),
        ("5위안", "오 위안"),
        ("CNY 5위안", "오 위안"),
        ("3개짜리", "세 개짜리"),
        ("3개가량", "세 개가량"),
        ("3호실", "삼 호실"),
        ("101동", "백일 동"),
        ("1단계", "일 단계"),
        ("3종", "삼 종"),
        ("집 2채", "집 두 채"),
        ("나무 3그루", "나무 세 그루"),
        ("연필 4자루", "연필 네 자루"),
        ("배추 5포기", "배추 다섯 포기"),
        ("3〜5kg", "삼에서 오 킬로그램"),
    ],
)
def test_general_scale_currency_counter_and_range_forms(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


def test_ordinal_and_numbered_location_use_sino_context(
    normalizer: KoreanTTSNormalizer,
) -> None:
    assert normalizer("제 2장과 3번 출구") == "제 이 장과 삼 번 출구"
    assert normalizer("3번 문제를 풀었다") == "삼 번 문제를 풀었다"
    assert normalizer("세 번 시도") == "세 번 시도"


@pytest.mark.parametrize(
    "source",
    [
        "카드번호 1234 5678 9012 3456 99",
        "코드 1 2 3 4 5",
        "번호 123-456-789-012-345",
        "전화번호 1234",
        "비밀번호 1234",
        "좌석 12A",
    ],
)
def test_entire_identifier_context_is_preserved(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("119에 신고", "일 일 구에 신고"),
        ("112로 전화", "일 일 이로 전화"),
        ("신고는 112", "신고는 일 일 이"),
    ],
)
def test_emergency_numbers_are_read_digit_by_digit(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize("source", ["2 1/2", "2 1/2kg", "1 3/4L"])
def test_mixed_fraction_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()
    assert len(result.diagnostics) == 1


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("112 페이지를 참고하라", "백십이 페이지를 참고하라"),
        ("119 곳의 매장", "백십구 곳의 매장"),
    ],
)
def test_emergency_rule_does_not_capture_ordinary_quantities(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    """긴급 번호 읽기는 조사·문말 문맥에서만 발화한다."""

    assert normalizer(source) == expected


def test_unseparated_hotline_digits_use_learned_digitwise_reading(
    normalizer: KoreanTTSNormalizer,
) -> None:
    """구분자 없는 대표번호형 숫자는 학습 판별기가 자릿수대로 읽는다."""

    assert normalizer("15881234") == "일 오 팔 팔 일 이 삼 사"
    assert normalizer("예산은 18000000원이다") == "예산은 천팔백만 원이다"


def test_measurement_decimals_before_history_suffixes_stay_decimal(
    normalizer: KoreanTTSNormalizer,
) -> None:
    """역사 날짜 읽기는 닫힌 날짜 목록에만 적용한다."""

    assert normalizer("규모 5.8 사건 이후") == "규모 오 점 팔 사건 이후"


@pytest.mark.parametrize("source", ["2B 연필", "3A 등급"])
def test_context_disambiguates_single_letter_units(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()


@pytest.mark.parametrize(
    "source",
    [
        "인증번호 123456",
        "시리얼 번호 123456",
        "카드번호 1234 5678 9012 3456",
        "차량번호 12가3456",
        "GPT-5.1",
    ],
)
def test_identifier_context_is_not_misread_as_quantity(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete


def test_unknown_attached_korean_word_is_not_partially_read(
    normalizer: KoreanTTSNormalizer,
) -> None:
    result = normalizer.convert("3사과")

    assert result.normalized_text == "3사과"
    assert not result.complete
    assert result.rewrites == ()


@pytest.mark.parametrize("source", ["2월 30일", "13월 1일", "0월 1일", "4월 31일"])
def test_invalid_month_day_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()


def test_valid_month_day_is_verbalized_atomically(
    normalizer: KoreanTTSNormalizer,
) -> None:
    assert normalizer("2월 29일") == "이 월 이십구 일"


@pytest.mark.parametrize("source", ["날짜는 3/4입니다.", "생일은 3/4", "날짜는 3.4"])
def test_short_numeric_date_context_is_not_misread(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()


def test_malformed_spaced_phone_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
) -> None:
    result = normalizer.convert("010 123 45678")

    assert result.normalized_text == "010 123 45678"
    assert not result.complete
    assert result.rewrites == ()


@pytest.mark.parametrize("source", ["+82 10 123 45678", "010 12345 6789", "02 123 45678"])
def test_other_malformed_phone_candidates_are_atomic(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()


@pytest.mark.parametrize(
    "source",
    [
        "12:00-13:00",
        "2024-01-01 – 2024-01-31",
        "$1 - $2",
        "1kg - 3kg",
        "1% - 3%",
    ],
)
def test_unsupported_semantic_pairs_are_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()


def test_card_number_groups_are_not_partially_read_as_phone(
    normalizer: KoreanTTSNormalizer,
) -> None:
    """유효한 전화 접두가 아닌 숫자 그룹은 전화번호로 잘라 읽지 않는다."""

    source = "카드 0021 2832 5344 3074 확인"
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("02-312-4567로", "공 이 삼 일 이 사 오 육 칠로"),
        ("031-234-5678", "공 삼 일 이 삼 사 오 육 칠 팔"),
        ("070-1234-5678", "공 칠 공 일 이 삼 사 오 육 칠 팔"),
        ("088-1452-3265로", "공 팔 팔 일 사 오 이 삼 이 육 오로"),
        ("060-700-1234", "공 육 공 칠 공 공 일 이 삼 사"),
    ],
)
def test_known_phone_prefixes_are_still_read_as_phone(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "0021 2832 5344 3074로 이체했다",
        "0021 2832 5344 3074번 카드",
        "0021 2832 5344 3074했다",
        "010-1234-5678 2024년",
    ],
)
def test_digit_group_chain_after_phone_shape_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    """네 자리 묶음이 이어지는 사슬은 어떤 꼬리가 붙어도 잘라 읽지 않는다."""

    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete


@pytest.mark.parametrize(
    "source",
    [
        "3월 1일~3월 5일",
        "3월 1일~3월 5일까지",
        "1월 1일~2월 28일",
        "2024년 1월~2024년 3월",
    ],
)
def test_date_range_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    """날짜 범위는 읽기를 지원하기 전까지 부분 변환 없이 전체를 보존한다."""

    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete


def test_decimal_before_scale_word_is_not_partially_converted(
    normalizer: KoreanTTSNormalizer,
) -> None:
    """`3.000만 원`의 만을 조사로 오인해 소수만 읽는 부분 변환을 막는다."""

    source = "행사에 3.000만 원이 든다."
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete


@pytest.mark.parametrize(
    "source",
    [
        "커피가 5.000원이다.",
        "3.000만 원으로 계약했다.",
        "1.500만 원",
        "행사비는 2.000달러였다.",
    ],
)
def test_three_digit_fraction_currency_may_be_thousands_separator(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    """세 자리 소수부가 붙은 통화는 유럽식 천 단위 구분점일 수 있어 보존한다."""

    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete


@pytest.mark.parametrize(
    "source",
    [
        "가격은 €5.000이다.",
        "$2.000",
        "USD 2.000",
    ],
)
def test_three_digit_fraction_symbol_currency_is_also_preserved(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    """기호·ISO 통화 경로도 천 단위 구분점 가드를 똑같이 따른다."""

    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1.5만 원", "만오천 원"),
        ("환율이 1,350.25원이다.", "환율이 천삼백오십 점 이 오 원이다."),
        ("$19.99", "십구 점 구 구 달러"),
    ],
)
def test_short_fraction_currency_still_reads_as_decimal(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


def test_hangul_ton_homograph_is_preserved(
    normalizer: KoreanTTSNormalizer,
) -> None:
    """`톤`은 무게 단위와 색·음의 단계가 동형이라 한글 표기는 보존한다."""

    source = "피부 톤이 2톤 밝아졌다."
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete


def test_counter_with_only_particle_man_is_still_converted(
    normalizer: KoreanTTSNormalizer,
) -> None:
    assert normalizer("사과 3개만 먹어") == "사과 세 개만 먹어"


@pytest.mark.parametrize("source", ["1 ~ 3개발", "1 ~ 3세대"])
def test_range_does_not_consume_a_unit_prefix(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()


@pytest.mark.parametrize("source", ["*31", "∼31", "=31", "≤31", "÷31", "×31", "^31"])
def test_unsupported_leading_symbol_and_number_are_atomic(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()


@pytest.mark.parametrize(
    ("source", "expected"),
    [("값은 2.", "값은 이."), ("합계는 12,345,", "합계는 만이천삼백사십오,")],
)
def test_sentence_punctuation_after_number_is_preserved(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize("source", ["1$2.", "0$0$"])
def test_malformed_currency_cluster_is_atomic_and_idempotent(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    once = normalizer.convert(source)

    assert once.normalized_text == source
    assert not once.complete
    assert normalizer(once.normalized_text) == source


@pytest.mark.parametrize(
    "source",
    ["$\n5", "₩\r\n5", "3\nkg", "5\n%", "1\n~\n3개", "6\n월"],
)
def test_semantic_rules_never_consume_line_breaks(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    assert "\n" in normalizer(source)
    assert normalizer(source).count("\n") == source.count("\n")
    assert normalizer(source).count("\r") == source.count("\r")


@pytest.mark.parametrize("source", ["①개", "½개", "⁴개", "Ⅳ", "😀", "日本語", "жця", "ＡＢＣ", "\x00", "\u200d"])
def test_unsupported_unicode_is_preserved_but_never_reported_complete(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.diagnostics


@pytest.mark.parametrize("source", ["3①", "①3", "3½", "Ⅳ3", "3⁴", "3１", "3𝟙"])
def test_non_ascii_numeric_neighbors_are_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()
    assert len(result.diagnostics) == 1


@pytest.mark.parametrize(
    "source",
    [
        "1①g",
        "①1g",
        "1%①",
        "$1①",
        "+1①",
        "$€0",
        "€$1",
        "₩0Ⅳ",
        "𝟙+1.2",
        "𝟙₩1",
        "3:05*",
        "3:05Ⅳ",
        "2026-08-07÷",
        "2026-08-07Ⅳ",
        "31~1$",
        "31~1½",
        "+−1",
        "+−12,345",
        "++1",
        "+-1",
        "--1",
        "-+1",
        "+₩2/5",
        "+$2/5",
        "+€12,345kg",
        "31~1H>3",
        "H>3",
        "M>0",
        "V<9g",
        "H>3kg",
        "V<9%",
        "M>0V",
        "/3>0",
        "/H>3",
        "/2<2",
    ],
)
def test_mixed_numeric_cluster_with_unsupported_unicode_is_atomic(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()
    assert len(result.diagnostics) == 1


@pytest.mark.parametrize(
    "source",
    ["AI시01MHz$5시", "USB시01MHz$5시", "AI$2ㄱ", "AI₩2ㄱ", "USB₩2ㄱ"],
)
def test_acronym_next_to_ambiguous_numeric_cluster_is_idempotent(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    once = normalizer.convert(source)

    assert once.normalized_text == source
    assert not once.complete
    assert once.rewrites == ()
    assert normalizer(once.normalized_text) == once.normalized_text


@pytest.mark.parametrize("source", ["1 / 2", "1/ 2", "3*4", "3 − 4", "3 < 4"])
def test_unsupported_math_expression_is_preserved_atomically(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()
    assert len(result.diagnostics) == 1


def test_pathological_numeric_separator_run_is_linear_and_atomic(
    normalizer: KoreanTTSNormalizer,
) -> None:
    source = ("9." * 5_000)[:10_000]

    started = time.perf_counter()
    result = normalizer.convert(source)
    elapsed = time.perf_counter() - started

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()
    assert elapsed < 1.0


def test_output_growth_is_bounded() -> None:
    normalizer = KoreanTTSNormalizer(
        lexicon={"X": "가" * 100},
        max_output_length=500,
    )

    with pytest.raises(InputValidationError, match="출력"):
        normalizer("X " * 10)


@pytest.mark.parametrize("value", ["😀", "Minnie", "①", "ㅏ"])
def test_lexicon_rejects_output_that_is_not_tts_complete(value: str) -> None:
    with pytest.raises(InputValidationError, match="완결된 발화형"):
        KoreanTTSNormalizer(lexicon={"X": value})


def test_shared_normalizer_is_deterministic_across_threads(
    normalizer: KoreanTTSNormalizer,
) -> None:
    sources = ["3.3V", "오후 3:05", "2026. 8. 7.", "1∼3개"] * 100
    expected = [normalizer(source) for source in sources]

    with ThreadPoolExecutor(max_workers=16) as executor:
        actual = list(executor.map(normalizer, sources))

    assert actual == expected
