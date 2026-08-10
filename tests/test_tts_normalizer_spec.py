"""세 공개 자료를 종합한 독립 한국어 TTS 노멀라이저의 동작 명세다.

CoreaSpeech N2gk+의 한국어 수사 규칙, Raon 전처리의 통화·범위 보완,
KRAFTON TTS 규칙집의 음성 친화적 보존 원칙을 회귀 테스트로 고정한다.
"""

from __future__ import annotations

import pytest

from g2p_ko import InputValidationError
from g2p_ko.normalizer import KoreanTTSNormalizer


@pytest.fixture
def normalizer() -> KoreanTTSNormalizer:
    return KoreanTTSNormalizer()


def test_callable_is_text_only_convenience_api(
    normalizer: KoreanTTSNormalizer,
) -> None:
    source = "총 5개입니다."
    result = normalizer.convert(source)

    assert normalizer(source) == "총 다섯 개입니다."
    assert result.source == source
    assert result.normalized_text == "총 다섯 개입니다."
    assert result.complete


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("값은 0입니다.", "값은 영입니다."),
        ("값은 101입니다.", "값은 백일입니다."),
        ("참가자는 12,345명입니다.", "참가자는 만이천삼백사십오 명입니다."),
        ("정확도는 12.5%입니다.", "정확도는 십이 점 오 퍼센트입니다."),
        ("측정값은 3.10입니다.", "측정값은 삼 점 일 영입니다."),
        ("오차는 0.05%입니다.", "오차는 영 점 영 오 퍼센트입니다."),
    ],
)
def test_sino_numbers_and_decimals_do_not_lose_digits(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1개", "한 개"),
        ("2명", "두 명"),
        ("3마리", "세 마리"),
        ("4권", "네 권"),
        ("10개", "열 개"),
        ("20살", "스무 살"),
        ("21개", "스물한 개"),
        ("100개", "백 개"),
    ],
)
def test_counter_selects_native_number_and_falls_back_after_ninety_nine(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("가격은 $5입니다.", "가격은 오 달러입니다."),
        ("가격은 ₩1,200입니다.", "가격은 천이백 원입니다."),
        ("가격은 €12입니다.", "가격은 십이 유로입니다."),
        ("가격은 £9입니다.", "가격은 구 파운드입니다."),
        ("가격은 ¥300입니다.", "가격은 삼백 엔입니다."),
    ],
)
def test_prefix_currency_is_spoken_after_amount(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("3kg", "삼 킬로그램"),
        ("250mL", "이백오십 밀리리터"),
        ("12cm", "십이 센티미터"),
        ("3mm", "삼 밀리미터"),
        ("30km", "삼십 킬로미터"),
    ],
)
def test_longest_unit_match_uses_standard_korean_spelling(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1~3개", "하나에서 세 개"),
        ("2–4명", "둘에서 네 명"),
        ("1—3권", "하나에서 세 권"),
        ("40~100m", "사십에서 백 미터"),
        ("0.3~0.5도", "영 점 삼에서 영 점 오 도"),
        ("15~64살", "열다섯에서 예순네 살"),
        ("0~2살", "영에서 두 살"),
        ("20~30개", "스물에서 서른 개"),
        ("12~18세의", "십이에서 십팔 세의"),
        ("1~10월에", "일에서 시월에"),
    ],
)
def test_range_reads_unit_once_and_left_endpoint_standalone(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    """원문에 단위가 오른쪽에만 있으면 단위를 한 번만 읽는다."""

    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("4m~5m", "사 미터에서 오 미터"),
        ("5%~10%", "오 퍼센트에서 십 퍼센트"),
        ("1시간~2시간", "한 시간에서 두 시간"),
        ("5만원~10만원", "오만 원에서 십만 원"),
        ("1kg – 3kg", "일 킬로그램에서 삼 킬로그램"),
    ],
)
def test_range_with_unit_on_both_ends_reads_each_side(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    """원문이 양쪽에 단위를 쓰면 쓰인 대로 각각 읽는다."""

    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("4억~5억 원", "사억에서 오억 원"),
        ("4억~5억 원을", "사억에서 오억 원을"),
        ("3만~5만 원", "삼만에서 오만 원"),
    ],
)
def test_scaled_currency_range_reads_won_once(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    """축약 금액 범위는 양쪽 배율을 읽고 원은 한 번만 읽는다."""

    assert normalizer(source) == expected


def test_phone_is_recognized_before_hyphen_or_number_rules(
    normalizer: KoreanTTSNormalizer,
) -> None:
    assert normalizer("전화는 010-1234-5678입니다.") == (
        "전화는 공 일 공 일 이 삼 사 오 육 칠 팔입니다."
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("6월", "유월"),
        ("10월", "시월"),
        ("3.1절", "삼일절"),
        ("3.1", "삼 점 일"),
        ("2/5", "오분의 이"),
        ("-5", "마이너스 오"),
    ],
)
def test_context_specific_date_fraction_and_sign_rules_run_before_decimal(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("6월에", "유월에"),
        ("6월은", "유월은"),
        ("10월부터", "시월부터"),
        ("10월의", "시월의"),
        ("06월에", "유월에"),
        ("3월에", "삼 월에"),
    ],
)
def test_month_exception_survives_attached_particles(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    """조사가 붙어 다른 토큰 경로로 가도 유월·시월 예외를 유지한다."""

    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("AI와 GPU", "에이아이와 지피유"),
        ("ROI", "알오아이"),
        ("NASA와 FIFA", "나사와 피파"),
        ("KIA와 TV", "기아와 티비"),
        ("CD BC CPU", "씨디 비씨 씨피유"),
        ("CIA CNN BBC", "씨아이에이 씨엔엔 비비씨"),
        ("CCTV", "씨씨티비"),
    ],
)
def test_initialism_pronunciation_is_distinguished_from_lexical_acronym(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("AIDS", "에이즈"),
        ("APEC", "에이펙"),
        ("ASCII", "아스키"),
        ("ASEAN", "아세안"),
        ("BIOS", "바이오스"),
        ("CESCO", "세스코"),
        ("COEX", "코엑스"),
        ("DGIST", "디지스트"),
        ("FEDEX", "페덱스"),
        ("FILA", "휠라"),
        ("GIST", "지스트"),
        ("IKEA", "이케아"),
        ("KAIST", "카이스트"),
        ("KATUSA", "카투사"),
        ("KIST", "키스트"),
        ("KOSDAQ", "코스닥"),
        ("KOSPI", "코스피"),
        ("LAN", "랜"),
        ("LASEK", "라섹"),
        ("LASER", "레이저"),
        ("LASIK", "라식"),
        ("MERS", "메르스"),
        ("NASDAQ", "나스닥"),
        ("NATO", "나토"),
        ("NIMBY", "님비"),
        ("OLED", "올레드"),
        ("PASS", "패스"),
        ("PAYCO", "페이코"),
        ("PIN", "핀"),
        ("POSTECH", "포스텍"),
        ("PUBG", "펍지"),
        ("RADAR", "레이더"),
        ("RAM", "램"),
        ("ROM", "롬"),
        ("SARS", "사스"),
        ("SCUBA", "스쿠버"),
        ("SSGPAY", "쓱페이"),
        ("TOEFL", "토플"),
        ("TOEIC", "토익"),
        ("TVN", "티비엔"),
        ("UNESCO", "유네스코"),
        ("UNICEF", "유니세프"),
        ("UNIST", "유니스트"),
        ("USIM", "유심"),
    ],
)
def test_conventional_english_reading_is_applied_before_g2p(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


def test_ambiguous_basic_is_not_fixed_to_one_korean_reading(
    normalizer: KoreanTTSNormalizer,
) -> None:
    assert normalizer("BASIC") == "BASIC"


@pytest.mark.parametrize(
    "source",
    [
        "SUSBtype",
        "fooUSBbar",
        "USBtype",
        "typeUSB",
        "foo_USB_bar",
        "foo.USB.bar",
        "NASA_test",
        "foo.NASA.bar",
    ],
)
def test_known_english_token_inside_identifier_is_preserved(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    assert normalizer(source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("USB.", "유에스비."),
        ("NASA.", "나사."),
        ("USB의", "유에스비의"),
    ],
)
def test_known_english_token_keeps_sentence_and_korean_boundaries(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "calmly",
        "10calmly",
        "10mlly",
        "foo_10ml_bar",
        "10ml_test",
        "10ml.test",
        "3kg~5kg_test",
        "3kg~5kg.test",
    ],
)
def test_unit_like_text_inside_identifier_is_preserved(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    assert normalizer(source) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("10ml.", "십 밀리리터."),
        ("10ml의", "십 밀리리터의"),
        ("몇 ml", "몇 밀리리터"),
        ("3kg~5kg.", "삼 킬로그램에서 오 킬로그램."),
    ],
)
def test_units_keep_sentence_and_korean_boundaries(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("ㄱ", "기역"),
        ("ㄴ", "니은"),
        ("ㄷ", "디귿"),
        ("ㄹ", "리을"),
        ("ㅁ", "미음"),
        ("ㅂ", "비읍"),
        ("ㅅ", "시옷"),
        ("ㅇ", "이응"),
        ("ㅈ", "지읒"),
        ("ㅊ", "치읓"),
        ("ㅋ", "키읔"),
        ("ㅌ", "티읕"),
        ("ㅍ", "피읖"),
        ("ㅎ", "히읗"),
    ],
)
def test_standalone_consonant_jamo_uses_canonical_name(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize("source", ["ㅋㅋ", "ㅎㅎㅎ", "ㅇㅋ", "ㄱㅅ", "ㅋㅋ2", "ㅋㅋ2개"])
def test_consecutive_jamo_is_preserved_as_chat_notation(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    """연속 자모는 웃음·초성어일 수 있어 낱자 이름으로 읽지 않는다."""

    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete


def test_tts_punctuation_parentheses_newlines_and_repeated_spaces_are_preserved(
    normalizer: KoreanTTSNormalizer,
) -> None:
    source = "그는 3개를  샀다... (정말로 2개?)\n다음 줄!"
    expected = "그는 세 개를  샀다... (정말로 두 개?)\n다음 줄!"

    assert normalizer(source) == expected


@pytest.mark.parametrize(
    "source",
    [
        "https://example.com/v1.2?q=3에서 확인하세요.",
        "user@example.com으로 보내세요.",
        r"경로 C:\work\v1.2\model3을 확인하세요.",
        "버전 `v1.2.3`을 사용합니다.",
        "코드 ``5개``를 확인합니다.",
        "https://example.com/a(5)에서 확인합니다.",
        r"C:\Program Files\App 2\run.exe를 실행합니다.",
    ],
)
def test_protected_structures_are_not_partially_normalized(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    assert normalizer(source) == source


@pytest.mark.parametrize(
    ("source", "diagnostic_code"),
    [
        ("고유명사 Minnie를 부른다.", "unconverted_latin"),
        ("주문번호 AB12를 확인한다.", "ambiguous_identifier"),
        ("식별번호 12345678901을 확인한다.", "ambiguous_numeric"),
        ("漢字를 확인한다.", "unconverted_han"),
        ("범위 1-3개를 확인한다.", "ambiguous_hyphen"),
    ],
)
def test_ambiguous_text_is_preserved_with_diagnostic(
    normalizer: KoreanTTSNormalizer,
    source: str,
    diagnostic_code: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert diagnostic_code in {item.code for item in result.diagnostics}
    assert result.complete is False


@pytest.mark.parametrize(
    "source",
    [
        "1.2.3",
        "1..2",
        "1,,2원",
    ],
)
def test_malformed_numeric_cluster_is_preserved_and_diagnosed_as_one_token(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert result.complete is False
    assert any(
        span.start == 0 and span.end == len(source) and span.surface == source
        for item in result.diagnostics
        for span in item.source_spans
    )


@pytest.mark.parametrize(
    "source",
    [
        "2026-08/07",
        "2026-08-07.1",
        "010-1234-5678-9",
        "999.999.999.999",
        "1.2.3.4.5",
    ],
)
def test_malformed_structured_number_does_not_allow_prefix_conversion(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert result.complete is False
    assert any(
        span.start == 0 and span.end == len(source) and span.surface == source
        for item in result.diagnostics
        for span in item.source_spans
    )


def test_valid_calendar_date_and_ipv4_still_take_their_full_structure(
    normalizer: KoreanTTSNormalizer,
) -> None:
    date_result = normalizer.convert("2026-08-07")
    ipv4_result = normalizer.convert("192.168.0.1")

    assert date_result.normalized_text == "이천이십육 년 팔 월 칠 일"
    assert date_result.complete
    assert ipv4_result.normalized_text == "192.168.0.1"
    assert ipv4_result.complete
    assert ipv4_result.diagnostics == ()


@pytest.mark.parametrize(
    "source",
        [
            "01/2",
            "1/02",
            "1/2개",
        ],
)
def test_ambiguous_fraction_is_preserved_without_partial_conversion(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert result.complete is False
    assert any(
        span.start == 0 and span.end == len(source) and span.surface == source
        for item in result.diagnostics
        for span in item.source_spans
    )


def test_normalization_does_not_apply_g2p_phonology(
    normalizer: KoreanTTSNormalizer,
) -> None:
    assert normalizer("국물 2개") == "국물 두 개"


def test_rewrite_keeps_rule_id_and_original_source_coordinates(
    normalizer: KoreanTTSNormalizer,
) -> None:
    source = "총 5개"
    result = normalizer.convert(source)

    assert result.rewrites
    assert all(item.rule_id for item in result.rewrites)
    assert any(
        span.start == 2 and span.end >= 3 and span.surface.startswith("5")
        for item in result.rewrites
        for span in item.source_spans
    )


@pytest.mark.parametrize(
    "source",
    [
        "가격은 ₩1,200이고 3개 남았습니다.",
        "전화는 010-1234-5678입니다.",
        "3.1절과 2/5를 읽습니다.",
        "AI 모델의 정확도는 12.50%입니다.",
    ],
)
def test_normalization_is_idempotent(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    once = normalizer(source)

    assert normalizer(once) == once


def test_normalize_many_preserves_order_and_matches_single_call(
    normalizer: KoreanTTSNormalizer,
) -> None:
    sources = ["1개", "$5", "국물", "3.10%"]
    results = list(normalizer.normalize_many(iter(sources)))

    assert results == [normalizer(item) for item in sources]
    detailed = list(normalizer.convert_many(iter(sources)))
    assert [item.source for item in detailed] == sources
    assert [item.normalized_text for item in detailed] == results


def test_empty_text_is_valid_but_non_string_is_rejected(
    normalizer: KoreanTTSNormalizer,
) -> None:
    result = normalizer.convert("")

    assert result.normalized_text == ""
    assert result.rewrites == ()
    assert result.diagnostics == ()
    assert result.complete
    with pytest.raises(InputValidationError, match="문자열"):
        normalizer.convert(None)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, 1, True, [], {}])
def test_non_string_ambiguous_policy_uses_public_validation_error(value: object) -> None:
    with pytest.raises(InputValidationError, match="on_ambiguous"):
        KoreanTTSNormalizer(on_ambiguous=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("매일 5킬로미터를 달렸다.", "매일 오 킬로미터를 달렸다."),
        ("500그램", "오백 그램"),
        ("3.5리터", "삼 점 오 리터"),
        ("40~100미터", "사십에서 백 미터"),
        ("4미터~5미터", "사 미터에서 오 미터"),
    ],
)
def test_si_units_written_in_hangul_read_like_their_symbols(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


def test_hangul_unit_prefix_of_longer_word_is_preserved(
    normalizer: KoreanTTSNormalizer,
) -> None:
    source = "5마일리지 적립"
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("1조 원", "일조 원"),
        ("3조 원을 편성했다.", "삼조 원을 편성했다."),
        ("1.5조 원", "일조오천억 원"),
        ("2조~3조 원", "이조에서 삼조 원"),
    ],
)
def test_jo_scale_currency_reads_like_man_and_eok(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


def test_jo_without_won_stays_preserved(
    normalizer: KoreanTTSNormalizer,
) -> None:
    source = "학생들을 3조로 나눴다."
    result = normalizer.convert(source)

    assert "3조로" in result.normalized_text
    assert not result.complete


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("3억 원으로 시작했다.", "삼억 원으로 시작했다."),
        ("5만 원을 냈다.", "오만 원을 냈다."),
        ("1.5만 원이 남았다.", "만오천 원이 남았다."),
        ("1조 원은 큰돈이다.", "일조 원은 큰돈이다."),
    ],
)
def test_scaled_currency_accepts_particles_after_won(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("정답은 3이야.", "정답은 삼이야."),
        ("사과 3개랑 배 2개", "사과 세 개랑 배 두 개"),
        ("친구 2명한테 줬다.", "친구 두 명한테 줬다."),
        ("남은 건 5개뿐이다.", "남은 건 다섯 개뿐이다."),
        ("3명조차 오지 않았다.", "세 명조차 오지 않았다."),
        ("대표 1명으로서 참석했다.", "대표 한 명으로서 참석했다."),
    ],
)
def test_additional_particles_are_recognized(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    """표준 조사 목록의 빈칸 때문에 숫자가 통째로 보존되지 않아야 한다."""

    assert normalizer(source) == expected


def test_non_particle_tail_is_still_preserved(
    normalizer: KoreanTTSNormalizer,
) -> None:
    """조사가 아닌 꼬리는 여전히 보존한다."""

    source = "5마일리지 적립"
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
