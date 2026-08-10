"""숫자 결합 단위와 독립 단위 문맥의 보수적 변환을 검증한다."""

from __future__ import annotations

import pytest

from g2p_ko.normalizer import KoreanTTSNormalizer


@pytest.fixture
def normalizer() -> KoreanTTSNormalizer:
    return KoreanTTSNormalizer()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2.5 kg", "이 점 오 킬로그램"),
        ("3,000억 km", "삼천억 킬로미터"),
        ("200여 m", "이백여 미터"),
        ("23만km를", "이십삼만 킬로미터를"),
    ],
)
def test_quantity_unit_accepts_spacing_scale_and_approximation(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2/5kg", "오분의 이 킬로그램"),
        ("3/2 kg", "이분의 삼 킬로그램"),
        ("1/8g", "팔분의 일 그램"),
    ],
)
def test_fraction_unit_accepts_proper_and_improper_fractions(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


def test_slash_before_mmhg_is_preserved_as_a_ratio(
    normalizer: KoreanTTSNormalizer,
) -> None:
    source = "120/80mmHg"
    result = normalizer.convert(source)

    assert result.normalized_text == source
    assert not result.complete
    assert result.rewrites == ()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("몇 mg 넣어야 해?", "몇 밀리그램 넣어야 해?"),
        ("수 kg 내외가 소모된다", "수 킬로그램 내외가 소모된다"),
        ("몇 L이면 돼?", "몇 리터이면 돼?"),
        ("kg당은", "킬로그램당은"),
        ("t당", "톤당"),
    ],
)
def test_non_numeric_units_require_a_strong_quantity_context(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("㎏은", "킬로그램은"),
        ("ℓ당", "리터당"),
        ("℃", "도씨"),
        ("㎢", "제곱킬로미터"),
    ],
)
def test_unambiguous_unicode_unit_symbols_can_stand_alone(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2ha", "이 헥타르"),
        ("3ℓ", "삼 리터"),
        ("4㎢", "사 제곱킬로미터"),
        ("5㎾h", "오 킬로와트시"),
        ("6㎿", "육 메가와트"),
        ("0.4GWh", "영 점 사 기가와트시"),
        ("7kJ", "칠 킬로줄"),
        ("8cal", "팔 칼로리"),
        ("120mmHg", "백이십 수은주밀리미터"),
        ("9mile", "구 마일"),
        ("1.6W/kg", "일 점 육 와트 퍼 킬로그램"),
        ("9.7km/l", "구 점 칠 킬로미터 퍼 리터"),
        ("1TEU", "일 티이유"),
        ("1FEU", "일 에프이유"),
        ("3ppm", "삼 피피엠"),
        ("2400dpi", "이천사백 디피아이"),
    ],
)
def test_observed_standard_units_have_deterministic_readings(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("112km/h", "시속 백십이 킬로미터"),
        ("6m/s", "초속 육 미터"),
        ("3km/s", "초속 삼 킬로미터"),
        ("290㎞/h", "시속 이백구십 킬로미터"),
        ("0~100km/h", "시속 영에서 백 킬로미터"),
        ("1.6~3.3m/s", "초속 일 점 육에서 삼 점 삼 미터"),
    ],
)
def test_speed_units_move_the_rate_word_before_the_quantity(
    normalizer: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    assert normalizer(source) == expected
@pytest.mark.parametrize(
    "source",
    [
        "kg 단위",
        "981P 적립",
        "22P 그릇세트",
        "241.1p 하락",
        "4천167㎢",
        "1Km",
        "3kg/s",
        "3kg²",
        "10±2kg",
        "175/102A",
    ],
)
def test_ambiguous_or_out_of_scope_unit_like_text_is_preserved(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    assert normalizer(source) == source


def test_quantity_unit_trace_uses_one_source_span(
    normalizer: KoreanTTSNormalizer,
) -> None:
    source = "값은 2/5kg이다."
    result = normalizer.convert(source)
    rewrite = next(item for item in result.rewrites if item.rule_id == "normalizer.fraction_unit.v1")

    assert rewrite.before == "2/5kg"
    assert rewrite.after == "오분의 이 킬로그램"
    assert len(rewrite.source_spans) == 1
    assert source[rewrite.source_spans[0].start : rewrite.source_spans[0].end] == "2/5kg"


@pytest.mark.parametrize("source", ["2/5kg", "몇 mg", "0~100km/h", "㎏은"])
def test_unit_normalization_is_idempotent(
    normalizer: KoreanTTSNormalizer,
    source: str,
) -> None:
    once = normalizer(source)
    assert normalizer(once) == once
