"""고정된 노멀라이저와 G2P 조합의 순서를 검증한다."""

from __future__ import annotations

import pytest

from g2p_ko import G2P, Handler, KoreanTTSNormalizer
from tests._internal_pipeline import convert_pipeline


def test_internal_pipeline_keeps_intermediate_and_pronunciation_outputs() -> None:
    result = convert_pipeline("국물")

    assert result.normalized_text == "국물"
    assert result.surface_pronunciation == "궁물"
    assert any(item.stage.value == "phonology" for item in result.rewrites)


@pytest.mark.parametrize(
    ("source", "normalized", "pronunciation"),
    [
        (
            "010-1234-5678",
            "공 일 공 일 이 삼 사 오 육 칠 팔",
            "공 일 공 일 이 삼 사 오 육 칠 팔",
        ),
        ("16%", "십육 퍼센트", "심뉵 퍼센트"),
        ("26개", "스물여섯 개", "스물려섣 깨"),
    ],
)
def test_g2p_applies_tts_normalizer_before_pronunciation(
    source: str,
    normalized: str,
    pronunciation: str,
) -> None:
    assert KoreanTTSNormalizer()(source) == normalized
    assert G2P()(source) == pronunciation


def test_english_letter_names_are_always_enabled() -> None:
    result = convert_pipeline("API OpenAI hello ray's")

    assert result.normalized_text == "에이피아이 오피이엔에이아이 에이치이엘엘오 알에이와이에스"
    assert [
        item.after
        for item in result.rewrites
        if item.rule_id == "english.letter_names.v1"
    ] == [
        "에이피아이",
        "오피이엔에이아이",
        "에이치이엘엘오",
        "알에이와이에스",
    ]
    assert result.complete


def test_converted_english_participates_in_korean_phonology() -> None:
    result = convert_pipeline("L는")

    assert result.normalized_text == "엘는"
    assert result.surface_pronunciation == "엘른"
    assert result.normalized_segments[0].handler is Handler.ENGLISH


def test_nfd_source_map_points_phonology_rewrite_to_raw_span() -> None:
    source = "\u1100\u1161꽃"
    result = convert_pipeline(source)
    rewrite = next(item for item in result.rewrites if item.rule_id == "ko.coda.9")

    assert result.normalized_text == "가꽃"
    assert result.surface_pronunciation == "가꼳"
    assert rewrite.source_spans == (result.normalized_segments[0].source_spans[1],)
    assert (rewrite.source_spans[0].start, rewrite.source_spans[0].end) == (2, 3)
