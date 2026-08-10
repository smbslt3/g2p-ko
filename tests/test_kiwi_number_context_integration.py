# -*- coding: utf-8 -*-
"""실제 Kiwi와 배포 모델로 문맥 기반 숫자 읽기가 동작하는지 확인한다."""

from __future__ import annotations

import unicodedata

import pytest
from kiwipiepy import Kiwi

import g2p_ko.analyzer as analyzer_module
from g2p_ko import KoreanTTSNormalizer


@pytest.fixture(scope="module")
def kiwi() -> Kiwi:
    """모델 적재 비용이 커서 모듈 단위로 재사용한다."""

    return Kiwi()


@pytest.fixture(autouse=True)
def use_real_kiwi(monkeypatch: pytest.MonkeyPatch, kiwi: Kiwi) -> None:
    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", kiwi)


@pytest.fixture(scope="module")
def contextual() -> KoreanTTSNormalizer:
    return KoreanTTSNormalizer()


def test_identifier_number_is_read_digit_by_digit(
    contextual: KoreanTTSNormalizer,
) -> None:
    """번호 문맥의 숫자는 수량이 아니라 낱자리로 읽는다."""

    assert contextual("우편번호는 54130입니다.") == "우편번호는 오 사 일 삼 공입니다."


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("사과 3개를 샀다.", "사과 세 개를 샀다."),
        ("영화를 3번 봤다.", "영화를 세 번 봤다."),
        ("책 3권을 빌렸다.", "책 세 권을 빌렸다."),
        ("시험에서 95점 받았어.", "시험에서 구십오 점 받았어."),
    ],
)
def test_confident_rule_readings_are_kept(
    contextual: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    """규칙이 이미 맞는 자리는 판별기가 건드리지 않는다."""

    assert contextual(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("2024년 3월 5일에 만나요.", "이천이십사 년 삼 월 오 일에 만나요."),
        ("가격은 5만 원입니다.", "가격은 오만 원입니다."),
        ("전화는 010-1234-5678입니다.", "전화는 공 일 공 일 이 삼 사 오 육 칠 팔입니다."),
        ("정확도는 12.5%입니다.", "정확도는 십이 점 오 퍼센트입니다."),
        ("안녕하세요.", "안녕하세요."),
    ],
)
def test_structured_and_plain_text_keep_expected_output(
    contextual: KoreanTTSNormalizer,
    source: str,
    expected: str,
) -> None:
    """구조 규칙이 담당하는 입력은 판별기가 바꾸지 않는다."""

    assert contextual(source) == expected


def test_decomposed_input_uses_normalized_offsets(
    contextual: KoreanTTSNormalizer,
) -> None:
    """NFC 정규화 뒤 좌표로 분석해야 자모 분해 입력도 어긋나지 않는다."""

    composed = "우편번호는 54130입니다."
    decomposed = unicodedata.normalize("NFD", composed)

    assert decomposed != composed
    assert contextual(decomposed) == contextual(composed)


def test_combined_unit_and_following_word_feature_picks_number_reading(
    contextual: KoreanTTSNormalizer,
) -> None:
    """단위·뒤 낱말 결합 자질(`un=`)의 유일한 실질 성과를 고정한다.

    `3번`처럼 단위(`번`)가 같아도 뒤 낱말이 `창구`면 횟수가 아니라 번호로
    읽어야 한다. 결합 자질을 추가하기 전에는 규칙 기본값인 횟수(`세 번`)로
    남아 이 사례가 대표적인 실패였다(계획 문서 Task 4 참고).
    """

    assert contextual("3번 창구로 가시면 됩니다.") == "삼 번 창구로 가시면 됩니다."


def test_expanded_particles_allow_bare_number_context_reading(
    contextual: KoreanTTSNormalizer,
) -> None:
    """숫자 뒤 조사 확장(`이야` 등)이 없으면 통째로 보존되던 자리를 고정한다.

    조사 목록에 `이야`가 없던 시절에는 `계좌번호는 3456이야`가 낱자리 판별
    이전에 원문 그대로 보존됐다(Task 3의 동기 사례). 조사 확장 이후에는
    번호 문맥으로 인식돼 낱자리로 읽힌다.
    """

    assert contextual("계좌번호는 3456이야.") == "계좌번호는 삼 사 오 육이야."


def test_analysis_runs_once_per_conversion(
    monkeypatch: pytest.MonkeyPatch,
    kiwi: Kiwi,
) -> None:
    """같은 문장을 여러 번 분석하지 않는다."""

    class CountingKiwi:
        def __init__(self, inner: Kiwi) -> None:
            self._inner = inner
            self.calls = 0

        def split(self, text: str):
            self.calls += 1
            return self._inner.split(text)

    counting = CountingKiwi(kiwi)
    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", counting)
    normalizer = KoreanTTSNormalizer()
    normalizer("3개와 5개와 7개를 세었다.")

    assert counting.calls == 1
