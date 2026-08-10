# -*- coding: utf-8 -*-
"""내장 Kiwi 숫자 판별기의 교정·기권·실패 폴백을 검증한다.

테스트에서는 외부 공개 옵션 대신 내부 분석기 조회만 가짜 구현으로 바꾼다.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

import g2p_ko.analyzer as analyzer_module
from g2p_ko import KoreanTTSNormalizer
from g2p_ko.errors import BackendUnavailableError
from g2p_ko.normalizer.number_context import (
    NumberReadingClassifier,
    NumberReadingModel,
)


def _model(weights: dict[str, list[float]], thresholds: dict[str, float]) -> NumberReadingModel:
    return NumberReadingModel(
        classes=("native", "sino", "digitwise"),
        bias=(0.0, 0.0, 0.0),
        weights={key: tuple(value) for key, value in weights.items()},
        thresholds={
            name: {"unit": value, "bare": value} for name, value in thresholds.items()
        },
    )


class _FakeKiwi:
    """미리 정한 분해만 돌려주는 Kiwi 대역이다."""

    def __init__(self, table: dict[str, list[tuple[str, str, int, int]]]) -> None:
        self._table = table
        self.calls: list[str] = []

    def split(self, text: str):
        self.calls.append(text)
        if text not in self._table:
            raise AssertionError(f"준비되지 않은 입력입니다: {text!r}")
        return [
            SimpleNamespace(form=form, tag=tag, start=start, len=length)
            for form, tag, start, length in self._table[text]
        ]


class _BrokenAnalyzer:
    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    def analyze(self, text: str):
        self.calls += 1
        raise self._error


def _install_model(normalizer: KoreanTTSNormalizer, model: NumberReadingModel) -> None:
    """생성자를 거치지 않고 시험용 계수를 주입한다."""

    normalizer._number_reader._model = model  # noqa: SLF001


@pytest.fixture(autouse=True)
def reset_shared_kiwi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", None)


def _with_kiwi(kiwi: _FakeKiwi, **options) -> KoreanTTSNormalizer:
    analyzer_module._runtime_kiwi = kiwi
    return KoreanTTSNormalizer(**options)


def _with_analyzer(analyzer: object) -> KoreanTTSNormalizer:
    normalizer = KoreanTTSNormalizer()
    normalizer._number_reader = NumberReadingClassifier(analyzer)  # type: ignore[arg-type]  # noqa: SLF001
    return normalizer


_ROOM_TABLE = {
    "3번 문제를 풀어라": [
        ("3", "SN", 0, 1),
        ("번", "NNB", 1, 1),
        ("문제", "NNG", 3, 2),
        ("를", "JKO", 5, 1),
        ("풀", "VV", 7, 1),
        ("어라", "EF", 8, 2),
    ],
    "사과 3개를 샀다": [
        ("사과", "NNG", 0, 2),
        ("3", "SN", 3, 1),
        ("개", "NNB", 4, 1),
        ("를", "JKO", 5, 1),
        ("사", "VV", 7, 1),
        ("었", "EP", 7, 1),
        ("다", "EF", 8, 1),
    ],
}


def _normalizer(model: NumberReadingModel, table: dict) -> tuple[KoreanTTSNormalizer, _FakeKiwi]:
    kiwi = _FakeKiwi(table)
    normalizer = _with_kiwi(kiwi)
    _install_model(normalizer, model)
    return normalizer, kiwi


class TestOverride:
    """확신할 때만 규칙의 읽기를 바꾼다."""

    def test_confident_sino_replaces_native_counter(self) -> None:
        normalizer, _ = _normalizer(
            _model({"nf=문제/NNG": [-6.0, 6.0, -6.0]}, {"sino": 0.9}), _ROOM_TABLE
        )

        assert normalizer("3번 문제를 풀어라") == "삼 번 문제를 풀어라"

    def test_numbered_item_uses_deterministic_rule_id(self) -> None:
        normalizer, _ = _normalizer(
            _model({"nf=문제/NNG": [-6.0, 6.0, -6.0]}, {"sino": 0.9}), _ROOM_TABLE
        )
        result = normalizer.convert("3번 문제를 풀어라")

        assert [item.rule_id for item in result.rewrites] == [
            "normalizer.number_ordinal_sino.v1"
        ]

    def test_numbered_item_does_not_depend_on_model_confidence(self) -> None:
        normalizer, _ = _normalizer(
            _model({"nf=문제/NNG": [0.0, 0.4, 0.0]}, {"sino": 0.9}), _ROOM_TABLE
        )

        assert normalizer("3번 문제를 풀어라") == "삼 번 문제를 풀어라"

    def test_agreement_keeps_rule_rule_id(self) -> None:
        normalizer, _ = _normalizer(
            _model({"nf=문제/NNG": [6.0, -6.0, -6.0]}, {"native": 0.5}), _ROOM_TABLE
        )
        result = normalizer.convert("사과 3개를 샀다")

        assert result.normalized_text == "사과 세 개를 샀다"
        assert [item.rule_id for item in result.rewrites] == [
            "normalizer.number_native_counter.v1"
        ]


class TestDigitwise:
    """번호처럼 읽어야 하는 숫자는 낱자리로 고친다."""

    def test_bare_number_becomes_digitwise(self) -> None:
        table = {
            "내 번호는 5566": [
                ("내", "NP", 0, 1),
                ("번호", "NNG", 2, 2),
                ("는", "JX", 4, 1),
                ("5566", "SN", 6, 4),
            ]
        }
        normalizer, _ = _normalizer(
            _model({"pf=는/JX": [-6.0, -6.0, 6.0]}, {"digitwise": 0.9}), table
        )

        assert normalizer("내 번호는 5566") == "내 번호는 오 오 육 육"

    def test_bare_number_never_becomes_native(self) -> None:
        table = {
            "내 번호는 5566": [
                ("내", "NP", 0, 1),
                ("번호", "NNG", 2, 2),
                ("는", "JX", 4, 1),
                ("5566", "SN", 6, 4),
            ]
        }
        normalizer, _ = _normalizer(
            _model({"pf=는/JX": [6.0, -6.0, -6.0]}, {"native": 0.5}), table
        )

        assert normalizer("내 번호는 5566") == "내 번호는 오천오백육십육"


class TestStructuredRulesKeepAuthority:
    """날짜·시간·통화 같은 구조 규칙은 판별기에 묻지 않는다."""

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("2024년 3월 5일", "이천이십사 년 삼 월 오 일"),
            ("3시 30분", "세 시 삼십 분"),
            ("5만 원", "오만 원"),
            ("제3장", "제 삼 장"),
            ("1번째", "첫 번째"),
        ],
    )
    def test_structured_tokens_are_untouched(self, source: str, expected: str) -> None:
        kiwi = _FakeKiwi({})
        normalizer = _with_kiwi(kiwi)
        _install_model(normalizer, _model({}, {"sino": 0.0, "native": 0.0}))

        assert normalizer(source) == expected
        assert kiwi.calls == []


class TestLaziness:
    """숫자 판별이 필요할 때만 문장을 분석한다."""

    def test_analysis_is_skipped_without_candidates(self) -> None:
        kiwi = _FakeKiwi({})
        normalizer = _with_kiwi(kiwi)

        assert normalizer("안녕하세요") == "안녕하세요"
        assert kiwi.calls == []

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("100km/h", "시속 백 킬로미터"),
            ("90℃", "구십 도씨"),
            ("3㎏", "삼 킬로그램"),
        ],
    )
    def test_analysis_is_skipped_for_unambiguous_unit_symbols(
        self, source: str, expected: str
    ) -> None:
        kiwi = _FakeKiwi({})
        normalizer = _with_kiwi(kiwi)

        assert normalizer(source) == expected
        assert kiwi.calls == []

    def test_analysis_runs_once_per_conversion(self) -> None:
        table = {
            "3개와 5개": [
                ("3", "SN", 0, 1),
                ("개", "NNB", 1, 1),
                ("와", "JC", 2, 1),
                ("5", "SN", 4, 1),
                ("개", "NNB", 5, 1),
            ]
        }
        normalizer, kiwi = _normalizer(_model({}, {}), table)
        normalizer("3개와 5개")

        assert kiwi.calls == ["3개와 5개"]

    def test_lexicon_validation_does_not_analyze(self) -> None:
        kiwi = _FakeKiwi({})
        _with_kiwi(kiwi, lexicon={"TTS": "티티에스"})

        assert kiwi.calls == []


class TestFailureFallback:
    """분석기가 실패해도 규칙 결과를 그대로 내고 진단만 남긴다."""

    def test_backend_unavailable_falls_back_with_diagnostic(self) -> None:
        analyzer = _BrokenAnalyzer(BackendUnavailableError("kiwipiepy가 없습니다."))
        normalizer = _with_analyzer(analyzer)
        result = normalizer.convert("사과 3개를 샀다")

        assert result.normalized_text == "사과 세 개를 샀다"
        assert result.complete
        assert [item.code for item in result.diagnostics] == ["analyzer_unavailable"]

    def test_unexpected_error_is_analyzer_failed(self) -> None:
        analyzer = _BrokenAnalyzer(RuntimeError("깨짐"))
        normalizer = _with_analyzer(analyzer)
        result = normalizer.convert("사과 3개를 샀다")

        assert result.normalized_text == "사과 세 개를 샀다"
        assert [item.code for item in result.diagnostics] == ["analyzer_failed"]

    def test_diagnostic_is_emitted_once_per_conversion(self) -> None:
        analyzer = _BrokenAnalyzer(RuntimeError("깨짐"))
        normalizer = _with_analyzer(analyzer)
        result = normalizer.convert("3개와 5개와 7개")

        assert [item.code for item in result.diagnostics] == ["analyzer_failed"]
        assert analyzer.calls == 1

    def test_fast_path_does_not_raise_on_failure(self) -> None:
        analyzer = _BrokenAnalyzer(RuntimeError("깨짐"))
        normalizer = _with_analyzer(analyzer)

        assert normalizer("사과 3개를 샀다") == "사과 세 개를 샀다"


class TestConstructor:
    """고정 Kiwi 분석기와 숫자 판별 모델이 기본 구성에 포함된다."""

    def test_default_normalizer_has_reader(self) -> None:
        assert KoreanTTSNormalizer()._number_reader is not None  # noqa: SLF001


class TestConcurrency:
    """같은 인스턴스를 여러 스레드가 써도 결과가 흔들리지 않는다."""

    def test_parallel_conversions_are_deterministic(self) -> None:
        normalizer, _ = _normalizer(
            _model({"nf=문제/NNG": [-6.0, 6.0, -6.0]}, {"sino": 0.9}), _ROOM_TABLE
        )
        with ThreadPoolExecutor(max_workers=16) as pool:
            results = list(pool.map(normalizer, ["3번 문제를 풀어라"] * 64))

        assert set(results) == {"삼 번 문제를 풀어라"}
