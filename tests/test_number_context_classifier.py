# -*- coding: utf-8 -*-
"""문맥 기반 숫자 읽기 판별기의 순수 단위 동작을 검증한다.

Kiwi나 학습된 모델 없이, 손으로 만든 초소형 모델과 가짜 형태소열로
자질 추출·확률 계산·임계값·실패 처리를 확인한다.
"""

from __future__ import annotations

import pytest

from g2p_ko.model import AnalysisToken
from g2p_ko.errors import BackendUnavailableError
from g2p_ko.normalizer import number_context
from g2p_ko.normalizer.number_context import (
    NumberReadingClassifier,
    NumberReadingModel,
)
from g2p_ko.normalizer.numbers import read_digitwise


class TestReadDigitwise:
    """낱자리 읽기는 0을 공으로 읽고 낱자를 공백으로 잇는다."""

    @pytest.mark.parametrize(
        ("surface", "expected"),
        [
            ("6630", "육 육 삼 공"),
            ("55", "오 오"),
            ("2021", "이 공 이 일"),
            ("7", "칠"),
        ],
    )
    def test_digits_are_spaced_and_zero_is_gong(self, surface: str, expected: str) -> None:
        assert read_digitwise(surface) == expected


def _model(weights: dict[str, list[float]], thresholds: dict[str, float]) -> NumberReadingModel:
    """세 클래스 순서 (native, sino, digitwise)의 초소형 모델을 만든다.

    임계값은 판별 자리별로 갈리지만 시험에서는 두 자리에 같은 값을 쓴다.
    """

    return NumberReadingModel(
        classes=("native", "sino", "digitwise"),
        bias=(0.0, 0.0, 0.0),
        weights={key: tuple(value) for key, value in weights.items()},
        thresholds={
            name: {"unit": value, "bare": value} for name, value in thresholds.items()
        },
    )


class _FakeAnalyzer:
    """고정된 형태소열을 돌려주는 분석기다."""

    def __init__(self, tokens, *, error: Exception | None = None) -> None:
        self._tokens = tokens
        self._error = error
        self.calls: list[str] = []

    def analyze(self, text: str):
        self.calls.append(text)
        if self._error is not None:
            raise self._error
        return tuple(
            AnalysisToken(form, tag, start, end)
            for form, tag, start, end in self._tokens
        )


def _bus_analyzer() -> _FakeAnalyzer:
    """`버스 12번은` 문장의 형태소열이다."""

    return _FakeAnalyzer(
        [
            ("버스", "NNG", 0, 2),
            ("12", "SN", 3, 5),
            ("번", "NNB", 5, 6),
            ("은", "JX", 6, 7),
        ]
    )


class TestDecision:
    """규칙 기본값과 다르고 확신할 때만 교정한다."""

    def test_confident_disagreement_overrides(self) -> None:
        classifier = NumberReadingClassifier(
            _bus_analyzer(),
            model=_model({"nf=번/NNB": [-5.0, 5.0, -5.0]}, {"sino": 0.9}),
        )
        run = classifier.new_run("버스 12번은")

        assert run.decide(3, 5, unit="번", site="unit", default="native") == "sino"

    def test_agreement_returns_none(self) -> None:
        classifier = NumberReadingClassifier(
            _bus_analyzer(),
            model=_model({"nf=번/NNB": [5.0, -5.0, -5.0]}, {"native": 0.9}),
        )
        run = classifier.new_run("버스 12번은")

        assert run.decide(3, 5, unit="번", site="unit", default="native") is None

    def test_low_confidence_abstains(self) -> None:
        classifier = NumberReadingClassifier(
            _bus_analyzer(),
            model=_model({"nf=번/NNB": [0.0, 0.1, 0.0]}, {"sino": 0.9}),
        )
        run = classifier.new_run("버스 12번은")

        assert run.decide(3, 5, unit="번", site="unit", default="native") is None

    def test_bare_site_only_accepts_digitwise(self) -> None:
        classifier = NumberReadingClassifier(
            _FakeAnalyzer([("55", "SN", 0, 2), ("는", "JX", 2, 3)]),
            model=_model({"nf=는/JX": [5.0, -5.0, -5.0]}, {"native": 0.5}),
        )
        run = classifier.new_run("55는")

        assert run.decide(0, 2, unit="", site="bare", default="sino") is None

    def test_bare_site_digitwise_override(self) -> None:
        classifier = NumberReadingClassifier(
            _FakeAnalyzer([("55", "SN", 0, 2), ("는", "JX", 2, 3)]),
            model=_model({"nf=는/JX": [-5.0, -5.0, 5.0]}, {"digitwise": 0.9}),
        )
        run = classifier.new_run("55는")

        assert run.decide(0, 2, unit="", site="bare", default="sino") == "digitwise"


class TestAnalysisReuse:
    """한 번의 변환에서 형태소 분석은 한 번만 한다."""

    def test_analyze_called_once_per_run(self) -> None:
        analyzer = _bus_analyzer()
        classifier = NumberReadingClassifier(
            analyzer, model=_model({}, {"sino": 0.9})
        )
        run = classifier.new_run("버스 12번은")
        run.decide(3, 5, unit="번", site="unit", default="native")
        run.decide(3, 5, unit="번", site="unit", default="native")

        assert analyzer.calls == ["버스 12번은"]

    def test_no_analysis_without_decide(self) -> None:
        analyzer = _bus_analyzer()
        classifier = NumberReadingClassifier(analyzer, model=_model({}, {}))
        classifier.new_run("버스 12번은")

        assert analyzer.calls == []


class TestFailure:
    """분석기가 실패하면 판별을 포기하고 실패를 한 번만 알린다."""

    def test_backend_unavailable_is_reported_once(self) -> None:
        classifier = NumberReadingClassifier(
            _FakeAnalyzer([], error=BackendUnavailableError("kiwipiepy 없음")),
            model=_model({}, {}),
        )
        run = classifier.new_run("버스 12번은")

        assert run.decide(3, 5, unit="번", site="unit", default="native") is None
        failure = run.take_failure()
        assert failure is not None and failure[0] == "analyzer_unavailable"
        assert run.take_failure() is None

    def test_other_errors_are_analyzer_failed(self) -> None:
        classifier = NumberReadingClassifier(
            _FakeAnalyzer([], error=RuntimeError("깨짐")), model=_model({}, {})
        )
        run = classifier.new_run("버스 12번은")

        assert run.decide(3, 5, unit="번", site="unit", default="native") is None
        failure = run.take_failure()
        assert failure is not None and failure[0] == "analyzer_failed"

    def test_failure_stops_further_analysis(self) -> None:
        analyzer = _FakeAnalyzer([], error=RuntimeError("깨짐"))
        classifier = NumberReadingClassifier(analyzer, model=_model({}, {}))
        run = classifier.new_run("버스 12번은")
        run.decide(3, 5, unit="번", site="unit", default="native")
        run.decide(3, 5, unit="번", site="unit", default="native")

        assert len(analyzer.calls) == 1

    def test_token_offsets_outside_text_are_failure(self) -> None:
        classifier = NumberReadingClassifier(
            _FakeAnalyzer([("버스", "NNG", 0, 99)]), model=_model({}, {})
        )
        run = classifier.new_run("버스 12번은")

        assert run.decide(3, 5, unit="번", site="unit", default="native") is None
        failure = run.take_failure()
        assert failure is not None and failure[0] == "analyzer_failed"

    def test_token_form_mismatch_is_failure(self) -> None:
        classifier = NumberReadingClassifier(
            _FakeAnalyzer([("다른말", "NNG", 0, 3)]), model=_model({}, {})
        )
        run = classifier.new_run("버스 12번은")

        assert run.decide(3, 5, unit="번", site="unit", default="native") is None
        assert run.take_failure() is not None

    @pytest.mark.parametrize(
        "tokens",
        [
            [("버스", "NNG", 0, 2), ("스", "NNG", 1, 2)],
            [("버스", "", 0, 2)],
            [],
        ],
    )
    def test_incompatible_analysis_is_failure(self, tokens: list[tuple]) -> None:
        classifier = NumberReadingClassifier(
            _FakeAnalyzer(tokens), model=_model({}, {})
        )
        run = classifier.new_run("버스 12번은")

        assert run.decide(3, 5, unit="번", site="unit", default="native") is None
        failure = run.take_failure()
        assert failure is not None and failure[0] == "analyzer_failed"


class TestFeatures:
    """자질 키는 학습·추론이 같은 규칙을 쓰도록 고정한다."""

    def test_adjacent_morphemes_and_shape(self) -> None:
        analyzer = _FakeAnalyzer(
            [
                ("버스", "NNG", 0, 2),
                ("12", "SN", 3, 5),
                ("번", "NNB", 5, 6),
                ("은", "JX", 6, 7),
                ("빨강", "NNG", 8, 10),
            ]
        )
        keys = set(
            number_context.extract_features(
                analyzer.analyze("버스 12번은 빨강"),
                "버스 12번은 빨강",
                start=3,
                end=5,
                context_end=7,
                unit="번",
            )
        )

        assert "pf=버스/NNG" in keys
        assert "u=번" in keys
        assert "len=2" in keys
        # 단위와 조사는 꼬리 자질로, 다음 형태소는 단위 너머에서 찾는다.
        assert "sf=은/JX" in keys
        assert "nf=빨강/NNG" in keys

    def test_following_morpheme_is_looked_up_past_the_unit(self) -> None:
        """`3번 문제`와 `3번 봤다`를 가르는 신호는 단위 뒤에 있다."""

        analyzer = _FakeAnalyzer(
            [("3", "SN", 0, 1), ("번", "NNB", 1, 2), ("문제", "NNG", 3, 5)]
        )
        keys = set(
            number_context.extract_features(
                analyzer.analyze("3번 문제"),
                "3번 문제",
                start=0,
                end=1,
                context_end=2,
                unit="번",
            )
        )

        assert "nf=문제/NNG" in keys

    def test_small_values_expose_the_number_itself(self) -> None:
        """`20번`은 스무 번, `37번`은 삼십칠 번으로 기울어 값이 신호다."""

        analyzer = _FakeAnalyzer(
            [("20", "SN", 0, 2), ("번", "NNB", 2, 3), ("봤", "VV+EP", 4, 6)]
        )
        keys = set(
            number_context.extract_features(
                analyzer.analyze("20번 봤다"),
                "20번 봤다",
                start=0,
                end=2,
                context_end=3,
                unit="번",
            )
        )

        assert "v=20" in keys
        assert "uv=번:20" in keys

    def test_large_values_omit_the_value_feature(self) -> None:
        """세 자리 이상은 고유어 수사를 쓰지 않아 값이 신호가 되지 않는다."""

        analyzer = _FakeAnalyzer([("54130", "SN", 0, 5)])
        keys = set(
            number_context.extract_features(
                analyzer.analyze("54130"), "54130", start=0, end=5, unit=""
            )
        )

        assert not any(key.startswith(("v=", "uv=")) for key in keys)

    def test_content_word_skips_particles(self) -> None:
        """조사 너머의 명사가 판별 단서이므로 따로 자질로 남긴다."""

        analyzer = _FakeAnalyzer(
            [
                ("우편번호", "NNG", 0, 4),
                ("는", "JX", 4, 5),
                ("54130", "SN", 6, 11),
            ]
        )
        keys = set(
            number_context.extract_features(
                analyzer.analyze("우편번호는 54130"),
                "우편번호는 54130",
                start=6,
                end=11,
                unit="",
            )
        )

        assert "pf=는/JX" in keys
        assert "pc1=우편번호/NNG" in keys

    def test_second_content_word_is_kept(self) -> None:
        """가장 가까운 내용어에 막혀 진짜 단서를 놓치지 않는다."""

        analyzer = _FakeAnalyzer(
            [
                ("계좌번호", "NNG", 0, 4),
                ("뒷자리", "NNG", 5, 8),
                ("5566", "SN", 9, 13),
            ]
        )
        keys = set(
            number_context.extract_features(
                analyzer.analyze("계좌번호 뒷자리 5566"),
                "계좌번호 뒷자리 5566",
                start=9,
                end=13,
                unit="",
            )
        )

        assert "pc1=뒷자리/NNG" in keys
        assert "pc2=계좌번호/NNG" in keys

    def test_sentence_boundaries_are_marked(self) -> None:
        analyzer = _FakeAnalyzer([("55", "SN", 0, 2)])
        keys = set(
            number_context.extract_features(
                analyzer.analyze("55"), "55", start=0, end=2, unit=""
            )
        )

        assert "bos=1" in keys
        assert "eos=1" in keys

    def test_conjunction_features_pair_unit_with_next_word(self) -> None:
        """`한 자리 + 번`의 횟수 편향을 뒤 낱말이 뒤집을 수 있어야 한다."""

        analyzer = _FakeAnalyzer(
            [("7", "SN", 0, 1), ("번", "NNB", 1, 2), ("문제", "NNG", 3, 5)]
        )
        keys = set(
            number_context.extract_features(
                analyzer.analyze("7번 문제"),
                "7번 문제",
                start=0,
                end=1,
                context_end=2,
                unit="번",
            )
        )

        assert "un=번|문제/NNG" in keys
        assert "ln=1|문제/NNG" in keys

    def test_conjunction_features_absent_without_following_word(self) -> None:
        analyzer = _FakeAnalyzer([("7", "SN", 0, 1), ("번", "NNB", 1, 2)])
        keys = set(
            number_context.extract_features(
                analyzer.analyze("7번"), "7번", start=0, end=1, context_end=2, unit="번"
            )
        )

        assert not any(key.startswith(("un=", "ln=")) for key in keys)


class TestModelLoading:
    """배포된 가중치는 필요할 때 한 번만 읽는다."""

    def test_packaged_model_loads_and_has_three_classes(self) -> None:
        model = number_context.load_model()

        assert model.classes == ("native", "sino", "digitwise")
        assert model.weights
        assert set(model.thresholds) <= set(model.classes)
        assert all(
            set(per_site) <= {"unit", "bare"} for per_site in model.thresholds.values()
        )

    def test_load_model_is_cached(self) -> None:
        assert number_context.load_model() is number_context.load_model()


class TestModelValidation:
    """손상된 계수 파일은 조용히 통과시키지 않는다."""

    @staticmethod
    def _payload(**overrides) -> dict:
        payload = {
            "schema_version": 2,
            "feature_version": 3,
            "classes": ["native", "sino", "digitwise"],
            "bias": [0.0, 0.0, 0.0],
            "weights": {"u=번": [0.1, 0.2, 0.3]},
            "thresholds": {"sino": {"unit": 0.9, "bare": 1.01}},
        }
        payload.update(overrides)
        return payload

    def test_valid_payload_is_accepted(self) -> None:
        assert number_context._parse_model(self._payload()).classes == (
            "native",
            "sino",
            "digitwise",
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"bias": [0.0, float("inf"), 0.0]},
            {"weights": {"u=번": [float("nan"), 0.0, 0.0]}},
            {"thresholds": {"sino": {"unit": -1.0}}},
            {"thresholds": {"sino": {"unit": 1.5}}},
            {"thresholds": {"sino": {"unit": float("nan")}}},
            {"thresholds": {"sino": 0.9}},
        ],
        ids=[
            "무한대bias",
            "NaN가중치",
            "음수임계값",
            "1초과임계값",
            "NaN임계값",
            "자리별아님",
        ],
    )
    def test_non_finite_or_out_of_range_values_are_rejected(self, overrides) -> None:
        with pytest.raises(number_context.ModelFormatError):
            number_context._parse_model(self._payload(**overrides))
