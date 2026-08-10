"""형태소 분석 annotation을 소비하는 한국어 음운 규칙을 검증한다."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import patch

import pytest

from g2p_ko.errors import BackendUnavailableError
import g2p_ko.korean.engine as engine_module
from g2p_ko.korean.engine import pronounce
from g2p_ko.model import AnalysisToken, Boundary, Handler, OutputSegment, SourceSpan
from g2p_ko.pipeline import Pipeline
from tests._internal_pipeline import convert_pipeline


class FakeAnalyzer:
    """호출 횟수와 반환 token을 완전히 통제하는 테스트 분석기다."""

    def __init__(
        self,
        factory: Callable[[str], tuple[AnalysisToken, ...]],
    ) -> None:
        self._factory = factory
        self.calls = 0

    def analyze(self, text: str) -> tuple[AnalysisToken, ...]:
        self.calls += 1
        return self._factory(text)


class RaisingAnalyzer:
    """선택 분석기의 미설치·실패 경로를 재현한다."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls = 0

    def analyze(self, text: str) -> tuple[AnalysisToken, ...]:
        self.calls += 1
        raise self._error


def _segment(source: str) -> OutputSegment:
    return OutputSegment(
        source,
        (SourceSpan.from_source(source, 0, len(source)),),
        Handler.KOREAN,
        "test",
    )


def _token(
    form: str,
    tag: str,
    start: int,
    end: int,
) -> AnalysisToken:
    return AnalysisToken(form, tag, start, end)


def _pronounce(
    segments: tuple[OutputSegment, ...],
    analyzer: FakeAnalyzer | RaisingAnalyzer,
):
    """공개 입력을 늘리지 않고 내부 Kiwi 응답만 테스트 대역으로 바꾼다."""

    with patch.object(engine_module, "KiwiAnalyzer", return_value=analyzer):
        return pronounce(segments)


@pytest.fixture(scope="module")
def default_pipeline() -> Pipeline:
    """기본 Kiwi 모델의 초기화 비용을 형태소 통합 테스트에서 재사용한다."""

    return Pipeline(max_length=10_000)


def test_safe_vowel_rules_do_not_need_analyzer() -> None:
    source = "가져 희"
    analyzer = FakeAnalyzer(lambda _text: ())

    result = _pronounce((_segment(source),), analyzer)

    assert result.text == "가저 히"
    assert analyzer.calls == 0
    assert {item.rule_id for item in result.rewrites} >= {
        "ko.vowel.jyeo",
        "ko.vowel.consonant_ui",
    }


def test_particle_ui_uses_confirmed_particle_pos_and_analyzes_once() -> None:
    source = "우리의"
    analyzer = FakeAnalyzer(
        lambda _text: (
            _token("우리", "NP", 0, 2),
            _token("의", "JKG", 2, 3),
        )
    )

    result = _pronounce((_segment(source),), analyzer)

    assert result.text == "우리에"
    assert analyzer.calls == 1
    assert [item.rule_id for item in result.rewrites] == ["ko.ui.particle"]


def test_default_kiwi_confirms_particle_ui_candidate(default_pipeline: Pipeline) -> None:
    result = default_pipeline.run("우리의")

    assert result.surface_pronunciation == "우리에"
    assert result.complete
    assert not result.diagnostics
    assert "ko.ui.particle" in [item.rule_id for item in result.rewrites]


def test_word_initial_ui_is_not_a_particle_candidate() -> None:
    for source in ("의사", "의자"):
        result = convert_pipeline(source)

        assert result.surface_pronunciation == source
        assert result.complete
        assert not any(item.code == "morphology_rule_skipped" for item in result.diagnostics)


def test_ui_after_space_or_punctuation_is_not_a_particle_candidate() -> None:
    for source in ("가 의사", "가,의자"):
        result = convert_pipeline(source)

        assert result.complete
        assert not any(item.code == "morphology_rule_skipped" for item in result.diagnostics)


def test_ui_after_hard_boundary_is_not_a_particle_candidate() -> None:
    source = "가의사"
    segments = (
        OutputSegment(
            "가",
            (SourceSpan.from_source(source, 0, 1),),
            Handler.KOREAN,
            "test",
            boundary_after=Boundary.HARD,
        ),
        OutputSegment(
            "의사",
            (SourceSpan.from_source(source, 1, 3),),
            Handler.KOREAN,
            "test",
            boundary_before=Boundary.HARD,
        ),
    )

    result = pronounce(segments)

    assert result.text == source
    assert not result.diagnostics


def test_multiple_confirmed_candidates_share_one_analysis_call() -> None:
    source = "우리의 맑게"
    analyzer = FakeAnalyzer(
        lambda _text: (
            _token("우리", "NP", 0, 2),
            _token("의", "JKG", 2, 3),
            _token("맑", "VA", 4, 5),
            _token("게", "EC", 5, 6),
        )
    )

    result = _pronounce((_segment(source),), analyzer)

    assert result.text == "우리에 말께"
    assert analyzer.calls == 1
    assert {item.rule_id for item in result.rewrites} >= {
        "ko.ui.particle",
        "ko.stem.rieul_giyeok",
    }


def test_stem_and_modifier_rules_require_matching_pos() -> None:
    source = "앉고 넓게 할 것을"
    analyzer = FakeAnalyzer(
        lambda _text: (
            _token("앉", "VV", 0, 1),
            _token("고", "EC", 1, 2),
            _token("넓", "VA", 3, 4),
            _token("게", "EC", 4, 5),
            _token("할", "ETM", 6, 7),
            _token("것", "NNB", 8, 9),
            _token("을", "JKO", 9, 10),
        )
    )

    result = _pronounce((_segment(source),), analyzer)

    assert result.text == "안꼬 널께 할 꺼슬"
    assert {item.rule_id for item in result.rewrites} >= {
        "ko.stem.tensing.24",
        "ko.stem.tensing.25",
        "ko.modifier.tensing.27",
    }


def test_modifier_rule_accepts_combined_surface_tag() -> None:
    """분석기가 주는 결합 품사 태그도 형태소 규칙에 사용할 수 있다."""

    source = "할 것을"
    analyzer = FakeAnalyzer(
        lambda _text: (
            _token("할", "VV+ETM", 0, 1),
            _token("것", "NNB", 2, 3),
            _token("을", "JKO", 3, 4),
        )
    )

    result = _pronounce((_segment(source),), analyzer)

    assert result.text == "할 꺼슬"
    assert "ko.modifier.tensing.27" in [item.rule_id for item in result.rewrites]


def test_stem_tensing_covers_all_ported_final_groups() -> None:
    cases = (
        ("신고", "신꼬", "ko.stem.tensing.24"),
        ("삼고", "삼꼬", "ko.stem.tensing.24"),
        ("닮고", "담꼬", "ko.stem.tensing.24"),
        ("핥다", "할따", "ko.stem.tensing.25"),
    )
    for source, expected, rule_id in cases:
        analyzer = FakeAnalyzer(
            lambda _text: (
                _token(source[0], "VV", 0, 1),
                _token(source[1], "EC", 1, 2),
            )
        )

        result = _pronounce((_segment(source),), analyzer)

        assert result.text == expected
        assert analyzer.calls == 1
        assert rule_id in [item.rule_id for item in result.rewrites]


def test_deferred_balb_candidate_skips_positive_and_negative_analyzers() -> None:
    source = "밟다"
    factories = (
        lambda _text: (
            _token("밟", "VV", 0, 1),
            _token("다", "EF", 1, 2),
        ),
        lambda _text: (
            _token("밟", "NNG", 0, 1),
            _token("다", "JKS", 1, 2),
        ),
    )

    for factory in factories:
        analyzer = FakeAnalyzer(factory)

        result = _pronounce((_segment(source),), analyzer)

        assert result.text == source
        assert analyzer.calls == 0
        assert any(
            item.code == "morphology_rule_skipped"
            and item.rule_id == "ko.lexical.balb_neolb"
            for item in result.diagnostics
        )
        assert "ko.stem.tensing.25" not in [item.rule_id for item in result.rewrites]


def test_deferred_neolb_prefixes_block_only_the_exact_lexical_pair() -> None:
    for source, prefix in (("넓죽하다", "넓죽"), ("넓둥글다", "넓둥")):
        analyzer = FakeAnalyzer(lambda _text: ())

        result = _pronounce((_segment(source),), analyzer)

        assert result.text.startswith(prefix)
        assert analyzer.calls == 0
        assert any(
            item.code == "morphology_rule_skipped"
            and item.rule_id == "ko.lexical.balb_neolb"
            for item in result.diagnostics
        )
        assert "ko.stem.tensing.25" not in [item.rule_id for item in result.rewrites]


def test_deferred_balb_does_not_suppress_confirmed_neolb_stem_rule() -> None:
    source = "밟다 넓게"
    analyzer = FakeAnalyzer(
        lambda _text: (
            _token("밟", "NNG", 0, 1),
            _token("다", "JKS", 1, 2),
            _token("넓", "VA", 3, 4),
            _token("게", "EC", 4, 5),
        )
    )

    result = _pronounce((_segment(source),), analyzer)

    assert result.text == "밟다 널께"
    assert analyzer.calls == 1
    assert any(item.rule_id == "ko.lexical.balb_neolb" for item in result.diagnostics)
    assert "ko.stem.tensing.25" in [item.rule_id for item in result.rewrites]


def test_default_engine_preserves_deferred_balb_coda_but_keeps_vowel_liaison() -> None:
    for source in ("밟", "밟.", "밟다"):
        result = convert_pipeline(source)

        assert result.surface_pronunciation == source
        assert not result.complete
        assert any(
            item.code == "morphology_rule_skipped"
            and item.rule_id == "ko.lexical.balb_neolb"
            for item in result.diagnostics
        )

    liaison = convert_pipeline("밟아")

    assert liaison.surface_pronunciation == "발바"
    assert liaison.complete
    assert not any(item.rule_id == "ko.lexical.balb_neolb" for item in liaison.diagnostics)


def test_non_ending_pos_does_not_guess_stem_tensing() -> None:
    source = "안기다"
    analyzer = FakeAnalyzer(
        lambda _text: (
            _token("안", "VV", 0, 1),
            _token("기", "XSV", 1, 2),
            _token("다", "EF", 2, 3),
        )
    )

    result = _pronounce((_segment(source),), analyzer)

    assert result.text == source
    assert "ko.stem.tensing.24" not in [item.rule_id for item in result.rewrites]


def test_analyzer_unavailable_preserves_morphology_candidate_pair() -> None:
    source = "맑게"
    analyzer = RaisingAnalyzer(BackendUnavailableError("테스트용 미설치"))

    result = _pronounce((_segment(source),), analyzer)

    assert result.text == source
    assert analyzer.calls == 1
    assert any(item.code == "analyzer_unavailable" for item in result.diagnostics)
    assert any(item.code == "morphology_rule_skipped" for item in result.diagnostics)


def test_analyzer_failure_and_offset_mismatch_do_not_apply_rule() -> None:
    source = "맑게"
    failed = RaisingAnalyzer(RuntimeError("테스트용 분석 실패"))

    failed_result = _pronounce((_segment(source),), failed)

    assert failed_result.text == source
    assert any(item.code == "analyzer_failed" for item in failed_result.diagnostics)

    mismatch = FakeAnalyzer(
        lambda _text: (
            AnalysisToken("맑게", "VA", 0, 2),
            AnalysisToken("게", "EC", 1, 2),
        )
    )
    mismatch_result = _pronounce((_segment(source),), mismatch)

    assert mismatch_result.text == source
    assert mismatch.calls == 1
    assert any(item.code == "analyzer_failed" for item in mismatch_result.diagnostics)


def test_default_kiwi_confirms_stem_pair(default_pipeline: Pipeline) -> None:
    result = default_pipeline.run("맑게")

    assert result.surface_pronunciation == "말께"
    assert result.complete
    assert not result.diagnostics
    assert "ko.stem.rieul_giyeok" in [item.rule_id for item in result.rewrites]


def test_non_candidate_rieul_word_keeps_context_independent_path() -> None:
    result = convert_pipeline("갈비")

    assert result.surface_pronunciation == "갈비"
    assert result.complete
    assert not result.diagnostics


def test_tab_and_line_feed_do_not_join_morphology_or_phonology_neighbors() -> None:
    for source in ("할\n것", "할\t것"):
        result = convert_pipeline(source)

        assert result.complete
        assert not result.diagnostics
        assert all(item.rule_id != "ko.modifier.tensing.27" for item in result.rewrites)
