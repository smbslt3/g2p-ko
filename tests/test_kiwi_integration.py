"""Git 커밋에 고정한 실제 Kiwi.split 통합 경로를 검증한다."""

from __future__ import annotations

import pytest
from kiwipiepy import Kiwi, SplitToken

import g2p_ko.analyzer as analyzer_module
from g2p_ko.analyzer import KiwiAnalyzer
from g2p_ko.model import AnalysisToken
from tests._internal_pipeline import convert_pipeline


@pytest.fixture(scope="module")
def kiwi() -> Kiwi:
    """비용이 큰 형태소 모델을 모듈 내 실제 통합 테스트에서 재사용한다."""

    return Kiwi()


@pytest.fixture(autouse=True)
def use_real_kiwi(monkeypatch: pytest.MonkeyPatch, kiwi: Kiwi) -> None:
    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", kiwi)


def test_installed_kiwi_exposes_source_aligned_split(kiwi: Kiwi) -> None:
    assert kiwi.split("했다") == [
        SplitToken("했", "VV+EP", 0, 1),
        SplitToken("다", "EF", 1, 1),
    ]
    assert kiwi.split("A\u2800B") == [
        SplitToken("A", "SL", 0, 1),
        SplitToken("B", "SL", 2, 1),
    ]


def test_real_split_tokens_map_directly_to_analysis_tokens(kiwi: Kiwi) -> None:
    source = "우리의 맑게 할 것을"

    tokens = KiwiAnalyzer().analyze(source)

    assert tokens == (
        AnalysisToken("우리", "NP", 0, 2),
        AnalysisToken("의", "JKG", 2, 3),
        AnalysisToken("맑", "VA", 4, 5),
        AnalysisToken("게", "EC", 5, 6),
        AnalysisToken("할", "VX+ETM", 7, 8),
        AnalysisToken("것", "NNB", 9, 10),
        AnalysisToken("을", "JKO", 10, 11),
    )


def test_real_kiwi_drives_morphology_rules_after_nfc_normalization(kiwi: Kiwi) -> None:
    source = "\u110b\u116e리의 맑게 할 것을"

    result = convert_pipeline(source)

    assert result.normalized_text == "우리의 맑게 할 것을"
    assert result.surface_pronunciation == "우리에 말께 할 꺼슬"
    assert result.complete
    assert not result.diagnostics
    assert {
        "ko.ui.particle",
        "ko.stem.rieul_giyeok",
        "ko.modifier.tensing.27",
    } <= {item.rule_id for item in result.rewrites}
    particle = next(item for item in result.rewrites if item.rule_id == "ko.ui.particle")
    assert [(span.start, span.end) for span in particle.source_spans] == [(3, 4)]


def test_real_kiwi_analyzes_many_candidates_in_one_split_call(
    monkeypatch: pytest.MonkeyPatch,
    kiwi: Kiwi,
) -> None:
    count = 1_000
    source = "맑게 " * count
    calls = 0

    class CountingKiwi:
        """실제 Kiwi에 위임하면서 split 호출 수만 기록한다."""

        def split(self, text: str) -> list[SplitToken]:
            nonlocal calls
            calls += 1
            return kiwi.split(text)

    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", CountingKiwi())
    result = convert_pipeline(source, max_length=len(source))

    assert calls == 1
    assert result.surface_pronunciation == "말께 " * count
    assert result.complete
    assert not result.diagnostics
