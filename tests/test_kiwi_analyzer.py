"""공식 Kiwi.split 기반 분석기의 직접 좌표 변환을 검증한다."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import g2p_ko.analyzer as analyzer_module
from g2p_ko.analyzer import KiwiAnalyzer
from g2p_ko.errors import BackendUnavailableError
from g2p_ko.model import AnalysisToken


class FakeKiwi:
    """split 호출과 반환값을 통제하는 가벼운 Kiwi 대역이다."""

    def __init__(self, parts: list[SimpleNamespace]) -> None:
        self.parts = parts
        self.calls: list[str] = []

    def split(self, text: str) -> list[SimpleNamespace]:
        self.calls.append(text)
        return self.parts


def _part(form: str, tag: str, start: int, length: int) -> SimpleNamespace:
    return SimpleNamespace(form=form, tag=tag, start=start, len=length)


def test_split_tokens_are_mapped_without_coordinate_search(monkeypatch: pytest.MonkeyPatch) -> None:
    kiwi = FakeKiwi([
        _part("했", "VV+EP", 0, 1),
        _part("다", "EF", 1, 1),
    ])

    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", kiwi)
    tokens = KiwiAnalyzer().analyze("했다")

    assert kiwi.calls == ["했다"]
    assert tokens == (
        AnalysisToken("했", "VV+EP", 0, 1),
        AnalysisToken("다", "EF", 1, 2),
    )


def test_split_coordinates_preserve_gaps_and_combined_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    kiwi = FakeKiwi([
        _part("할", "VV+ETM", 0, 1),
        _part("것", "NNB", 2, 1),
    ])

    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", kiwi)
    tokens = KiwiAnalyzer().analyze("할 것")

    assert [(item.form, item.tag, item.normalized_start, item.normalized_end) for item in tokens] == [
        ("할", "VV+ETM", 0, 1),
        ("것", "NNB", 2, 3),
    ]


def test_kiwi_is_created_lazily_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[FakeKiwi] = []

    def create_kiwi() -> FakeKiwi:
        kiwi = FakeKiwi([_part("가", "VV", 0, 1)])
        created.append(kiwi)
        return kiwi

    monkeypatch.setattr(analyzer_module, "_create_kiwi", create_kiwi)
    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", None)
    analyzer = KiwiAnalyzer()

    assert not created
    analyzer.analyze("가")
    analyzer.analyze("가")

    assert len(created) == 1
    assert created[0].calls == ["가", "가"]


def test_missing_kiwi_reports_dependency_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "kiwipiepy", None)
    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", None)
    analyzer = KiwiAnalyzer()

    with pytest.raises(
        BackendUnavailableError,
        match="프로젝트 의존성을 설치하세요",
    ) as captured:
        analyzer.analyze("했다")

    assert isinstance(captured.value.__cause__, ImportError)


def test_legacy_kiwi_without_split_reports_unavailable_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", object())
    analyzer = KiwiAnalyzer()

    with pytest.raises(BackendUnavailableError, match="Kiwi.split"):
        analyzer.analyze("했다")


def test_large_split_result_is_converted_in_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    count = 5_000
    text = "가" * count
    kiwi = FakeKiwi([_part("가", "NNG", index, 1) for index in range(count)])

    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", kiwi)
    tokens = KiwiAnalyzer().analyze(text)

    assert kiwi.calls == [text]
    assert len(tokens) == count
    assert tokens[-1].normalized_end == count
