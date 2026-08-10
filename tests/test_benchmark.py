"""성능 측정 도구의 통계와 측정 단계를 검증한다."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import statistics
import sys
import tracemalloc
from types import SimpleNamespace

import pytest


_BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "tools" / "benchmark.py"
_SPEC = importlib.util.spec_from_file_location("_g2p_ko_benchmark", _BENCHMARK_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BENCHMARK
_SPEC.loader.exec_module(_BENCHMARK)


@pytest.mark.parametrize("seconds", [(0.9, 0.1, 0.4), (0.8, 0.2, 0.4, 0.6)])
def test_summary_keeps_existing_statistics_and_adds_odd_even_median(seconds: tuple[float, ...]) -> None:
    samples = [
        {"seconds": value, "allocated_bytes": index + 1, "peak_bytes": (index + 1) * 10}
        for index, value in enumerate(seconds)
    ]

    summary = _BENCHMARK._summary(samples)

    assert summary["seconds"] == {
        "min": min(seconds),
        "mean": statistics.fmean(seconds),
        "median": statistics.median(seconds),
        "max": max(seconds),
    }


def test_measurement_separates_timing_and_memory_calls() -> None:
    tracing_states: list[bool] = []

    def call() -> object:
        tracing_states.append(tracemalloc.is_tracing())
        return object()

    sample = _BENCHMARK._measurement(call)

    assert tracing_states == [False, True]
    assert set(sample) == {"seconds", "allocated_bytes", "peak_bytes"}


def test_cold_import_uses_separate_subprocesses_for_time_and_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    phases: list[str] = []

    def run(command: list[str], **_: object) -> SimpleNamespace:
        phase = command[-1]
        phases.append(phase)
        stdout = '{"seconds": 0.125}' if phase == "seconds" else '{"peak_bytes": 4096}'
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(_BENCHMARK.subprocess, "run", run)

    result = _BENCHMARK._fresh_import(2)

    assert phases == ["seconds", "memory", "seconds", "memory"]
    assert result["status"] == "ok"
    assert result["iterations"] == 2
    assert result["seconds"]["median"] == 0.125
    assert result["memory"]["peak_bytes_max"] == 4096


def test_cold_and_warm_workloads_use_separate_time_and_memory_calls() -> None:
    instances: list[object] = []

    class RecordingG2P:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, text: str) -> str:
            self.calls.append(text)
            return text

    def factory() -> RecordingG2P:
        instance = RecordingG2P()
        instances.append(instance)
        return instance

    scenario = _BENCHMARK.Scenario("test", factory, "입력")
    result = _BENCHMARK._run_workload(scenario, "입력", repeats=3)

    assert result["cold"]["iterations"] == 3
    assert result["warm"]["iterations"] == 3
    assert len(instances) == 7
    assert all(instance.calls == ["입력"] for instance in instances[:6])
    assert instances[6].calls == ["입력"] * 7


def test_morphology_scenario_has_candidates_and_separate_activation() -> None:
    scenario = _BENCHMARK._morphology_scenario()
    corpus = _BENCHMARK._corpus(scenario.sample, 100)

    assert scenario.name == "kiwi_morphology"
    assert scenario.cold_workloads is False
    assert scenario.activation_text == "우리의 맑게"
    assert "맑게" in corpus
    assert "할 것을" in corpus


def test_default_scenario_does_not_activate_morphology_analyzer(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    import g2p_ko.analyzer as analyzer_module
    from g2p_ko import G2P

    calls = 0

    class CountingKiwi:
        def split(self, _text: str) -> tuple[object, ...]:
            nonlocal calls
            calls += 1
            return ()

    scenario = _BENCHMARK._default_scenario()
    text = _BENCHMARK._corpus(scenario.sample, 10_000)

    monkeypatch.setattr(analyzer_module, "_runtime_kiwi", CountingKiwi())
    G2P()(text)

    assert calls == 0


def test_tts_normalizer_scenario_uses_only_the_string_fast_path() -> None:
    scenario = _BENCHMARK._tts_normalizer_scenario()
    instance = scenario.factory()

    assert instance("3개에 ₩1,200") == "세 개에 천이백 원"


def test_scenario_can_skip_repeated_cold_workloads() -> None:
    instances: list[object] = []

    class RecordingG2P:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, text: str) -> str:
            self.calls.append(text)
            return text

    def factory() -> RecordingG2P:
        instance = RecordingG2P()
        instances.append(instance)
        return instance

    scenario = _BENCHMARK.Scenario(
        "test",
        factory,
        "입력",
        cold_workloads=False,
    )
    result = _BENCHMARK._run_workload(scenario, "입력", repeats=2)

    assert result["cold"]["status"] == "not_measured"
    assert result["warm"]["iterations"] == 2
    assert len(instances) == 1
    assert instances[0].calls == ["입력"] * 5


def test_activation_uses_fresh_instance_for_time_and_memory() -> None:
    instances: list[object] = []

    class RecordingG2P:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def __call__(self, text: str) -> str:
            self.calls.append(text)
            return text

    def factory() -> RecordingG2P:
        instance = RecordingG2P()
        instances.append(instance)
        return instance

    scenario = _BENCHMARK.Scenario(
        "test",
        factory,
        "입력",
        activation_text="우리의 맑게",
    )
    result = _BENCHMARK._run_activation(scenario, repeats=2)

    assert result is not None
    assert result["iterations"] == 2
    assert result["input_code_points"] == len("우리의 맑게")
    assert len(instances) == 4
    assert all(
        instance.calls == ["우리의 맑게"]
        for instance in instances
    )


def test_benchmark_schema_exposes_default_and_morphology_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = {"seconds": 0.1, "allocated_bytes": 10, "peak_bytes": 20}
    workload_calls: list[tuple[str, int, int]] = []

    monkeypatch.setattr(_BENCHMARK, "_SIZES", (10,))
    monkeypatch.setattr(_BENCHMARK, "_fresh_import", lambda _repeats: {"status": "ok"})
    monkeypatch.setattr(_BENCHMARK, "_measurement", lambda _call: sample)
    monkeypatch.setattr(
        _BENCHMARK,
        "_run_activation",
        lambda scenario, _repeats: (
            {"status": "ok", "input_code_points": len(scenario.activation_text)}
            if scenario.activation_text is not None
            else None
        ),
    )

    def workload(scenario, text: str, repeats: int):  # type: ignore[no-untyped-def]
        workload_calls.append((scenario.name, len(text), repeats))
        return {"cold": {"status": "ok"}, "warm": {"status": "ok"}}

    monkeypatch.setattr(_BENCHMARK, "_run_workload", workload)

    result = _BENCHMARK.run_benchmark(repeats=2)

    assert result["schema_version"] == 3
    assert set(result["scenarios"]) == {"default", "kiwi_morphology", "tts_normalizer"}
    assert "activation" not in result["scenarios"]["default"]
    assert result["scenarios"]["kiwi_morphology"]["activation"]["status"] == "ok"
    assert result["metadata"]["semantics"]["warm"].startswith("같은 처리 인스턴스")
    assert workload_calls == [
        ("default", 10, 2),
        ("kiwi_morphology", 10, 2),
        ("tts_normalizer", 10, 2),
    ]
