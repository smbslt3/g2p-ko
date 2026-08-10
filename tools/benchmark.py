"""CPU 전용 릴리스 후보의 재현 가능한 기본 성능 측정 도구다."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Callable


_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
_SIZES = (100, 1_000, 10_000)
_KOREAN_SAMPLE = "국물이 좋아요. 아기가 웃어요. 공백은 그대로 둬요. "
_MORPHOLOGY_SAMPLE = "우리의 맑게 앉고 넓게 할 것을 정확히 읽습니다. "
_MORPHOLOGY_ACTIVATION_TEXT = "우리의 맑게"
_TTS_NORMALIZER_SAMPLE = "사과 3개는 ₩1,200이고 정확도는 12.50%입니다. "


@dataclass(frozen=True, slots=True)
class Scenario:
    """동일한 측정 규칙으로 실행할 G2P 구성 하나다."""

    name: str
    factory: Callable[[], Any]
    sample: str
    description: str = ""
    cold_workloads: bool = True
    activation_text: str | None = None


class _TTSNormalizerAdapter:
    """공통 workload 호출 규격에 문자열 전용 노멀라이저를 맞춘다."""

    def __init__(self) -> None:
        from g2p_ko import KoreanTTSNormalizer

        self._normalizer = KoreanTTSNormalizer()

    def __call__(self, text: str) -> str:
        """상세 trace를 만들지 않는 대량 처리 경로만 측정한다."""

        return self._normalizer(text)


def _measurement(call: Callable[[], Any]) -> dict[str, float | int]:
    """시간과 Python 할당량을 서로 다른 호출에서 측정한다.

    ``tracemalloc``의 추적 비용이 경과 시간에 섞이지 않게 시간 호출을 먼저
    실행한다.
    """

    started = time.perf_counter()
    call()
    elapsed = time.perf_counter() - started

    tracemalloc.start()
    try:
        before_current, _ = tracemalloc.get_traced_memory()
        memory_value = call()
        after_current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    del memory_value
    return {
        "seconds": elapsed,
        "allocated_bytes": max(0, after_current - before_current),
        "peak_bytes": peak,
    }


def _summary(samples: list[dict[str, float | int]]) -> dict[str, Any]:
    """개별 반복 값을 남기면서 비교에 쓸 요약 통계를 만든다."""

    seconds = [float(item["seconds"]) for item in samples]
    peaks = [int(item["peak_bytes"]) for item in samples]
    allocations = [int(item["allocated_bytes"]) for item in samples]
    return {
        "status": "ok",
        "iterations": len(samples),
        "samples": samples,
        "seconds": {
            "min": min(seconds),
            "mean": statistics.fmean(seconds),
            "median": statistics.median(seconds),
            "max": max(seconds),
        },
        "memory": {
            "peak_bytes_max": max(peaks),
            "allocated_bytes_mean": statistics.fmean(allocations),
        },
    }


def _exception_result(error: Exception) -> dict[str, str]:
    """측정 실패를 예외 형식과 함께 JSON에 기록한다."""

    name = type(error).__name__
    return {"status": "error", "exception": name, "reason": str(error)}


def _corpus(sample: str, size: int) -> str:
    """Python code point 수가 정확히 요청 크기인 결정적 입력을 만든다."""

    return (sample * ((size // len(sample)) + 1))[:size]


def _fresh_import(repeats: int) -> dict[str, Any]:
    """별도 Python 프로세스에서 cold import 시간과 메모리를 분리 측정한다."""

    script = """
import importlib
import json
import sys
import time
import tracemalloc
sys.path.insert(0, sys.argv[1])
phase = sys.argv[2]
if phase == 'seconds':
    started = time.perf_counter()
    importlib.import_module('g2p_ko')
    print(json.dumps({'seconds': time.perf_counter() - started}))
elif phase == 'memory':
    tracemalloc.start()
    importlib.import_module('g2p_ko')
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(json.dumps({'peak_bytes': peak}))
else:
    raise ValueError(f'알 수 없는 측정 단계: {phase}')
"""
    samples: list[dict[str, float | int]] = []
    for _ in range(repeats):
        try:
            values: dict[str, float | int] = {"allocated_bytes": 0}
            for phase in ("seconds", "memory"):
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", script, str(_SRC), phase],
                    cwd=_ROOT,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode:
                    return {
                        "status": "error",
                        "exception": "FreshImportFailed",
                        "reason": completed.stderr.strip() or completed.stdout.strip(),
                    }
                value = json.loads(completed.stdout)
                if phase == "seconds":
                    values["seconds"] = float(value["seconds"])
                else:
                    values["peak_bytes"] = int(value["peak_bytes"])
            samples.append(values)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return _exception_result(error)
    return _summary(samples)


def _default_scenario() -> Scenario:
    from g2p_ko import G2P

    return Scenario(
        "default",
        G2P,
        _KOREAN_SAMPLE,
        "일반 한국어 정규화와 기본 발음 규칙 경로",
    )


def _morphology_scenario() -> Scenario:
    from g2p_ko import G2P

    return Scenario(
        "kiwi_morphology",
        G2P,
        _MORPHOLOGY_SAMPLE,
        "형태소 근거가 필요한 후보를 반복해 Kiwi.split 경로를 실행",
        cold_workloads=False,
        activation_text=_MORPHOLOGY_ACTIVATION_TEXT,
    )


def _tts_normalizer_scenario() -> Scenario:
    return Scenario(
        "tts_normalizer",
        _TTSNormalizerAdapter,
        _TTS_NORMALIZER_SAMPLE,
        "독립형 TTS written-to-spoken 노멀라이저의 문자열 빠른 경로",
    )


def _run_activation(scenario: Scenario, repeats: int) -> dict[str, Any] | None:
    """새 인스턴스의 첫 형태소 분석으로 Kiwi 지연 초기화 비용을 분리한다."""

    text = scenario.activation_text
    if text is None:
        return None
    samples: list[dict[str, float | int]] = []
    for _ in range(repeats):
        try:
            sample = _measurement(
                lambda: scenario.factory()(text)
            )
        except Exception as error:
            return _exception_result(error)
        samples.append(sample)
    result = _summary(samples)
    result["input_code_points"] = len(text)
    return result


def _run_workload(scenario: Scenario, text: str, repeats: int) -> dict[str, Any]:
    """선택한 cold 정책과 준비가 끝난 단일 인스턴스의 warm 성능을 측정한다."""

    if scenario.cold_workloads:
        cold_samples: list[dict[str, float | int]] = []
        for _ in range(repeats):
            try:
                sample = _measurement(lambda: scenario.factory()(text))
            except Exception as error:  # 공개 API의 구성 실패도 JSON으로 남겨야 한다.
                return {"cold": _exception_result(error), "warm": {"status": "not_run"}}
            cold_samples.append(sample)
        cold_result: dict[str, Any] = _summary(cold_samples)
    else:
        cold_result = {
            "status": "not_measured",
            "reason": "Kiwi 지연 초기화는 scenario activation에서 한 번만 분리 측정합니다.",
        }

    try:
        instance = scenario.factory()
    except Exception as error:
        return {"cold": cold_result, "warm": _exception_result(error)}

    try:
        # warm 수치에는 첫 변환의 지연 초기화 비용을 포함하지 않는다.
        instance(text)
    except Exception as error:
        warm_error = _exception_result(error)
        warm_error["status"] = "error"
        warm_error["phase"] = "warmup"
        return {"cold": cold_result, "warm": warm_error}

    warm_samples: list[dict[str, float | int]] = []
    for _ in range(repeats):
        try:
            sample = _measurement(lambda: instance(text))
        except Exception as error:
            return {"cold": cold_result, "warm": _exception_result(error)}
        warm_samples.append(sample)
    return {"cold": cold_result, "warm": _summary(warm_samples)}


def run_benchmark(*, repeats: int) -> dict[str, Any]:
    """CLI와 테스트가 공용으로 쓸 전체 benchmark 결과를 만든다."""

    if repeats <= 0:
        raise ValueError("repeats는 양의 정수여야 합니다.")
    # 소스 checkout에서 실행해도 설치본과 같은 import 대상을 사용하게 한다.
    sys.path.insert(0, str(_SRC))
    scenarios = (_default_scenario(), _morphology_scenario(), _tts_normalizer_scenario())
    result: dict[str, Any] = {
        "schema_version": 3,
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "code_point_sizes": list(_SIZES),
            "repeats": repeats,
            "clock": "time.perf_counter",
            "memory": "tracemalloc Python allocation peak; native RSS는 포함하지 않음",
            "measurement_phases": {
                "seconds": "tracemalloc을 끈 별도 호출에서 측정합니다.",
                "memory": "tracemalloc을 켠 별도 호출에서 측정하며 해당 호출의 시간은 seconds에 포함하지 않습니다.",
            },
            "warmup": "warm workload는 같은 입력으로 1회 비측정 priming 뒤에 측정합니다.",
            "semantics": {
                "construction": "처리 인스턴스를 생성합니다. G2P의 Kiwi 모델은 아직 만들지 않습니다.",
                "activation": "새 G2P의 첫 형태소 처리로 Kiwi 생성과 첫 split을 측정합니다.",
                "cold": "새 G2P 생성과 변환을 함께 측정하며 Kiwi scenario에서는 activation으로 대체합니다.",
                "warm": "같은 처리 인스턴스를 준비 호출한 뒤 내부 상태를 재사용해 측정합니다.",
            },
        },
        "baseline": {"cold_import": _fresh_import(repeats)},
        "scenarios": {},
    }

    for scenario in scenarios:
        try:
            construction = _measurement(scenario.factory)
            construction_result: dict[str, Any] = _summary([construction])
        except Exception as error:
            construction_result = _exception_result(error)
        scenario_result: dict[str, Any] = {
            "description": scenario.description,
            "construction": construction_result,
            "workloads": {},
        }
        activation = _run_activation(scenario, repeats)
        if activation is not None:
            scenario_result["activation"] = activation
        for size in _SIZES:
            text = _corpus(scenario.sample, size)
            workload_result = {
                "input_code_points": len(text),
            }
            workload_result.update(_run_workload(scenario, text, repeats))
            scenario_result["workloads"][str(size)] = workload_result
        result["scenarios"][scenario.name] = scenario_result
    return result


def main(argv: list[str] | None = None) -> int:
    """명시한 옵션만 사용해 network 없는 benchmark JSON을 출력한다."""

    parser = argparse.ArgumentParser(description="g2p-ko CPU benchmark를 JSON으로 출력합니다.")
    parser.add_argument("--repeats", type=int, default=5, help="cold/warm 반복 횟수 (기본값: 5)")
    parser.add_argument("--output", type=Path, help="stdout 대신 결과를 쓸 JSON 파일")
    args = parser.parse_args(argv)
    try:
        result = run_benchmark(repeats=args.repeats)
    except ValueError as error:
        parser.error(str(error))
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
