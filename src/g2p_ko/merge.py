"""비겹침 marker를 원문 순서 segment로 병합하고 경계를 보존한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .errors import InternalInvariantError
from .model import Boundary, Handler, OutputSegment, SourceSpan
from .unicode import NormalizedText


@dataclass(frozen=True, slots=True)
class Marker:
    """하나의 정규화 범위에 적용할 handler 결과 또는 보존 지시다."""

    start: int
    end: int
    text: str
    handler: Handler
    reason: str
    rule_id: str | None = None
    boundary: Boundary = Boundary.WORD

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise InternalInvariantError("Marker 범위가 올바르지 않습니다.")


def _source_spans(
    normalized: NormalizedText,
    start: int,
    end: int,
    output_text: str,
) -> tuple[SourceSpan, ...]:
    """무변환 문자는 문자 단위 source map을, 길이 변경은 전체 근거를 보존한다."""

    if len(output_text) == end - start:
        return normalized.character_spans[start:end]
    return (normalized.source_span(start, end),)


def merge_segments(normalized: NormalizedText, markers: Sequence[Marker]) -> tuple[OutputSegment, ...]:
    """marker와 비표시 구간을 빠짐없이 병합해 정규화 segment를 만든다."""

    ordered = sorted(markers, key=lambda item: (item.start, item.end))
    result: list[OutputSegment] = []
    cursor = 0
    for marker in ordered:
        if marker.end > len(normalized.text) or marker.start < cursor:
            raise InternalInvariantError("겹치거나 범위를 벗어난 marker가 발견됐습니다.")
        if cursor < marker.start:
            unchanged = normalized.text[cursor : marker.start]
            result.append(
                OutputSegment(
                    unchanged,
                    normalized.character_spans[cursor : marker.start],
                    Handler.KOREAN,
                    "unchanged",
                )
            )
        result.append(
            OutputSegment(
                marker.text,
                _source_spans(normalized, marker.start, marker.end, marker.text),
                marker.handler,
                marker.reason,
                marker.rule_id,
                marker.boundary,
                marker.boundary,
            )
        )
        cursor = marker.end
    if cursor < len(normalized.text):
        unchanged = normalized.text[cursor:]
        result.append(
            OutputSegment(
                unchanged,
                normalized.character_spans[cursor:],
                Handler.KOREAN,
                "unchanged",
            )
        )
    return tuple(result)


def join_segments(segments: Sequence[OutputSegment]) -> str:
    """순서가 이미 검증된 segment 문자열을 결합한다."""

    return "".join(item.text for item in segments)
