"""최종 출력의 허용 문자와 완전성 진단을 계산한다."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import unicodedata
from typing import Sequence

from .model import Diagnostic, Handler, OutputSegment, RewriteStage, Severity, SourceSpan
from .routing import is_han
from .unicode import is_numeric_character


_ASCII_PUNCTUATION = frozenset(r"!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~")
_GENERIC_NEUTRAL = frozenset("。？！、・")


def _is_hangul(character: str) -> bool:
    code = ord(character)
    return (
        "가" <= character <= "힣"
        or 0x1100 <= code <= 0x11FF
        or 0x3130 <= code <= 0x318F
        or 0xA960 <= code <= 0xA97C
        or 0xD7B0 <= code <= 0xD7FB
    )


def _allowed(character: str) -> bool:
    if _is_hangul(character) or character in " \t\r\n" or is_numeric_character(character):
        return True
    if character.isascii() and character in _ASCII_PUNCTUATION:
        return True
    return character in _GENERIC_NEUTRAL


def _issue(character: str) -> tuple[str, str]:
    code = ord(character)
    if unicodedata.category(character) in {"Cc", "Cf", "Co", "Cn", "Cs"}:
        return "unsupported_character", "제어·형식·비공개 문자가 최종 출력에 남았습니다."
    if is_han(character):
        return "unconverted_han", "정규화되지 않은 Han이 최종 출력에 남았습니다."
    if character.isascii() and character.isalpha():
        return "unconverted_latin", "정규화되지 않은 Latin 문자가 최종 출력에 남았습니다."
    if 0x0400 <= code <= 0x052F:
        return "unsupported_cyrillic", "지원하지 않는 키릴 문자가 최종 출력에 남았습니다."
    if 0x1F000 <= code <= 0x1FAFF or 0x2600 <= code <= 0x27BF:
        return "unsupported_emoji", "지원하지 않는 이모지가 최종 출력에 남았습니다."
    if character.isalpha() or unicodedata.category(character).startswith("L"):
        return "unsupported_script", "지원하지 않는 문자 체계가 최종 출력에 남았습니다."
    return "unsupported_symbol", "지원하지 않는 기호가 최종 출력에 남았습니다."


@dataclass(frozen=True, slots=True)
class _SpanIndex:
    """정렬·병합한 미해결 원문 범위를 이진 탐색하는 작은 색인이다."""

    starts: tuple[int, ...]
    ends: tuple[int, ...]

    @classmethod
    def build(cls, spans: Sequence[SourceSpan]) -> _SpanIndex:
        merged: list[tuple[int, int]] = []
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            if merged and span.start <= merged[-1][1]:
                start, end = merged[-1]
                merged[-1] = start, max(end, span.end)
            else:
                merged.append((span.start, span.end))
        return cls(
            tuple(start for start, _ in merged),
            tuple(end for _, end in merged),
        )

    def overlaps(self, span: SourceSpan) -> bool:
        index = bisect_left(self.starts, span.end) - 1
        return index >= 0 and self.ends[index] > span.start


def _run_span(segment: OutputSegment, start: int, end: int) -> SourceSpan:
    """원문과 길이가 같은 무변환 조각은 문제 문자 범위만 좁혀 반환한다."""

    if not segment.source_spans:
        if segment.anchor is None:
            raise ValueError("삽입 segment에 anchor가 없습니다.")
        return segment.anchor
    if len(segment.source_spans) == len(segment.text):
        selected = segment.source_spans[start:end]
        unique: list[SourceSpan] = []
        for span in selected:
            if not unique or unique[-1] != span:
                unique.append(span)
        source_start = unique[0].start
        source_end = unique[-1].end
        if all(left.end == right.start for left, right in zip(unique, unique[1:])):
            return SourceSpan(
                source_start,
                source_end,
                "".join(item.surface for item in unique),
                unique[0].source_length,
            )
    source_span = segment.source_spans[0]
    if len(segment.source_spans) == 1 and len(segment.text) == len(source_span.surface):
        return SourceSpan(
            source_span.start + start,
            source_span.start + end,
            source_span.surface[start:end],
            source_span.source_length,
        )
    return source_span


def validate_output(
    segments: Sequence[OutputSegment],
    *,
    known_unresolved: Sequence[SourceSpan] = (),
) -> tuple[Diagnostic, ...]:
    """허용 출력 밖 문자만 중복 없이 진단한다.

    외국어 라우팅이 이미 같은 span의 보존 진단을 만들었다면 여기서는 중복하지 않는다.
    """

    diagnostics: list[Diagnostic] = []
    known_index = _SpanIndex.build(known_unresolved)
    for segment in segments:
        span = segment.source_spans[0] if segment.source_spans else segment.anchor
        if span is None:
            raise ValueError("segment의 원문 span 또는 anchor가 필요합니다.")
        if segment.handler is Handler.PROTECTED:
            continue
        index = 0
        while index < len(segment.text):
            if _allowed(segment.text[index]):
                index += 1
                continue
            start = index
            issue_code, message = _issue(segment.text[index])
            index += 1
            while index < len(segment.text) and not _allowed(segment.text[index]):
                next_code, _ = _issue(segment.text[index])
                if next_code != issue_code:
                    break
                index += 1
            run_span = _run_span(segment, start, index)
            if not known_index.overlaps(run_span):
                diagnostics.append(
                    Diagnostic(
                        issue_code,
                        message,
                        Severity.WARNING,
                        (run_span,),
                        "output.validate",
                        RewriteStage.VALIDATION,
                    )
                )
    return tuple(diagnostics)
