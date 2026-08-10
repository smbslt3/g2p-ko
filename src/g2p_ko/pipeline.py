"""단계 순서·실패 격리·trace 병합을 담당하는 정규화 파이프라인이다."""

from __future__ import annotations

from dataclasses import dataclass

from .english import resolve_english
from .korean import pronounce
from .merge import Marker, join_segments, merge_segments
from .model import (
    Boundary,
    Diagnostic,
    Handler,
    OutputSegment,
    Rewrite,
    RewriteStage,
    Severity,
    SourceSpan,
)
from .validation import validate_output
from .routing import Candidate, is_unsupported_script_character, route_candidates
from .scanner import ProtectedRange, scan_protected
from .unicode import NormalizedText, normalize_nfc


@dataclass(frozen=True, slots=True)
class _Resolution:
    """외국어 후보의 치환 결과다."""

    replacement: str
    reason: str
    rule_id: str


@dataclass(frozen=True, slots=True)
class _PipelineResult:
    """내부 G2P 단계의 중간 결과와 검증 정보를 묶는다."""

    normalized_text: str
    surface_pronunciation: str
    normalized_segments: tuple[OutputSegment, ...]
    rewrites: tuple[Rewrite, ...]
    diagnostics: tuple[Diagnostic, ...]
    complete: bool


class Pipeline:
    """발화형 문자열을 영문 철자화와 표면 발음 결과로 조합한다."""

    def __init__(
        self,
        *,
        max_length: int,
    ) -> None:
        self._max_length = max_length

    def run(
        self,
        source: str,
    ) -> _PipelineResult:
        """영문 철자화와 한국어 표면 발음 단계를 모두 실행한다."""

        normalized = normalize_nfc(source, max_length=self._max_length)
        protected = self._snap_protected(normalized, scan_protected(normalized.text))
        routed = route_candidates(normalized.text, protected)

        diagnostics: list[Diagnostic] = []
        unresolved: list[SourceSpan] = []
        rewrites: list[Rewrite] = list(normalized.rewrites)
        markers = self._base_markers(normalized, protected, routed.numeric_ranges)
        self._apply_protected_script_policy(
            diagnostics,
            unresolved,
            normalized,
            protected,
        )

        for candidate in routed.opaque_candidates:
            self._preserve_opaque_candidate(
                diagnostics,
                unresolved,
                normalized,
                candidate,
            )

        for candidate in routed.candidates:
            resolution = self._resolve_candidate(
                diagnostics,
                unresolved,
                normalized,
                candidate,
            )
            if resolution is None:
                markers.append(self._preserved_marker(normalized, candidate))
                continue
            markers.append(
                Marker(
                    candidate.start,
                    candidate.end,
                    resolution.replacement,
                    candidate.handler,
                    resolution.reason,
                    resolution.rule_id,
                    Boundary.MORPHEME,
                )
            )
            span = normalized.source_span(candidate.start, candidate.end)
            rewrites.append(
                Rewrite(
                    (span,),
                    normalized.text[candidate.start : candidate.end],
                    resolution.replacement,
                    candidate.handler,
                    resolution.rule_id,
                    RewriteStage.LEXICAL,
                )
            )

        segments = merge_segments(normalized, markers)
        normalized_text = join_segments(segments)
        validation_diagnostics = validate_output(segments, known_unresolved=unresolved)
        diagnostics.extend(validation_diagnostics)
        complete = not unresolved and not validation_diagnostics

        pronounced = pronounce(segments)
        pronunciation_text = join_segments(pronounced.segments)
        pronunciation_segments = pronounced.segments
        rewrites.extend(pronounced.rewrites)
        diagnostics.extend(pronounced.diagnostics)
        pronunciation_validation_diagnostics = validate_output(
            pronunciation_segments,
            known_unresolved=unresolved,
        )
        diagnostics.extend(pronunciation_validation_diagnostics)
        complete = (
            complete
            and not any(item.severity is Severity.WARNING for item in pronounced.diagnostics)
            and not pronunciation_validation_diagnostics
        )

        diagnostics = self._sorted_unique_diagnostics(diagnostics)
        return _PipelineResult(
            normalized_text,
            pronunciation_text,
            segments,
            tuple(rewrites),
            tuple(diagnostics),
            complete,
        )

    @staticmethod
    def _base_markers(
        normalized: NormalizedText,
        protected: tuple[ProtectedRange, ...],
        numeric_ranges: tuple[tuple[int, int], ...],
    ) -> list[Marker]:
        markers: list[Marker] = []
        for item in protected:
            markers.append(
                Marker(
                    item.start,
                    item.end,
                    normalized.text[item.start : item.end],
                    Handler.PROTECTED,
                    f"protected_{item.kind}",
                    f"protected.{item.kind}",
                    Boundary.HARD,
                )
            )
        for start, end in numeric_ranges:
            markers.append(
                Marker(
                    start,
                    end,
                    normalized.text[start:end],
                    Handler.NUMERIC,
                    "numeric_passthrough",
                    "numeric.passthrough",
                    Boundary.HARD,
                )
            )
        return markers

    @staticmethod
    def _preserved_marker(normalized: NormalizedText, candidate: Candidate) -> Marker:
        return Marker(
            candidate.start,
            candidate.end,
            normalized.text[candidate.start : candidate.end],
            candidate.handler,
            candidate.reason,
            None,
            Boundary.HARD,
        )

    def _resolve_candidate(
        self,
        diagnostics: list[Diagnostic],
        unresolved: list[SourceSpan],
        normalized: NormalizedText,
        candidate: Candidate,
    ) -> _Resolution | None:
        source = normalized.text[candidate.start : candidate.end]
        span = normalized.source_span(candidate.start, candidate.end)
        if candidate.handler is Handler.ENGLISH:
            resolved = resolve_english(source)
            if resolved is not None:
                return _Resolution(
                    resolved,
                    "english_letter_names",
                    "english.letter_names.v1",
                )
            self._diagnose(
                diagnostics,
                "unconverted_latin",
                "정규화되지 않은 Latin span을 원문 그대로 보존했습니다.",
                span,
                "routing.latin",
            )
        else:
            if candidate.reason == "ambiguous_han":
                code = "unconverted_han"
                message = "Han-only span을 보존했습니다."
                rule_id = "routing.ambiguous_han"
            else:
                code = "unsupported_script"
                message = "미지원 문자 체계를 보존했습니다."
                rule_id = "routing.unsupported_script"
            self._diagnose(
                diagnostics,
                code,
                message,
                span,
                rule_id,
            )
        unresolved.append(span)
        return None

    @staticmethod
    def _preserve_opaque_candidate(
        diagnostics: list[Diagnostic],
        unresolved: list[SourceSpan],
        normalized: NormalizedText,
        candidate: Candidate,
    ) -> None:
        """숫자 구조 내부 후보는 resolver에 보내지 않고 문자 체계만 보고한다."""

        span = normalized.source_span(candidate.start, candidate.end)
        if candidate.handler is Handler.ENGLISH:
            Pipeline._diagnose(
                diagnostics,
                "unconverted_latin",
                "숫자 포함 Latin 구조는 부분 변환하지 않고 보존했습니다.",
                span,
                "routing.numeric_latin",
            )
        else:
            Pipeline._diagnose(
                diagnostics,
                "unconverted_han",
                "숫자 포함 Han 구조는 부분 변환하지 않고 보존했습니다.",
                span,
                "routing.numeric_han",
            )
        unresolved.append(span)

    @staticmethod
    def _diagnose(
        diagnostics: list[Diagnostic],
        code: str,
        message: str,
        span: SourceSpan,
        rule_id: str,
    ) -> None:
        diagnostics.append(
            Diagnostic(code, message, Severity.WARNING, (span,), rule_id, RewriteStage.LEXICAL)
        )

    @staticmethod
    def _apply_protected_script_policy(
        diagnostics: list[Diagnostic],
        unresolved: list[SourceSpan],
        normalized: NormalizedText,
        protected: tuple[ProtectedRange, ...],
    ) -> None:
        """보호 구간 내부의 미지원 alpha도 진단하고 보존한다."""

        for item in protected:
            index = item.start
            while index < item.end:
                if not is_unsupported_script_character(normalized.text[index]):
                    index += 1
                    continue
                start = index
                while (
                    index < item.end
                    and is_unsupported_script_character(normalized.text[index])
                ):
                    index += 1
                span = normalized.source_span(start, index)
                Pipeline._diagnose(
                    diagnostics,
                    "unsupported_script",
                    "보호 구간 내 미지원 문자 체계를 보존했습니다.",
                    span,
                    "routing.protected_unsupported_script",
                )
                unresolved.append(span)

    @staticmethod
    def _snap_protected(
        normalized: NormalizedText,
        ranges: tuple[ProtectedRange, ...],
    ) -> tuple[ProtectedRange, ...]:
        result: list[ProtectedRange] = []
        for item in ranges:
            start, end = normalized.snap_range(item.start, item.end)
            if result and start < result[-1].end:
                previous = result[-1]
                result[-1] = ProtectedRange(previous.start, max(previous.end, end), previous.kind)
            else:
                result.append(ProtectedRange(start, end, item.kind))
        return tuple(result)

    @staticmethod
    def _sorted_unique_diagnostics(items: list[Diagnostic]) -> list[Diagnostic]:
        seen: set[tuple[str, int, int, str | None]] = set()
        result: list[Diagnostic] = []
        for item in sorted(
            items,
            key=lambda value: (
                value.source_spans[0].start if value.source_spans else -1,
                value.code,
                value.rule_id or "",
            ),
        ):
            span = item.source_spans[0] if item.source_spans else None
            key = (item.code, span.start if span else -1, span.end if span else -1, item.rule_id)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
