"""경계와 trace를 보존하는 첫 한국어 음운 엔진이다."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from io import StringIO
from typing import Mapping

from ..analyzer import KiwiAnalyzer
from ..errors import BackendUnavailableError
from ..model import (
    AnalysisToken,
    Boundary,
    Diagnostic,
    Handler,
    OutputSegment,
    Rewrite,
    RewriteStage,
    Severity,
    SourceSpan,
)
from .hangul import HangulToken, character_source_spans, decompose_syllable, is_modern_syllable
from .rules import (
    CODA_TRANSITIONS,
    HIEUH_ASSIMILATION,
    HIEUH_FINALS,
    HIEUH_REMAINDER,
    LIAISON_CLUSTER,
    NASAL_CODA,
    RULES,
    TENSE_ONSET,
)


_MORPHOLOGY_CODAS = frozenset({"ㄴ", "ㄵ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄾ", "ㅁ"})
_SINO_N_INSERTION_LEFT = frozenset({"십", "백", "천", "만", "억"})
_NATIVE_N_INSERTION_LEFT = frozenset({"열", "물", "른", "흔", "쉰", "순", "든"})


@dataclass(slots=True)
class _Unit:
    """한 출력 문자와 변환 중 바뀔 수 있는 음운 토큰을 함께 보관한다."""

    segment_index: int
    handler: Handler
    source_spans: tuple[SourceSpan, ...]
    raw: str | None = None
    syllable: HangulToken | None = None

    @property
    def text(self) -> str:
        if self.syllable is not None:
            return self.syllable.text
        assert self.raw is not None
        return self.raw


@dataclass(frozen=True, slots=True)
class _MorphologyCandidate:
    """분석 적용 또는 보수적 차단에 쓰는 한 음운 후보다."""

    rule_ids: tuple[str, ...]
    left: int
    right: int
    crossed_space: bool
    deferred: bool = False
    block_unit: bool = False


@dataclass(frozen=True, slots=True)
class _Annotation:
    """검증된 분석 token을 normalized offset으로 조회하는 내부 색인이다."""

    _tokens_ending_at: Mapping[int, AnalysisToken]
    _tokens_starting_at: Mapping[int, AnalysisToken]

    @classmethod
    def from_tokens(cls, tokens: tuple[AnalysisToken, ...]) -> _Annotation:
        """원래 token 순서를 보존한 offset 색인을 한 번만 만든다."""

        return cls(
            {token.normalized_end: token for token in tokens},
            {token.normalized_start: token for token in tokens},
        )

    def ending_at(self, offset: int) -> AnalysisToken | None:
        """지정한 code point 직후에 끝나는 token을 반환한다."""

        return self._tokens_ending_at.get(offset)

    def starting_at(self, offset: int) -> AnalysisToken | None:
        """지정한 code point에서 시작하는 token을 반환한다."""

        return self._tokens_starting_at.get(offset)


@dataclass(frozen=True, slots=True)
class _Pronunciation:
    """내장 음운 규칙이 파이프라인에 넘기는 최소 결과다."""

    segments: tuple[OutputSegment, ...]
    rewrites: tuple[Rewrite, ...]
    diagnostics: tuple[Diagnostic, ...]

    @property
    def text(self) -> str:
        return "".join(segment.text for segment in self.segments)


def pronounce(
    segments: Sequence[OutputSegment],
) -> _Pronunciation:
    """정규화 segment에 문맥 독립 한국어 음운 규칙을 적용한다.

    ``HARD`` 경계와 한국어 이외 segment는 절대로 넘지 않는다. 공백은 보존하되,
    문장부호는 음운 이웃 관계를 끊는다. 첫 슬라이스에서 품사·형태소 경계가
    필수인 규칙은 적용하지 않고 진단만 남긴다.
    """

    items = tuple(segments)
    text = "".join(item.text for item in items)

    units = _build_units(items)
    rewrites: list[Rewrite] = []

    # 구개음화는 종성+ㅎ 격음화보다 먼저 판정해야 ‘굳히다’를 [구치다]로 만든다.
    _apply_jyeo(units, rewrites)
    _apply_consonant_ui(units, rewrites)
    diagnostics, blocked_pairs, blocked_units = _apply_morphology_rules(
        text, units, items, KiwiAnalyzer(), rewrites
    )
    _apply_palatalization(units, items, rewrites, blocked_pairs, blocked_units)
    _apply_hieuh_consonants(units, items, rewrites, blocked_pairs, blocked_units)
    _apply_hieuh_before_vowel(units, items, rewrites, blocked_pairs, blocked_units)
    _apply_numeral_n_insertion(units, items, rewrites, blocked_pairs, blocked_units)
    _apply_liaison(units, items, rewrites, blocked_pairs, blocked_units)
    _apply_coda_neutralization(units, items, rewrites, blocked_pairs, blocked_units)
    _apply_liquid_assimilation(units, items, rewrites, blocked_pairs, blocked_units)
    _apply_nasalization(units, items, rewrites, blocked_pairs, blocked_units)
    _apply_tensification(units, items, rewrites, blocked_pairs, blocked_units)

    pronunciation_segments = _render_segments(units, items)
    return _Pronunciation(
        pronunciation_segments,
        tuple(rewrites),
        tuple(diagnostics),
    )


def _build_units(segments: tuple[OutputSegment, ...]) -> list[_Unit]:
    units: list[_Unit] = []
    for segment_index, segment in enumerate(segments):
        provenance = segment.source_spans or ((segment.anchor,) if segment.anchor is not None else ())
        char_spans = character_source_spans(segment.text, provenance)
        # 영어 provenance는 출력 segment에 유지하되, 문자 이름 규칙으로 생성된 한글만
        # 한국어 음운 이웃으로 취급한다. 미변환 Latin segment에는 적용하지 않는다.
        phonological_handler = (
            Handler.KOREAN
            if segment.handler is Handler.ENGLISH and segment.rule_id == "english.letter_names.v1"
            else segment.handler
        )
        for character, spans in zip(segment.text, char_spans):
            if phonological_handler is Handler.KOREAN and is_modern_syllable(character):
                units.append(
                    _Unit(
                        segment_index,
                        phonological_handler,
                        spans,
                        syllable=decompose_syllable(character, spans),
                    )
                )
            else:
                units.append(_Unit(segment_index, phonological_handler, spans, raw=character))
    return units


def _render_segments(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
) -> tuple[OutputSegment, ...]:
    rendered: list[str] = []
    unit_index = 0
    for segment_index in range(len(segments)):
        buffer = StringIO()
        while unit_index < len(units) and units[unit_index].segment_index == segment_index:
            buffer.write(units[unit_index].text)
            unit_index += 1
        rendered.append(buffer.getvalue())
    return tuple(
        OutputSegment(
            rendered[index],
            segment.source_spans,
            segment.handler,
            segment.reason,
            segment.rule_id,
            segment.boundary_before,
            segment.boundary_after,
            segment.insertion_rule,
            segment.anchor,
        )
        for index, segment in enumerate(segments)
    )


def _next_syllable(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    index: int,
) -> tuple[int, bool] | None:
    """공백만 건너뛴 다음 한국어 음절과 단어 경계 여부를 찾는다."""

    crossed_space = False
    for candidate in range(index + 1, len(units)):
        unit = units[candidate]
        if unit.syllable is not None:
            if unit.handler is not Handler.KOREAN:
                return None
            if _has_hard_boundary(units[index].segment_index, unit.segment_index, segments):
                return None
            return candidate, crossed_space
        # U+0020만 단어 사이 공백으로 허용한다. 탭·줄바꿈은 구·문장 경계를 넘지 않는다.
        if unit.handler is not Handler.KOREAN or unit.raw != " ":
            return None
        crossed_space = True
    return None


def _has_hard_boundary(
    left_segment: int,
    right_segment: int,
    segments: tuple[OutputSegment, ...],
) -> bool:
    if left_segment == right_segment:
        return False
    start, end = sorted((left_segment, right_segment))
    for index in range(start, end):
        if (
            segments[index].boundary_after is Boundary.HARD
            or segments[index + 1].boundary_before is Boundary.HARD
        ):
            return True
    return False


def _pair_text(units: list[_Unit], left: int, right: int) -> str:
    return "".join(unit.text for unit in units[left : right + 1])


def _record(
    rewrites: list[Rewrite],
    units: list[_Unit],
    left: int,
    right: int,
    before: str,
    rule_id: str,
) -> None:
    after = _pair_text(units, left, right)
    if before == after:
        return
    source_spans = _ordered_unique_spans(
        span for unit in units[left : right + 1] for span in unit.source_spans
    )
    rewrites.append(
        Rewrite(source_spans, before, after, Handler.KOREAN, rule_id, RewriteStage.PHONOLOGY)
    )


def _ordered_unique_spans(spans: Iterable[SourceSpan]) -> tuple[SourceSpan, ...]:
    unique: dict[tuple[int, int, str], SourceSpan] = {}
    for span in spans:
        unique.setdefault((span.start, span.end, span.surface), span)
    return tuple(sorted(unique.values(), key=lambda item: (item.start, item.end, item.surface)))


def _apply_jyeo(units: list[_Unit], rewrites: list[Rewrite]) -> None:
    """용언 활용형에서 안전하게 관찰되는 져·쪄·쳐 계열을 처리한다."""

    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None or token.onset not in {"ㅈ", "ㅉ", "ㅊ"} or token.vowel != "ㅕ":
            continue
        before = unit.text
        unit.syllable = token.changed(vowel="ㅓ")
        _record(rewrites, units, index, index, before, RULES["ko.vowel.jyeo"].id)


def _apply_consonant_ui(units: list[_Unit], rewrites: list[Rewrite]) -> None:
    """자음 초성 음절의 ㅢ를 문맥 없이 ㅣ로 실현한다."""

    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None or token.onset == "ㅇ" or token.vowel != "ㅢ":
            continue
        before = unit.text
        unit.syllable = token.changed(vowel="ㅣ")
        _record(rewrites, units, index, index, before, RULES["ko.vowel.consonant_ui"].id)


def _apply_morphology_rules(
    text: str,
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    analyzer: KiwiAnalyzer,
    rewrites: list[Rewrite],
) -> tuple[list[Diagnostic], frozenset[tuple[int, int]], frozenset[int]]:
    """분석 규칙과 의도적으로 미룬 어휘 후보를 분리해 처리한다."""

    candidates = _collect_morphology_candidates(units, segments)
    if not candidates:
        return [], frozenset(), frozenset()
    deferred = tuple(item for item in candidates if item.deferred)
    analyzable = tuple(item for item in candidates if not item.deferred)
    diagnostics = _skipped_rule_diagnostics(units, deferred)
    blocked_pairs = _blocked_pairs(deferred)
    blocked_units = _blocked_units(deferred)
    if not analyzable:
        return diagnostics, blocked_pairs, blocked_units
    annotation, analyzer_diagnostics = _analyze_once(text, analyzer, units, analyzable)
    diagnostics.extend(analyzer_diagnostics)
    if annotation is None:
        # 초기 M2의 호환 진단은 분석 실패를 감추지 않되, 규칙이 적용되지 않았음을
        # 호출자가 기존처럼 식별할 수 있게 함께 남긴다.
        diagnostics.extend(_skipped_rule_diagnostics(units, analyzable))
        return (
            diagnostics,
            blocked_pairs | _blocked_pairs(analyzable),
            blocked_units | _blocked_units(analyzable),
        )

    for candidate in analyzable:
        if candidate.rule_ids == (RULES["ko.ui.particle"].id,):
            _apply_particle_ui(units, candidate, annotation, rewrites)
        elif candidate.rule_ids == (RULES["ko.stem.rieul_giyeok"].id,):
            _apply_rieul_giyeok_stem(units, candidate, annotation, rewrites)
        elif candidate.rule_ids == (RULES["ko.stem.tensing.24"].id,):
            _apply_stem_tensing_24(units, candidate, annotation, rewrites)
        elif candidate.rule_ids == (RULES["ko.stem.tensing.25"].id,):
            _apply_stem_tensing_25(units, candidate, annotation, rewrites)
        elif candidate.rule_ids == (RULES["ko.modifier.tensing.27"].id,):
            _apply_modifier_tensing_27(units, candidate, annotation, rewrites)
    return diagnostics, blocked_pairs, blocked_units


def _blocked_pairs(
    candidates: tuple[_MorphologyCandidate, ...],
) -> frozenset[tuple[int, int]]:
    """적용을 미룬 후보의 인접 음절 pair를 만든다."""

    return frozenset((item.left, item.right) for item in candidates)


def _blocked_units(candidates: tuple[_MorphologyCandidate, ...]) -> frozenset[int]:
    """어휘 보류 후보에서 어떤 규칙도 바꾸지 않을 음절을 만든다."""

    return frozenset(
        index
        for item in candidates
        if item.block_unit
        for index in range(item.left, item.right + 1)
    )


def _is_blocked_pair(
    blocked_pairs: frozenset[tuple[int, int]],
    blocked_units: frozenset[int],
    left: int,
    right: int,
) -> bool:
    return left in blocked_units or right in blocked_units or (left, right) in blocked_pairs


def _collect_morphology_candidates(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
) -> tuple[_MorphologyCandidate, ...]:
    """표면형만으로 가능한 후보를 모으되, 실제 적용은 분석 뒤로 미룬다."""

    candidates: list[_MorphologyCandidate] = []
    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None:
            continue
        if _is_adjacent_korean_syllable_before(units, segments, index) and (
            token.onset == "ㅇ" and token.vowel == "ㅢ" and token.coda is None
        ):
            candidates.append(_MorphologyCandidate((RULES["ko.ui.particle"].id,), index, index, False))
        if token.coda not in _MORPHOLOGY_CODAS:
            continue
        neighbor = _next_syllable(units, segments, index)
        deferred = (
            _deferred_lexical_candidate(units, segments, index, neighbor)
            if token.coda == "ㄼ" and unit.text in {"밟", "넓"}
            else None
        )
        if deferred is not None:
            candidates.append(deferred)
            continue
        if neighbor is None:
            continue
        right, crossed_space = neighbor
        next_token = units[right].syllable
        assert next_token is not None
        if token.coda == "ㄺ" and next_token.onset == "ㄱ" and not crossed_space:
            candidates.append(
                _MorphologyCandidate((RULES["ko.stem.rieul_giyeok"].id,), index, right, False)
            )
        if (
            token.coda in {"ㄴ", "ㄵ", "ㅁ", "ㄻ"}
            and next_token.onset in {"ㄱ", "ㄷ", "ㅅ", "ㅈ"}
            and not crossed_space
        ):
            candidates.append(
                _MorphologyCandidate((RULES["ko.stem.tensing.24"].id,), index, right, False)
            )
        if (
            token.coda in {"ㄼ", "ㄾ"}
            and next_token.onset in {"ㄱ", "ㄷ", "ㅅ", "ㅈ"}
            and not crossed_space
        ):
            candidates.append(
                _MorphologyCandidate((RULES["ko.stem.tensing.25"].id,), index, right, False)
            )
        if (
            token.coda == "ㄹ"
            and crossed_space
            and next_token.onset in {"ㄱ", "ㄷ", "ㅂ", "ㅅ", "ㅈ"}
        ):
            candidates.append(
                _MorphologyCandidate((RULES["ko.modifier.tensing.27"].id,), index, right, True)
            )
    return tuple(candidates)


def _deferred_lexical_candidate(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    index: int,
    neighbor: tuple[int, bool] | None,
) -> _MorphologyCandidate | None:
    """근거 corpus가 없는 밟·넓 예외는 발음을 추측하지 않고 차단한다."""

    unit = units[index]
    surface = unit.text
    if surface == "밟":
        if neighbor is None:
            return _MorphologyCandidate(
                (RULES["ko.lexical.balb_neolb"].id,), index, index, False, True, True
            )
        right, crossed_space = neighbor
        next_token = units[right].syllable
        assert next_token is not None
        if not crossed_space and next_token.onset == "ㅇ":
            # 모음 앞 겹받침 연음은 이미 검증된 일반 규칙이므로 보존 차단하지 않는다.
            return None
        if crossed_space:
            return _MorphologyCandidate(
                (RULES["ko.lexical.balb_neolb"].id,), index, index, True, True, True
            )
        return _MorphologyCandidate(
            (RULES["ko.lexical.balb_neolb"].id,), index, right, False, True, True
        )
    if surface != "넓":
        return None
    for prefix in ("넓죽", "넓둥글"):
        if _matches_lexical_prefix(units, segments, index, prefix):
            assert neighbor is not None
            right, crossed_space = neighbor
            if not crossed_space:
                return _MorphologyCandidate(
                    (RULES["ko.lexical.balb_neolb"].id,), index, right, False, True, True
                )
    return None


def _matches_lexical_prefix(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    index: int,
    prefix: str,
) -> bool:
    """공백·HARD 경계를 넘지 않는 정확한 어휘 표면 접두사만 확인한다."""

    if index + len(prefix) > len(units):
        return False
    for offset, character in enumerate(prefix):
        unit = units[index + offset]
        if unit.handler is not Handler.KOREAN or unit.syllable is None or unit.text != character:
            return False
        if offset and _has_hard_boundary(
            units[index + offset - 1].segment_index,
            unit.segment_index,
            segments,
        ):
            return False
    return True


def _is_adjacent_korean_syllable_before(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    index: int,
) -> bool:
    """조사 후보 바로 앞이 같은 한국어 island의 연속 음절인지 확인한다."""

    if index == 0:
        return False
    previous = units[index - 1]
    current = units[index]
    return (
        previous.handler is Handler.KOREAN
        and previous.syllable is not None
        and not _has_hard_boundary(previous.segment_index, current.segment_index, segments)
    )


def _analyze_once(
    text: str,
    analyzer: KiwiAnalyzer,
    units: list[_Unit],
    candidates: tuple[_MorphologyCandidate, ...],
) -> tuple[_Annotation | None, list[Diagnostic]]:
    """후보가 있을 때만 분석기를 한 번 호출하고 offset 계약을 검증한다."""

    try:
        tokens = tuple(analyzer.analyze(text))
        _validate_analysis_tokens(tokens, text)
    except BackendUnavailableError:
        return None, _analyzer_failure_diagnostics(
            units,
            candidates,
            "analyzer_unavailable",
            "형태소 의존 규칙 후보가 있지만 분석기를 사용할 수 없어 적용하지 않았습니다.",
        )
    except Exception:
        return None, _analyzer_failure_diagnostics(
            units,
            candidates,
            "analyzer_failed",
            "형태소 분석 결과를 검증할 수 없어 후보 규칙을 적용하지 않았습니다.",
        )
    return _Annotation.from_tokens(tokens), []


def _validate_analysis_tokens(
    tokens: tuple[AnalysisToken, ...],
    text: str,
) -> None:
    """분석 token의 표면형과 normalized 좌표를 현재 문자열에 맞춰 검증한다."""

    previous_end = 0
    for token in tokens:
        if not isinstance(token, AnalysisToken):
            raise ValueError("분석기는 AnalysisToken만 반환해야 합니다.")
        if not (0 <= token.normalized_start < token.normalized_end <= len(text)):
            raise ValueError("분석 token의 normalized 좌표가 문자열 범위를 벗어났습니다.")
        if token.normalized_start < previous_end:
            raise ValueError("분석 token은 원문 순서의 비중첩 범위여야 합니다.")
        if not token.tag or not token.form:
            raise ValueError("분석 token에는 form과 tag가 필요합니다.")
        if token.form != text[token.normalized_start : token.normalized_end]:
            raise ValueError("분석 token의 표면형이 normalized 좌표와 다릅니다.")
        previous_end = token.normalized_end


def _analyzer_failure_diagnostics(
    units: list[_Unit],
    candidates: tuple[_MorphologyCandidate, ...],
    code: str,
    message: str,
) -> list[Diagnostic]:
    diagnostics: list[Diagnostic] = []
    seen: set[tuple[str, int, int]] = set()
    for candidate in candidates:
        for rule_id in candidate.rule_ids:
            key = (rule_id, candidate.left, candidate.right)
            if key in seen:
                continue
            seen.add(key)
            diagnostics.append(
                Diagnostic(
                    code,
                    message,
                    Severity.WARNING,
                    _candidate_spans(units, candidate),
                    rule_id,
                    RewriteStage.PHONOLOGY,
                )
            )
    return diagnostics


def _skipped_rule_diagnostics(
    units: list[_Unit],
    candidates: tuple[_MorphologyCandidate, ...],
) -> list[Diagnostic]:
    """분석 실패 뒤에도 규칙 미적용을 식별하는 호환 진단을 남긴다."""

    diagnostics: list[Diagnostic] = []
    seen: set[tuple[str, int, int]] = set()
    for candidate in candidates:
        for rule_id in candidate.rule_ids:
            key = (rule_id, candidate.left, candidate.right)
            if key in seen:
                continue
            seen.add(key)
            diagnostics.append(
                Diagnostic(
                    "morphology_rule_skipped",
                    "형태소 분석 결과가 없어 형태소 의존 규칙을 적용하지 않았습니다.",
                    Severity.WARNING,
                    _candidate_spans(units, candidate),
                    rule_id,
                    RewriteStage.PHONOLOGY,
                )
            )
    return diagnostics


def _candidate_spans(
    units: list[_Unit],
    candidate: _MorphologyCandidate,
) -> tuple[SourceSpan, ...]:
    return _ordered_unique_spans(
        span for unit in units[candidate.left : candidate.right + 1] for span in unit.source_spans
    )


def _has_tag_ending_at(annotation: _Annotation, offset: int, prefix: str) -> bool:
    item = annotation.ending_at(offset)
    return item is not None and _tag_has_prefix(item.tag, prefix)


def _has_tag_starting_at(annotation: _Annotation, offset: int, prefix: str) -> bool:
    item = annotation.starting_at(offset)
    return item is not None and _tag_has_prefix(item.tag, prefix)


def _tag_has_prefix(tag: str, prefix: str) -> bool:
    """분석기의 결합 품사 태그도 개별 품사로 안전하게 판정한다."""

    return any(part.startswith(prefix) for part in tag.split("+"))


def _apply_particle_ui(
    units: list[_Unit],
    candidate: _MorphologyCandidate,
    annotation: _Annotation,
    rewrites: list[Rewrite],
) -> None:
    unit = units[candidate.left]
    token = unit.syllable
    assert token is not None
    item = annotation.starting_at(candidate.left)
    if (
        item is None
        or item.normalized_end != candidate.left + 1
        or not _tag_has_prefix(item.tag, "J")
    ):
        return
    before = unit.text
    unit.syllable = token.changed(vowel="ㅔ")
    _record(rewrites, units, candidate.left, candidate.left, before, RULES["ko.ui.particle"].id)


def _is_verb_stem_before_ending(
    annotation: _Annotation,
    left: int,
    right: int,
) -> bool:
    return _has_tag_ending_at(annotation, left + 1, "V") and _has_tag_starting_at(
        annotation, right, "E"
    )


def _apply_rieul_giyeok_stem(
    units: list[_Unit],
    candidate: _MorphologyCandidate,
    annotation: _Annotation,
    rewrites: list[Rewrite],
) -> None:
    if not _is_verb_stem_before_ending(annotation, candidate.left, candidate.right):
        return
    left = units[candidate.left]
    right = units[candidate.right]
    token = left.syllable
    next_token = right.syllable
    assert token is not None and next_token is not None
    before = _pair_text(units, candidate.left, candidate.right)
    left.syllable = token.changed(coda="ㄹ")
    right.syllable = next_token.changed(onset="ㄲ")
    _record(
        rewrites,
        units,
        candidate.left,
        candidate.right,
        before,
        RULES["ko.stem.rieul_giyeok"].id,
    )


def _apply_stem_tensing_24(
    units: list[_Unit],
    candidate: _MorphologyCandidate,
    annotation: _Annotation,
    rewrites: list[Rewrite],
) -> None:
    if not _is_verb_stem_before_ending(annotation, candidate.left, candidate.right):
        return
    left = units[candidate.left]
    right = units[candidate.right]
    token = left.syllable
    next_token = right.syllable
    assert token is not None and next_token is not None
    coda = {"ㄴ": "ㄴ", "ㄵ": "ㄴ", "ㅁ": "ㅁ", "ㄻ": "ㅁ"}[token.coda or ""]
    tense = TENSE_ONSET.get(next_token.onset)
    if tense is None:
        return
    before = _pair_text(units, candidate.left, candidate.right)
    left.syllable = token.changed(coda=coda)
    right.syllable = next_token.changed(onset=tense)
    _record(
        rewrites,
        units,
        candidate.left,
        candidate.right,
        before,
        RULES["ko.stem.tensing.24"].id,
    )


def _apply_stem_tensing_25(
    units: list[_Unit],
    candidate: _MorphologyCandidate,
    annotation: _Annotation,
    rewrites: list[Rewrite],
) -> None:
    if not _is_verb_stem_before_ending(annotation, candidate.left, candidate.right):
        return
    left = units[candidate.left]
    right = units[candidate.right]
    token = left.syllable
    next_token = right.syllable
    assert token is not None and next_token is not None
    tense = TENSE_ONSET.get(next_token.onset)
    if tense is None:
        return
    before = _pair_text(units, candidate.left, candidate.right)
    left.syllable = token.changed(coda="ㄹ")
    right.syllable = next_token.changed(onset=tense)
    _record(
        rewrites,
        units,
        candidate.left,
        candidate.right,
        before,
        RULES["ko.stem.tensing.25"].id,
    )


def _apply_modifier_tensing_27(
    units: list[_Unit],
    candidate: _MorphologyCandidate,
    annotation: _Annotation,
    rewrites: list[Rewrite],
) -> None:
    if not _has_tag_ending_at(annotation, candidate.left + 1, "ETM"):
        return
    right = units[candidate.right]
    next_token = right.syllable
    assert next_token is not None
    tense = TENSE_ONSET.get(next_token.onset)
    if tense is None:
        return
    before = _pair_text(units, candidate.left, candidate.right)
    right.syllable = next_token.changed(onset=tense)
    _record(
        rewrites,
        units,
        candidate.left,
        candidate.right,
        before,
        RULES["ko.modifier.tensing.27"].id,
    )


def _apply_palatalization(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    rewrites: list[Rewrite],
    blocked_pairs: frozenset[tuple[int, int]],
    blocked_units: frozenset[int],
) -> None:
    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None or token.coda not in {"ㄷ", "ㅌ", "ㄾ"}:
            continue
        neighbor = _next_syllable(units, segments, index)
        if neighbor is None:
            continue
        right, crossed_space = neighbor
        if _is_blocked_pair(blocked_pairs, blocked_units, index, right):
            continue
        next_token = units[right].syllable
        assert next_token is not None
        if (
            not crossed_space
            and next_token.onset == "ㅇ"
            and next_token.vowel in {"ㅣ", "ㅕ"}
        ):
            before = _pair_text(units, index, right)
            remaining = "ㄹ" if token.coda == "ㄾ" else None
            onset = "ㅈ" if token.coda == "ㄷ" else "ㅊ"
            unit.syllable = token.changed(coda=remaining)
            units[right].syllable = next_token.changed(onset=onset)
            _record(rewrites, units, index, right, before, RULES["ko.palatalization"].id)
        elif (
            not crossed_space
            and token.coda == "ㄷ"
            and next_token.onset == "ㅎ"
            and next_token.vowel == "ㅣ"
        ):
            before = _pair_text(units, index, right)
            unit.syllable = token.changed(coda=None)
            units[right].syllable = next_token.changed(onset="ㅊ")
            _record(rewrites, units, index, right, before, RULES["ko.palatalization"].id)


def _apply_hieuh_consonants(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    rewrites: list[Rewrite],
    blocked_pairs: frozenset[tuple[int, int]],
    blocked_units: frozenset[int],
) -> None:
    aspiration = {"ㄱ": "ㅋ", "ㄷ": "ㅌ", "ㅈ": "ㅊ"}
    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None or (
            token.coda not in HIEUH_FINALS and token.coda not in HIEUH_ASSIMILATION
        ):
            continue
        neighbor = _next_syllable(units, segments, index)
        if neighbor is None:
            continue
        right, _ = neighbor
        if _is_blocked_pair(blocked_pairs, blocked_units, index, right):
            continue
        next_token = units[right].syllable
        assert next_token is not None
        if token.coda in HIEUH_FINALS:
            remainder = HIEUH_REMAINDER[token.coda]
            if next_token.onset not in aspiration and next_token.onset not in {"ㅅ", "ㄴ"}:
                continue
            before = _pair_text(units, index, right)
            if next_token.onset in aspiration:
                unit.syllable = token.changed(coda=remainder)
                units[right].syllable = next_token.changed(onset=aspiration[next_token.onset])
            elif next_token.onset == "ㅅ":
                unit.syllable = token.changed(coda=remainder)
                units[right].syllable = next_token.changed(onset="ㅆ")
            elif next_token.onset == "ㄴ":
                # 홑받침 ㅎ은 ㄴ 앞에서 [ㄴ]이 되고, ㄶ·ㅀ은 ㅎ만 탈락한다.
                unit.syllable = token.changed(coda="ㄴ" if token.coda == "ㅎ" else remainder)
            else:
                continue
            _record(rewrites, units, index, right, before, RULES["ko.hieuh.consonant"].id)
            continue
        if next_token.onset != "ㅎ" or token.coda not in HIEUH_ASSIMILATION:
            continue
        before = _pair_text(units, index, right)
        remainder, onset = HIEUH_ASSIMILATION[token.coda]
        unit.syllable = token.changed(coda=remainder)
        units[right].syllable = next_token.changed(onset=onset)
        _record(rewrites, units, index, right, before, RULES["ko.hieuh.consonant"].id)


def _apply_hieuh_before_vowel(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    rewrites: list[Rewrite],
    blocked_pairs: frozenset[tuple[int, int]],
    blocked_units: frozenset[int],
) -> None:
    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None or token.coda not in HIEUH_FINALS:
            continue
        neighbor = _next_syllable(units, segments, index)
        if neighbor is None:
            continue
        right, crossed_space = neighbor
        if _is_blocked_pair(blocked_pairs, blocked_units, index, right):
            continue
        next_token = units[right].syllable
        assert next_token is not None
        if crossed_space or next_token.onset != "ㅇ":
            continue
        before = _pair_text(units, index, right)
        remainder = HIEUH_REMAINDER[token.coda]
        unit.syllable = token.changed(coda=None)
        if remainder is not None:
            units[right].syllable = next_token.changed(onset=remainder)
        _record(rewrites, units, index, right, before, RULES["ko.hieuh.vowel"].id)


def _apply_liaison(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    rewrites: list[Rewrite],
    blocked_pairs: frozenset[tuple[int, int]],
    blocked_units: frozenset[int],
) -> None:
    for index, unit in enumerate(units):
        token = unit.syllable
        # 종성 ㅇ은 초성으로 옮길 수 없으므로 모음 앞에서도 그대로 둔다.
        if token is None or token.coda is None or token.coda == "ㅇ":
            continue
        neighbor = _next_syllable(units, segments, index)
        if neighbor is None:
            continue
        right, crossed_space = neighbor
        if _is_blocked_pair(blocked_pairs, blocked_units, index, right):
            continue
        next_token = units[right].syllable
        assert next_token is not None
        if crossed_space or next_token.onset != "ㅇ":
            continue
        before = _pair_text(units, index, right)
        if token.coda in LIAISON_CLUSTER:
            remaining, onset = LIAISON_CLUSTER[token.coda]
            unit.syllable = token.changed(coda=remaining)
            units[right].syllable = next_token.changed(onset=onset)
            rule_id = RULES["ko.liaison.cluster"].id
        else:
            unit.syllable = token.changed(coda=None)
            units[right].syllable = next_token.changed(onset=token.coda)
            rule_id = RULES["ko.liaison.single"].id
        _record(rewrites, units, index, right, before, rule_id)


def _apply_numeral_n_insertion(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    rewrites: list[Rewrite],
    blocked_pairs: frozenset[tuple[int, int]],
    blocked_units: frozenset[int],
) -> None:
    """노멀라이저가 만드는 수사 합성어에만 보수적으로 ㄴ을 첨가한다."""

    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None or token.coda is None:
            continue
        neighbor = _next_syllable(units, segments, index)
        if neighbor is None:
            continue
        right, crossed_space = neighbor
        if crossed_space or _is_blocked_pair(blocked_pairs, blocked_units, index, right):
            continue
        next_token = units[right].syllable
        assert next_token is not None
        if next_token.onset != "ㅇ":
            continue

        left_text = unit.text
        right_text = units[right].text
        sino = right_text == "육" and left_text in _SINO_N_INSERTION_LEFT
        native = (
            right_text == "여"
            and left_text in _NATIVE_N_INSERTION_LEFT
            and right + 1 < len(units)
            and units[right + 1].syllable is not None
            and units[right + 1].text in {"섯", "덟"}
        )
        if not sino and not native:
            continue
        before = _pair_text(units, index, right)
        units[right].syllable = next_token.changed(onset="ㄴ")
        _record(
            rewrites,
            units,
            index,
            right,
            before,
            RULES["ko.n_insertion.numeral"].id,
        )


def _apply_coda_neutralization(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    rewrites: list[Rewrite],
    blocked_pairs: frozenset[tuple[int, int]],
    blocked_units: frozenset[int],
) -> None:
    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None or token.coda not in CODA_TRANSITIONS:
            continue
        if index in blocked_units:
            continue
        neighbor = _next_syllable(units, segments, index)
        if neighbor is not None:
            right, crossed_space = neighbor
            if _is_blocked_pair(blocked_pairs, blocked_units, index, right):
                continue
            next_token = units[right].syllable
            assert next_token is not None
            if not crossed_space and next_token.onset == "ㅇ":
                continue
            last = right
        else:
            last = index
        transition = CODA_TRANSITIONS[token.coda]
        before = _pair_text(units, index, last)
        unit.syllable = token.changed(coda=transition.target)
        _record(rewrites, units, index, last, before, transition.rule_id)


def _apply_liquid_assimilation(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    rewrites: list[Rewrite],
    blocked_pairs: frozenset[tuple[int, int]],
    blocked_units: frozenset[int],
) -> None:
    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None or token.coda not in {"ㄱ", "ㄴ", "ㄹ", "ㅁ", "ㅂ", "ㅇ"}:
            continue
        neighbor = _next_syllable(units, segments, index)
        if neighbor is None:
            continue
        right, _ = neighbor
        if _is_blocked_pair(blocked_pairs, blocked_units, index, right):
            continue
        next_token = units[right].syllable
        assert next_token is not None
        if token.coda == "ㄴ" and next_token.onset == "ㄹ":
            before = _pair_text(units, index, right)
            unit.syllable = token.changed(coda="ㄹ")
            units[right].syllable = next_token.changed(onset="ㄹ")
            _record(rewrites, units, index, right, before, RULES["ko.liquid.20"].id)
        elif token.coda == "ㄹ" and next_token.onset == "ㄴ":
            before = _pair_text(units, index, right)
            units[right].syllable = next_token.changed(onset="ㄹ")
            _record(rewrites, units, index, right, before, RULES["ko.liquid.20"].id)
        elif next_token.onset == "ㄹ" and token.coda in {"ㄱ", "ㅂ"}:
            before = _pair_text(units, index, right)
            unit.syllable = token.changed(coda="ㅇ" if token.coda == "ㄱ" else "ㅁ")
            units[right].syllable = next_token.changed(onset="ㄴ")
            _record(rewrites, units, index, right, before, RULES["ko.liquid.19"].id)
        elif next_token.onset == "ㄹ" and token.coda in {"ㅁ", "ㅇ"}:
            before = _pair_text(units, index, right)
            units[right].syllable = next_token.changed(onset="ㄴ")
            _record(rewrites, units, index, right, before, RULES["ko.liquid.19"].id)


def _apply_nasalization(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    rewrites: list[Rewrite],
    blocked_pairs: frozenset[tuple[int, int]],
    blocked_units: frozenset[int],
) -> None:
    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None or token.coda not in NASAL_CODA:
            continue
        neighbor = _next_syllable(units, segments, index)
        if neighbor is None:
            continue
        right, _ = neighbor
        if _is_blocked_pair(blocked_pairs, blocked_units, index, right):
            continue
        next_token = units[right].syllable
        assert next_token is not None
        if next_token.onset not in {"ㄴ", "ㅁ"}:
            continue
        before = _pair_text(units, index, right)
        unit.syllable = token.changed(coda=NASAL_CODA[token.coda])
        _record(rewrites, units, index, right, before, RULES["ko.nasalization"].id)


def _apply_tensification(
    units: list[_Unit],
    segments: tuple[OutputSegment, ...],
    rewrites: list[Rewrite],
    blocked_pairs: frozenset[tuple[int, int]],
    blocked_units: frozenset[int],
) -> None:
    for index, unit in enumerate(units):
        token = unit.syllable
        if token is None or token.coda not in {"ㄱ", "ㄷ", "ㅂ"}:
            continue
        neighbor = _next_syllable(units, segments, index)
        if neighbor is None:
            continue
        right, _ = neighbor
        if _is_blocked_pair(blocked_pairs, blocked_units, index, right):
            continue
        next_token = units[right].syllable
        assert next_token is not None
        tense = TENSE_ONSET.get(next_token.onset)
        if tense is None:
            continue
        before = _pair_text(units, index, right)
        units[right].syllable = next_token.changed(onset=tense)
        _record(rewrites, units, index, right, before, RULES["ko.tensification"].id)
