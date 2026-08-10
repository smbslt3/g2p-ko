"""보호 구간을 제외하고 숫자·외국어 후보를 보수적으로 라우팅한다."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum
import re

from .model import Handler
from .scanner import ProtectedRange, overlaps
from .unicode import is_numeric_character


@dataclass(frozen=True, slots=True)
class Candidate:
    """정규화 문자열 좌표의 외국어 후보와 보존 사유다."""

    start: int
    end: int
    handler: Handler
    reason: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("Candidate는 비어 있지 않은 범위여야 합니다.")


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """숫자 불투명 범위와 일반·불투명 외국어 후보를 분리한 결과다."""

    numeric_ranges: tuple[tuple[int, int], ...]
    candidates: tuple[Candidate, ...]
    opaque_candidates: tuple[Candidate, ...]


_LATIN = re.compile(
    r"(?<![A-Za-z0-9])[A-Za-z](?:[A-Za-z0-9]|[+#]+(?![A-Za-z0-9])|[._'\u2019-](?=[A-Za-z0-9]))*"
)
_STRUCTURAL_PUNCTUATION = frozenset(".#_'\u2019+-")


class _CharacterKind(IntEnum):
    OTHER = 0
    STRUCTURAL = 1
    NUMBER = 2
    LATIN = 3
    HANGUL = 4
    HAN = 5
    UNSUPPORTED = 6


_STRUCTURE_KINDS = frozenset(
    {
        _CharacterKind.STRUCTURAL,
        _CharacterKind.NUMBER,
        _CharacterKind.LATIN,
        _CharacterKind.HANGUL,
        _CharacterKind.HAN,
    }
)
_NON_HAN_FOREIGN_KINDS = frozenset(
    {_CharacterKind.LATIN, _CharacterKind.HANGUL}
)


def _is_hangul(character: str) -> bool:
    code = ord(character)
    return "가" <= character <= "힣" or 0x1100 <= code <= 0x11FF


def is_han(character: str) -> bool:
    """CJK 통합 한자와 확장 블록을 Han-only 보존 대상으로 판정한다."""

    code = ord(character)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
        or 0x30000 <= code <= 0x323AF
    )


def _character_kind(character: str) -> _CharacterKind:
    """라우팅에 필요한 문자 분류를 우선순위에 따라 한 번만 계산한다."""

    if character.isascii():
        if character.isalpha():
            return _CharacterKind.LATIN
        if character.isnumeric():
            return _CharacterKind.NUMBER
        return (
            _CharacterKind.STRUCTURAL
            if character in _STRUCTURAL_PUNCTUATION
            else _CharacterKind.OTHER
        )
    if _is_hangul(character):
        return _CharacterKind.HANGUL
    # 수치 속성이 함께 있는 `京`도 어휘 한자로 우선 분류한다.
    if is_han(character):
        return _CharacterKind.HAN
    if is_numeric_character(character):
        return _CharacterKind.NUMBER
    if character.isalpha():
        return _CharacterKind.UNSUPPORTED
    if character in _STRUCTURAL_PUNCTUATION:
        return _CharacterKind.STRUCTURAL
    return _CharacterKind.OTHER


def _unprotected_ranges(
    text_length: int,
    protected: tuple[ProtectedRange, ...],
) -> Iterator[tuple[int, int]]:
    """정렬된 보호 구간 사이의 처리 가능한 범위를 순서대로 반환한다."""

    cursor = 0
    for item in protected:
        if cursor < item.start:
            yield cursor, item.start
        cursor = max(cursor, item.end)
    if cursor < text_length:
        yield cursor, text_length


def _latin_candidates(text: str, protected: tuple[ProtectedRange, ...]) -> tuple[Candidate, ...]:
    result: list[Candidate] = []
    for match in _LATIN.finditer(text):
        start, end = match.span()
        if overlaps(protected, start, end):
            continue
        value = match.group()
        reason = "structural_latin_token" if any(not item.isalpha() for item in value) else "latin_candidate"
        result.append(Candidate(start, end, Handler.ENGLISH, reason))
    return tuple(result)


def is_unsupported_script_character(character: str) -> bool:
    """영문 외의 alpha 계열 문자 중 한글/한자 후보로 분류되지 않은 항목."""

    return _character_kind(character) is _CharacterKind.UNSUPPORTED


def _script_candidate_kind(
    character: str,
) -> tuple[Handler, str] | None:
    kind = _character_kind(character)
    if kind is _CharacterKind.HAN:
        return Handler.KOREAN, "ambiguous_han"
    if kind is _CharacterKind.UNSUPPORTED:
        return Handler.KOREAN, "unsupported_script"
    return None


def _script_candidates(
    text: str,
    protected: tuple[ProtectedRange, ...],
) -> tuple[Candidate, ...]:
    """Han·Kana·미지원 문자 후보를 보호되지 않은 구간에서 한 번에 찾는다."""

    result: list[Candidate] = []
    for range_start, range_end in _unprotected_ranges(len(text), protected):
        index = range_start
        while index < range_end:
            kind = _script_candidate_kind(text[index])
            if kind is None:
                index += 1
                continue
            start = index
            index += 1
            while index < range_end and _script_candidate_kind(text[index]) == kind:
                index += 1
            if kind[1] == "ambiguous_han" and text[start:index].isnumeric():
                continue
            result.append(Candidate(start, index, *kind))
    return tuple(result)


def _numeric_ranges(
    text: str,
    protected: tuple[ProtectedRange, ...],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """숫자 범위와 숫자를 포함한 구조 토큰을 한 번의 순회로 찾는다."""

    result: list[tuple[int, int]] = []
    structural: list[tuple[int, int]] = []
    for range_start, range_end in _unprotected_ranges(len(text), protected):
        index = range_start
        while index < range_end:
            kind = _character_kind(text[index])
            if kind not in _STRUCTURE_KINDS:
                index += 1
                continue
            start = index
            number_runs: list[tuple[int, int]] = []
            number_start: int | None = None
            has_foreign = False
            has_explicit_number = False
            has_non_numeric_han = False
            while index < range_end and kind in _STRUCTURE_KINDS:
                numeric_han = (
                    kind is _CharacterKind.HAN
                    and is_numeric_character(text[index])
                )
                if kind is _CharacterKind.NUMBER or numeric_han:
                    if number_start is None:
                        number_start = index
                elif number_start is not None:
                    number_runs.append((number_start, index))
                    number_start = None
                has_explicit_number = has_explicit_number or kind is _CharacterKind.NUMBER
                non_numeric_han = kind is _CharacterKind.HAN and not numeric_han
                has_non_numeric_han = has_non_numeric_han or non_numeric_han
                has_foreign = (
                    has_foreign
                    or kind in _NON_HAN_FOREIGN_KINDS
                    or non_numeric_han
                )
                index += 1
                if index < range_end:
                    kind = _character_kind(text[index])
            if number_start is not None:
                number_runs.append((number_start, index))
            if has_non_numeric_han and not has_explicit_number:
                number_runs.clear()
            if number_runs and has_foreign:
                structural.append((start, index))
                result.append((start, index))
            else:
                result.extend(number_runs)
    return tuple(result), tuple(structural)


def _partition_candidates(
    candidates: tuple[Candidate, ...],
    structural: tuple[tuple[int, int], ...],
) -> tuple[tuple[Candidate, ...], tuple[Candidate, ...], set[tuple[int, int]]]:
    """정렬된 후보와 구조 범위를 선형으로 병합해 겹치는 후보를 분리한다."""

    regular: list[Candidate] = []
    opaque: list[Candidate] = []
    covered: set[tuple[int, int]] = set()
    structural_index = 0
    for candidate in candidates:
        while (
            structural_index < len(structural)
            and structural[structural_index][1] <= candidate.start
        ):
            structural_index += 1
        index = structural_index
        overlaps_structure = False
        while index < len(structural) and structural[index][0] < candidate.end:
            overlaps_structure = True
            covered.add(structural[index])
            index += 1
        (opaque if overlaps_structure else regular).append(candidate)
    return tuple(regular), tuple(opaque), covered


def route_candidates(text: str, protected: tuple[ProtectedRange, ...]) -> RoutingResult:
    """후보 탐지와 숫자 불투명 범위 승격을 한 번의 순서 계약으로 수행한다."""

    latin = _latin_candidates(text, protected)
    scripts = _script_candidates(text, protected)
    numeric, structural = _numeric_ranges(text, protected)
    all_candidates = tuple(
        sorted((*latin, *scripts), key=lambda item: (item.start, item.end, item.handler.value))
    )
    candidates, opaque, covered_structural = _partition_candidates(all_candidates, structural)
    synthetic_opaque = tuple(
        Candidate(
            start,
            end,
            Handler.ENGLISH,
            "numeric_structural_token",
        )
        for start, end in structural
        if (start, end) not in covered_structural
        # `3A`처럼 숫자 뒤에서 Latin 정규식이 시작하지 못한 경우만 보완한다.
        # 한글+숫자는 변환 실패가 아니라 숫자 passthrough이므로 후보를 만들지 않는다.
        and any(character.isascii() and character.isalpha() for character in text[start:end])
    )
    return RoutingResult(
        numeric,
        candidates,
        tuple(sorted((*opaque, *synthetic_opaque), key=lambda item: (item.start, item.end, item.handler.value))),
    )
