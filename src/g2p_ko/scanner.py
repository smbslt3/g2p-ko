"""보호해야 할 구조 토큰을 선형 스캔으로 찾아 재처리를 차단한다."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True, slots=True, order=True)
class ProtectedRange:
    """정규화 문자열의 보호 구간과 검출 종류다."""

    start: int
    end: int
    kind: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("ProtectedRange는 비어 있지 않은 범위여야 합니다.")


_URL = re.compile(r"https?://[^\s<>\[\]{}]+", re.IGNORECASE)
_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?![A-Za-z0-9_.-])"
)
_WINDOWS_SEGMENT = r"[A-Za-z0-9_.$()\-]+(?: [A-Za-z0-9_.$()\-]+)*"
_WINDOWS_PATH = re.compile(
    rf"(?<![A-Za-z0-9_])[A-Za-z]:\\(?:"
    rf"(?:{_WINDOWS_SEGMENT}\\)+{_WINDOWS_SEGMENT}"
    r"|[^\s<>\[\]{}()]+)"
)
_RELATIVE_PATH_SEGMENT = r"[A-Za-z0-9_.$()\-]+"
_WINDOWS_UNC_PATH = re.compile(
    rf"(?<![A-Za-z0-9_\\])\\\\{_RELATIVE_PATH_SEGMENT}"
    rf"\\{_RELATIVE_PATH_SEGMENT}(?:\\{_RELATIVE_PATH_SEGMENT})*"
    rf"\\?(?![A-Za-z0-9_.$()\\-])"
)
_WINDOWS_EXPLICIT_RELATIVE_PATH = re.compile(
    rf"(?<![\w.\\-])(?:\.{{1,2}}\\)+{_RELATIVE_PATH_SEGMENT}"
    rf"(?:\\{_RELATIVE_PATH_SEGMENT})*\\?(?![A-Za-z0-9_.$()\\-])"
)
_WINDOWS_RELATIVE_PATH_CANDIDATE = re.compile(
    rf"(?<![A-Za-z0-9_.$()\\-]){_RELATIVE_PATH_SEGMENT}(?:\\{_RELATIVE_PATH_SEGMENT})+"
    rf"\\?(?![A-Za-z0-9_.$()\\-])"
)
_POSIX_RELATIVE_PATH = re.compile(
    rf"(?<![\w./-])(?:\.\.?/)+{_RELATIVE_PATH_SEGMENT}"
    rf"(?:/{_RELATIVE_PATH_SEGMENT})*/?"
    rf"(?![A-Za-z0-9_.$()/-])"
)
# `./`과 `../`는 위 정규식이 시작점부터 소유하게 해 일부 경로 보호를 막는다.
_POSIX_PATH = re.compile(
    r"(?<![A-Za-z0-9_/\.\u3300-\u33FF])/(?:[^\s<>\[\]{}()])+"
    r"(?![^\s<>\[\]{}()]|[=<>≤≥])"
)
_TRAILING = frozenset(".,!?;:\"'”’")
_KOREAN_PARTICLES = (
    "에서만",
    "에게만",
    "한테만",
    "으로만",
    "부터만",
    "까지만",
    "에서는",
    "에게는",
    "한테는",
    "으로부터",
    "에게서",
    "한테서",
    "으로는",
    "으로",
    "에게",
    "한테",
    "부터",
    "까지",
    "에서",
    "에는",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "의",
    "와",
    "과",
    "도",
    "만",
    "로",
)


def overlaps(ranges: tuple[ProtectedRange, ...] | list[ProtectedRange], start: int, end: int) -> bool:
    """정렬된 보호 범위와 주어진 반열린 구간의 겹침을 이진 탐색한다."""

    if not ranges:
        return False
    index = bisect_left(ranges, start, key=lambda item: item.end)
    return index < len(ranges) and ranges[index].start < end


def _trim_trailing(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in _TRAILING:
        end -= 1
    while True:
        for particle in _KOREAN_PARTICLES:
            particle_start = end - len(particle)
            if particle_start <= start or not text.endswith(particle, start, end):
                continue
            previous = text[particle_start - 1]
            if previous.isascii() and (previous.isalnum() or previous in "._/-\\)"):
                end = particle_start
                break
        else:
            return end


def _inline_code_ranges(text: str) -> tuple[ProtectedRange, ...]:
    """같은 길이의 backtick run으로 닫힌 인라인 코드를 보호한다."""

    result: list[ProtectedRange] = []
    index = 0
    while index < len(text):
        if text[index] != "`":
            index += 1
            continue
        start = index
        while index < len(text) and text[index] == "`":
            index += 1
        delimiter = text[start:index]
        closed = False
        while index < len(text):
            if text[index] == "\\" and index + 1 < len(text):
                index += 2
                continue
            if text.startswith(delimiter, index):
                end = index + len(delimiter)
                result.append(ProtectedRange(start, end, "inline_code"))
                index = end
                closed = True
                break
            if text[index] in "\r\n":
                break
            index += 1
        if not closed and index >= len(text):
            break
    return tuple(result)


def _trim_unbalanced_url_parentheses(text: str, start: int, end: int) -> int:
    """문장 닫는 괄호는 제외하고 URL 내부의 균형 잡힌 괄호는 남긴다."""

    balance = text.count("(", start, end) - text.count(")", start, end)
    while end > start and text[end - 1] == ")" and balance < 0:
        end -= 1
        balance += 1
    return end


def _is_strong_windows_relative_path(surface: str) -> bool:
    r"""일반적인 `word\word` escape와 상대 경로 후보를 보수적으로 구분한다."""

    components = surface.rstrip("\\").split("\\")
    if len(components) >= 3:
        return True
    return any(
        "." in component
        or "$" in component
        or any(character.isdigit() for character in component)
        for component in components
    )


def _safe_protected_end(text: str, start: int, end: int) -> int:
    """보호 구간이 숨기면 안 되는 제어·형식·비공개 문자 앞에서 끊는다."""

    for index in range(start, end):
        character = text[index]
        if character not in "\t\r\n" and unicodedata.category(character) in {
            "Cc",
            "Cf",
            "Co",
            "Cn",
            "Cs",
        }:
            return index
    return end


def scan_protected(text: str) -> tuple[ProtectedRange, ...]:
    """URL·이메일·Windows/POSIX 경로·인라인 코드를 겹치지 않게 반환한다."""

    found: list[ProtectedRange] = []

    def add_range(kind: str, start: int, end: int) -> None:
        end = _trim_trailing(text, start, end)
        if kind == "url":
            end = _trim_unbalanced_url_parentheses(text, start, end)
        end = _safe_protected_end(text, start, end)
        if start >= end:
            return
        index = bisect_left(found, start, key=lambda item: item.end)
        if index < len(found) and found[index].start < end:
            return
        found.insert(index, ProtectedRange(start, end, kind))

    def add(kind: str, match: re.Match[str]) -> None:
        add_range(kind, *match.span())

    for item in _inline_code_ranges(text):
        add_range(item.kind, item.start, item.end)

    if "://" in text:
        for match in _URL.finditer(text):
            add("url", match)
    if "@" in text:
        for match in _EMAIL.finditer(text):
            add("email", match)
    if "\\" in text:
        for match in _WINDOWS_PATH.finditer(text):
            add("windows_path", match)
        for match in _WINDOWS_UNC_PATH.finditer(text):
            add("windows_path", match)
        for match in _WINDOWS_EXPLICIT_RELATIVE_PATH.finditer(text):
            add("windows_path", match)
        for match in _WINDOWS_RELATIVE_PATH_CANDIDATE.finditer(text):
            if _is_strong_windows_relative_path(match.group(0)):
                add("windows_path", match)
    if "/" in text:
        for match in _POSIX_RELATIVE_PATH.finditer(text):
            add("posix_path", match)
        for match in _POSIX_PATH.finditer(text):
            add("posix_path", match)
    return tuple(found)
