"""대규모 사용자 사전을 검증하고 trie로 컴파일한다."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
import unicodedata

from ..errors import InputValidationError


_DEFAULT_MAX_KEY_LENGTH = 256
_DEFAULT_MAX_VALUE_LENGTH = 4_096
_DEFAULT_MAX_ENTRIES = 100_000
_DEFAULT_MAX_TOTAL_SIZE = 16_000_000


@dataclass(frozen=True, slots=True)
class LexiconMatch:
    """원문 좌표와 해소된 사전 값을 함께 반환하는 한 매치다."""

    start: int
    end: int
    key: str
    value: str


class _TrieNode:
    """문자별 자식과 종단 키만 보관하는 가벼운 trie 노드다."""

    __slots__ = ("children", "key")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.key: str | None = None


class CompiledLexicon(Mapping[str, str]):
    """검증·의존성 해소가 끝난 불변 사전과 leftmost-longest matcher다."""

    def __init__(
        self,
        entries: Mapping[str, str],
        *,
        base_normalize: Callable[[str], str],
        max_key_length: int = _DEFAULT_MAX_KEY_LENGTH,
        max_value_length: int = _DEFAULT_MAX_VALUE_LENGTH,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        max_total_size: int = _DEFAULT_MAX_TOTAL_SIZE,
    ) -> None:
        limits = {
            "max_key_length": max_key_length,
            "max_value_length": max_value_length,
            "max_entries": max_entries,
            "max_total_size": max_total_size,
        }
        for name, value in limits.items():
            _validate_positive_limit(name, value)
        if not isinstance(entries, Mapping):
            raise InputValidationError("사전은 문자열 mapping이어야 합니다.")
        if not callable(base_normalize):
            raise InputValidationError("base_normalize는 호출 가능한 함수여야 합니다.")
        if len(entries) > max_entries:
            raise InputValidationError(f"사전 항목 수가 최대 {max_entries}개를 초과했습니다.")

        copied = _copy_and_validate_entries(
            entries,
            max_key_length=max_key_length,
            max_value_length=max_value_length,
            max_entries=max_entries,
            max_total_size=max_total_size,
        )
        normalized_values = _normalize_values(
            copied,
            base_normalize=base_normalize,
            max_value_length=max_value_length,
            max_total_size=max_total_size,
        )
        self._entries: Mapping[str, str] = MappingProxyType(normalized_values)
        self._trie = _build_trie(normalized_values)

    def __getitem__(self, key: str) -> str:
        return self._entries[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def iter_matches(
        self,
        text: str,
        *,
        start: int = 0,
        end: int | None = None,
    ) -> Iterator[LexiconMatch]:
        """지정 범위에서 겹치지 않는 leftmost-longest 매치를 순서대로 낸다."""

        if not isinstance(text, str):
            raise InputValidationError("사전 매칭 입력은 문자열이어야 합니다.")
        if not isinstance(start, int) or isinstance(start, bool):
            raise InputValidationError("사전 매칭 start는 정수여야 합니다.")
        if end is None:
            end = len(text)
        if not isinstance(end, int) or isinstance(end, bool):
            raise InputValidationError("사전 매칭 end는 정수여야 합니다.")
        if not 0 <= start <= end <= len(text):
            raise InputValidationError("사전 매칭 범위가 입력 좌표를 벗어났습니다.")

        for match_start, match_end, key in _iter_key_matches(
            text,
            self._trie,
            start=start,
            end=end,
        ):
            yield LexiconMatch(
                match_start,
                match_end,
                key,
                self._entries[key],
            )


def _validate_positive_limit(name: str, value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputValidationError(f"{name}은 양의 정수여야 합니다.")


def _validate_text_surface(text: str, *, label: str, max_length: int) -> None:
    if not text:
        raise InputValidationError(f"사전 {label}은 비어 있을 수 없습니다.")
    if len(text) > max_length:
        raise InputValidationError(
            f"사전 {label} 길이가 최대 {max_length}자를 초과했습니다."
        )
    for character in text:
        code_point = ord(character)
        if 0xD800 <= code_point <= 0xDFFF:
            raise InputValidationError(f"사전 {label}에 surrogate를 사용할 수 없습니다.")
        if unicodedata.category(character) in {"Cc", "Cf"}:
            raise InputValidationError(f"사전 {label}에 제어 문자를 사용할 수 없습니다.")


def _copy_and_validate_entries(
    entries: Mapping[str, str],
    *,
    max_key_length: int,
    max_value_length: int,
    max_entries: int,
    max_total_size: int,
) -> dict[str, str]:
    copied: dict[str, str] = {}
    raw_total = 0
    normalized_total = 0
    for key, value in entries.items():
        if len(copied) >= max_entries:
            raise InputValidationError(f"사전 항목 수가 최대 {max_entries}개를 초과했습니다.")
        if not isinstance(key, str) or not isinstance(value, str):
            raise InputValidationError("사전 키와 값은 문자열이어야 합니다.")
        _validate_text_surface(key, label="키", max_length=max_key_length)
        _validate_text_surface(value, label="값", max_length=max_value_length)
        raw_total += len(key) + len(value)
        if raw_total > max_total_size:
            raise InputValidationError(
                f"사전 전체 크기가 최대 {max_total_size}자를 초과했습니다."
            )

        normalized_key = unicodedata.normalize("NFC", key)
        normalized_value = unicodedata.normalize("NFC", value)
        _validate_text_surface(
            normalized_key,
            label="NFC 키",
            max_length=max_key_length,
        )
        _validate_text_surface(
            normalized_value,
            label="NFC 값",
            max_length=max_value_length,
        )
        if normalized_key in copied:
            raise InputValidationError("사전 키가 NFC 정규화 후 중복됩니다.")
        normalized_total += len(normalized_key) + len(normalized_value)
        if normalized_total > max_total_size:
            raise InputValidationError(
                f"NFC 사전 전체 크기가 최대 {max_total_size}자를 초과했습니다."
            )
        copied[normalized_key] = normalized_value
    return copied


def _normalize_values(
    entries: Mapping[str, str],
    *,
    base_normalize: Callable[[str], str],
    max_value_length: int,
    max_total_size: int,
) -> dict[str, str]:
    normalized: dict[str, str] = {}
    total_size = 0
    for key, value in entries.items():
        rendered = base_normalize(value)
        if not isinstance(rendered, str):
            raise InputValidationError("base_normalize 결과는 문자열이어야 합니다.")
        rendered = unicodedata.normalize("NFC", rendered)
        _validate_text_surface(
            rendered,
            label="기본 정규화 값",
            max_length=max_value_length,
        )
        total_size += len(key) + len(rendered)
        if total_size > max_total_size:
            raise InputValidationError(
                f"기본 정규화 사전 크기가 최대 {max_total_size}자를 초과했습니다."
            )
        normalized[key] = rendered
    return normalized


def _build_trie(entries: Mapping[str, str]) -> _TrieNode:
    root = _TrieNode()
    for key in entries:
        node = root
        for character in key:
            node = node.children.setdefault(character, _TrieNode())
        node.key = key
    return root


def _is_ascii_alphanumeric(character: str) -> bool:
    return (
        "0" <= character <= "9"
        or "A" <= character <= "Z"
        or "a" <= character <= "z"
    )


def _is_hangul_syllable(character: str) -> bool:
    return "가" <= character <= "힣"


def _boundary_allows_left(text: str, start: int, key: str) -> bool:
    if start == 0:
        return True
    previous = text[start - 1]
    if _is_ascii_alphanumeric(previous):
        return False
    return not (_is_hangul_syllable(key[0]) and _is_hangul_syllable(previous))


def _boundary_allows_right(text: str, end: int, limit: int, key: str) -> bool:
    if end >= limit:
        return True
    following = text[end]
    if _is_ascii_alphanumeric(following):
        return False
    return not (_is_hangul_syllable(key[-1]) and _is_hangul_syllable(following))


def _iter_key_matches(
    text: str,
    trie: _TrieNode,
    *,
    start: int = 0,
    end: int | None = None,
) -> Iterator[tuple[int, int, str]]:
    limit = len(text) if end is None else end
    position = start
    while position < limit:
        node = trie.children.get(text[position])
        if node is None:
            position += 1
            continue

        cursor = position + 1
        best_end = -1
        best_key: str | None = None
        if (
            node.key is not None
            and _boundary_allows_left(text, position, node.key)
            and _boundary_allows_right(text, cursor, limit, node.key)
        ):
            best_end = cursor
            best_key = node.key
        while cursor < limit:
            child = node.children.get(text[cursor])
            if child is None:
                break
            node = child
            cursor += 1
            if (
                node.key is not None
                and _boundary_allows_left(text, position, node.key)
                and _boundary_allows_right(text, cursor, limit, node.key)
            ):
                best_end = cursor
                best_key = node.key

        if best_key is None:
            position += 1
            continue
        yield position, best_end, best_key
        position = best_end
