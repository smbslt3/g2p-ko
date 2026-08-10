"""부동소수점을 쓰지 않는 한국어 수사 렌더러다."""

from __future__ import annotations

from dataclasses import dataclass
import re


class NumberFormatError(ValueError):
    """손실 없이 읽을 수 없는 숫자 표면형이다."""


@dataclass(frozen=True, slots=True)
class ParsedNumber:
    """부호와 모든 원문 숫자를 보존한 파싱 결과다."""

    sign: str
    integer: str
    fraction: str | None


_DIGITS = ("영", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
_PHONE_DIGITS = ("공", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구")
_SMALL_UNITS = ("", "십", "백", "천")
_LARGE_UNITS = ("", "만", "억", "조", "경", "해", "자", "양")
_NATIVE_TENS = {
    10: "열",
    20: "스물",
    30: "서른",
    40: "마흔",
    50: "쉰",
    60: "예순",
    70: "일흔",
    80: "여든",
    90: "아흔",
}
_NATIVE_ONES = {
    1: "한",
    2: "두",
    3: "세",
    4: "네",
    5: "다섯",
    6: "여섯",
    7: "일곱",
    8: "여덟",
    9: "아홉",
}
_NATIVE_STANDALONE_ONES = {
    1: "하나",
    2: "둘",
    3: "셋",
    4: "넷",
    5: "다섯",
    6: "여섯",
    7: "일곱",
    8: "여덟",
    9: "아홉",
}


def parse_number(surface: str) -> ParsedNumber:
    """쉼표 묶음과 소수부를 검증하되 숫자를 정수·실수로 바꾸지 않는다."""

    sign = ""
    if surface[:1] in {"+", "-", "−"}:
        sign, surface = surface[0], surface[1:]
    if not surface:
        raise NumberFormatError("숫자 본문이 없습니다.")

    integer_part, separator, fraction = surface.partition(".")
    if separator and (not fraction or not fraction.isascii() or not fraction.isdigit()):
        raise NumberFormatError("소수부가 올바르지 않습니다.")

    groups = integer_part.split(",")
    if any(not item or not item.isascii() or not item.isdigit() for item in groups):
        raise NumberFormatError("정수부가 올바르지 않습니다.")
    if len(groups) > 1 and (not 1 <= len(groups[0]) <= 3 or any(len(item) != 3 for item in groups[1:])):
        raise NumberFormatError("쉼표의 세 자리 묶음이 올바르지 않습니다.")

    integer = "".join(groups)
    if len(integer) > len(_LARGE_UNITS) * 4:
        raise NumberFormatError("지원하는 큰 수 범위를 넘었습니다.")
    return ParsedNumber(sign, integer, fraction if separator else None)


def has_ambiguous_leading_zero(number: ParsedNumber) -> bool:
    """전화·식별자일 가능성이 있는 여러 자리 선행 0을 찾는다."""

    return len(number.integer) > 1 and number.integer.startswith("0")


def _read_small(chunk: str) -> str:
    result: list[str] = []
    width = len(chunk)
    for index, character in enumerate(chunk):
        digit = ord(character) - ord("0")
        if digit == 0:
            continue
        unit = _SMALL_UNITS[width - index - 1]
        if digit != 1 or not unit:
            result.append(_DIGITS[digit])
        result.append(unit)
    return "".join(result)


def _read_integer(digits: str) -> str:
    stripped = digits.lstrip("0")
    if not stripped:
        return "영"

    chunks: list[str] = []
    end = len(stripped)
    while end > 0:
        start = max(0, end - 4)
        chunks.append(stripped[start:end])
        end = start

    result: list[str] = []
    for unit_index in range(len(chunks) - 1, -1, -1):
        chunk = chunks[unit_index]
        spoken = _read_small(chunk)
        if not spoken:
            continue
        large_unit = _LARGE_UNITS[unit_index]
        if unit_index == 1 and int(chunk) == 1:
            spoken = ""
        result.extend((spoken, large_unit))
    return "".join(result)


def read_sino(surface: str) -> str:
    """정수와 소수를 한자어 수사로 읽고 후행 0도 유지한다."""

    number = parse_number(surface)
    result = _read_integer(number.integer)
    if number.fraction is not None:
        result += " 점 " + " ".join(_DIGITS[ord(item) - ord("0")] for item in number.fraction)
    if number.sign in {"-", "−"}:
        return "마이너스 " + result
    if number.sign == "+":
        return "플러스 " + result
    return result


def read_native_or_sino(surface: str) -> str:
    """1~99는 관형형 고유어 수사로, 나머지는 한자어 수사로 읽는다."""

    number = parse_number(surface)
    if number.sign or number.fraction is not None or has_ambiguous_leading_zero(number):
        return read_sino(surface)
    value = int(number.integer)
    if not 1 <= value <= 99:
        return read_sino(surface)
    if value == 20:
        return "스무"
    tens, ones = divmod(value, 10)
    return (_NATIVE_TENS.get(tens * 10, "") if tens else "") + _NATIVE_ONES.get(ones, "")


def read_native_standalone_or_sino(surface: str) -> str:
    """1~99는 단독형 고유어 수사로, 나머지는 한자어 수사로 읽는다.

    관형형(한·두·세)은 단위 없이 단독으로 설 수 없으므로 범위의 왼쪽처럼
    수사만 남는 자리는 하나·둘·셋 단독형을 쓴다.
    """

    number = parse_number(surface)
    if number.sign or number.fraction is not None or has_ambiguous_leading_zero(number):
        return read_sino(surface)
    value = int(number.integer)
    if not 1 <= value <= 99:
        return read_sino(surface)
    tens, ones = divmod(value, 10)
    return (_NATIVE_TENS.get(tens * 10, "") if tens else "") + _NATIVE_STANDALONE_ONES.get(ones, "")


def read_phone_digits(surface: str) -> str:
    """전화번호의 낱자 경계를 보존하고 0을 공으로 읽는다."""

    return " ".join(_PHONE_DIGITS[ord(item) - ord("0")] for item in surface)


def read_digitwise(surface: str) -> str:
    """숫자를 수량이 아니라 낱자리로 하나씩 띄어 읽는다.

    번호·코드처럼 자릿값이 없는 숫자를 위한 읽기다. 전화번호와 같은 0=공
    관습과 낱자 경계를 쓴다(`6630` → `육 육 삼 공`).
    """

    return " ".join(_PHONE_DIGITS[ord(item) - ord("0")] for item in surface)


def read_history_digits(surface: str) -> str:
    """역사일의 첫 묶음은 수사로, 뒤 묶음은 낱자리로 이어 읽는다."""

    first, *rest = re.split(r"[.·]", surface)
    suffix = "".join(
        _DIGITS[ord(character) - ord("0")]
        for group in rest
        for character in group
    )
    return read_sino(first) + suffix
