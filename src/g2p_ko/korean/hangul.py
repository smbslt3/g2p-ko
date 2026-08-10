"""현대 한글 음절을 음운 규칙용 토큰으로 분해하고 다시 조립한다.

이 모듈은 입력 문자열을 전역 치환하지 않는다. 각 음절 토큰은 자신이 유래한
원문 span을 보관하므로, 이후 음운 규칙이 바꾼 소리를 원문 위치로 설명할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..model import SourceSpan


CHOSEONG = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)
JUNGSEONG = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)
JONGSEONG = (
    None, "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

_CHOSEONG_INDEX = {value: index for index, value in enumerate(CHOSEONG)}
_JUNGSEONG_INDEX = {value: index for index, value in enumerate(JUNGSEONG)}
_JONGSEONG_INDEX = {value: index for index, value in enumerate(JONGSEONG) if value}


def is_modern_syllable(value: str) -> bool:
    """현대 한글 완성형 한 음절인지 판정한다."""

    return len(value) == 1 and "가" <= value <= "힣"


@dataclass(frozen=True, slots=True)
class HangulToken:
    """분해된 한글 음절과 그 음절의 원문 근거다."""

    onset: str
    vowel: str
    coda: str | None
    source_spans: tuple[SourceSpan, ...]

    def __post_init__(self) -> None:
        if self.onset not in _CHOSEONG_INDEX:
            raise ValueError(f"지원하지 않는 초성입니다: {self.onset}")
        if self.vowel not in _JUNGSEONG_INDEX:
            raise ValueError(f"지원하지 않는 중성입니다: {self.vowel}")
        if self.coda is not None and self.coda not in _JONGSEONG_INDEX:
            raise ValueError(f"지원하지 않는 종성입니다: {self.coda}")
        if not self.source_spans:
            raise ValueError("HangulToken에는 적어도 하나의 원문 span이 필요합니다.")

    @property
    def text(self) -> str:
        """현재 초성·중성·종성으로 조립한 현대 한글 음절이다."""

        return compose_syllable(self.onset, self.vowel, self.coda)

    def changed(
        self,
        *,
        onset: str | None = None,
        vowel: str | None = None,
        coda: str | None | object = ...,  # ``None``도 유효한 종성 값이다.
    ) -> "HangulToken":
        """선택한 음운 성분만 바꾼 새 토큰을 반환한다."""

        values: dict[str, object] = {}
        if onset is not None:
            values["onset"] = onset
        if vowel is not None:
            values["vowel"] = vowel
        if coda is not ...:
            values["coda"] = coda
        return replace(self, **values)


def decompose_syllable(value: str, source_spans: tuple[SourceSpan, ...]) -> HangulToken:
    """완성형 한글 한 음절을 ``HangulToken``으로 분해한다."""

    if not is_modern_syllable(value):
        raise ValueError(f"현대 한글 완성형 한 음절이 아닙니다: {value!r}")
    offset = ord(value) - 0xAC00
    onset_index = offset // (21 * 28)
    vowel_index = (offset // 28) % 21
    coda_index = offset % 28
    return HangulToken(
        CHOSEONG[onset_index],
        JUNGSEONG[vowel_index],
        JONGSEONG[coda_index],
        source_spans,
    )


def compose_syllable(onset: str, vowel: str, coda: str | None = None) -> str:
    """호환 자모 초·중·종성을 현대 한글 완성형으로 조립한다."""

    try:
        onset_index = _CHOSEONG_INDEX[onset]
        vowel_index = _JUNGSEONG_INDEX[vowel]
        coda_index = 0 if coda is None else _JONGSEONG_INDEX[coda]
    except KeyError as error:
        raise ValueError("지원하지 않는 현대 한글 자모 조합입니다.") from error
    return chr(0xAC00 + (onset_index * 21 + vowel_index) * 28 + coda_index)


def character_source_spans(
    text: str,
    source_spans: tuple[SourceSpan, ...],
) -> tuple[tuple[SourceSpan, ...], ...]:
    """출력 문자마다 가능한 가장 좁은 원문 span을 배정한다.

    일반 한국어 segment는 원문과 길이가 같고 하나의 연속 span으로 들어온다.
    이 경우 한 글자 단위로 span을 복원한다. 어휘 치환처럼 길이가 달라진 조각은
    모든 글자가 기존 근거 span을 공유하게 하여 거짓 정밀도를 만들지 않는다.
    """

    if not text:
        return ()
    if len(source_spans) == len(text):
        return tuple((span,) for span in source_spans)
    if len(source_spans) == 1:
        span = source_spans[0]
        if span.surface == text and span.end - span.start == len(text):
            return tuple(
                (
                    SourceSpan(
                        span.start + index,
                        span.start + index + 1,
                        character,
                        span.source_length,
                    ),
                )
                for index, character in enumerate(text)
            )
    return tuple(source_spans for _ in text)
