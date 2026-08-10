"""우선순위 토큰 스캔으로 written form을 한국어 발화형으로 바꾼다."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import date
import re
import unicodedata

from ..analyzer import KiwiAnalyzer
from ..english.classify import spell_ascii_letters
from ..errors import ConversionPolicyError, InputValidationError, InternalInvariantError
from ..model import Diagnostic, Handler, Rewrite, RewriteStage, Severity, SourceSpan
from ..scanner import scan_protected
from ..unicode import NormalizedText, normalize_nfc, normalize_nfc_text
from .conventional_english import CONVENTIONAL_READINGS, INITIALISMS
from .model import NormalizationResult
from .lexicon import CompiledLexicon
from .number_context import NumberReadingClassifier, NumberReadingRun
from .numbers import (
    NumberFormatError,
    ParsedNumber,
    has_ambiguous_leading_zero,
    parse_number,
    read_digitwise,
    read_history_digits,
    read_native_or_sino,
    read_native_standalone_or_sino,
    read_phone_digits,
    read_sino,
)


_NATIVE_COUNTERS = frozenset(
    {
        "개",
        "명",
        "사람",
        "마리",
        "번째",
        "시",
        "배",
        "가구",
        "게임",
        "건",
        "세트",
        "가지",
        "개비",
        "잔",
        "번",
        "장",
        "병",
        "권",
        "벌",
        "곳",
        "시간",
        "척",
        "차례",
        "바퀴",
        "경기",
        "골",
        "살",
        "달",
        "글자",
        "알",
        "켤레",
        "송이",
        "컵",
        "쪽",
        "채",
        "자루",
        "그루",
        "포기",
    }
)
_UNIT_READINGS = {
    "km/h": "킬로미터 퍼 아워",
    "㎞/h": "킬로미터 퍼 아워",
    "m/s": "미터 퍼 세컨드",
    "km/s": "킬로미터 퍼 세컨드",
    "MB/s": "메가바이트 퍼 세컨드",
    "GB/s": "기가바이트 퍼 세컨드",
    "W/kg": "와트 퍼 킬로그램",
    "km/l": "킬로미터 퍼 리터",
    "kg": "킬로그램",
    "g/mL": "그램 퍼 밀리리터",
    "g/ml": "그램 퍼 밀리리터",
    "mg": "밀리그램",
    "g": "그램",
    "t": "톤",
    "mL": "밀리리터",
    "ml": "밀리리터",
    "L": "리터",
    "l": "리터",
    "cm": "센티미터",
    "mm": "밀리미터",
    "km": "킬로미터",
    "m": "미터",
    "mi": "마일",
    "mile": "마일",
    "ha": "헥타르",
    "ℓ": "리터",
    "nm": "나노미터",
    "μm": "마이크로미터",
    "µm": "마이크로미터",
    "m²": "제곱미터",
    "cm²": "제곱센티미터",
    "km²": "제곱킬로미터",
    "㎢": "제곱킬로미터",
    "m³": "세제곱미터",
    "cm³": "세제곱센티미터",
    "km³": "세제곱킬로미터",
    "㎡": "제곱미터",
    "㎥": "세제곱미터",
    "cc": "씨씨",
    "kcal": "킬로칼로리",
    "cal": "칼로리",
    "kJ": "킬로줄",
    "mmHg": "수은주밀리미터",
    "Hz": "헤르츠",
    "kHz": "킬로헤르츠",
    "MHz": "메가헤르츠",
    "GHz": "기가헤르츠",
    "W": "와트",
    "kW": "킬로와트",
    "MW": "메가와트",
    "Wh": "와트시",
    "kWh": "킬로와트시",
    "GWh": "기가와트시",
    "V": "볼트",
    "mV": "밀리볼트",
    "mA": "밀리암페어",
    "mAh": "밀리암페어시",
    "dB": "데시벨",
    "bpm": "비피엠",
    "rpm": "알피엠",
    "TEU": "티이유",
    "FEU": "에프이유",
    "ppm": "피피엠",
    "dpi": "디피아이",
    "KB": "킬로바이트",
    "MB": "메가바이트",
    "GB": "기가바이트",
    "TB": "테라바이트",
    "°C": "도씨",
    "°c": "도씨",
    "℃": "도씨",
    "㎏": "킬로그램",
    "㎎": "밀리그램",
    "㎖": "밀리리터",
    "㎞": "킬로미터",
    "㎝": "센티미터",
    "㎜": "밀리미터",
    "㎾h": "킬로와트시",
    "㎿": "메가와트",
}

# 이동 속도만 한국어 관용 어순으로 전치한다. 같은 슬래시 표기라도 전송률이나
# 단위 질량당 값은 `_UNIT_READINGS`의 "퍼" 읽기를 그대로 사용한다.
_SPEED_UNIT_READINGS = {
    "km/h": ("시속", "킬로미터"),
    "㎞/h": ("시속", "킬로미터"),
    "m/s": ("초속", "미터"),
    "km/s": ("초속", "킬로미터"),
}

# 문자 자체가 단위 전용이라 일반 영문 약어와 충돌하지 않는 표기만 독립적으로
# 읽는다. ASCII 단위는 숫자·수량사·`당` 문맥이 없으면 보존한다.
_STANDALONE_UNIT_SYMBOLS = frozenset(
    {
        "ℓ",
        "°C",
        "°c",
        "℃",
        "㎡",
        "㎥",
        "㎏",
        "㎎",
        "㎖",
        "㎞",
        "㎝",
        "㎜",
        "㎢",
        "㎾h",
        "㎿",
    }
)
# 이동 속도 표기와 단위 전용 유니코드 기호는 식별자와 겹치지 않으므로
# 숫자를 언제나 한자어 수사로 읽는다. 통화도 금액이라는 의미가 확정되므로
# 자릿수 읽기로 바꾸지 않는다. 문맥 판별기에 보낼 이유가 없다.
_CONTEXT_INDEPENDENT_SINO_UNITS = (
    frozenset(_SPEED_UNIT_READINGS)
    | _STANDALONE_UNIT_SYMBOLS
    | frozenset(
        {
            "호실",
            "동",
            "단계",
            "종",
            "원",
            "달러",
            "유로",
            "엔",
            "파운드",
            "위안",
        }
    )
)
_SINO_UNITS = frozenset(
    {
        "초",
        "분",
        "일",
        "주",
        "개월",
        "월",
        "년",
        "점",
        "포인트",
        "퍼센트",
        "레벨",
        "점수",
        "등급",
        "등",
        "원",
        "달러",
        "유로",
        "엔",
        "파운드",
        "회",
        "차",
        "기",
        "호",
        "페이지",
        "도",
        "층",
        "학년",
        "학기",
        "학점",
        "교시",
        "세기",
        "라운드",
        "세",
        "차원",
        "위안",
        "호실",
        "동",
        "단계",
        "종",
    }
)
# 이미 발화형으로 쓰는 단위의 한글 표기는 원문에서도 같은 단위로 인식한다.
# 인식 어휘를 발화 어휘와 같게 묶어 새 읽기를 도입하지 않으며, 여러 낱말로
# 읽는 단위(`km/h` → 킬로미터 퍼 아워)와 다른 뜻의 낱말과 표기가 겹치는
# `톤`(색·음의 단계는 두 톤처럼 고유어로 센다)은 제외한다.
_HANGUL_SPELLED_UNITS = frozenset(
    reading for reading in _UNIT_READINGS.values() if " " not in reading
) - {"톤"}
_ALL_UNITS = tuple(
    sorted(
        _NATIVE_COUNTERS | _SINO_UNITS | _HANGUL_SPELLED_UNITS | set(_UNIT_READINGS),
        key=lambda item: (-len(item), item),
    )
)
_UNIT_PATTERN = "(?:" + "|".join(re.escape(item) for item in _ALL_UNITS) + ")"
_WRITTEN_UNIT_PATTERN = "(?:" + "|".join(
    re.escape(item)
    for item in sorted(_UNIT_READINGS, key=lambda item: (-len(item), item))
) + ")"
_WRITTEN_UNIT_INITIAL_PATTERN = "[" + re.escape(
    "".join(sorted({item[0] for item in _UNIT_READINGS}))
) + "]"
_STANDALONE_UNIT_SYMBOL_PATTERN = "(?:" + "|".join(
    re.escape(item)
    for item in sorted(_STANDALONE_UNIT_SYMBOLS, key=lambda item: (-len(item), item))
) + ")"
_HANGUL_UNITS = tuple(
    item
    for item in _ALL_UNITS
    if item in _NATIVE_COUNTERS or item in _SINO_UNITS or item in _HANGUL_SPELLED_UNITS
)
_HANGUL_UNIT_PATTERN = "(?:" + "|".join(re.escape(item) for item in _HANGUL_UNITS) + ")"
_NUMBER_PATTERN = r"[+\-−]?[0-9](?:[0-9,]*[0-9])?(?:\.[0-9]+)?"
_UNSIGNED_NUMBER_PATTERN = r"[0-9](?:[0-9,]*[0-9])?(?:\.[0-9]+)?"
_DECIMAL_PATTERN = r"[+\-−]?[0-9](?:[0-9,]*[0-9])?\.[0-9]+"

_SPOKEN_ACRONYMS = frozenset(INITIALISMS | set(CONVENTIONAL_READINGS))
_ACRONYM_PATTERN = (
    "(?:"
    + "|".join(
        re.escape(item)
        for item in sorted(_SPOKEN_ACRONYMS, key=lambda item: (-len(item), item))
    )
    + ")"
)
_JAMO_NAMES = {
    "ㄱ": "기역",
    "ㄲ": "쌍기역",
    "ㄳ": "기역시옷",
    "ㄴ": "니은",
    "ㄵ": "니은지읒",
    "ㄶ": "니은히읗",
    "ㄷ": "디귿",
    "ㄸ": "쌍디귿",
    "ㄹ": "리을",
    "ㄺ": "리을기역",
    "ㄻ": "리을미음",
    "ㄼ": "리을비읍",
    "ㄽ": "리을시옷",
    "ㄾ": "리을티읕",
    "ㄿ": "리을피읖",
    "ㅀ": "리을히읗",
    "ㅁ": "미음",
    "ㅂ": "비읍",
    "ㅃ": "쌍비읍",
    "ㅄ": "비읍시옷",
    "ㅅ": "시옷",
    "ㅆ": "쌍시옷",
    "ㅇ": "이응",
    "ㅈ": "지읒",
    "ㅉ": "쌍지읒",
    "ㅊ": "치읓",
    "ㅋ": "키읔",
    "ㅌ": "티읕",
    "ㅍ": "피읖",
    "ㅎ": "히읗",
}
_CURRENCIES = {"$": "달러", "₩": "원", "€": "유로", "£": "파운드", "¥": "엔"}
_ISO_CURRENCIES = {
    "KRW": "원",
    "USD": "달러",
    "EUR": "유로",
    "JPY": "엔",
    "CNY": "위안",
    "GBP": "파운드",
}
# 통화 금액의 세 자리 소수부는 유럽식 천 단위 구분점(5.000원 = 오천 원)일 수
# 있어 소수로 단정해 읽지 않는다.
_CURRENCY_UNIT_WORDS = frozenset(_CURRENCIES.values()) | frozenset(_ISO_CURRENCIES.values())


def _may_be_thousands_separator(parsed: ParsedNumber) -> bool:
    return parsed.fraction is not None and len(parsed.fraction) == 3


def _expected_match(pattern: re.Pattern[str], token: str) -> re.Match[str]:
    """스캐너가 이미 확정한 토큰의 내부 정규식 불일치를 즉시 드러낸다."""

    match = pattern.fullmatch(token)
    if match is None:
        raise InternalInvariantError(f"스캐너와 렌더러의 토큰 규칙이 다릅니다: {token!r}")
    return match


def _is_plain_korean_text(text: str) -> bool:
    """변환·진단 후보가 전혀 없는 완성형 한글 문장을 빠르게 확인한다."""

    return all(
        "가" <= character <= "힣"
        or character.isspace()
        or (
            unicodedata.category(character).startswith("P")
            and character not in "%％"
        )
        for character in text
    )


# 역사 날짜는 날짜와 사건 이름의 조합 자체가 어휘화된 경우에만 낱자리로 읽는다.
# 날짜와 접미사를 따로 나열하면 실제 사건이 아닌 교차 조합까지 허용하게 된다.
_HISTORY_EVENTS = (
    ("3.1", "절"),
    ("3.1", "운동"),
    ("4.3", "사건"),
    ("4.3", "항쟁"),
    ("4.19", "혁명"),
    ("5.18", "민주화운동"),
    ("6.25", "전쟁"),
    ("8.15", "광복절"),
    ("10.26", "사건"),
)
_HISTORY_PATTERN = (
    "(?:"
    + "|".join(
        date_surface.replace(".", "[.·]") + r"[ \t]*" + re.escape(event)
        for date_surface, event in _HISTORY_EVENTS
    )
    + ")"
)
_IDENTIFIER_NUMBER_PREFIXES = (
    "휴대폰",
    "시리얼",
    "전화",
    "비밀",
    "제품",
    "카드",
    "차량",
    "인증",
    "계좌",
    "주문",
    "식별",
    "우편",
    "팩스",
)
_STANDALONE_IDENTIFIER_LABELS = (
    "연락처",
    "PIN",
    "계좌",
    "버스",
    "번호",
    "좌석",
    "코드",
    "호실",
)
_IDENTIFIER_LABEL_PATTERN = (
    "(?:"
    + "|".join(
        re.escape(item) + r"[ \t]*번호"
        for item in sorted(_IDENTIFIER_NUMBER_PREFIXES, key=lambda item: (-len(item), item))
    )
    + "|"
    + "|".join(
        re.escape(item)
        for item in sorted(_STANDALONE_IDENTIFIER_LABELS, key=lambda item: (-len(item), item))
    )
    + ")"
)
_IDENTIFIER_BODY_PATTERN = (
    r"(?:[0-9]{2,4}[가-힣][0-9]{4}|[0-9]+[A-Za-z]|[0-9]+(?:[ \t-][0-9]+)*)"
)
_KNOWN_EXTENSIONS = "(?:json|ya?ml|toml|csv|tsv|txt|md|py|js|ts|wav|mp3|flac)"
# 숫자 뒤에 붙어도 읽기를 바꾸지 않는 조사·서술격 어미의 닫힌 목록이다.
# 여기 없는 꼬리는 더 긴 낱말일 수 있어 토큰 전체를 보존한다.
_KOREAN_SUFFIXES = frozenset(
    {
        "가",
        "간",
        "과",
        "까지",
        "까지는",
        "께",
        "다",
        "도",
        "라고",
        "라도",
        "로",
        "를",
        "마다",
        "만",
        "밖에",
        "보다",
        "부터",
        "부터는",
        "당",
        "에",
        "에는",
        "에게",
        "에서",
        "와",
        "은",
        "을",
        "으로",
        "의",
        "이",
        "이다",
        "이었다",
        "이라",
        "이라고",
        "이라도",
        "이나",
        "이고",
        "인데",
        "이며",
        "이면",
        "이지만",
        "입니다",
        "였다",
        "처럼",
        "는",
        "쯤",
        "씩",
        "야",
        "이야",
        "랑",
        "이랑",
        "한테",
        "한테서",
        "조차",
        "마저",
        "뿐",
        "로서",
        "으로서",
        "로써",
        "으로써",
        "라서",
        "이라서",
        "짜리",
        "가량",
    }
)

_KOREAN_SUFFIX_PATTERN = "(?:" + "|".join(
    re.escape(item) for item in sorted(_KOREAN_SUFFIXES, key=lambda item: (-len(item), item))
) + ")"
_HSPACE = r"[ \t]*"
_ASCII_IDENTIFIER_LEFT_BOUNDARY = r"(?<![A-Za-z0-9_])(?<![A-Za-z0-9_]\.)"
_ASCII_IDENTIFIER_RIGHT_BOUNDARY = r"(?![A-Za-z0-9_])(?!\.[A-Za-z0-9_])"
_DOTTED_IDENTIFIER_RIGHT_GUARD = r"(?!\.[A-Za-z0-9_])"
_RANGE_SEPARATOR = r"[~〜～∼–—]"
_RANGE_UNIT_PATTERN = rf"(?:{_UNIT_PATTERN}|%[pP]|%|％|만원|억원|조원)"
# 구조화 날짜·시간 패턴을 구성하는 단위가 범위 양쪽에 서면 `3월 1일~3월 5일`의
# 가운데를 잘라 갈 수 있으므로 양쪽 단위 범위에서는 제외한다.
_RANGE_BOTH_EXCLUDED_UNITS = frozenset({"년", "월", "일", "시", "분", "초"})
_RANGE_BOTH_UNIT_PATTERN = (
    "(?:"
    + "|".join(
        re.escape(item)
        for item in sorted(
            set(_ALL_UNITS) - _RANGE_BOTH_EXCLUDED_UNITS,
            key=lambda item: (-len(item), item),
        )
    )
    + "|%[pP]|%|％|만원|억원|조원)"
)
_ISO_CURRENCY_PATTERN = "(?:" + "|".join(_ISO_CURRENCIES) + ")"

# 대표번호는 구분자 표기만 전화로 읽는다. 무구분 8자리는 1,500만 같은
# 일반 수와 구별할 수 없어 일반 숫자 규칙에 맡긴다.
_PHONE_PATTERN = (
    r"(?:\+82[-. ]?10[-. ]?[0-9]{3,4}[-. ][0-9]{4}"
    r"|0[0-9]{1,3}[-. ][0-9]{3,4}[-. ][0-9]{4}"
    r"|1[568][0-9]{2}[- ][0-9]{4})"
)
# 전화 뒤에 네 자리 묶음이 하나 더 이어지면 카드번호류 사슬의 앞부분일 수
# 있으므로 전화로 잘라 읽지 않고 malformed_phone 보존 경로로 넘긴다.
_CARD_CHAIN_GUARD = r"(?![-. ][0-9]{4}(?![0-9]))"
_PHONE_EXPRESSION = rf"{_PHONE_PATTERN}{_CARD_CHAIN_GUARD}(?:{_KOREAN_SUFFIX_PATTERN})?"
_DATE_SUFFIX_PATTERN = rf"(?:\.?{_KOREAN_SUFFIX_PATTERN})?"
_DATE_PATTERN = rf"""
    [0-9]{{4}}(?:
        {_HSPACE}-{_HSPACE}[0-9]{{1,2}}{_HSPACE}-{_HSPACE}[0-9]{{1,2}}
      | {_HSPACE}/{_HSPACE}[0-9]{{1,2}}{_HSPACE}/{_HSPACE}[0-9]{{1,2}}
      | {_HSPACE}\.{_HSPACE}[0-9]{{1,2}}{_HSPACE}\.{_HSPACE}[0-9]{{1,2}}
    ){_DATE_SUFFIX_PATTERN}
"""
_KOREAN_DATE_PATTERN = rf"""
    [0-9]{{4}}{_HSPACE}년{_HSPACE}[0-9]{{1,2}}{_HSPACE}월
    {_HSPACE}[0-9]{{1,2}}{_HSPACE}일(?:{_KOREAN_SUFFIX_PATTERN})?
"""
_MONTH_DAY_PATTERN = rf"""
    [0-9]{{1,2}}{_HSPACE}월{_HSPACE}[0-9]{{1,2}}{_HSPACE}일
    (?:{_KOREAN_SUFFIX_PATTERN})?
"""
_YEAR_MONTH_PATTERN = rf"""
    [0-9]{{4}}{_HSPACE}년(?:{_HSPACE}[0-9]{{1,2}}{_HSPACE}월)?
    (?:{_KOREAN_SUFFIX_PATTERN})?
"""
# 날짜 범위 읽기는 아직 지원하지 않으므로 공백을 포함한 사슬 전체를 한 토큰으로
# 보존한다. 그렇지 않으면 malformed_structured가 공백에서 끊겨 뒤쪽 날짜가
# 부분 변환된다.
_DATE_RANGE_OPERAND_PATTERN = rf"""
    (?:{_KOREAN_DATE_PATTERN}|{_MONTH_DAY_PATTERN}|{_YEAR_MONTH_PATTERN})
"""
_COLON_TIME_PATTERN = rf"""
    (?:(?:오전|오후){_HSPACE})?[0-9]{{1,2}}:[0-9]{{2}}(?::[0-9]{{2}})?
    (?:{_KOREAN_SUFFIX_PATTERN})?
"""
_KOREAN_TIME_PATTERN = rf"""
    (?:(?:오전|오후){_HSPACE})?[0-9]{{1,2}}{_HSPACE}시
    (?:{_HSPACE}[0-9]{{1,2}}{_HSPACE}분(?:{_HSPACE}[0-9]{{1,2}}{_HSPACE}초)?)?
    (?:{_KOREAN_SUFFIX_PATTERN})?
"""
_TIME_RANGE_OPERAND_PATTERN = rf"""
    (?!(?:(?:오전|오후){_HSPACE})?[0-9]{{1,2}}{_HSPACE}시간)
    (?:{_COLON_TIME_PATTERN}|{_KOREAN_TIME_PATTERN})
"""
_TIME_RANGE_PATTERN = rf"""
    (?=(?:(?:오전|오후){_HSPACE})?[0-9]{{1,2}}(?:{_HSPACE}시|:))
    (?P<left>{_TIME_RANGE_OPERAND_PATTERN}){_HSPACE}{_RANGE_SEPARATOR}{_HSPACE}
    (?P<right>{_TIME_RANGE_OPERAND_PATTERN})
"""
# 단일 패스 스캐너에서는 거대한 조사 목록을 매 위치마다 대조하지 않는다.
# 여기서는 범위 전체만 잡고, 정확한 조사·시각 유효성은 렌더러가 위 문법으로 검증한다.
_TIME_RANGE_SCAN_OPERAND_PATTERN = rf"""
    (?:(?:오전|오후){_HSPACE})?[0-9]{{1,2}}
    (?: :[0-9]{{2}}(?::[0-9]{{2}})?
      | {_HSPACE}시(?:{_HSPACE}[0-9]{{1,2}}{_HSPACE}분
        (?:{_HSPACE}[0-9]{{1,2}}{_HSPACE}초)?)?)
    [가-힣]*
"""
_TIME_RANGE_SCAN_PATTERN = rf"""
    {_TIME_RANGE_SCAN_OPERAND_PATTERN}{_HSPACE}{_RANGE_SEPARATOR}{_HSPACE}
    {_TIME_RANGE_SCAN_OPERAND_PATTERN}
"""
_MONTH_PATTERN = rf"[0-9]{{1,2}}{_HSPACE}월(?:{_KOREAN_SUFFIX_PATTERN})?"
_MIXED_FRACTION_PATTERN = rf"""
    {_UNSIGNED_NUMBER_PATTERN}[ \t]+{_UNSIGNED_NUMBER_PATTERN}/{_UNSIGNED_NUMBER_PATTERN}
    (?:{_HSPACE}{_WRITTEN_UNIT_PATTERN})?(?:{_KOREAN_SUFFIX_PATTERN})?
"""
_CURRENCY_WORD_PATTERN = "(?:" + "|".join(sorted(_CURRENCY_UNIT_WORDS, key=len, reverse=True)) + ")"
_CURRENCY_PATTERN = rf"[\$₩€£¥]{_HSPACE}{_NUMBER_PATTERN}(?:{_HSPACE}{_CURRENCY_WORD_PATTERN})?"
_ISO_CURRENCY_EXPRESSION = rf"""
    (?:{_ISO_CURRENCY_PATTERN}{_HSPACE}{_NUMBER_PATTERN}
      | {_NUMBER_PATTERN}{_HSPACE}{_ISO_CURRENCY_PATTERN})
    (?:{_HSPACE}{_CURRENCY_WORD_PATTERN})?
"""
_SCALE_POWERS = {"십": 1, "백": 2, "천": 3, "만": 4, "억": 8, "조": 12}
_SCALE_PATTERN = "(?:" + "|".join(_SCALE_POWERS) + ")"
_SCALED_CURRENCY_PATTERN = rf"{_NUMBER_PATTERN}{_HSPACE}{_SCALE_PATTERN}{_HSPACE}원"
_MATH_OPERATOR_PATTERN = r"(?:/|\*|\+|<|>|≤|≥|÷|=)"
_NON_ASCII_NUMERIC_RANGES = (
    r"\u00BC-\u00BE\u2070-\u209F\u2150-\u218F\u2460-\u24FF"
    r"\uFF10-\uFF19\U0001D7CE-\U0001D7FF"
)
_NUMERIC_GUARD_CHARACTERS = (
    rf"0-9A-Za-z_.,:/+−^~～∼–—×±÷=*<>≤≥%％²³$₩€£¥°℃㎡㎥μµ{_NON_ASCII_NUMERIC_RANGES}\-"
)
_NUMERIC_GUARD_SIGNAL = (
    rf"A-Za-z_.,:/+−^~～∼–—×±÷=*<>≤≥%％²³$₩€£¥°℃㎡㎥μµ{_NON_ASCII_NUMERIC_RANGES}\-"
)
_NUMERIC_SINGLE_PREFIX = r".,:/+\-"
_NUMERIC_STRONG_PREFIX = (
    rf"−^~～∼–—×±÷=*<>≤≥%％²³$₩€£¥°℃㎡㎥μµ{_NON_ASCII_NUMERIC_RANGES}"
)
_STRUCTURED_TOKEN_CONTINUATION = (
    rf"A-Za-z0-9_가-힣:/+−^~～∼–—×±÷=*<>≤≥%％²³$₩€£¥°℃㎡㎥μµ"
    rf"{_NON_ASCII_NUMERIC_RANGES}\-"
)
# 수량 뒤의 조사·어미는 허용하되 식별자·수식 사슬은 기존 보존 규칙에
# 넘긴다. 구조화 토큰 경계와 같은 집합을 쓰고 한글 범위만 제외한다.
_QUANTITY_CONTINUATION = _STRUCTURED_TOKEN_CONTINUATION.replace("가-힣", "")
_FRACTION_UNIT_PATTERN = rf"""
    (?=[0-9][0-9,.]*/[0-9])
    {_UNSIGNED_NUMBER_PATTERN}/{_UNSIGNED_NUMBER_PATTERN}
    {_HSPACE}(?={_WRITTEN_UNIT_INITIAL_PATTERN}){_WRITTEN_UNIT_PATTERN}
"""
_NUMBER_UNIT_PATTERN = rf"""
    (?=[+\-−]?[0-9][0-9,.]*{_HSPACE}
        (?:{_SCALE_PATTERN}{_HSPACE})?(?:여{_HSPACE})?
        {_WRITTEN_UNIT_INITIAL_PATTERN})
    {_NUMBER_PATTERN}(?:{_HSPACE}{_SCALE_PATTERN})?
    {_HSPACE}(?:여{_HSPACE})?
    (?={_WRITTEN_UNIT_INITIAL_PATTERN}){_WRITTEN_UNIT_PATTERN}
"""
_QUANTITY_UNIT_PATTERN = rf"(?:{_FRACTION_UNIT_PATTERN}|{_NUMBER_UNIT_PATTERN})"
_INDEFINITE_UNIT_PATTERN = rf"(?:몇|수){_HSPACE}{_WRITTEN_UNIT_PATTERN}"
_UNIT_PER_PATTERN = rf"{_WRITTEN_UNIT_PATTERN}당(?:{_KOREAN_SUFFIX_PATTERN})?"
_STANDALONE_UNIT_SYMBOL_EXPRESSION = (
    rf"{_STANDALONE_UNIT_SYMBOL_PATTERN}(?:{_KOREAN_SUFFIX_PATTERN})?"
)
_SEMANTIC_CLUSTER_PATTERN = rf"""
    (?:[+\-]{{2,}}(?:[{_NUMERIC_GUARD_CHARACTERS}]|[^\x00-\x7F\s])*
      | [{_NUMERIC_SINGLE_PREFIX}][{_NUMERIC_STRONG_PREFIX}]
        (?:[{_NUMERIC_GUARD_CHARACTERS}]|[^\x00-\x7F\s])*
      | [{_NUMERIC_SINGLE_PREFIX}][0-9](?:[{_NUMERIC_GUARD_CHARACTERS}]|[^\x00-\x7F\s])*
      | [{_NUMERIC_STRONG_PREFIX}](?:[{_NUMERIC_GUARD_CHARACTERS}]|[^\x00-\x7F\s])*
      | [0-9]+(?:[{_NUMERIC_GUARD_SIGNAL}가-힣]|[^\x00-\x7F\s])
        (?:[{_NUMERIC_GUARD_CHARACTERS}]|[^\x00-\x7F\s])*)
"""
_MIXED_SUFFIX_PATTERN = rf"""
    (?:[\$₩€£¥]{_HSPACE})?[0-9][{_NUMERIC_GUARD_CHARACTERS}가-힣]*
    [가-힣]+[A-Za-z][{_NUMERIC_GUARD_CHARACTERS}가-힣]*
"""
_MIXED_CURRENCY_PATTERN = rf"""
    (?:
        [+\-−]?[$₩€£¥]{_HSPACE}{_NUMBER_PATTERN}{_HSPACE}{_ISO_CURRENCY_PATTERN}
      | {_ISO_CURRENCY_PATTERN}{_HSPACE}[$₩€£¥]{_HSPACE}{_NUMBER_PATTERN}
      | [+\-−][$₩€£¥]{_HSPACE}{_NUMBER_PATTERN}
    )
"""
_SEMANTIC_OPERAND_PATTERN = rf"""
    (?:{_DATE_PATTERN}|{_COLON_TIME_PATTERN}|{_CURRENCY_PATTERN}
      | {_NUMBER_PATTERN}{_HSPACE}(?:{_UNIT_PATTERN}|%[pP]|%|％))
"""
_SEMANTIC_PAIR_PATTERN = rf"""
    {_SEMANTIC_OPERAND_PATTERN}{_HSPACE}[-−–—~～∼/+*=<>≤≥±÷×]{_HSPACE}
    {_SEMANTIC_OPERAND_PATTERN}
"""
_MALFORMED_PHONE_PATTERN = rf"""
    (?:\+[0-9]{{1,3}}|0[0-9]{{1,3}})(?:[-. ][0-9]+){{2,}}
"""
# 긴급 번호는 조사가 바로 붙거나(112에) 뒤에 다른 낱말이 이어지지 않을 때만
# 낱자리로 읽는다. `112 페이지` 같은 수량 문맥은 일반 숫자 규칙에 맡긴다.
_EMERGENCY_PATTERN = rf"(?:112|119)(?:{_KOREAN_SUFFIX_PATTERN}|(?![ \t]+[가-힣A-Za-z]))"
_CONTEXTUAL_SHORT_DATE_PATTERN = rf"""
    (?:날짜|생일)(?:은|는)?{_HSPACE}[0-9]{{1,2}}[./][0-9]{{1,2}}
    (?:{_KOREAN_SUFFIX_PATTERN})?
"""
_MALFORMED_STRUCTURED_PATTERN = rf"""
    (?:{_PHONE_EXPRESSION}|{_COLON_TIME_PATTERN}|{_KOREAN_TIME_PATTERN}
      | {_DATE_PATTERN}|{_KOREAN_DATE_PATTERN}|{_MONTH_DAY_PATTERN})
    (?:[{_STRUCTURED_TOKEN_CONTINUATION}]+
      | \.[0-9](?:[{_NUMERIC_GUARD_CHARACTERS}]|[^\x00-\x7F\s])*)
"""
_UNSUPPORTED_RANGE_PATTERN = rf"""
    {_NUMBER_PATTERN}{_HSPACE}{_RANGE_SEPARATOR}{_HSPACE}{_NUMBER_PATTERN}
    {_HSPACE}[A-Za-z%％°℃㎡㎥μµ가-힣²³]+
"""
_UNSUPPORTED_COMPARISON_PATTERN = rf"""
    /?[A-Za-z][A-Za-z0-9_.-]*{_HSPACE}[<>≤≥=]{_HSPACE}{_NUMBER_PATTERN}
    (?:{_HSPACE}(?:{_UNIT_PATTERN}|%[pP]|%|％))?
"""
_NON_ASCII_NUMERIC_CLUSTER_PATTERN = rf"""
    (?:[0-9]+[^\x00-\x7F\s]
      | [^\x00-\x7F\s][0-9])
    (?:[{_NUMERIC_GUARD_CHARACTERS}]|[^\x00-\x7F\s])*
"""
_MIXED_ACRONYM_NUMERIC_CONTEXT_PATTERN = rf"""
    (?:{_ACRONYM_PATTERN}[가-힣]+[0-9]
        (?:[{_NUMERIC_GUARD_CHARACTERS}]|[^\x00-\x7F\s])*
      | {_ACRONYM_PATTERN}[{_NUMERIC_SINGLE_PREFIX}{_NUMERIC_STRONG_PREFIX}]+[0-9]
        [{_NUMERIC_GUARD_CHARACTERS}]*[ㄱ-ㅎㅏ-ㅣ])
"""

# 기본 스캐너와 공백 없는 결합 토큰 재분류기가 공유하는 실제 문법이다.
# 삽입 순서가 더 구체적인 형식에서 일반 숫자 순서로 이어지는 우선순위다.
_SEMANTIC_TOKEN_PATTERNS = {
    "phone": _PHONE_EXPRESSION,
    "time": rf"(?:{_COLON_TIME_PATTERN}|{_KOREAN_TIME_PATTERN})",
    "korean_date": _KOREAN_DATE_PATTERN,
    "date": _DATE_PATTERN,
    "history": _HISTORY_PATTERN,
    "currency": _CURRENCY_PATTERN,
    "scaled_currency": _SCALED_CURRENCY_PATTERN,
    "range": (
        rf"{_NUMBER_PATTERN}{_HSPACE}{_RANGE_SEPARATOR}{_HSPACE}{_NUMBER_PATTERN}"
        rf"(?:{_HSPACE}{_RANGE_UNIT_PATTERN}(?:{_KOREAN_SUFFIX_PATTERN})?)?"
    ),
    "ambiguous_hyphen": (
        rf"{_UNSIGNED_NUMBER_PATTERN}{_HSPACE}-{_HSPACE}{_UNSIGNED_NUMBER_PATTERN}"
        rf"(?:{_HSPACE}{_UNIT_PATTERN})?"
    ),
    "exponent": rf"{_UNSIGNED_NUMBER_PATTERN}\^(?:[-−])?[0-9]+",
    "fraction_unit": _FRACTION_UNIT_PATTERN,
    "fraction": rf"{_UNSIGNED_NUMBER_PATTERN}/{_UNSIGNED_NUMBER_PATTERN}",
    "percentage": rf"{_NUMBER_PATTERN}{_HSPACE}(?:%[pP]|%|％)",
    "month": _MONTH_PATTERN,
    "number_hangul_unit": (
        rf"{_NUMBER_PATTERN}{_HSPACE}{_HANGUL_UNIT_PATTERN}[가-힣]*"
    ),
    "number_unit": _NUMBER_UNIT_PATTERN,
    "decimal": _DECIMAL_PATTERN,
    "number": _NUMBER_PATTERN,
}

_BASE_TOKEN_PATTERN = rf"""
    (?P<contextual_short_date>(?<![A-Za-z0-9가-힣]){_CONTEXTUAL_SHORT_DATE_PATTERN}(?![A-Za-z0-9가-힣/]|\.[0-9]))
  | (?P<contextual_identifier>(?<![A-Za-z0-9가-힣]){_IDENTIFIER_LABEL_PATTERN}{_HSPACE}(?:[:#]{_HSPACE})?{_IDENTIFIER_BODY_PATTERN}(?![A-Za-z0-9]))
  | (?P<unsupported_comparison>(?<![A-Za-z0-9]){_UNSUPPORTED_COMPARISON_PATTERN}(?![{_STRUCTURED_TOKEN_CONTINUATION}]|[.,][0-9]))
  | (?P<date_range>(?<![A-Za-z0-9가-힣]){_DATE_RANGE_OPERAND_PATTERN}{_HSPACE}{_RANGE_SEPARATOR}{_HSPACE}{_DATE_RANGE_OPERAND_PATTERN}(?![{_STRUCTURED_TOKEN_CONTINUATION}]))
  | (?P<mixed_fraction>(?<![A-Za-z0-9]){_MIXED_FRACTION_PATTERN}(?![{_STRUCTURED_TOKEN_CONTINUATION}]))
  | (?P<range_both>{_ASCII_IDENTIFIER_LEFT_BOUNDARY}{_NUMBER_PATTERN}{_HSPACE}{_RANGE_BOTH_UNIT_PATTERN}{_HSPACE}{_RANGE_SEPARATOR}{_HSPACE}{_NUMBER_PATTERN}{_HSPACE}{_RANGE_BOTH_UNIT_PATTERN}(?:{_KOREAN_SUFFIX_PATTERN})?(?![{_STRUCTURED_TOKEN_CONTINUATION}]|\.[0-9]){_DOTTED_IDENTIFIER_RIGHT_GUARD})
  | (?P<time_range>(?<![A-Za-z0-9가-힣]){_TIME_RANGE_SCAN_PATTERN}(?![{_STRUCTURED_TOKEN_CONTINUATION}]))
  | (?P<unsupported_semantic_pair>(?<![A-Za-z0-9]){_SEMANTIC_PAIR_PATTERN}(?![A-Za-z0-9가-힣._/+−-]))
  | (?P<mixed_currency>(?<![A-Za-z0-9]){_MIXED_CURRENCY_PATTERN}(?![{_STRUCTURED_TOKEN_CONTINUATION}]|[.,][0-9]))
  | (?P<mixed_numeric_context>{_ASCII_IDENTIFIER_LEFT_BOUNDARY}{_MIXED_ACRONYM_NUMERIC_CONTEXT_PATTERN}(?![A-Za-z0-9_가-힣]){_DOTTED_IDENTIFIER_RIGHT_GUARD})
  | (?P<version>(?<![A-Za-z0-9])[vV][0-9]+(?:\.[0-9]+){{1,}}(?![A-Za-z0-9]))
  | (?P<filename>(?<![A-Za-z0-9_.-])[A-Za-z0-9_-]+\.{_KNOWN_EXTENSIONS}(?![A-Za-z0-9_.-]))
  | (?P<phone>(?<![A-Za-z0-9._/+\-]){_SEMANTIC_TOKEN_PATTERNS["phone"]}(?![{_STRUCTURED_TOKEN_CONTINUATION}]|\.[0-9]))
  | (?P<malformed_phone>(?<![A-Za-z0-9._/+\-]){_MALFORMED_PHONE_PATTERN}[가-힣]*(?![{_STRUCTURED_TOKEN_CONTINUATION}]|\.[0-9]))
  | (?P<emergency>(?<![A-Za-z0-9]){_EMERGENCY_PATTERN}(?![{_STRUCTURED_TOKEN_CONTINUATION}]))
  | (?P<ipv4>(?<![0-9._/-])(?:[0-9]{{1,3}}\.){{3}}[0-9]{{1,3}}(?![{_STRUCTURED_TOKEN_CONTINUATION}]|\.[0-9]))
  | (?P<time>(?<![A-Za-z0-9가-힣]){_SEMANTIC_TOKEN_PATTERNS["time"]}(?![{_STRUCTURED_TOKEN_CONTINUATION}]|\.[0-9]))
  | (?P<korean_date>(?<![A-Za-z0-9가-힣]){_SEMANTIC_TOKEN_PATTERNS["korean_date"]}(?![{_STRUCTURED_TOKEN_CONTINUATION}]))
  | (?P<date>(?<![A-Za-z0-9]){_SEMANTIC_TOKEN_PATTERNS["date"]}(?![{_STRUCTURED_TOKEN_CONTINUATION}]|\.[0-9]))
  | (?P<month_day>(?<![A-Za-z0-9가-힣]){_MONTH_DAY_PATTERN}(?![{_STRUCTURED_TOKEN_CONTINUATION}]))
  | (?P<year_month>(?<![A-Za-z0-9가-힣]){_YEAR_MONTH_PATTERN}(?![{_STRUCTURED_TOKEN_CONTINUATION}]))
  | (?P<malformed_structured>(?<![A-Za-z0-9]){_MALFORMED_STRUCTURED_PATTERN}(?![{_STRUCTURED_TOKEN_CONTINUATION}.]))
  | (?P<ordinal>(?<![A-Za-z0-9가-힣])제{_HSPACE}{_NUMBER_PATTERN}{_HSPACE}{_HANGUL_UNIT_PATTERN}[가-힣]*)
  | (?P<numbered_item>(?<![A-Za-z0-9]){_NUMBER_PATTERN}{_HSPACE}번(?=[ \t]+(?:출구|버스|문제|문항|노선|좌석|채널)))
  | (?P<history>(?<![0-9]){_SEMANTIC_TOKEN_PATTERNS["history"]})
  | (?P<currency>(?<![A-Za-z0-9]){_SEMANTIC_TOKEN_PATTERNS["currency"]}(?![{_NUMERIC_GUARD_CHARACTERS}가-힣]))
  | (?P<iso_currency>(?<![A-Za-z0-9]){_ISO_CURRENCY_EXPRESSION}(?![{_NUMERIC_GUARD_CHARACTERS}가-힣]))
  | (?P<scaled_currency>(?<![A-Za-z0-9]){_SEMANTIC_TOKEN_PATTERNS["scaled_currency"]}(?:{_KOREAN_SUFFIX_PATTERN})?(?![{_NUMERIC_GUARD_CHARACTERS}가-힣]))
  | (?P<unsupported_math>(?<![A-Za-z0-9]){_NUMBER_PATTERN}(?:(?:[ \t]+{_MATH_OPERATOR_PATTERN}{_HSPACE}|{_HSPACE}{_MATH_OPERATOR_PATTERN}[ \t]+)|[ \t]+[-−][ \t]+){_NUMBER_PATTERN}(?![A-Za-z0-9._/+−-]))
  | (?P<scaled_range>(?<![A-Za-z0-9]){_NUMBER_PATTERN}{_HSPACE}{_SCALE_PATTERN}{_HSPACE}{_RANGE_SEPARATOR}{_HSPACE}{_NUMBER_PATTERN}{_HSPACE}{_SCALE_PATTERN}{_HSPACE}원(?:{_KOREAN_SUFFIX_PATTERN})?(?![A-Za-z0-9가-힣]))
  | (?P<range>{_ASCII_IDENTIFIER_LEFT_BOUNDARY}{_SEMANTIC_TOKEN_PATTERNS["range"]}(?![{_STRUCTURED_TOKEN_CONTINUATION}]|\.[0-9]){_DOTTED_IDENTIFIER_RIGHT_GUARD})
  | (?P<unsupported_range>(?<![A-Za-z0-9]){_UNSUPPORTED_RANGE_PATTERN}(?![{_STRUCTURED_TOKEN_CONTINUATION}]|[.,][0-9]))
  | (?P<standalone_unit_symbol>{_ASCII_IDENTIFIER_LEFT_BOUNDARY}{_STANDALONE_UNIT_SYMBOL_EXPRESSION}(?![A-Za-z0-9_가-힣]){_DOTTED_IDENTIFIER_RIGHT_GUARD})
  | (?P<quantity_unit>{_ASCII_IDENTIFIER_LEFT_BOUNDARY}{_QUANTITY_UNIT_PATTERN}(?![{_QUANTITY_CONTINUATION}]|[.,][0-9]){_DOTTED_IDENTIFIER_RIGHT_GUARD})
  | (?P<mixed_numeric_suffix>(?<![A-Za-z0-9]){_MIXED_SUFFIX_PATTERN}(?![A-Za-z0-9가-힣]))
  | (?P<semantic_cluster>(?<![A-Za-z0-9]){_SEMANTIC_CLUSTER_PATTERN}(?![A-Za-z0-9가-힣]))
  | (?P<non_ascii_decimal>(?<![0-9])(?=[^0-9])\d+(?:{_HSPACE}{_UNIT_PATTERN})?(?![0-9]))
  | (?P<exponent>(?<![A-Za-z0-9]){_SEMANTIC_TOKEN_PATTERNS["exponent"]}(?![A-Za-z0-9가-힣]))
  | (?P<fraction>(?<![A-Za-z0-9]){_SEMANTIC_TOKEN_PATTERNS["fraction"]}(?![A-Za-z0-9./]))
  | (?P<percentage>(?<![A-Za-z0-9]){_SEMANTIC_TOKEN_PATTERNS["percentage"]}(?![A-Za-z0-9%]))
  | (?P<number_hangul_unit>{_ASCII_IDENTIFIER_LEFT_BOUNDARY}{_SEMANTIC_TOKEN_PATTERNS["number_hangul_unit"]}(?![A-Za-z0-9_]){_DOTTED_IDENTIFIER_RIGHT_GUARD})
  | (?P<number_suffix>(?<![A-Za-z0-9]){_NUMBER_PATTERN}{_KOREAN_SUFFIX_PATTERN}(?![A-Za-z0-9가-힣]))
  | (?P<decimal>(?<![A-Za-z0-9]){_SEMANTIC_TOKEN_PATTERNS["decimal"]}(?![A-Za-z0-9.,:/+^~～∼–—×±÷=*<>≤≥%％²³\-]))
  | (?P<unsupported_prefixed_numeric>(?<![A-Za-z0-9])[\$₩€£¥]{_HSPACE}[0-9][{_NUMERIC_GUARD_CHARACTERS}]*[가-힣]*)
  | (?P<unsupported_numeric>(?<![A-Za-z0-9])(?:
        [+\-−]?\.[0-9]+
      | {_NUMBER_PATTERN}[eE][+\-]?[0-9]+
      | {_NUMBER_PATTERN}{_HSPACE}[\$₩€£¥]
      | [-−]{_UNSIGNED_NUMBER_PATTERN}/{_UNSIGNED_NUMBER_PATTERN}
      | {_UNSIGNED_NUMBER_PATTERN}\^(?:[-−])?[0-9]+{_HSPACE}{_UNIT_PATTERN}
      | {_NUMBER_PATTERN}{_HSPACE}{_UNIT_PATTERN}/[A-Za-z]+
      | {_NUMBER_PATTERN}{_HSPACE}{_UNIT_PATTERN}[²³]
      | {_NUMBER_PATTERN}{_HSPACE}[×±]{_HSPACE}{_NUMBER_PATTERN}(?:{_HSPACE}{_UNIT_PATTERN})?
    )(?![A-Za-z0-9]))
  | (?P<ambiguous_hyphen>(?<![0-9]){_SEMANTIC_TOKEN_PATTERNS["ambiguous_hyphen"]}(?![A-Za-z0-9._/+−-]))
  | (?P<number_punctuation>(?<![A-Za-z0-9]){_NUMBER_PATTERN}[.,](?![0-9.,]))
  | (?P<numeric_guard>(?<![A-Za-z0-9])(?=[0-9][{_NUMERIC_GUARD_CHARACTERS}]*[{_NUMERIC_GUARD_SIGNAL}])[0-9][{_NUMERIC_GUARD_CHARACTERS}]*[가-힣]*)
  | (?P<non_ascii_numeric_cluster>(?<![A-Za-z0-9]){_NON_ASCII_NUMERIC_CLUSTER_PATTERN}(?![A-Za-z0-9]))
  | (?P<ambiguous_numeric>(?<![A-Za-z0-9])[0-9]{{11,}}(?![0-9]))
  | (?P<ambiguous_number_word>(?<![A-Za-z0-9]){_NUMBER_PATTERN}[가-힣]+(?![A-Za-z0-9가-힣]))
  | (?P<ambiguous_latin_hyphen>(?<![A-Za-z])[A-Za-z]+[-‑][A-Za-z]+(?![A-Za-z]))
  | (?P<ambiguous_identifier>(?<![A-Za-z0-9])(?:[A-Za-z][A-Za-z0-9_.-]*[0-9][A-Za-z0-9_.-]*|[+\-.]?[0-9]+[A-Za-z][A-Za-z0-9_.-]*)(?![A-Za-z0-9]))
  | (?P<number>(?<![A-Za-z0-9]){_SEMANTIC_TOKEN_PATTERNS["number"]}(?![A-Za-z0-9가-힣]))
  | (?P<indefinite_unit>(?<![A-Za-z0-9_가-힣])(?<![A-Za-z0-9_]\.){_INDEFINITE_UNIT_PATTERN}(?![A-Za-z0-9_]){_DOTTED_IDENTIFIER_RIGHT_GUARD})
  | (?P<unit_per>{_ASCII_IDENTIFIER_LEFT_BOUNDARY}{_UNIT_PER_PATTERN}(?![A-Za-z0-9_가-힣]){_DOTTED_IDENTIFIER_RIGHT_GUARD})
  | (?P<acronym>{_ASCII_IDENTIFIER_LEFT_BOUNDARY}{_ACRONYM_PATTERN}{_ASCII_IDENTIFIER_RIGHT_BOUNDARY})
  | (?P<jamo_cluster>[ㄱ-ㅎ]{{2,}}[A-Za-z0-9]*)
  | (?P<jamo>[ㄱ-ㅎ])
  | (?P<han>[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]+)
  | (?P<latin>(?<![A-Za-z])[A-Za-z]+(?:['’][A-Za-z]+)?(?![A-Za-z]))
  | (?P<unsupported_symbol>[\$₩€£¥%％×±²³⁴=÷^≤≥*<>∼])
"""
_BASE_TOKEN_RE = re.compile(_BASE_TOKEN_PATTERN, re.VERBOSE)

# 공백 없는 숫자 결합 토큰도 위와 같은 문법과 우선순위를 사용한다.
_SEMANTIC_TOKEN_RE = re.compile(
    "(?:"
    + "|".join(
        rf"(?P<{kind}>{pattern})"
        for kind, pattern in _SEMANTIC_TOKEN_PATTERNS.items()
    )
    + ")",
    re.VERBOSE,
)
_PHONE_RE = re.compile(
    rf"(?P<number>{_PHONE_PATTERN})(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_EMERGENCY_RE = re.compile(
    rf"(?P<number>112|119)(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_DECIMAL_RE = re.compile(_DECIMAL_PATTERN)
_RANGE_RE = re.compile(
    rf"(?P<left>{_NUMBER_PATTERN}){_HSPACE}{_RANGE_SEPARATOR}{_HSPACE}"
    rf"(?P<right>{_NUMBER_PATTERN})(?:{_HSPACE}(?P<unit>{_RANGE_UNIT_PATTERN})"
    rf"(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?)?"
)
_RANGE_BOTH_RE = re.compile(
    rf"(?P<left>{_NUMBER_PATTERN}){_HSPACE}(?P<left_unit>{_RANGE_BOTH_UNIT_PATTERN})"
    rf"{_HSPACE}{_RANGE_SEPARATOR}{_HSPACE}"
    rf"(?P<right>{_NUMBER_PATTERN}){_HSPACE}(?P<unit>{_RANGE_BOTH_UNIT_PATTERN})"
    rf"(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_SCALED_RANGE_RE = re.compile(
    rf"(?P<left>{_NUMBER_PATTERN}){_HSPACE}(?P<left_scale>{_SCALE_PATTERN}){_HSPACE}{_RANGE_SEPARATOR}{_HSPACE}"
    rf"(?P<right>{_NUMBER_PATTERN}){_HSPACE}(?P<right_scale>{_SCALE_PATTERN}){_HSPACE}원"
    rf"(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_QUANTITY_UNIT_RE = re.compile(
    rf"(?:"
    rf"(?P<numerator>{_UNSIGNED_NUMBER_PATTERN})/"
    rf"(?P<denominator>{_UNSIGNED_NUMBER_PATTERN})"
    rf"|(?P<number>{_NUMBER_PATTERN})(?:{_HSPACE}(?P<scale>{_SCALE_PATTERN}))?"
    rf"{_HSPACE}(?P<approximation>여)?"
    rf"){_HSPACE}(?P<unit>{_WRITTEN_UNIT_PATTERN})"
)
_INDEFINITE_UNIT_RE = re.compile(
    rf"(?P<quantifier>몇|수){_HSPACE}(?P<unit>{_WRITTEN_UNIT_PATTERN})"
)
_UNIT_PER_RE = re.compile(
    rf"(?P<unit>{_WRITTEN_UNIT_PATTERN})당(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_STANDALONE_UNIT_SYMBOL_RE = re.compile(
    rf"(?P<unit>{_STANDALONE_UNIT_SYMBOL_PATTERN})"
    rf"(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_NUMBER_HANGUL_UNIT_RE = re.compile(
    rf"(?P<number>{_NUMBER_PATTERN}){_HSPACE}(?P<word>{_HANGUL_UNIT_PATTERN}[가-힣]*)"
)
_FRACTION_RE = re.compile(
    rf"(?P<numerator>{_UNSIGNED_NUMBER_PATTERN})/"
    rf"(?P<denominator>{_UNSIGNED_NUMBER_PATTERN})"
)
_PERCENT_RE = re.compile(rf"(?P<number>{_NUMBER_PATTERN}){_HSPACE}(?P<unit>%[pP]|%|％)")
_DATE_RE = re.compile(
    rf"(?P<year>[0-9]{{4}}){_HSPACE}(?P<separator>[-./]){_HSPACE}"
    rf"(?P<month>[0-9]{{1,2}}){_HSPACE}(?P=separator){_HSPACE}"
    rf"(?P<day>[0-9]{{1,2}})(?:\.?(?P<suffix>{_KOREAN_SUFFIX_PATTERN}))?"
)
_MONTH_DAY_RE = re.compile(
    rf"(?P<month>[0-9]{{1,2}}){_HSPACE}월{_HSPACE}"
    rf"(?P<day>[0-9]{{1,2}}){_HSPACE}일(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_MONTH_RE = re.compile(
    rf"(?P<month>[0-9]{{1,2}}){_HSPACE}월(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_YEAR_MONTH_RE = re.compile(
    rf"(?P<year>[0-9]{{4}}){_HSPACE}년"
    rf"(?:{_HSPACE}(?P<month>[0-9]{{1,2}}){_HSPACE}월)?"
    rf"(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_KOREAN_DATE_RE = re.compile(
    rf"(?P<year>[0-9]{{4}}){_HSPACE}년{_HSPACE}(?P<month>[0-9]{{1,2}})"
    rf"{_HSPACE}월{_HSPACE}(?P<day>[0-9]{{1,2}}){_HSPACE}일"
    rf"(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_COLON_TIME_RE = re.compile(
    rf"(?:(?P<period>오전|오후){_HSPACE})?(?P<hour>[0-9]{{1,2}}):"
    rf"(?P<minute>[0-9]{{2}})(?::(?P<second>[0-9]{{2}}))?"
    rf"(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_KOREAN_TIME_RE = re.compile(
    rf"(?:(?P<period>오전|오후){_HSPACE})?(?P<hour>[0-9]{{1,2}}){_HSPACE}시"
    rf"(?:{_HSPACE}(?P<minute>[0-9]{{1,2}}){_HSPACE}분"
    rf"(?:{_HSPACE}(?P<second>[0-9]{{1,2}}){_HSPACE}초)?)?"
    rf"(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_TIME_RANGE_RE = re.compile(_TIME_RANGE_PATTERN, re.VERBOSE)
_CURRENCY_RE = re.compile(
    rf"(?P<symbol>[\$₩€£¥]){_HSPACE}(?P<number>{_NUMBER_PATTERN})"
    rf"(?:{_HSPACE}(?P<spoken_unit>{_CURRENCY_WORD_PATTERN}))?"
)
_ISO_CURRENCY_RE = re.compile(
    rf"(?:(?P<prefix>{_ISO_CURRENCY_PATTERN}){_HSPACE}(?P<prefix_number>{_NUMBER_PATTERN})"
    rf"|(?P<suffix_number>{_NUMBER_PATTERN}){_HSPACE}(?P<suffix>{_ISO_CURRENCY_PATTERN}))"
    rf"(?:{_HSPACE}(?P<spoken_unit>{_CURRENCY_WORD_PATTERN}))?"
)
_SCALED_CURRENCY_RE = re.compile(
    rf"(?P<number>{_NUMBER_PATTERN}){_HSPACE}(?P<scale>{_SCALE_PATTERN}){_HSPACE}원"
    rf"(?P<suffix>{_KOREAN_SUFFIX_PATTERN})?"
)
_ORDINAL_RE = re.compile(rf"제{_HSPACE}(?P<body>{_NUMBER_PATTERN}{_HSPACE}{_HANGUL_UNIT_PATTERN}[가-힣]*)")
_NUMBER_SUFFIX_RE = re.compile(
    rf"(?P<number>{_NUMBER_PATTERN})(?P<suffix>{_KOREAN_SUFFIX_PATTERN})"
)
_EXPONENT_RE = re.compile(
    rf"(?P<base>{_UNSIGNED_NUMBER_PATTERN})\^(?P<exponent>[-−]?[0-9]+)"
)
_POLICIES = frozenset({"preserve", "warn", "error"})

_UNSUPPORTED_NUMERIC_KINDS = frozenset(
    {
        "unsupported_numeric",
        "unsupported_prefixed_numeric",
        "unsupported_math",
        "unsupported_semantic_pair",
        "malformed_phone",
        "malformed_structured",
        "mixed_currency",
        "unsupported_range",
        "unsupported_comparison",
        "non_ascii_numeric_cluster",
    }
)
_AMBIGUOUS_TOKEN_MESSAGES = {
    "contextual_identifier": (
        "ambiguous_numeric",
        "문맥상 식별번호인 숫자는 수량으로 읽지 않습니다.",
    ),
    "contextual_short_date": (
        "ambiguous_numeric",
        "연도가 없는 날짜 표기의 월·일 순서를 확정할 수 없습니다.",
    ),
    "non_ascii_decimal": (
        "ambiguous_numeric",
        "ASCII가 아닌 숫자의 읽기를 확정할 수 없습니다.",
    ),
    "mixed_numeric_suffix": (
        "ambiguous_identifier",
        "숫자 뒤에 문자 체계가 혼합된 토큰은 전체 원문을 보존합니다.",
    ),
    "mixed_numeric_context": (
        "ambiguous_identifier",
        "숫자 뒤에 문자 체계가 혼합된 토큰은 전체 원문을 보존합니다.",
    ),
    "ambiguous_hyphen": (
        "ambiguous_hyphen",
        "하이픈이 범위인지 식별자인지 확정할 수 없습니다.",
    ),
    "ambiguous_latin_hyphen": (
        "unconverted_latin",
        "하이픈 영문 표현의 관용 읽기를 확정할 수 없습니다.",
    ),
    "ambiguous_identifier": (
        "ambiguous_identifier",
        "영숫자 식별자의 의도한 읽기를 확정할 수 없습니다.",
    ),
    "ambiguous_numeric": (
        "ambiguous_numeric",
        "긴 숫자가 식별번호인지 수량인지 확정할 수 없습니다.",
    ),
    "ambiguous_number_word": (
        "ambiguous_numeric",
        "숫자 뒤 표기가 조사인지 낱말인지 확정할 수 없습니다.",
    ),
    "date_range": (
        "ambiguous_numeric",
        "날짜 범위 읽기는 지원하지 않아 원문을 보존합니다.",
    ),
    "mixed_fraction": (
        "ambiguous_numeric",
        "대분수 읽기는 지원하지 않아 원문을 보존합니다.",
    ),
    "jamo_cluster": (
        "unsupported_script",
        "연속된 자모는 웃음 표기나 초성어일 수 있어 원문을 보존합니다.",
    ),
    "han": (
        "unconverted_han",
        "한자의 문맥상 한국어 읽기를 확정할 수 없습니다.",
    ),
    "latin": (
        "unconverted_latin",
        "영문 고유명사의 관용 읽기를 확정할 수 없습니다.",
    ),
    "unsupported_symbol": (
        "unsupported_symbol",
        "발화 규칙이 없는 의미 기호는 원문을 보존합니다.",
    ),
}
_SIMPLE_RENDER_METHODS = {
    "phone": "_render_phone",
    "emergency": "_render_emergency",
    "time": "_render_time",
    "time_range": "_render_time_range",
    "month_day": "_render_month_day",
    "year_month": "_render_year_month",
    "month": "_render_month",
    "ordinal": "_render_ordinal",
    "exponent": "_render_exponent",
    "history": "_render_history",
    "currency": "_render_currency",
    "iso_currency": "_render_iso_currency",
    "scaled_currency": "_render_scaled_currency",
    "fraction": "_render_fraction",
    "percentage": "_render_percentage",
    "acronym": "_render_acronym",
}


@dataclass(frozen=True, slots=True)
class _Rendered:
    """한 토큰의 출력과 provenance다."""

    text: str
    handler: Handler
    rule_id: str
    unresolved: bool = False
    diagnostic_code: str | None = None
    diagnostic_message: str | None = None


@dataclass(frozen=True, slots=True)
class _TokenMatch:
    """기본 규칙과 사용자 사전 매치를 같은 방식으로 순회하기 위한 뷰다."""

    start: int
    end: int
    kind: str
    token: str


@dataclass(frozen=True, slots=True)
class _TextOnlyNormalized:
    """문자열 API에서 code point별 원문 span 할당을 생략하는 뷰다."""

    text: str

    def source_span(self, start: int, end: int) -> SourceSpan:
        """문자열 전용 경로에서는 호출되지 않아야 한다."""

        raise InternalInvariantError(f"문자열 전용 정규화에는 원문 span이 없습니다: {start}:{end}")


class KoreanTTSNormalizer:
    """문장 구조를 보존하며 비표준 표기를 한국어 발화형으로 바꾼다.

    내장 Kiwi가 형태소 문맥으로 숫자 읽기를 고른다. 확신이 임계값에 못 미치면
    판단을 포기하고 규칙의 읽기를 그대로 쓴다. Kiwi는 실제 판별 지점에서만
    만들어지고 이후 호출에서 재사용된다.
    """

    def __init__(
        self,
        *,
        lexicon: Mapping[str, str] | None = None,
        on_ambiguous: str = "warn",
        max_length: int = 10_000,
        max_output_length: int | None = None,
    ) -> None:
        if not isinstance(on_ambiguous, str) or on_ambiguous not in _POLICIES:
            raise InputValidationError("on_ambiguous는 preserve, warn, error 중 하나여야 합니다.")
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise InputValidationError("max_length는 양의 정수여야 합니다.")
        if max_output_length is None:
            max_output_length = max_length * 16
        if (
            not isinstance(max_output_length, int)
            or isinstance(max_output_length, bool)
            or max_output_length <= 0
        ):
            raise InputValidationError("max_output_length는 양의 정수여야 합니다.")
        # 사전 값도 기본 규칙으로 한 번 읽어 반복 호출의 결과를 같게 만든다.
        self._on_ambiguous = "preserve"
        # 사전 자기검증이 판별기 없이 결정적으로 돌도록 마지막에 켠다.
        self._number_reader: NumberReadingClassifier | None = None
        self._max_length = max_length
        self._max_output_length = max_output_length
        self._lexicon: CompiledLexicon | None = None
        if lexicon is not None:
            compiled_lexicon = CompiledLexicon(
                lexicon,
                base_normalize=self.normalize,
            )
            if compiled_lexicon:
                self._lexicon = compiled_lexicon
                for key, value in compiled_lexicon.items():
                    validation = self.convert(value)
                    if validation.normalized_text != value or not validation.complete:
                        raise InputValidationError(
                            "lexicon 값은 한 번의 변환 뒤 완결된 발화형이어야 "
                            f"합니다: {key!r}"
                        )
        self._on_ambiguous = on_ambiguous
        self._number_reader = NumberReadingClassifier(KiwiAnalyzer())

    def __call__(self, text: str) -> str:
        """상세 객체를 만들지 않고 정규화 문자열만 반환한다."""

        return self.normalize(text)

    def normalize(self, text: str) -> str:
        """정규화 문자열만 반환하는 빠른 경로다."""

        return self._run(text, trace=False).normalized_text

    def convert(self, text: str) -> NormalizationResult:
        """정규화 문자열과 rewrite·diagnostic을 모두 반환한다."""

        return self._run(text, trace=True)

    def normalize_many(self, texts: Iterable[str]) -> Iterator[str]:
        """입력을 미리 모으지 않고 문자열을 순서대로 정규화한다."""

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Iterable):
            raise InputValidationError("texts는 문자열이 아닌 문자열 iterable이어야 합니다.")
        return map(self.normalize, texts)

    def convert_many(self, texts: Iterable[str]) -> Iterator[NormalizationResult]:
        """입력을 미리 모으지 않고 상세 결과를 순서대로 변환한다."""

        if isinstance(texts, (str, bytes)) or not isinstance(texts, Iterable):
            raise InputValidationError("texts는 문자열이 아닌 문자열 iterable이어야 합니다.")
        return map(self.convert, texts)

    def _run(self, source: str, *, trace: bool) -> NormalizationResult:
        normalized: NormalizedText | _TextOnlyNormalized
        if trace or self._on_ambiguous == "error":
            normalized = normalize_nfc(source, max_length=self._max_length)
            initial_rewrites = normalized.rewrites if trace else ()
        else:
            normalized = _TextOnlyNormalized(
                normalize_nfc_text(source, max_length=self._max_length),
            )
            initial_rewrites = ()
        if not normalized.text:
            return NormalizationResult(source, "", initial_rewrites)
        if self._lexicon is None and _is_plain_korean_text(normalized.text):
            return NormalizationResult(source, normalized.text, initial_rewrites)

        protected = scan_protected(normalized.text)
        chunks: list[str] = []
        rewrites: list[Rewrite] = list(initial_rewrites)
        diagnostics: list[Diagnostic] = []
        unresolved = False
        cursor = 0
        output_length = 0
        # 판별 문맥은 호출마다 새로 만들어 같은 인스턴스를 여러 스레드가
        # 동시에 써도 분석 결과가 섞이지 않게 한다.
        number_run = (
            self._number_reader.new_run(normalized.text)
            if self._number_reader is not None
            else None
        )

        def append(chunk: str) -> None:
            nonlocal output_length
            output_length += len(chunk)
            if output_length > self._max_output_length:
                raise InputValidationError(
                    f"정규화 출력이 최대 길이 {self._max_output_length}를 초과했습니다."
                )
            chunks.append(chunk)

        for item in protected:
            segment, segment_unresolved = self._normalize_range(
                normalized,
                cursor,
                item.start,
                trace=trace,
                rewrites=rewrites,
                diagnostics=diagnostics,
                output_budget=self._max_output_length - output_length,
                number_run=number_run,
            )
            append(segment)
            unresolved = unresolved or segment_unresolved
            append(normalized.text[item.start : item.end])
            cursor = item.end

        segment, segment_unresolved = self._normalize_range(
            normalized,
            cursor,
            len(normalized.text),
            trace=trace,
            rewrites=rewrites,
            diagnostics=diagnostics,
            output_budget=self._max_output_length - output_length,
            number_run=number_run,
        )
        append(segment)
        unresolved = unresolved or segment_unresolved
        return NormalizationResult(
            source,
            "".join(chunks),
            tuple(rewrites),
            tuple(diagnostics),
            not unresolved,
        )

    def _normalize_range(
        self,
        normalized: NormalizedText | _TextOnlyNormalized,
        start: int,
        end: int,
        *,
        trace: bool,
        rewrites: list[Rewrite],
        diagnostics: list[Diagnostic],
        output_budget: int,
        number_run: NumberReadingRun | None = None,
    ) -> tuple[str, bool]:
        if start >= end:
            return "", False
        chunks: list[str] = []
        cursor = start
        unresolved = False
        output_length = 0

        def append(chunk: str) -> None:
            nonlocal output_length
            output_length += len(chunk)
            if output_length > output_budget:
                raise InputValidationError(
                    f"정규화 출력이 최대 길이 {self._max_output_length}를 초과했습니다."
                )
            chunks.append(chunk)

        for match in self._iter_token_matches(normalized.text, start, end):
            if match.start < cursor:
                continue
            append(normalized.text[cursor : match.start])
            unresolved = unresolved or self._inspect_unmatched(
                normalized,
                cursor,
                match.start,
                trace=trace,
                rewrites=rewrites,
                diagnostics=diagnostics,
            )
            token_end = match.end
            while token_end < end and self._is_unsafe_attached_character(
                normalized.text[token_end]
            ):
                token_end += 1
            if token_end != match.end:
                token = normalized.text[match.start:token_end]
                rendered = self._ambiguous(
                    token,
                    "unsupported_script",
                    "토큰에 결합할 수 없는 문자가 붙어 있어 원문을 보존합니다.",
                )
            else:
                token = match.token
                rendered = self._render(
                    match.kind,
                    token,
                    previous=normalized.text[match.start - 1 : match.start],
                    number_run=number_run,
                    start=match.start,
                )
            append(rendered.text)
            unresolved = unresolved or rendered.unresolved
            if trace:
                span = normalized.source_span(match.start, token_end)
                self._trace(rendered, token, span, rewrites, diagnostics)
                if number_run is not None:
                    failure = number_run.take_failure()
                    if failure is not None:
                        # 판별 실패는 규칙 경로로 폴백하므로 완결성에는 영향이 없다.
                        diagnostics.append(
                            Diagnostic(
                                failure[0],
                                failure[1],
                                Severity.WARNING,
                                (span,),
                                None,
                                RewriteStage.VERBALIZATION,
                            )
                        )
            elif rendered.unresolved and self._on_ambiguous == "error":
                span = normalized.source_span(match.start, token_end)
                self._raise_ambiguous(rendered, span)
            cursor = token_end
        append(normalized.text[cursor:end])
        unresolved = unresolved or self._inspect_unmatched(
            normalized,
            cursor,
            end,
            trace=trace,
            rewrites=rewrites,
            diagnostics=diagnostics,
        )
        return "".join(chunks), unresolved

    def _iter_token_matches(
        self,
        text: str,
        start: int,
        end: int,
    ) -> Iterator[_TokenMatch]:
        """기본 규칙과 trie 사전을 시작점 우선, 동률이면 사전 우선으로 합친다."""

        def base_matches() -> Iterator[_TokenMatch]:
            for match in _BASE_TOKEN_RE.finditer(text, start, end):
                kind = match.lastgroup
                if kind is None:
                    raise InternalInvariantError("토큰 정규식에 이름이 없습니다.")
                yield _TokenMatch(match.start(), match.end(), kind, match.group(0))

        lexicon = self._lexicon
        lexicon_matches = (
            iter(())
            if lexicon is None
            else (
                _TokenMatch(item.start, item.end, "user_lexicon", item.key)
                for item in lexicon.iter_matches(text, start=start, end=end)
            )
        )
        base_iterator = base_matches()
        lexicon_iterator = iter(lexicon_matches)
        base = next(base_iterator, None)
        lexical = next(lexicon_iterator, None)
        occupied_until = start
        while base is not None or lexical is not None:
            use_lexicon = lexical is not None and (
                base is None or lexical.start <= base.start
            )
            if use_lexicon:
                candidate = lexical
                lexical = next(lexicon_iterator, None)
            else:
                candidate = base
                base = next(base_iterator, None)
            assert candidate is not None
            if candidate.start < occupied_until:
                continue
            yield candidate
            occupied_until = candidate.end

    def _inspect_unmatched(
        self,
        normalized: NormalizedText | _TextOnlyNormalized,
        start: int,
        end: int,
        *,
        trace: bool,
        rewrites: list[Rewrite],
        diagnostics: list[Diagnostic],
    ) -> bool:
        """규칙에 잡히지 않은 문자 중 TTS가 해석할 수 없는 범위를 진단한다."""

        unresolved = False
        cursor = start
        while cursor < end:
            classification = self._unsupported_character(normalized.text[cursor])
            if classification is None:
                cursor += 1
                continue
            code, message = classification
            token_end = cursor + 1
            while (
                token_end < end
                and self._unsupported_character(normalized.text[token_end]) == classification
            ):
                token_end += 1
            token = normalized.text[cursor:token_end]
            rendered = self._ambiguous(token, code, message)
            unresolved = True
            if trace:
                span = normalized.source_span(cursor, token_end)
                self._trace(rendered, token, span, rewrites, diagnostics)
            elif self._on_ambiguous == "error":
                span = normalized.source_span(cursor, token_end)
                self._raise_ambiguous(rendered, span)
            cursor = token_end
        return unresolved

    @staticmethod
    def _is_unsafe_attached_character(character: str) -> bool:
        code_point = ord(character)
        return (
            unicodedata.category(character).startswith("M")
            or unicodedata.category(character) == "Cf"
            or 0x1100 <= code_point <= 0x11FF
            or 0xA960 <= code_point <= 0xA97F
            or 0xD7B0 <= code_point <= 0xD7FF
        )

    @staticmethod
    def _unsupported_character(character: str) -> tuple[str, str] | None:
        """허용한 한글·공백·문장부호 밖의 미해석 문자를 분류한다."""

        code_point = ord(character)
        category = unicodedata.category(character)
        if character in "\t\r\n" or character.isspace():
            return None
        if category in {"Cc", "Cf", "Co", "Cn", "Cs"}:
            return "unsupported_character", "제어·형식·비공개 문자는 TTS 입력으로 확정할 수 없습니다."
        if character.isnumeric() and not ("0" <= character <= "9"):
            return "ambiguous_numeric", "ASCII가 아닌 숫자의 읽기를 확정할 수 없습니다."
        if "가" <= character <= "힣":
            return None
        if 0x3131 <= code_point <= 0x318E or 0x1100 <= code_point <= 0x11FF:
            return "unsupported_script", "지원하지 않는 한글 자모 표기를 원문으로 보존합니다."
        if category.startswith("L") or category.startswith("M"):
            return "unsupported_script", "지원하지 않는 문자 체계는 원문으로 보존합니다."
        if category.startswith("S") or character in "=*<>^~∼":
            return "unsupported_symbol", "발화 규칙이 없는 의미 기호는 원문을 보존합니다."
        return None

    def _trace(
        self,
        rendered: _Rendered,
        before: str,
        span: SourceSpan,
        rewrites: list[Rewrite],
        diagnostics: list[Diagnostic],
    ) -> None:
        if rendered.unresolved:
            if self._on_ambiguous == "error":
                self._raise_ambiguous(rendered, span)
            if self._on_ambiguous == "warn":
                diagnostics.append(
                    Diagnostic(
                        rendered.diagnostic_code or "ambiguous_text",
                        rendered.diagnostic_message or "안전하게 읽을 수 없어 원문을 보존했습니다.",
                        Severity.WARNING,
                        (span,),
                        rendered.rule_id,
                        RewriteStage.VERBALIZATION,
                    )
                )
        if rendered.text != before:
            rewrites.append(
                Rewrite(
                    (span,),
                    before,
                    rendered.text,
                    rendered.handler,
                    rendered.rule_id,
                    RewriteStage.VERBALIZATION,
                )
            )

    @staticmethod
    def _raise_ambiguous(rendered: _Rendered, span: SourceSpan) -> None:
        raise ConversionPolicyError(
            rendered.diagnostic_code or "ambiguous_text",
            rendered.diagnostic_message or "안전하게 읽을 수 없어 원문을 보존했습니다.",
            source_span=span,
            rule_id=rendered.rule_id,
        )

    def _render(
        self,
        kind: str,
        token: str,
        *,
        previous: str = "",
        number_run: NumberReadingRun | None = None,
        start: int | None = None,
        context_end: int | None = None,
    ) -> _Rendered:
        ambiguity = _AMBIGUOUS_TOKEN_MESSAGES.get(kind)
        if ambiguity is not None:
            return self._ambiguous(token, *ambiguity)
        if kind in _UNSUPPORTED_NUMERIC_KINDS:
            return self._ambiguous(
                token,
                "ambiguous_numeric",
                "지원하지 않는 숫자 구조는 원문을 보존합니다.",
            )
        simple_method = _SIMPLE_RENDER_METHODS.get(kind)
        if simple_method is not None:
            return getattr(self, simple_method)(token)
        if kind == "semantic_cluster":
            return self._render_semantic_cluster(token, number_run=number_run, start=start)
        if kind in {"range", "range_both", "scaled_range"}:
            return self._render_range(token, kind=kind)
        if kind in {"version", "filename"}:
            return _Rendered(token, Handler.PROTECTED, f"protected.{kind}")
        if kind == "ipv4":
            if all(int(octet) <= 255 for octet in token.split(".")):
                return _Rendered(token, Handler.PROTECTED, "protected.ipv4")
            return self._ambiguous(token, "ambiguous_numeric", "유효하지 않은 IPv4 주소입니다.")
        if kind == "user_lexicon":
            if self._lexicon is None:
                raise InternalInvariantError("사용자 사전 매치에 컴파일된 사전이 없습니다.")
            return _Rendered(self._lexicon[token], Handler.KOREAN, "normalizer.user_lexicon.v1")
        if kind in {"date", "korean_date"}:
            return self._render_date(token, korean=kind == "korean_date")
        if kind == "numbered_item":
            return self._render_number_hangul_unit(token, force_sino=True)
        if kind == "numeric_guard":
            if re.search(r"[A-Za-z]", token):
                return self._ambiguous(
                    token,
                    "ambiguous_identifier",
                    "영숫자 식별자의 의도한 읽기를 확정할 수 없습니다.",
                )
            return self._ambiguous(
                token,
                "ambiguous_numeric",
                "지원하지 않는 숫자 구조는 원문을 보존합니다.",
            )
        if kind == "number_hangul_unit":
            return self._render_number_hangul_unit(
                token,
                force_sino=previous == "제",
                number_run=number_run,
                start=start,
                context_end=context_end,
            )
        if kind in {"fraction_unit", "number_unit", "quantity_unit"}:
            return self._render_quantity_unit(
                token,
                force_sino=previous == "제",
                number_run=number_run,
                start=start,
                context_end=(
                    context_end
                    if context_end is not None
                    else None if start is None else start + len(token)
                ),
            )
        if kind == "indefinite_unit":
            return self._render_indefinite_unit(token)
        if kind == "unit_per":
            return self._render_unit_per(token)
        if kind == "standalone_unit_symbol":
            return self._render_standalone_unit_symbol(token)
        if kind == "number_suffix":
            match = _expected_match(_NUMBER_SUFFIX_RE, token)
            number = match.group("number")
            rendered = self._render_number(
                number,
                rule_id="normalizer.number.sino.v1",
                original=token,
            )
            if rendered.unresolved:
                return rendered
            rendered = self._apply_bare_number_context(
                rendered,
                number,
                number_run,
                None if start is None else start + match.start("number"),
                None if start is None else start + len(token),
            )
            return _Rendered(
                rendered.text + match.group("suffix"),
                rendered.handler,
                rendered.rule_id,
            )
        if kind == "decimal":
            try:
                parsed = parse_number(token)
            except NumberFormatError:
                return self._ambiguous(token, "ambiguous_numeric", "소수를 손실 없이 읽을 수 없습니다.")
            if len(parsed.integer) > 10:
                return self._ambiguous(
                    token,
                    "ambiguous_numeric",
                    "긴 숫자가 식별번호인지 수량인지 확정할 수 없습니다.",
                )
            return self._render_number(token, rule_id="normalizer.number.sino.v1")
        if kind == "number":
            rendered = self._render_number(token, rule_id="normalizer.number.sino.v1")
            if rendered.unresolved:
                return rendered
            return self._apply_bare_number_context(
                rendered,
                token,
                number_run,
                start,
                (
                    context_end
                    if context_end is not None
                    else None if start is None else start + len(token)
                ),
            )
        if kind == "number_punctuation":
            rendered = self._render_number(
                token[:-1],
                rule_id="normalizer.number.sino.v1",
                original=token,
            )
            if rendered.unresolved:
                return rendered
            rendered = self._apply_bare_number_context(
                rendered,
                token[:-1],
                number_run,
                start,
                None if start is None else start + len(token) - 1,
            )
            return _Rendered(
                rendered.text + token[-1],
                rendered.handler,
                rendered.rule_id,
            )
        if kind == "jamo":
            return _Rendered(_JAMO_NAMES[token], Handler.KOREAN, "normalizer.jamo_name.v1")
        raise InternalInvariantError(f"알 수 없는 normalizer 토큰입니다: {kind}")

    def _render_phone(self, token: str) -> _Rendered:
        match = _expected_match(_PHONE_RE, token)
        number = match.group("number")
        groups = re.findall(r"[0-9]+", number)
        prefix = "플러스 " if number.startswith("+") else ""
        return _Rendered(
            prefix
            + " ".join(read_phone_digits(item) for item in groups)
            + (match.group("suffix") or ""),
            Handler.NUMERIC,
            "normalizer.phone.v1",
        )

    def _render_emergency(self, token: str) -> _Rendered:
        match = _expected_match(_EMERGENCY_RE, token)
        return _Rendered(
            read_phone_digits(match.group("number")) + (match.group("suffix") or ""),
            Handler.NUMERIC,
            "normalizer.emergency_number.v1",
        )

    def _render_ordinal(self, token: str) -> _Rendered:
        match = _expected_match(_ORDINAL_RE, token)
        rendered = self._render_number_hangul_unit(match.group("body"), force_sino=True)
        if rendered.unresolved:
            return self._ambiguous(token, "ambiguous_numeric", "서수 표현을 손실 없이 읽을 수 없습니다.")
        return _Rendered(
            "제 " + rendered.text,
            Handler.NUMERIC,
            "normalizer.ordinal_context.v1",
        )

    def _render_history(self, token: str) -> _Rendered:
        numeric_match = re.match(r"[0-9]{1,2}[.·][0-9]{1,2}", token)
        if numeric_match is None:
            return self._ambiguous(token, "ambiguous_numeric", "역사 날짜를 분리할 수 없습니다.")
        try:
            spoken = read_history_digits(numeric_match.group(0))
        except NumberFormatError:
            return self._ambiguous(
                token,
                "ambiguous_numeric",
                "역사 날짜를 손실 없이 읽을 수 없습니다.",
            )
        return _Rendered(
            spoken + token[numeric_match.end() :],
            Handler.NUMERIC,
            "normalizer.historical_date.v1",
        )

    def _render_currency(self, token: str) -> _Rendered:
        match = _expected_match(_CURRENCY_RE, token)
        unit = _CURRENCIES[match.group("symbol")]
        spoken_unit = match.group("spoken_unit")
        if spoken_unit is not None and spoken_unit != unit:
            return self._ambiguous(token, "ambiguous_numeric", "통화 기호와 단위가 서로 다릅니다.")
        return self._render_number(
            match.group("number"),
            suffix=unit,
            rule_id="normalizer.currency_prefix.v1",
            original=token,
        )

    def _render_acronym(self, token: str) -> _Rendered:
        reading = CONVENTIONAL_READINGS.get(token)
        if reading is not None:
            return _Rendered(
                reading,
                Handler.ENGLISH,
                "normalizer.english_acronym_exception.v1",
            )
        return _Rendered(
            self._read_letters(token),
            Handler.ENGLISH,
            "normalizer.english_initialism.v1",
        )

    def _render_semantic_cluster(
        self,
        token: str,
        *,
        number_run: NumberReadingRun | None = None,
        start: int | None = None,
    ) -> _Rendered:
        """공백 없는 숫자 인접 토큰을 전체 검증해 부분 변환을 막는다."""

        if not re.search(r"[0-9]", token):
            if any(character.isnumeric() for character in token) and not all(
                character in "²³⁴" for character in token
            ):
                return self._ambiguous(
                    token,
                    "ambiguous_numeric",
                    "ASCII가 아닌 숫자의 읽기를 확정할 수 없습니다.",
                )
            return self._ambiguous(
                token,
                "unsupported_symbol",
                "발화 규칙이 없는 의미 기호는 원문을 보존합니다.",
            )
        punctuation = token[-1] if token[-1:] in {".", ","} else ""
        body = token[:-1] if punctuation else token
        # 조사 후보를 떼어 내며 재분류하더라도 문맥 경계는 원래 토큰의 끝으로
        # 유지한다. 그래야 다른 경로와 같은 자질(꼬리=조사, 다음=뒷낱말)을 본다.
        body_end = None if start is None else start + len(body)
        rendered = self._classify_semantic_cluster(
            body, number_run=number_run, start=start, context_end=body_end
        )
        unresolved = rendered
        suffix = ""
        if rendered.unresolved:
            for candidate in sorted(_KOREAN_SUFFIXES, key=lambda item: (-len(item), item)):
                if not body.endswith(candidate) or len(body) == len(candidate):
                    continue
                remainder = body[: -len(candidate)]
                if candidate == "만" and _DECIMAL_RE.fullmatch(remainder):
                    # `3.000만 원`의 만은 배율일 수 있어 소수+조사로 읽지 않는다.
                    continue
                candidate_rendered = self._classify_semantic_cluster(
                    remainder, number_run=number_run, start=start, context_end=body_end
                )
                if candidate_rendered.unresolved:
                    if candidate_rendered.diagnostic_code != "ambiguous_numeric":
                        unresolved = candidate_rendered
                    continue
                rendered = candidate_rendered
                suffix = candidate
                break
        if rendered.unresolved:
            return self._ambiguous(
                token,
                unresolved.diagnostic_code or "ambiguous_numeric",
                unresolved.diagnostic_message or "지원하지 않는 숫자 구조는 원문을 보존합니다.",
            )
        return _Rendered(
            rendered.text + suffix + punctuation,
            rendered.handler,
            rendered.rule_id,
        )

    def _classify_semantic_cluster(
        self,
        token: str,
        *,
        number_run: NumberReadingRun | None = None,
        start: int | None = None,
        context_end: int | None = None,
    ) -> _Rendered:
        """공백 없는 숫자 후보 하나를 정확히 한 규칙으로 분류한다.

        ``context_end``는 조사 후보를 떼기 전 토큰의 끝이다. 판별 자질이
        다른 경로와 같은 문맥을 보게 하려고 분류 대상 길이와 따로 받는다.
        """

        if context_end is None and start is not None:
            context_end = start + len(token)

        match = _SEMANTIC_TOKEN_RE.fullmatch(token)
        if match is not None:
            kind = match.lastgroup
            if kind is None:
                raise InternalInvariantError("숫자 결합 토큰 규칙에 이름이 없습니다.")
            return self._render(
                kind,
                token,
                number_run=number_run,
                start=start,
                context_end=context_end,
            )
        if re.fullmatch(r"[+\-−]?[0-9][0-9,]*(?:\.[0-9]+)?[eE][+\-]?[0-9]+", token):
            code = "ambiguous_numeric"
            message = "지원하지 않는 과학적 표기법은 원문을 보존합니다."
        elif re.search(r"[A-Za-z]", token):
            code = "ambiguous_identifier"
            message = "영숫자 식별자의 의도한 읽기를 확정할 수 없습니다."
        else:
            code = "ambiguous_numeric"
            message = "지원하지 않는 숫자 구조는 원문을 보존합니다."
        return self._ambiguous(token, code, message)

    def _render_time(self, token: str) -> _Rendered:
        """12시간·24시간 표기를 값 검증 뒤 시·분·초 발화형으로 바꾼다."""

        match = _COLON_TIME_RE.fullmatch(token) or _KOREAN_TIME_RE.fullmatch(token)
        if match is None:
            return self._ambiguous(token, "ambiguous_numeric", "시간 구성요소를 분리할 수 없습니다.")
        period = match.group("period")
        hour = int(match.group("hour"))
        minute_surface = match.group("minute")
        second_surface = match.group("second")
        minute = int(minute_surface) if minute_surface is not None else None
        second = int(second_surface) if second_surface is not None else None
        valid_hour = 1 <= hour <= 12 if period is not None else 0 <= hour <= 24
        valid_24 = hour != 24 or ((minute or 0) == 0 and (second or 0) == 0)
        if (
            not valid_hour
            or not valid_24
            or (minute is not None and not 0 <= minute <= 59)
            or (second is not None and not 0 <= second <= 59)
        ):
            return self._ambiguous(token, "ambiguous_numeric", "유효하지 않은 시각 표기입니다.")

        if period is not None or 1 <= hour <= 12:
            spoken_hour = read_native_or_sino(str(hour))
        else:
            spoken_hour = read_sino(str(hour))
        parts = [f"{spoken_hour} 시"]
        if minute is not None:
            parts.append(f"{read_sino(str(minute))} 분")
        if second is not None:
            parts.append(f"{read_sino(str(second))} 초")
        prefix = f"{period} " if period is not None else ""
        suffix = match.group("suffix") or ""
        return _Rendered(
            prefix + " ".join(parts) + suffix,
            Handler.NUMERIC,
            "normalizer.clock_time.v1",
        )

    def _render_time_range(self, token: str) -> _Rendered:
        """양쪽 시각이 완전히 쓰인 범위만 `에서`로 연결한다."""

        match = _TIME_RANGE_RE.fullmatch(token)
        if match is None:
            return self._ambiguous(token, "ambiguous_numeric", "시각 범위를 분리할 수 없습니다.")
        left_token = match.group("left")
        right_token = match.group("right")
        left_match = _COLON_TIME_RE.fullmatch(left_token) or _KOREAN_TIME_RE.fullmatch(left_token)
        if left_match is None or left_match.group("suffix") is not None:
            return self._ambiguous(
                token,
                "ambiguous_numeric",
                "범위 중간에 조사가 붙은 시각은 읽기를 확정할 수 없습니다.",
            )
        left = self._render_time(left_token)
        right = self._render_time(right_token)
        if left.unresolved or right.unresolved:
            return self._ambiguous(token, "ambiguous_numeric", "유효하지 않은 시각 범위입니다.")
        return _Rendered(
            f"{left.text}에서 {right.text}",
            Handler.NUMERIC,
            "normalizer.clock_range.v1",
        )

    def _render_iso_currency(self, token: str) -> _Rendered:
        match = _expected_match(_ISO_CURRENCY_RE, token)
        code = match.group("prefix") or match.group("suffix")
        number = match.group("prefix_number") or match.group("suffix_number")
        assert code is not None and number is not None
        spoken_unit = match.group("spoken_unit")
        if spoken_unit is not None and spoken_unit != _ISO_CURRENCIES[code]:
            return self._ambiguous(
                token,
                "ambiguous_numeric",
                "ISO 통화 코드와 발화 단위가 서로 다릅니다.",
            )
        return self._render_number(
            number,
            suffix=_ISO_CURRENCIES[code],
            rule_id="normalizer.currency_iso.v1",
            original=token,
        )

    def _render_scaled_currency(self, token: str) -> _Rendered:
        match = _expected_match(_SCALED_CURRENCY_RE, token)
        try:
            if _may_be_thousands_separator(parse_number(match.group("number"))):
                return self._ambiguous(
                    token,
                    "ambiguous_numeric",
                    "점이 천 단위 구분점일 수 있어 원문을 보존합니다.",
                )
            scaled = self._scale_decimal(match.group("number"), _SCALE_POWERS[match.group("scale")])
        except NumberFormatError:
            return self._ambiguous(token, "ambiguous_numeric", "축약 금액을 손실 없이 읽을 수 없습니다.")
        rendered = self._render_number(
            scaled,
            suffix="원",
            rule_id="normalizer.currency_scaled.v1",
            original=token,
            allow_long=True,
        )
        suffix = match.group("suffix") or ""
        if rendered.unresolved or not suffix:
            return rendered
        return _Rendered(rendered.text + suffix, rendered.handler, rendered.rule_id)

    @staticmethod
    def _scale_decimal(surface: str, power: int) -> str:
        """float 없이 10의 거듭제곱을 곱해 축약 금액의 모든 자릿수를 보존한다."""

        parsed = parse_number(surface)
        fraction = parsed.fraction or ""
        digits = parsed.integer + fraction
        remaining_fraction = len(fraction) - power
        if remaining_fraction <= 0:
            integer = digits + "0" * (-remaining_fraction)
            scaled_fraction = ""
        else:
            split_at = len(digits) - remaining_fraction
            integer = digits[:split_at] or "0"
            scaled_fraction = digits[split_at:]
        integer = integer.lstrip("0") or "0"
        result = integer
        if scaled_fraction:
            result += "." + scaled_fraction
        return parsed.sign + result

    def _render_number_hangul_unit(
        self,
        token: str,
        *,
        force_sino: bool,
        number_run: NumberReadingRun | None = None,
        start: int | None = None,
        context_end: int | None = None,
    ) -> _Rendered:
        match = _expected_match(_NUMBER_HANGUL_UNIT_RE, token)
        word = match.group("word")
        unit = next((item for item in _ALL_UNITS if word.startswith(item)), "")
        suffix = word[len(unit) :]
        if not unit or (suffix and suffix not in _KOREAN_SUFFIXES):
            return self._ambiguous(
                token,
                "ambiguous_numeric",
                "단위인지 더 긴 낱말인지 확정할 수 없습니다.",
            )
        rendered = self._render_number_unit(
            match.group("number"),
            unit,
            original=token,
            force_sino=force_sino,
            number_run=number_run,
            number_start=None if start is None else start + match.start("number"),
            context_end=(
                context_end
                if context_end is not None
                else (None if start is None else start + len(token))
            ),
        )
        if rendered.unresolved:
            return rendered
        return _Rendered(rendered.text + suffix, rendered.handler, rendered.rule_id)

    @staticmethod
    def _consultable_number(number: str) -> bool:
        """판별기에 물어볼 수 있는 단순 정수 표기인지 본다.

        부호·쉼표·소수점이 있거나 선행 0이 있거나 아주 긴 숫자는 규칙이
        이미 보존이나 별도 경로로 다루므로 문맥 판별 대상이 아니다.
        """

        return (
            number.isascii()
            and number.isdigit()
            and 1 <= len(number) <= 10
            and not (len(number) > 1 and number.startswith("0"))
        )

    def _apply_bare_number_context(
        self,
        rendered: _Rendered,
        number: str,
        number_run: NumberReadingRun | None,
        start: int | None,
        context_end: int | None = None,
    ) -> _Rendered:
        """단위 없는 숫자를 문맥에 따라 낱자리로 고쳐 읽는다.

        고유어 수사는 단위 없이 설 수 없으므로 이 자리에서는 한자어 기본값을
        낱자리로 바꾸는 교정만 허용한다.
        """

        if number_run is None or start is None or not self._consultable_number(number):
            return rendered
        decision = number_run.decide(
            start,
            start + len(number),
            unit="",
            site="bare",
            default="sino",
            context_end=context_end,
        )
        if decision != "digitwise":
            return rendered
        return _Rendered(
            read_digitwise(number),
            Handler.NUMERIC,
            "normalizer.number_context_digitwise.v1",
        )

    @staticmethod
    def _format_quantity(
        spoken_number: str,
        unit: str,
        *,
        number_suffix: str = "",
        include_speed_prefix: bool = True,
    ) -> str:
        """숫자 발화와 단위를 결합하고 이동 속도만 한국어 어순으로 전치한다."""

        speed = _SPEED_UNIT_READINGS.get(unit)
        if speed is not None:
            body = f"{spoken_number}{number_suffix} {speed[1]}"
            return f"{speed[0]} {body}" if include_speed_prefix else body
        return f"{spoken_number}{number_suffix} {_UNIT_READINGS.get(unit, unit)}"

    def _render_quantity_unit(
        self,
        token: str,
        *,
        force_sino: bool,
        number_run: NumberReadingRun | None = None,
        start: int | None = None,
        context_end: int | None = None,
    ) -> _Rendered:
        """일반 수·분수·단일 배율 수와 단위를 한 토큰으로 읽는다."""

        match = _expected_match(_QUANTITY_UNIT_RE, token)
        unit = match.group("unit")
        numerator = match.group("numerator")
        if numerator is not None:
            # 혈압의 `120/80mmHg`는 분수가 아니라 두 측정값의 비율 표기다.
            if unit == "mmHg":
                return self._ambiguous(
                    token,
                    "ambiguous_numeric",
                    "수은주밀리미터 앞의 슬래시는 혈압 비율일 수 있어 원문을 보존합니다.",
                )
            fraction_surface = f"{numerator}/{match.group('denominator')}"
            rendered = self._render_fraction(fraction_surface)
            if rendered.unresolved:
                return self._ambiguous(
                    token,
                    rendered.diagnostic_code or "ambiguous_numeric",
                    rendered.diagnostic_message or "분수를 손실 없이 읽을 수 없습니다.",
                )
            return _Rendered(
                self._format_quantity(rendered.text, unit),
                Handler.NUMERIC,
                "normalizer.fraction_unit.v1",
            )

        number = match.group("number")
        if number is None:
            raise InternalInvariantError("수량 단위 토큰에 숫자가 없습니다.")
        scale = match.group("scale")
        approximation = match.group("approximation")
        if scale is None and approximation is None:
            return self._render_number_unit(
                number,
                unit,
                original=token,
                force_sino=force_sino,
                number_run=number_run,
                number_start=None if start is None else start + match.start("number"),
                context_end=context_end,
            )

        try:
            parsed = parse_number(number)
            if len(parsed.integer) > 10:
                return self._ambiguous(
                    token,
                    "ambiguous_numeric",
                    "긴 숫자가 식별번호인지 수량인지 확정할 수 없습니다.",
                )
            if scale is not None and _may_be_thousands_separator(parsed):
                return self._ambiguous(
                    token,
                    "ambiguous_numeric",
                    "점이 천 단위 구분점일 수 있어 원문을 보존합니다.",
                )
            scaled = (
                self._scale_decimal(number, _SCALE_POWERS[scale])
                if scale is not None
                else number
            )
        except NumberFormatError:
            return self._ambiguous(token, "ambiguous_numeric", "수량을 손실 없이 읽을 수 없습니다.")
        rendered_number = self._render_number(
            scaled,
            rule_id="normalizer.number.sino.v1",
            original=token,
            allow_long=scale is not None,
        )
        if rendered_number.unresolved:
            return rendered_number
        rule = "scaled_unit" if scale is not None else "approximate_unit"
        if scale is not None and approximation is not None:
            rule = "scaled_approximate_unit"
        return _Rendered(
            self._format_quantity(
                rendered_number.text,
                unit,
                number_suffix="여" if approximation is not None else "",
            ),
            Handler.NUMERIC,
            f"normalizer.{rule}.v1",
        )

    def _render_indefinite_unit(self, token: str) -> _Rendered:
        match = _expected_match(_INDEFINITE_UNIT_RE, token)
        return _Rendered(
            self._format_quantity(match.group("quantifier"), match.group("unit")),
            Handler.NUMERIC,
            "normalizer.indefinite_unit.v1",
        )

    def _render_unit_per(self, token: str) -> _Rendered:
        match = _expected_match(_UNIT_PER_RE, token)
        return _Rendered(
            _UNIT_READINGS[match.group("unit")] + "당" + (match.group("suffix") or ""),
            Handler.NUMERIC,
            "normalizer.unit_per.v1",
        )

    def _render_standalone_unit_symbol(self, token: str) -> _Rendered:
        match = _expected_match(_STANDALONE_UNIT_SYMBOL_RE, token)
        return _Rendered(
            _UNIT_READINGS[match.group("unit")] + (match.group("suffix") or ""),
            Handler.NUMERIC,
            "normalizer.standalone_unit_symbol.v1",
        )

    def _render_number_unit(
        self,
        number: str,
        unit: str,
        *,
        original: str | None = None,
        force_sino: bool = False,
        number_run: NumberReadingRun | None = None,
        number_start: int | None = None,
        context_end: int | None = None,
        include_speed_prefix: bool = True,
    ) -> _Rendered:
        source = original if original is not None else number + unit
        try:
            parsed = parse_number(number)
            month_number = parsed.integer.lstrip("0") or "0"
            if (
                unit == "월"
                and not force_sino
                and not parsed.sign
                and parsed.fraction is None
                and month_number in {"6", "10"}
            ):
                return _Rendered(
                    "유월" if month_number == "6" else "시월",
                    Handler.NUMERIC,
                    "normalizer.month_exception.v1",
                )
            if has_ambiguous_leading_zero(parsed) and parsed.fraction is None:
                return self._ambiguous(
                    source,
                    "ambiguous_numeric",
                    "선행 0이 있는 숫자의 용도를 확정할 수 없습니다.",
                )
            if unit in _CURRENCY_UNIT_WORDS and _may_be_thousands_separator(parsed):
                return self._ambiguous(
                    source,
                    "ambiguous_numeric",
                    "점이 천 단위 구분점일 수 있어 원문을 보존합니다.",
                )
            if len(parsed.integer) > 10:
                return self._ambiguous(
                    source,
                    "ambiguous_numeric",
                    "긴 숫자가 식별번호인지 수량인지 확정할 수 없습니다.",
                )
            if (
                unit == "번째"
                and not force_sino
                and parsed.integer == "1"
                and parsed.fraction is None
            ):
                spoken_number = "첫"
            elif force_sino or unit not in _NATIVE_COUNTERS:
                spoken_number = read_sino(number)
            else:
                spoken_number = read_native_or_sino(number)
        except NumberFormatError:
            return self._ambiguous(
                source,
                "ambiguous_numeric",
                "숫자 표면형을 손실 없이 읽을 수 없습니다.",
            )
        if force_sino:
            rule = "ordinal_sino"
        elif unit in _NATIVE_COUNTERS:
            rule = "native_counter"
        else:
            rule = "sino_unit"
        # `제N`처럼 표기가 읽기를 정하는 자리와 `1번째`→`첫`처럼 규칙이 이미
        # 특수형을 쓰는 자리는 문맥 판별 대상이 아니다.
        if (
            not force_sino
            and spoken_number != "첫"
            and unit not in _CONTEXT_INDEPENDENT_SINO_UNITS
        ):
            default = "native" if unit in _NATIVE_COUNTERS else "sino"
            contextual = self._decide_number_reading(
                number, unit, number_run, number_start, default, context_end
            )
            if contextual is not None:
                return _Rendered(
                    self._format_quantity(
                        contextual[0],
                        unit,
                        include_speed_prefix=include_speed_prefix,
                    ),
                    Handler.NUMERIC,
                    f"normalizer.number_context_{contextual[1]}.v1",
                )
        rule_id = (
            "normalizer.speed_unit.v1"
            if unit in _SPEED_UNIT_READINGS
            else f"normalizer.number_{rule}.v1"
        )
        return _Rendered(
            self._format_quantity(
                spoken_number,
                unit,
                include_speed_prefix=include_speed_prefix,
            ),
            Handler.NUMERIC,
            rule_id,
        )

    def _decide_number_reading(
        self,
        number: str,
        unit: str,
        number_run: NumberReadingRun | None,
        number_start: int | None,
        default: str,
        context_end: int | None = None,
    ) -> tuple[str, str] | None:
        """단위가 붙은 숫자를 문맥에 따라 다시 읽어야 하는지 판단한다."""

        if (
            number_run is None
            or number_start is None
            or not self._consultable_number(number)
        ):
            return None
        decision = number_run.decide(
            number_start,
            number_start + len(number),
            unit=unit,
            site="unit",
            default=default,
            context_end=context_end,
        )
        if decision is None:
            return None
        if decision == "native":
            return read_native_or_sino(number), "native"
        if decision == "digitwise":
            return read_digitwise(number), "digitwise"
        return read_sino(number), "sino"

    def _render_number(
        self,
        number: str,
        *,
        suffix: str = "",
        rule_id: str,
        original: str | None = None,
        allow_long: bool = False,
    ) -> _Rendered:
        source = original if original is not None else number + suffix
        try:
            parsed = parse_number(number)
            if has_ambiguous_leading_zero(parsed) and parsed.fraction is None:
                return self._ambiguous(
                    source,
                    "ambiguous_numeric",
                    "선행 0이 있는 숫자의 용도를 확정할 수 없습니다.",
                )
            if suffix in _CURRENCY_UNIT_WORDS and _may_be_thousands_separator(parsed):
                return self._ambiguous(
                    source,
                    "ambiguous_numeric",
                    "점이 천 단위 구분점일 수 있어 원문을 보존합니다.",
                )
            if not allow_long and len(parsed.integer) > 10:
                return self._ambiguous(
                    source,
                    "ambiguous_numeric",
                    "긴 숫자가 식별번호인지 수량인지 확정할 수 없습니다.",
                )
            spoken = read_sino(number)
        except NumberFormatError:
            return self._ambiguous(
                source,
                "ambiguous_numeric",
                "숫자 표면형을 손실 없이 읽을 수 없습니다.",
            )
        return _Rendered(spoken + (f" {suffix}" if suffix else ""), Handler.NUMERIC, rule_id)

    def _render_range(self, token: str, *, kind: str) -> _Rendered:
        if kind == "range_both":
            match = _expected_match(_RANGE_BOTH_RE, token)
            left = self._render_range_side(match.group("left"), match.group("left_unit"))
            right = self._render_range_side(match.group("right"), match.group("unit"))
            if left.unresolved or right.unresolved:
                return self._ambiguous(
                    token,
                    "ambiguous_numeric",
                    "범위의 숫자를 손실 없이 읽을 수 없습니다.",
                )
            return _Rendered(
                f"{left.text}에서 {right.text}{match.group('suffix') or ''}",
                Handler.NUMERIC,
                "normalizer.range_dual_unit.v1",
            )

        if kind == "scaled_range":
            match = _expected_match(_SCALED_RANGE_RE, token)
            try:
                if _may_be_thousands_separator(parse_number(match.group("left"))):
                    return self._ambiguous(
                        token,
                        "ambiguous_numeric",
                        "점이 천 단위 구분점일 수 있어 원문을 보존합니다.",
                    )
                left_scaled = self._scale_decimal(
                    match.group("left"), _SCALE_POWERS[match.group("left_scale")]
                )
                left_spoken = read_sino(left_scaled)
            except NumberFormatError:
                return self._ambiguous(
                    token,
                    "ambiguous_numeric",
                    "축약 금액을 손실 없이 읽을 수 없습니다.",
                )
            right = self._render_scaled_currency(
                match.group("right") + match.group("right_scale") + "원"
            )
            if right.unresolved:
                return self._ambiguous(
                    token,
                    "ambiguous_numeric",
                    "축약 금액을 손실 없이 읽을 수 없습니다.",
                )
            return _Rendered(
                f"{left_spoken}에서 {right.text}{match.group('suffix') or ''}",
                Handler.NUMERIC,
                "normalizer.range_scaled_currency.v1",
            )

        if kind != "range":
            raise InternalInvariantError(f"알 수 없는 범위 토큰입니다: {kind}")
        match = _expected_match(_RANGE_RE, token)
        unit = match.group("unit")
        grammatical_suffix = match.group("suffix") or ""
        if unit == "시":
            return self._ambiguous(
                token,
                "ambiguous_numeric",
                "축약된 시간 범위인지 수량 범위인지 확정할 수 없습니다.",
            )
        if unit is None:
            left_number = self._render_number(
                match.group("left"),
                rule_id="normalizer.range_number.v1",
            )
            right_number = self._render_number(
                match.group("right"),
                rule_id="normalizer.range_number.v1",
            )
            if left_number.unresolved or right_number.unresolved:
                return self._ambiguous(
                    token,
                    "ambiguous_numeric",
                    "범위의 숫자를 손실 없이 읽을 수 없습니다.",
                )
            return _Rendered(
                f"{left_number.text}에서 {right_number.text}{grammatical_suffix}",
                Handler.NUMERIC,
                "normalizer.range_number.v1",
            )
        left = self._render_range_left_endpoint(match.group("left"), unit)
        speed = _SPEED_UNIT_READINGS.get(unit)
        right = self._render_range_side(
            match.group("right"),
            unit,
            include_speed_prefix=speed is None,
        )
        if left.unresolved or right.unresolved:
            return self._ambiguous(token, "ambiguous_numeric", "범위의 숫자를 손실 없이 읽을 수 없습니다.")
        prefix = f"{speed[0]} " if speed is not None else ""
        return _Rendered(
            f"{prefix}{left.text}에서 {right.text}{grammatical_suffix}",
            Handler.NUMERIC,
            "normalizer.range_speed.v1" if speed is not None else "normalizer.range_unit.v2",
        )

    def _render_range_left_endpoint(self, number: str, unit: str) -> _Rendered:
        """단위 없이 수사만 남는 범위 왼쪽 숫자를 단위의 수사계로 읽는다."""

        try:
            parsed = parse_number(number)
            if has_ambiguous_leading_zero(parsed) and parsed.fraction is None:
                return self._ambiguous(
                    number,
                    "ambiguous_numeric",
                    "선행 0이 있는 숫자의 용도를 확정할 수 없습니다.",
                )
            if unit in _NATIVE_COUNTERS:
                spoken = read_native_standalone_or_sino(number)
            else:
                spoken = read_sino(number)
        except NumberFormatError:
            return self._ambiguous(
                number,
                "ambiguous_numeric",
                "숫자 표면형을 손실 없이 읽을 수 없습니다.",
            )
        return _Rendered(spoken, Handler.NUMERIC, "normalizer.range_unit.v2")

    def _render_range_side(
        self,
        number: str,
        unit: str,
        *,
        include_speed_prefix: bool = True,
    ) -> _Rendered:
        """범위 한쪽 끝의 숫자와 단위를 단일 수량과 같은 규칙으로 읽는다."""

        if unit in {"%", "％", "%p", "%P"}:
            suffix = "퍼센트포인트" if unit.lower() == "%p" else "퍼센트"
            return self._render_number(
                number,
                suffix=suffix,
                rule_id="normalizer.range_percentage.v2",
            )
        if unit in {"만원", "억원", "조원"}:
            return self._render_scaled_currency(number + unit)
        return self._render_number_unit(
            number,
            unit,
            include_speed_prefix=include_speed_prefix,
        )

    def _render_fraction(self, token: str) -> _Rendered:
        match = _expected_match(_FRACTION_RE, token)
        try:
            numerator_number = parse_number(match.group("numerator"))
            denominator_number = parse_number(match.group("denominator"))
            if numerator_number.fraction is not None or denominator_number.fraction is not None:
                return self._ambiguous(token, "ambiguous_numeric", "소수의 슬래시 표기를 분수로 단정하지 않습니다.")
            if has_ambiguous_leading_zero(numerator_number) or has_ambiguous_leading_zero(denominator_number):
                return self._ambiguous(token, "ambiguous_numeric", "선행 0이 있는 분수의 용도를 확정할 수 없습니다.")
            if int(denominator_number.integer) == 0:
                return self._ambiguous(token, "ambiguous_numeric", "분모가 0인 표기를 분수로 읽지 않습니다.")
            denominator = read_sino(match.group("denominator"))
            numerator = read_sino(match.group("numerator"))
        except NumberFormatError:
            return self._ambiguous(token, "ambiguous_numeric", "분수를 손실 없이 읽을 수 없습니다.")
        return _Rendered(f"{denominator}분의 {numerator}", Handler.NUMERIC, "normalizer.fraction.v1")

    def _render_percentage(self, token: str) -> _Rendered:
        match = _expected_match(_PERCENT_RE, token)
        suffix = "퍼센트포인트" if match.group("unit").lower() == "%p" else "퍼센트"
        return self._render_number(
            match.group("number"),
            suffix=suffix,
            rule_id="normalizer.percentage.v1",
            original=token,
        )

    def _render_date(self, token: str, *, korean: bool = False) -> _Rendered:
        match = _expected_match(_KOREAN_DATE_RE if korean else _DATE_RE, token)
        year, month, day = (int(match.group(item)) for item in ("year", "month", "day"))
        try:
            date(year, month, day)
        except ValueError:
            return self._ambiguous(token, "ambiguous_numeric", "유효하지 않은 달력 날짜입니다.")
        month_text = "유월" if month == 6 else "시월" if month == 10 else f"{read_sino(str(month))} 월"
        return _Rendered(
            f"{read_sino(str(year))} 년 {month_text} {read_sino(str(day))} 일"
            + (match.group("suffix") or ""),
            Handler.NUMERIC,
            "normalizer.calendar_date.v1",
        )

    def _render_month_day(self, token: str) -> _Rendered:
        """연도 없는 한국어 월·일 표기를 윤년 기준으로 검증한다."""

        match = _expected_match(_MONTH_DAY_RE, token)
        month, day = (int(match.group(item)) for item in ("month", "day"))
        try:
            date(2000, month, day)
        except ValueError:
            return self._ambiguous(token, "ambiguous_numeric", "유효하지 않은 월·일 표기입니다.")
        month_text = "유월" if month == 6 else "시월" if month == 10 else f"{read_sino(str(month))} 월"
        return _Rendered(
            f"{month_text} {read_sino(str(day))} 일" + (match.group("suffix") or ""),
            Handler.NUMERIC,
            "normalizer.calendar_month_day.v1",
        )

    def _render_month(self, token: str) -> _Rendered:
        """독립 월 표기를 1월부터 12월까지만 읽는다."""

        match = _expected_match(_MONTH_RE, token)
        month = int(match.group("month"))
        if not 1 <= month <= 12:
            return self._ambiguous(token, "ambiguous_numeric", "유효하지 않은 월 표기입니다.")
        spoken = "유월" if month == 6 else "시월" if month == 10 else f"{read_sino(str(month))} 월"
        return _Rendered(
            spoken + (match.group("suffix") or ""),
            Handler.NUMERIC,
            "normalizer.calendar_month.v1",
        )

    def _render_year_month(self, token: str) -> _Rendered:
        """연도와 선택적 월을 한 구조로 검증해 부분 변환을 막는다."""

        match = _expected_match(_YEAR_MONTH_RE, token)
        year = read_sino(match.group("year"))
        month_surface = match.group("month")
        if month_surface is None:
            spoken = f"{year} 년"
        else:
            month = int(month_surface)
            if not 1 <= month <= 12:
                return self._ambiguous(token, "ambiguous_numeric", "유효하지 않은 연월 표기입니다.")
            month_text = "유월" if month == 6 else "시월" if month == 10 else f"{read_sino(str(month))} 월"
            spoken = f"{year} 년 {month_text}"
        return _Rendered(
            spoken + (match.group("suffix") or ""),
            Handler.NUMERIC,
            "normalizer.calendar_year_month.v1",
        )

    def _render_exponent(self, token: str) -> _Rendered:
        match = _expected_match(_EXPONENT_RE, token)
        try:
            base = read_sino(match.group("base"))
            exponent = read_sino(match.group("exponent"))
        except NumberFormatError:
            return self._ambiguous(token, "ambiguous_numeric", "지수 표기를 손실 없이 읽을 수 없습니다.")
        return _Rendered(f"{base}의 {exponent}승", Handler.NUMERIC, "normalizer.math_exponent.v1")

    @staticmethod
    def _read_letters(token: str) -> str:
        return spell_ascii_letters(token)

    @staticmethod
    def _ambiguous(token: str, code: str, message: str) -> _Rendered:
        return _Rendered(token, Handler.PROTECTED, f"normalizer.{code}.v1", True, code, message)
