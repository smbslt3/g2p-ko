"""한국어 음운 규칙과 종성 전이를 선언형 자료구조로 정의한다."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuleDefinition:
    """하나의 음운 규칙을 설명한다."""

    id: str
    article: str
    title: str
    requires_morphology: bool = False


@dataclass(frozen=True, slots=True)
class CodaTransition:
    """종성 하나를 대표음으로 바꾸는 문맥 독립 전이다."""

    source: str
    target: str
    rule_id: str


RULES = {
    "ko.vowel.jyeo": RuleDefinition("ko.vowel.jyeo", "5.1", "져·쪄·쳐의 ㅕ 약화"),
    "ko.vowel.consonant_ui": RuleDefinition("ko.vowel.consonant_ui", "5.3", "자음 뒤 ㅢ의 ㅣ 발음"),
    "ko.hieuh.consonant": RuleDefinition("ko.hieuh.consonant", "12", "ㅎ 탈락과 거센소리되기"),
    "ko.palatalization": RuleDefinition("ko.palatalization", "17", "구개음화"),
    "ko.hieuh.vowel": RuleDefinition("ko.hieuh.vowel", "12.4", "ㅎ 탈락 뒤 모음 연결"),
    "ko.liaison.single": RuleDefinition("ko.liaison.single", "13", "홑받침 연음"),
    "ko.liaison.cluster": RuleDefinition("ko.liaison.cluster", "14", "겹받침 연음"),
    "ko.n_insertion.numeral": RuleDefinition(
        "ko.n_insertion.numeral", "29", "수사 합성어의 ㄴ 첨가"
    ),
    "ko.coda.9": RuleDefinition("ko.coda.9", "9", "받침 대표음"),
    "ko.coda.10": RuleDefinition("ko.coda.10", "10", "겹받침 대표음"),
    "ko.coda.11": RuleDefinition("ko.coda.11", "11", "겹받침 대표음"),
    "ko.nasalization": RuleDefinition("ko.nasalization", "18", "비음화"),
    "ko.liquid.19": RuleDefinition("ko.liquid.19", "19", "ㄹ의 비음화"),
    "ko.liquid.20": RuleDefinition("ko.liquid.20", "20", "ㄴ·ㄹ 유음화"),
    "ko.tensification": RuleDefinition("ko.tensification", "23", "받침 뒤 된소리되기"),
    "ko.ui.particle": RuleDefinition("ko.ui.particle", "5.4.2", "조사 의 발음", True),
    "ko.stem.rieul_giyeok": RuleDefinition("ko.stem.rieul_giyeok", "11.1", "용언 ㄺ 어간", True),
    "ko.stem.tensing.24": RuleDefinition("ko.stem.tensing.24", "24", "용언 ㄴ·ㅁ 계열 어간 뒤 된소리", True),
    "ko.stem.tensing.25": RuleDefinition("ko.stem.tensing.25", "25", "용언 ㄼ·ㄾ 어간 뒤 된소리", True),
    "ko.lexical.balb_neolb": RuleDefinition(
        "ko.lexical.balb_neolb", "10.1", "밟·넓 어휘 예외의 보수적 보존"
    ),
    "ko.modifier.tensing.27": RuleDefinition("ko.modifier.tensing.27", "27", "관형형 ㄹ 뒤 된소리", True),
    "ko.liaison.lexical": RuleDefinition("ko.liaison.lexical", "15", "실질 형태소 연음", True),
}

CODA_TRANSITIONS = {
    "ㄲ": CodaTransition("ㄲ", "ㄱ", "ko.coda.9"),
    "ㅋ": CodaTransition("ㅋ", "ㄱ", "ko.coda.9"),
    "ㅅ": CodaTransition("ㅅ", "ㄷ", "ko.coda.9"),
    "ㅆ": CodaTransition("ㅆ", "ㄷ", "ko.coda.9"),
    "ㅈ": CodaTransition("ㅈ", "ㄷ", "ko.coda.9"),
    "ㅊ": CodaTransition("ㅊ", "ㄷ", "ko.coda.9"),
    "ㅌ": CodaTransition("ㅌ", "ㄷ", "ko.coda.9"),
    "ㅎ": CodaTransition("ㅎ", "ㄷ", "ko.coda.9"),
    "ㅍ": CodaTransition("ㅍ", "ㅂ", "ko.coda.9"),
    "ㄳ": CodaTransition("ㄳ", "ㄱ", "ko.coda.10"),
    "ㄵ": CodaTransition("ㄵ", "ㄴ", "ko.coda.10"),
    "ㄶ": CodaTransition("ㄶ", "ㄴ", "ko.coda.10"),
    "ㄼ": CodaTransition("ㄼ", "ㄹ", "ko.coda.10"),
    "ㄽ": CodaTransition("ㄽ", "ㄹ", "ko.coda.10"),
    "ㄾ": CodaTransition("ㄾ", "ㄹ", "ko.coda.10"),
    "ㅄ": CodaTransition("ㅄ", "ㅂ", "ko.coda.10"),
    "ㄺ": CodaTransition("ㄺ", "ㄱ", "ko.coda.11"),
    "ㄻ": CodaTransition("ㄻ", "ㅁ", "ko.coda.11"),
    "ㄿ": CodaTransition("ㄿ", "ㅂ", "ko.coda.11"),
    "ㅀ": CodaTransition("ㅀ", "ㄹ", "ko.coda.11"),
}


LIAISON_CLUSTER = {
    "ㄳ": ("ㄱ", "ㅆ"),
    "ㄵ": ("ㄴ", "ㅈ"),
    "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"),
    "ㄻ": ("ㄹ", "ㅁ"),
    "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅆ"),
    "ㄾ": ("ㄹ", "ㅌ"),
    "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"),
    "ㅄ": ("ㅂ", "ㅆ"),
}

HIEUH_FINALS = frozenset({"ㅎ", "ㄶ", "ㅀ"})
HIEUH_REMAINDER = {"ㅎ": None, "ㄶ": "ㄴ", "ㅀ": "ㄹ"}

# 받침과 초성 ㅎ이 합쳐질 때 남는 종성과 새 초성이다.
HIEUH_ASSIMILATION = {
    "ㄱ": (None, "ㅋ"),
    "ㄲ": (None, "ㅋ"),
    "ㅋ": (None, "ㅋ"),
    "ㄳ": (None, "ㅋ"),
    "ㄺ": ("ㄹ", "ㅋ"),
    "ㄷ": (None, "ㅌ"),
    "ㅅ": (None, "ㅌ"),
    "ㅆ": (None, "ㅌ"),
    "ㅈ": (None, "ㅌ"),
    "ㅊ": (None, "ㅌ"),
    "ㅌ": (None, "ㅌ"),
    "ㄵ": ("ㄴ", "ㅊ"),
    "ㅂ": (None, "ㅍ"),
    "ㅍ": (None, "ㅍ"),
    "ㄼ": ("ㄹ", "ㅍ"),
    "ㄿ": ("ㄹ", "ㅍ"),
    "ㅄ": (None, "ㅍ"),
}

TENSE_ONSET = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ", "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
NASAL_CODA = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}
