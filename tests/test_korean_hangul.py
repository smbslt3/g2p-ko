"""현대 한글 음절 token의 분해·조립과 원문 span 보존을 검증한다."""

from g2p_ko.korean.hangul import compose_syllable, decompose_syllable
from g2p_ko.model import SourceSpan


def test_decompose_and_compose_preserve_modern_hangul_and_span() -> None:
    source = "값"
    span = SourceSpan.from_source(source, 0, 1)

    token = decompose_syllable("값", (span,))

    assert (token.onset, token.vowel, token.coda) == ("ㄱ", "ㅏ", "ㅄ")
    assert token.text == "값"
    assert token.source_spans == (span,)
    assert compose_syllable("ㄱ", "ㅏ", "ㅄ") == "값"
