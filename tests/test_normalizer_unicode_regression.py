from __future__ import annotations

from g2p_ko import Handler, RewriteStage, SourceSpan
from g2p_ko.normalizer import KoreanTTSNormalizer
from g2p_ko.unicode import normalize_nfc, normalize_nfc_text


def test_precomposed_hangul_lv_and_tail_share_one_nfc_source_span() -> None:
    source = "앞가\u11a8나"

    normalized = normalize_nfc(source)

    composed_span = SourceSpan(1, 3, source[1:3], len(source))
    assert normalized.text == "앞각나"
    assert normalized.text == normalize_nfc_text(source)
    assert normalized.character_spans == (
        SourceSpan(0, 1, "앞", len(source)),
        composed_span,
        SourceSpan(3, 4, "나", len(source)),
    )
    assert normalized.source_span(1, 2) == composed_span
    assert len(normalized.rewrites) == 1
    rewrite = normalized.rewrites[0]
    assert rewrite.source_spans == (composed_span,)
    assert rewrite.before == source[1:3]
    assert rewrite.after == "각"
    assert rewrite.handler is Handler.KOREAN
    assert rewrite.rule_id == "unicode.nfc"
    assert rewrite.stage is RewriteStage.UNICODE


def test_tts_normalizer_uses_the_same_hangul_nfc_in_fast_and_trace_paths() -> None:
    source = "가\u11a8 3개"
    normalizer = KoreanTTSNormalizer()

    normalized_text = normalizer.normalize(source)
    converted = normalizer.convert(source)

    assert normalized_text == "각 세 개"
    assert converted.normalized_text == normalized_text
    assert converted.source == source
    assert converted.complete
    unicode_rewrite = next(
        rewrite for rewrite in converted.rewrites if rewrite.stage is RewriteStage.UNICODE
    )
    assert unicode_rewrite.source_spans == (
        SourceSpan(0, 2, source[:2], len(source)),
    )
    assert unicode_rewrite.before == source[:2]
    assert unicode_rewrite.after == "각"
    numeric_rewrite = next(
        rewrite
        for rewrite in converted.rewrites
        if rewrite.stage is RewriteStage.VERBALIZATION
    )
    assert numeric_rewrite.source_spans == (
        SourceSpan(3, 5, "3개", len(source)),
    )
    assert normalizer.normalize(normalized_text) == normalized_text
