from __future__ import annotations

import unicodedata

from g2p_ko import (
    Boundary,
    G2PError,
    Handler,
    InputValidationError,
    OutputSegment,
    Rewrite,
    RewriteStage,
    SourceSpan,
)
from g2p_ko.unicode import normalize_nfc


def test_public_errors_share_neutral_base_class() -> None:
    assert issubclass(InputValidationError, G2PError)


def test_nfc_rewrite_keeps_raw_source_span() -> None:
    source = "\u1100\u1161"
    normalized = normalize_nfc(source)

    assert normalized.text == "가"
    assert normalized.source_span(0, 1) == SourceSpan(0, 2, source, 2)
    assert normalized.rewrites[0].stage is RewriteStage.UNICODE
    assert normalized.rewrites[0].before == source


def test_already_nfc_combining_cluster_keeps_atomic_source_span() -> None:
    source = "a\u0338"
    assert unicodedata.is_normalized("NFC", source)

    normalized = normalize_nfc(source)

    cluster = SourceSpan(0, len(source), source, len(source))
    assert normalized.character_spans == (cluster, cluster)


def test_inserted_segment_requires_explicit_anchor_contract() -> None:
    anchor = SourceSpan.from_source("가", 0, 1)
    inserted = OutputSegment(
        "ㅇ",
        (),
        Handler.KOREAN,
        "inserted",
        insertion_rule="test.insert",
        anchor=anchor,
    )

    assert inserted.source_spans == ()
    assert inserted.anchor == anchor
