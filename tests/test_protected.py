from __future__ import annotations

from time import perf_counter

import pytest

from g2p_ko import Handler
from g2p_ko.normalizer import KoreanTTSNormalizer
from g2p_ko.scanner import scan_protected
from tests._internal_pipeline import convert_pipeline


def test_url_particle_and_closing_punctuation_are_not_protected() -> None:
    source = "https://example.com/path을 확인하세요."
    ranges = scan_protected(source)

    assert source[ranges[0].start : ranges[0].end] == "https://example.com/path"
    result = convert_pipeline(source)
    protected = [item.text for item in result.normalized_segments if item.handler is Handler.PROTECTED]
    assert protected == ["https://example.com/path"]


def test_url_and_email_leave_nested_korean_particles_outside_protected_span() -> None:
    source = "https://example.com/path에서만 user@example.com에서만 확인"
    ranges = scan_protected(source)

    assert [source[item.start : item.end] for item in ranges] == [
        "https://example.com/path",
        "user@example.com",
    ]
    result = convert_pipeline(source)
    assert result.normalized_text == source
    assert "에서만" not in [item.text for item in result.normalized_segments if item.handler is Handler.PROTECTED]


def test_paths_email_and_inline_code_are_preserved_verbatim() -> None:
    source = r"C:\work\GPT를 /tmp/GPT를 user@example.com `GPT 3개`"
    result = convert_pipeline(source)

    assert result.normalized_text == source
    assert [item.handler for item in result.normalized_segments].count(Handler.PROTECTED) == 4
    assert result.complete


@pytest.mark.parametrize("unsafe", ["\x00", "\u200b", "\ue000"])
def test_unsafe_character_inside_url_is_not_hidden_by_protection(unsafe: str) -> None:
    source = f"https://example.com/a{unsafe}b"

    g2p_result = convert_pipeline(source)
    normalizer_result = KoreanTTSNormalizer().convert(source)
    assert g2p_result.normalized_text == f"https://example.com/a{unsafe}비"
    assert normalizer_result.normalized_text == source
    for result in (g2p_result, normalizer_result):
        assert not result.complete
        assert "unsupported_character" in {item.code for item in result.diagnostics}


def test_multibacktick_url_parentheses_and_spaced_windows_path_are_atomic() -> None:
    source = r"``5개`` https://example.com/a(5) C:\Program Files\App 2\run.exe"
    ranges = scan_protected(source)

    assert [source[item.start : item.end] for item in ranges] == [
        "``5개``",
        "https://example.com/a(5)",
        r"C:\Program Files\App 2\run.exe",
    ]


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\2026-08-07\5kg\model.wav",
        r"models\2026-08-07\5kg\model.wav",
        r".\models\2026-08-07\5kg\model.wav",
        r"..\models\3kg\model.wav",
        "./models/2026-08-07/5kg/model.wav",
        "../models/3kg/model.wav",
        "../../models/2026-08-07/model.wav",
    ],
)
def test_relative_and_unc_paths_are_protected_as_one_range(path: str) -> None:
    ranges = scan_protected(path)

    assert len(ranges) == 1
    assert ranges[0].start == 0
    assert ranges[0].end == len(path)
    assert path[ranges[0].start : ranges[0].end] == path


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\2026-08-07\5kg\model.wav",
        r"models\2026-08-07\5kg\model.wav",
        r".\models\2026-08-07\5kg\model.wav",
        r"..\models\3kg\model.wav",
        "./models/2026-08-07/5kg/model.wav",
        "../models/3kg/model.wav",
    ],
)
def test_normalizer_does_not_rewrite_numbers_inside_relative_or_unc_path(
    path: str,
) -> None:
    result = KoreanTTSNormalizer().convert(path)

    assert result.normalized_text == path
    assert result.complete
    assert result.rewrites == ()
    assert result.diagnostics == ()


def test_relative_path_particle_remains_outside_protected_range() -> None:
    path = r"models\2026-08-07\5kg\model.wav"
    source = path + "를 확인하세요."
    ranges = scan_protected(source)

    assert [source[item.start : item.end] for item in ranges] == [path]


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\2026-08-07\5kg\model.wav",
        r"models\2026-08-07\5kg\model.wav",
        "./models/2026-08-07/5kg/model.wav",
    ],
)
def test_relative_path_does_not_absorb_neighboring_english_words(path: str) -> None:
    source = f"open {path} now"
    ranges = scan_protected(source)

    assert [source[item.start : item.end] for item in ranges] == [path]


@pytest.mark.parametrize(
    "source",
    [
        r"escape \n and \t",
        r"A\B",
        "and/or",
        "3\\개를 읽습니다.",
        "일반 문장./다음 문장",
    ],
)
def test_path_detection_does_not_protect_common_text_or_escape_examples(
    source: str,
) -> None:
    assert scan_protected(source) == ()


def test_compatibility_unit_rate_is_not_mistaken_for_a_posix_path() -> None:
    assert scan_protected("290㎞/h") == ()


def test_long_relative_path_is_scanned_as_one_range() -> None:
    path = "root\\" + "\\".join(f"part{index}" for index in range(1_000))
    ranges = scan_protected(path)

    assert len(ranges) == 1
    assert (ranges[0].start, ranges[0].end) == (0, len(path))


@pytest.mark.parametrize(
    "source",
    [
        pytest.param("1$" * 25_000, id="dollar-run"),
        pytest.param("1%" * 25_000, id="percent-run"),
        pytest.param("$" * 49_995 + r"\a\b", id="backslash-tail"),
    ],
)
def test_protected_scanner_avoids_quadratic_symbol_search(source: str) -> None:
    started = perf_counter()
    scan_protected(source)

    assert perf_counter() - started < 1.0
