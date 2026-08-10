"""고정된 한국어 TTS 파이프라인의 공개 진입점이다."""

from __future__ import annotations

from collections.abc import Mapping

from .normalizer import KoreanTTSNormalizer
from .pipeline import Pipeline


class G2P:
    """원문을 한국어 발화형으로 바꾼 뒤 표면 발음을 반환한다."""

    def __init__(
        self,
        *,
        lexicon: Mapping[str, str] | None = None,
        max_length: int = 10_000,
    ) -> None:
        self._normalizer = KoreanTTSNormalizer(
            lexicon=lexicon,
            max_length=max_length,
        )
        self._pipeline = Pipeline(max_length=max_length * 16)

    def __call__(self, text: str) -> str:
        """정규화와 G2P를 순서대로 적용한 표면 발음을 반환한다."""

        normalized = self._normalizer(text)
        return self._pipeline.run(normalized).surface_pronunciation
