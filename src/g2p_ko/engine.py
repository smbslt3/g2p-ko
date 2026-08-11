"""고정된 한국어 TTS 파이프라인의 공개 진입점이다."""

from __future__ import annotations

from collections.abc import Mapping

from .errors import InputValidationError
from .normalizer import KoreanTTSNormalizer
from .pipeline import Pipeline
from .unicode import normalize_nfc_text


class G2P:
    """원문 또는 정규화된 발화형을 표면 발음으로 변환한다."""

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
        self._input_max_length = max_length

    def __call__(self, text: str, *, normalize: bool = True) -> str:
        """선택적으로 정규화한 뒤 표면 발음을 반환한다.

        ``normalize=False``는 이미 정규화된 입력에서 TTS 노멀라이저만 생략한다.
        기본 입력 검증과 NFC 정규화는 두 경로에서 모두 유지한다.
        """

        if not isinstance(normalize, bool):
            raise InputValidationError("normalize는 불리언이어야 합니다.")
        source = (
            self._normalizer(text)
            if normalize
            else normalize_nfc_text(text, max_length=self._input_max_length)
        )
        return self._pipeline.run(source).surface_pronunciation
