"""G2P와 독립적으로 사용할 수 있는 한국어 TTS 텍스트 노멀라이저다."""

from .engine import KoreanTTSNormalizer
from .model import NormalizationResult

__all__ = ["KoreanTTSNormalizer", "NormalizationResult"]
