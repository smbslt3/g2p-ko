"""한국어 TTS 입력 정규화의 공개 API다."""

from .engine import G2P
from .errors import (
    BackendUnavailableError,
    ConversionPolicyError,
    G2PError,
    InputValidationError,
)
from .model import (
    Boundary,
    Diagnostic,
    Handler,
    OutputSegment,
    Rewrite,
    RewriteStage,
    Severity,
    SourceSpan,
)
from .normalizer import KoreanTTSNormalizer, NormalizationResult

__all__ = [
    "BackendUnavailableError",
    "Boundary",
    "ConversionPolicyError",
    "Diagnostic",
    "G2P",
    "G2PError",
    "Handler",
    "InputValidationError",
    "KoreanTTSNormalizer",
    "NormalizationResult",
    "OutputSegment",
    "Rewrite",
    "RewriteStage",
    "Severity",
    "SourceSpan",
]
