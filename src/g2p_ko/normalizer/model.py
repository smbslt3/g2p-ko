"""독립 TTS 노멀라이저의 불변 공개 결과 모델이다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..model import Diagnostic, Rewrite


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """원문, 발화형 텍스트와 모든 변경·미해결 진단을 함께 반환한다."""

    source: str
    normalized_text: str
    rewrites: tuple[Rewrite, ...] = ()
    diagnostics: tuple[Diagnostic, ...] = ()
    complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON으로 직렬화할 수 있는 새 컨테이너를 반환한다."""

        return {
            "source": self.source,
            "normalized_text": self.normalized_text,
            "rewrites": [item.to_dict() for item in self.rewrites],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "complete": self.complete,
        }
