"""호출자가 복구 가능한 실패와 내부 오류를 구분할 예외 타입이다."""

from __future__ import annotations

from .model import SourceSpan


class G2PError(Exception):
    """패키지 예외의 기반 클래스다."""


class InputValidationError(G2PError, ValueError):
    """입력·옵션·사용자 사전이 공개 계약을 위반했을 때 발생한다."""


class BackendUnavailableError(G2PError, ImportError):
    """필요한 Kiwi backend를 사용할 수 없을 때 발생한다."""


class InternalInvariantError(G2PError, RuntimeError):
    """span 순서·경계 등 패키지 내부 불변식이 깨졌을 때 발생한다."""


class ConversionPolicyError(G2PError):
    """원문 보존 대신 오류 정책을 고른 호출에서 span 실패를 보고한다."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        source_span: SourceSpan | None = None,
        rule_id: str | None = None,
    ) -> None:
        self.code = code
        self.source_span = source_span
        self.rule_id = rule_id
        location = ""
        if source_span is not None:
            location = f" (span={source_span.start}:{source_span.end})"
        rule = f" [{rule_id}]" if rule_id else ""
        super().__init__(f"{code}{rule}{location}: {message}")
