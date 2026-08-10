"""공유 Kiwi 형태소 분석기를 필요할 때 한 번만 만든다."""

from __future__ import annotations

from threading import Lock

from .errors import BackendUnavailableError
from .model import AnalysisToken


_runtime_kiwi: object | None = None
_kiwi_lock = Lock()


def _create_kiwi() -> object:
    """첫 형태소 후보에서만 고정된 Kiwi 구현을 가져온다."""

    try:
        from kiwipiepy import Kiwi
    except ImportError as error:
        raise BackendUnavailableError(
            "kiwipiepy가 설치되어 있지 않습니다. 프로젝트 의존성을 설치하세요."
        ) from error
    return Kiwi()


def _get_kiwi() -> object:
    """프로세스 안에서 노멀라이저와 G2P가 같은 Kiwi를 재사용한다."""

    global _runtime_kiwi
    kiwi = _runtime_kiwi
    if kiwi is not None:
        return kiwi
    with _kiwi_lock:
        if _runtime_kiwi is None:
            _runtime_kiwi = _create_kiwi()
        return _runtime_kiwi


class KiwiAnalyzer:
    """공식 ``Kiwi.split`` 결과를 내부 분석 token으로 옮긴다."""

    __slots__ = ()

    def analyze(self, text: str) -> tuple[AnalysisToken, ...]:
        split = getattr(_get_kiwi(), "split", None)
        if split is None:
            raise BackendUnavailableError(
                "설치된 kiwipiepy에는 Kiwi.split이 없습니다. 0.23.2 이상이 필요합니다."
            )
        return tuple(
            AnalysisToken(part.form, part.tag, part.start, part.start + part.len)
            for part in split(text)
        )
