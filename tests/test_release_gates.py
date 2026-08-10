"""기본 배포 경로가 network·경쟁 조건에 의존하지 않음을 검사한다."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import http.client
import socket
import urllib.request

import pytest

from g2p_ko import G2P


_SAMPLE = "한국어 기본 경로는 여러 스레드에서도 같은 결과를 반환합니다."


@pytest.mark.parametrize("workers", [1, 4, 16])
def test_default_g2p_is_deterministic_under_concurrent_calls(workers: int) -> None:
    g2p = G2P()
    expected = g2p(_SAMPLE)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        actual = list(executor.map(lambda _: g2p(_SAMPLE), range(workers)))

    assert actual == [expected] * workers


def test_default_runtime_does_not_call_network_apis(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked(*args: object, **kwargs: object) -> object:
        raise AssertionError("기본 런타임은 네트워크 API를 호출하면 안 됩니다.")

    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(urllib.request, "urlopen", blocked)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", blocked)
    monkeypatch.setattr(http.client.HTTPSConnection, "connect", blocked)

    assert G2P()(_SAMPLE)
