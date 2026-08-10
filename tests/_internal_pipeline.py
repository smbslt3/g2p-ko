"""공개 API에 노출하지 않는 G2P 단계의 상세 결과를 테스트한다."""

from g2p_ko.pipeline import Pipeline


def convert_pipeline(source: str, *, max_length: int = 10_000):  # type: ignore[no-untyped-def]
    return Pipeline(max_length=max_length).run(source)
