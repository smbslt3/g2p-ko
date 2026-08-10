"""Kiwi 형태소 문맥으로 숫자 읽기를 판별하는 보조 장치다.

규칙만으로는 `12번`이 열두 번인지 십이 번인지, `55`가 오십오인지 오 오인지
가릴 수 없다. 숫자 토큰 주변 형태소에서 자질을 뽑아 희소 선형 모델로
읽기를 고르고, 확신이 임계값에 못 미치면 판단을
포기해 규칙의 기본값을 그대로 쓴다.

런타임 의존성을 늘리지 않도록 추론은 순수 파이썬으로만 한다.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import exp, isfinite
from threading import Lock

from ..analyzer import KiwiAnalyzer
from ..errors import BackendUnavailableError
from ..model import AnalysisToken


_MODEL_RESOURCE = "data/number_reading_model.json"
_SCHEMA_VERSION = 2
_FEATURE_VERSION = 3
CLASSES = ("native", "sino", "digitwise")
# 맨숫자 자리는 규칙의 한자어 읽기를 기본으로 삼고 낱자리만 교정한다.
# 고유어는 단위 없이는 설 수 없어 후보에서 제외한다.
_ALLOWED_OVERRIDES = {
    "unit": frozenset(CLASSES),
    "bare": frozenset({"digitwise"}),
}
_MAX_LENGTH_FEATURE = 8


class ModelFormatError(ValueError):
    """배포된 판별 모델을 신뢰할 수 없다."""


# 임계값으로 쓸 수 있는 최댓값. 1을 넘기면 어떤 확률도 도달할 수 없어
# 그 방향으로는 교정하지 않는다는 뜻이 된다.
_NEVER_OVERRIDE = 1.01


def _finite(value: object, label: str) -> float:
    """무한대·NaN 계수는 확신 판정을 무력화하므로 받아들이지 않는다."""

    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ModelFormatError(f"{label}이(가) 숫자가 아닙니다.") from error
    if not isfinite(number):
        raise ModelFormatError(f"{label}에 유한하지 않은 값이 있습니다.")
    return number


@dataclass(frozen=True, slots=True)
class NumberReadingModel:
    """희소 선형 판별기의 계수와 보정된 임계값이다.

    ``weights``와 ``thresholds``는 dict라 내용까지 얼어 있지는 않다. 적재 후에는
    아무도 고치지 않는다는 전제로 공유한다.
    """

    classes: tuple[str, ...]
    bias: tuple[float, ...]
    weights: dict[str, tuple[float, ...]]
    # {클래스: {판별 자리: 임계값}}. 단위가 붙은 자리와 맨숫자 자리는 신뢰도
    # 분포가 달라 따로 보정한다.
    thresholds: dict[str, dict[str, float]]

    def threshold(self, name: str, site: str) -> float:
        """보정된 임계값. 없으면 그 방향으로 교정하지 않는다."""

        return self.thresholds.get(name, {}).get(site, _NEVER_OVERRIDE)

    def predict(self, features: tuple[str, ...]) -> tuple[float, ...]:
        """자질 목록에 대한 클래스 확률을 반환한다."""

        # 클래스가 세 개로 고정되어 있으므로 작은 list와 enumerate를 매 자질마다
        # 만들지 않는다. 숫자 하나를 판별할 때 이 루프가 가장 자주 실행된다.
        native, sino, digitwise = self.bias
        weights = self.weights
        for feature in features:
            weight = weights.get(feature)
            if weight is None:
                continue
            native += weight[0]
            sino += weight[1]
            digitwise += weight[2]
        largest = max(native, sino, digitwise)
        native_probability = exp(native - largest)
        sino_probability = exp(sino - largest)
        digitwise_probability = exp(digitwise - largest)
        # 보정 임계값이 과거 확률과 정확히 같은 경계 사례가 있으므로 기존
        # ``sum(list)``와 동일하게 0에서 시작하는 덧셈 순서를 유지한다.
        total = sum((native_probability, sino_probability, digitwise_probability))
        return (
            native_probability / total,
            sino_probability / total,
            digitwise_probability / total,
        )


_model_cache: NumberReadingModel | None = None
_model_lock = Lock()


def _parse_model(payload: object) -> NumberReadingModel:
    if not isinstance(payload, dict):
        raise ModelFormatError("판별 모델의 최상위 구조가 객체가 아닙니다.")
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ModelFormatError("지원하지 않는 판별 모델 schema_version입니다.")
    if payload.get("feature_version") != _FEATURE_VERSION:
        raise ModelFormatError("지원하지 않는 판별 모델 feature_version입니다.")
    classes = tuple(payload.get("classes", ()))
    if classes != CLASSES:
        raise ModelFormatError("판별 모델의 클래스 목록이 계약과 다릅니다.")
    bias = tuple(_finite(value, "bias") for value in payload.get("bias", ()))
    if len(bias) != len(classes):
        raise ModelFormatError("판별 모델 bias 길이가 클래스 수와 다릅니다.")
    raw_weights = payload.get("weights")
    if not isinstance(raw_weights, dict):
        raise ModelFormatError("판별 모델 weights가 객체가 아닙니다.")
    weights: dict[str, tuple[float, ...]] = {}
    for key, value in raw_weights.items():
        row = tuple(_finite(item, f"자질 {key!r}의 계수") for item in value)
        if len(row) != len(classes):
            raise ModelFormatError(f"자질 {key!r}의 계수 길이가 올바르지 않습니다.")
        weights[key] = row
    raw_thresholds = payload.get("thresholds")
    if not isinstance(raw_thresholds, dict):
        raise ModelFormatError("판별 모델 thresholds가 객체가 아닙니다.")
    thresholds: dict[str, dict[str, float]] = {}
    for key, per_site in raw_thresholds.items():
        if not isinstance(per_site, dict):
            raise ModelFormatError(f"{key} 임계값이 판별 자리별 객체가 아닙니다.")
        row: dict[str, float] = {}
        for site, value in per_site.items():
            threshold = _finite(value, f"{key}/{site} 임계값")
            # 확률과 비교하는 값이므로 범위를 벗어나면 교정 여부를 통제할 수 없다.
            # 1을 넘는 값은 "이 방향으로는 교정하지 않음"을 뜻하도록 허용한다.
            if not 0.0 <= threshold <= _NEVER_OVERRIDE:
                raise ModelFormatError(f"{key}/{site} 임계값이 허용 범위를 벗어났습니다.")
            row[site] = threshold
        thresholds[key] = row
    if not set(thresholds) <= set(classes):
        raise ModelFormatError("판별 모델 thresholds에 알 수 없는 클래스가 있습니다.")
    return NumberReadingModel(classes, bias, weights, thresholds)


def load_model() -> NumberReadingModel:
    """배포된 판별 모델을 처음 필요한 시점에 한 번만 읽는다."""

    import json
    from importlib.resources import files

    global _model_cache
    model = _model_cache
    if model is not None:
        return model
    with _model_lock:
        if _model_cache is None:
            resource = files("g2p_ko").joinpath(_MODEL_RESOURCE)
            _model_cache = _parse_model(json.loads(resource.read_text(encoding="utf-8")))
        model = _model_cache
    return model


def extract_features(
    tokens: tuple[AnalysisToken, ...],
    text: str,
    *,
    start: int,
    end: int,
    context_end: int | None = None,
    unit: str,
    starts: list[int] | None = None,
) -> tuple[str, ...]:
    """숫자 구간 주변 형태소에서 학습·추론이 공유하는 자질 키를 만든다.

    ``start``·``end``는 숫자 자체의 구간이고, ``context_end``는 단위와 조사까지
    포함한 토큰의 끝이다. 판별을 가르는 신호는 대개 단위 **뒤** 낱말이므로
    (`3번 문제`의 문제 vs `3번 봤다`의 보다) 다음 형태소는 단위 너머에서 찾는다.

    학습 스크립트도 이 함수를 그대로 불러 쓴다. 자질 규칙이 갈리면 배포된
    계수가 무의미해지므로 정의를 한 곳에만 둔다.

    ``starts``는 토큰 시작 좌표를 미리 뽑아 둔 목록이다. 한 문장에서 여러 숫자를
    판별할 때 같은 목록을 다시 만들지 않도록 호출자가 넘길 수 있다.
    """

    if context_end is None or context_end < end:
        context_end = end
    if starts is None:
        starts = [token.normalized_start for token in tokens]
    digits = end - start
    features: list[str] = [f"len={min(digits, _MAX_LENGTH_FEATURE)}"]
    if unit:
        features.append(f"u={unit}")
    # 고유어 수사가 살아 있는 구간에서는 수의 값 자체가 신호다.
    # `20번`은 스무 번(횟수)으로 기울고 `37번`은 삼십칠 번(번호)으로 기운다.
    if digits <= 2:
        surface = text[start:end]
        if surface.isascii() and surface.isdigit():
            value = int(surface)
            features.append(f"v={value}")
            if unit:
                features.append(f"uv={unit}:{value}")

    # 숫자와 다음 낱말 사이의 형태소(단위·조사) 중 마지막 것이 조사 신호다.
    index = bisect_left(starts, end)
    trailing: AnalysisToken | None = None
    while index < len(tokens) and tokens[index].normalized_start < context_end:
        trailing = tokens[index]
        index += 1
    if trailing is not None:
        features.append(f"sf={trailing.form}/{trailing.tag}")
        features.append(f"st={trailing.tag}")

    if index < len(tokens):
        following = tokens[index]
        features.append(f"nf={following.form}/{following.tag}")
        features.append(f"nt={following.tag}")
        content_words = _content_words(tokens, index, 1)
        for rank, content in enumerate(content_words, start=1):
            features.append(f"nc{rank}={content.form}/{content.tag}")
        # 선형 모델은 "한 자리 수 + 번은 횟수, 단 뒤에 문제·창구가 오면 번호"
        # 같은 조건부 규칙을 따로 적어 주지 않으면 표현하지 못한다.
        # 이 결합 자질은 단위 뒤 낱말과만 짝짓는다. 구별 신호가 단위 앞에
        # 오는 반대 어순(`문제 7번을 틀렸다`)은 아직 자질이 없어 잡지
        # 못한다. 대칭 자질(`pn=`, `pl=`)을 앞 낱말에도 추가하는 것이
        # 후속 과제다.
        if content_words:
            head = f"{content_words[0].form}/{content_words[0].tag}"
            if unit:
                features.append(f"un={unit}|{head}")
            features.append(f"ln={min(digits, _MAX_LENGTH_FEATURE)}|{head}")
    else:
        features.append("eos=1")

    preceding_index = bisect_right(starts, start) - 1
    while preceding_index >= 0 and tokens[preceding_index].normalized_end > start:
        preceding_index -= 1
    if preceding_index >= 0:
        preceding = tokens[preceding_index]
        features.append(f"pf={preceding.form}/{preceding.tag}")
        features.append(f"pt={preceding.tag}")
        for rank, content in enumerate(_content_words(tokens, preceding_index, -1), start=1):
            features.append(f"pc{rank}={content.form}/{content.tag}")
    else:
        features.append("bos=1")
    return tuple(features)


# 조사와 어미는 무엇에 대한 숫자인지 알려주지 못한다. `우편번호는 54130`의
# 직전 형태소는 조사 `는`이지만 판별을 가르는 낱말은 그 앞의 `번호`다.
_FUNCTION_TAG_PREFIXES = ("J", "E")
_CONTENT_WORDS = 2
_CONTENT_SCAN_LIMIT = 8


def _content_words(
    tokens: tuple[AnalysisToken, ...], index: int, step: int
) -> list[AnalysisToken]:
    """조사·어미를 건너뛰고 가까운 내용어를 순서대로 모은다.

    가장 가까운 하나만 보면 `계좌번호 뒷자리 5566`에서 `뒷자리`에 막혀 정작
    중요한 `계좌번호`를 놓치므로 둘까지 본다.
    """

    found: list[AnalysisToken] = []
    scanned = 0
    while 0 <= index < len(tokens) and scanned < _CONTENT_SCAN_LIMIT:
        token = tokens[index]
        if not token.tag.startswith(_FUNCTION_TAG_PREFIXES):
            found.append(token)
            if len(found) == _CONTENT_WORDS:
                break
        index += step
        scanned += 1
    return found


class NumberReadingRun:
    """한 번의 변환 동안만 사는 판별 문맥이다.

    분석 결과와 실패 상태를 호출별 객체에 담아 같은 normalizer를 여러
    스레드가 동시에 써도 서로 간섭하지 않게 한다.
    """

    __slots__ = (
        "_analyzer",
        "_model",
        "_text",
        "_tokens",
        "_starts",
        "_analyzed",
        "_failure",
        "_reported",
    )

    def __init__(self, analyzer: KiwiAnalyzer, model: NumberReadingModel | None, text: str) -> None:
        self._analyzer = analyzer
        self._model = model
        self._text = text
        self._tokens: tuple[AnalysisToken, ...] = ()
        self._starts: list[int] = []
        self._analyzed = False
        self._failure: tuple[str, str] | None = None
        self._reported = False

    def _fail(self, code: str, message: str) -> None:
        if self._failure is None:
            self._failure = (code, message)

    def _ensure_analysis(self) -> tuple[AnalysisToken, ...] | None:
        """숫자 판별이 실제로 필요해진 시점에만 문장을 한 번 분석한다."""

        if self._failure is not None:
            return None
        if self._analyzed:
            return self._tokens
        self._analyzed = True
        # 외부 Kiwi 호출과 좌표 검증 실패는 모두 규칙 경로로 되돌린다.
        try:
            tokens = tuple(self._analyzer.analyze(self._text))
            valid = self._valid(tokens)
        except BackendUnavailableError as error:
            self._fail("analyzer_unavailable", str(error))
            return None
        except Exception as error:  # noqa: BLE001 - 외부 Kiwi의 모든 실패를 흡수한다
            self._fail("analyzer_failed", f"형태소 분석에 실패했습니다: {error}")
            return None
        if not valid:
            self._fail(
                "analyzer_failed",
                "형태소 분석 결과의 좌표나 표면형이 원문과 맞지 않습니다.",
            )
            return None
        self._tokens = tokens
        self._starts = [token.normalized_start for token in tokens]
        return tokens

    def _valid(self, tokens: tuple[AnalysisToken, ...]) -> bool:
        if not tokens:
            return False
        limit = len(self._text)
        previous_end = 0
        for token in tokens:
            if not isinstance(token, AnalysisToken) or not token.form or not token.tag:
                return False
            start, end = token.normalized_start, token.normalized_end
            if start < previous_end or end > limit or start >= end:
                return False
            if self._text[start:end] != token.form:
                return False
            previous_end = end
        return True

    def _get_model(self) -> NumberReadingModel | None:
        if self._model is not None:
            return self._model
        if self._failure is not None:
            # 이미 실패한 변환에서 손상된 파일을 매번 다시 읽지 않는다.
            return None
        try:
            self._model = load_model()
        except Exception as error:  # noqa: BLE001 - 손상된 배포 자산도 규칙 경로로 폴백한다
            self._fail(
                "analyzer_failed",
                f"문맥 판별에 필요한 자산을 준비할 수 없습니다: {error}",
            )
            return None
        return self._model

    def decide(
        self,
        start: int,
        end: int,
        *,
        unit: str,
        site: str,
        default: str,
        context_end: int | None = None,
    ) -> str | None:
        """규칙 기본값을 바꿔야 할 때만 새 읽기를 반환한다.

        기본값과 같거나 확신이 임계값에 못 미치면 ``None``을 돌려주어
        호출자가 규칙의 판단을 그대로 유지하게 한다.
        """

        model = self._get_model()
        if model is None:
            return None
        tokens = self._ensure_analysis()
        if tokens is None:
            return None
        features = extract_features(
            tokens,
            self._text,
            start=start,
            end=end,
            context_end=context_end,
            unit=unit,
            starts=self._starts,
        )
        probabilities = model.predict(features)
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        candidate = model.classes[best]
        if candidate == default:
            return None
        if candidate not in _ALLOWED_OVERRIDES.get(site, frozenset()):
            return None
        if probabilities[best] < model.threshold(candidate, site):
            return None
        return candidate

    def take_failure(self) -> tuple[str, str] | None:
        """분석 실패를 한 번만 진단으로 넘긴다.

        실패 상태 자체는 남겨 두어 이후 판별 요청이 계속 규칙 경로로
        폴백하게 하고, 진단만 중복되지 않도록 한다.
        """

        if self._failure is None or self._reported:
            return None
        self._reported = True
        return self._failure


class NumberReadingClassifier:
    """내장 Kiwi 분석기로 변환마다 판별 문맥을 만드는 얇은 진입점이다."""

    __slots__ = ("_analyzer", "_model")

    def __init__(self, analyzer: KiwiAnalyzer, *, model: NumberReadingModel | None = None) -> None:
        self._analyzer = analyzer
        self._model = model

    def new_run(self, text: str) -> NumberReadingRun:
        """이번 변환에서만 쓰는 판별 문맥을 만든다."""

        return NumberReadingRun(self._analyzer, self._model, text)
