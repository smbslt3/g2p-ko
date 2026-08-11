# g2p-ko

한국어 TTS 입력을 발화형으로 정규화하고 표면 발음으로 변환합니다.

[English](README.en.md)

## 기능

- 숫자·단위·날짜·시간·통화·분수·약어 정규화
  - `3kg~5kg` → `삼 킬로그램에서 오 킬로그램` / `2/5kg` → `오분의 이 킬로그램` / `6,000원` → `육천 원`
- 내장형 통계 모델로 문맥에 따라 달라지는 숫자 읽기 판단
  - 규칙으로 확정하기 어려운 숫자만 대상으로 고유어·한자어·낱자리 읽기 중 하나를 고릅니다. 규칙이 이미 확정했거나 모델의 확신이 부족하면 규칙 결과를 유지합니다.
  - `영화를 3번 봤다.` → `영화를 세 번 봤다.`
  - `3번 창구로 가세요.` → `삼 번 창구로 가세요.` (모델 판별)
  - `우편번호는 54130입니다.` → `우편번호는 오 사 일 삼 공입니다.` (모델 판별)
- 한국어 음운 규칙 적용
  - `국물` → `궁물`, `같이` → `가치`, `읽는` → `잉는`
- NASA·KIA·TV 같은 관용 읽기 우선 처리
  - `NASA` → `나사`, `KIA` → `기아`, `FIFA와 TV` → `피파와 티비`
- 나머지 영어를 알파벳 이름으로 변환
  - `ABC` → `에이비씨`, `PYTHON` → `피와이티에이치오엔`, `TEST` → `티이에스티`
- 불확실하거나 지원하지 않는 입력 보존
  - `A-1` → `A-1`, `1-3개` → `1-3개`, `192.168.0.1` → `192.168.0.1`
- 사용자 사전 지원
  - `NAVER` → `네이버`, `RIDI` → `리디`, `DAANGN` → `당근`

숫자 읽기 모델은 숫자의 길이와 값, 단위, 앞뒤 Kiwi 형태소의 표면형과 품사를 살펴봅니다. AIHub [숫자가 포함된 패턴 발화 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=484)를 기반으로 학습했으며, 자세한 구조와 데이터 구성은 [데이터 및 모델 고지](NOTICE)에 정리했습니다.

## 설치

Python 3.10 이상과 [uv](https://docs.astral.sh/uv/)가 필요합니다.

```powershell
git clone https://github.com/smbslt3/g2p-ko.git
cd g2p-ko
uv sync --dev --locked
```

PyPI 배포는 빌드 없이 `Kiwi.split()`을 제공하는 공식 wheel이 나온 뒤 진행합니다.

## 사용

```python
from g2p_ko import G2P, KoreanTTSNormalizer

normalizer = KoreanTTSNormalizer()
g2p = G2P()

normalizer("사과 3개는 6,000원입니다.")
# >>> '사과 세 개는 육천 원입니다.'
g2p("사과 3개는 6,000원입니다.")
# >>> '사과 세 개는 육천 워님니다.'

normalizer("3kg~5kg 정도 필요합니다.")
# >>> '삼 킬로그램에서 오 킬로그램 정도 필요합니다.'
g2p("3kg~5kg 정도 필요합니다.")
# >>> '삼 킬로그래메서 오 킬로그램 정도 피료함니다.'

normalizer("NASA와 TV 뉴스를 봤다.")
# >>> '나사와 티비 뉴스를 봤다.'
g2p("NASA와 TV 뉴스를 봤다.")
# >>> '나사와 티비 뉴스를 봗따.'
```

### 이미 정규화한 입력

`KoreanTTSNormalizer`의 출력은 `normalize=False`로 넘겨 중복 정규화를 생략할 수 있습니다. 이 경로는 입력 검증과 NFC를 유지하고, 영어 철자화와 한국어 음운 규칙만 적용합니다.

```python
normalized = normalizer("사과 3개는 6,000원입니다.")
g2p(normalized, normalize=False)
# >>> '사과 세 개는 육천 워님니다.'
```

원문에 `normalize=False`를 사용하면 숫자·단위·관용 읽기·사용자 사전 처리가 생략됩니다.

### 사용자 사전

JSON 형식의 사용자 사전을 `lexicon` 인자로 전달하면 기본 읽기를 원하는 발화형으로 바꿀 수 있습니다.

```python
import json
from pathlib import Path
from g2p_ko import G2P

g2p = G2P()
g2p("NAVER")
# >>> '에네이브이이알'

# lexicon.json은 {"원문 표기": "완결된 한국어 발화형"} 형태의 JSON 객체입니다.
# 예: {"NAVER": "네이버", "RIDI": "리디", "DAANGN": "당근"}
lexicon = json.loads(Path("lexicon.json").read_text(encoding="utf-8"))
g2p = G2P(lexicon=lexicon)
g2p("NAVER")
# >>> '네이버'
```

## 참고

선행 한국어 G2P 구현: [g2pK](https://github.com/Kyubyong/g2pK) · [g2pK+](https://github.com/harmlessman/g2pkk) · [g2pk2](https://github.com/tenebo/g2pk2) · [g2pk3](https://github.com/kdrkdrkdr/g2pk3)

[MIT 라이선스](LICENSE) · [데이터 출처 고지](NOTICE)
