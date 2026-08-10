# g2p-ko

한국어 TTS 입력을 발화형으로 정규화하고 표면 발음으로 변환합니다.

[English](README.en.md)

## 기능

- 숫자·단위·날짜·시간·통화·분수·약어 정규화
  - `3kg~5kg` → `삼 킬로그램에서 오 킬로그램`
- 간단한 통계 모델로 문맥에 따라 달라지는 숫자 읽기 판단
  - `사과 3개` → `사과 세 개`
- 한국어 음운 규칙 적용
  - `국물` → `궁물`
- NASA·KIA·TV 같은 관용 읽기 우선 처리
  - `NASA와 TV` → `나사와 티비`
- 나머지 영어를 알파벳 이름으로 변환
  - `ABC` → `에이비씨`
- 불확실하거나 지원하지 않는 입력 보존
  - `A-1` → `A-1`
- 사용자 사전 지원
  - `NAVER` → `네이버`

숫자 읽기 모델은 AIHub [숫자가 포함된 패턴 발화 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=484)를 기반으로 학습했습니다. 자세한 내용은 [데이터 및 모델 고지](NOTICE)를 참고하세요.

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

JSON 사용자 사전은 읽어서 `lexicon`에 넘기면 됩니다.

```python
import json
from pathlib import Path
from g2p_ko import G2P

# lexicon.json은 {"원문 표기": "완결된 한국어 발화형"} 형태의 JSON 객체입니다.
# 예: {"NAVER": "네이버", "RIDI": "리디"}
lexicon = json.loads(Path("lexicon.json").read_text(encoding="utf-8"))
g2p = G2P(lexicon=lexicon)
g2p("NAVER 뉴스")
# >>> '네이버 뉴스'
```

## 참고

선행 한국어 G2P 구현: [g2pK](https://github.com/Kyubyong/g2pK) · [g2pK+](https://github.com/harmlessman/g2pkk) · [g2pk2](https://github.com/tenebo/g2pk2) · [g2pk3](https://github.com/kdrkdrkdr/g2pk3)

[MIT 라이선스](LICENSE) · [데이터 출처 고지](NOTICE)
