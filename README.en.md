# g2p-ko

Normalizes Korean TTS input into spoken forms and converts it to surface pronunciation.

[한국어](README.md)

## Features

- Normalizes numbers, units, dates, times, currencies, fractions, and abbreviations
  - `3kg~5kg` → `삼 킬로그램에서 오 킬로그램`
- Uses a small statistical model to choose context-dependent number readings
  - `사과 3개` → `사과 세 개`
- Applies Korean phonological rules
  - `국물` → `궁물`
- Prioritizes conventional readings such as NASA, KIA, and TV
  - `NASA와 TV` → `나사와 티비`
- Spells out other English text using Korean letter names
  - `ABC` → `에이비씨`
- Preserves uncertain or unsupported input
  - `A-1` → `A-1`
- Supports a user lexicon
  - `NAVER` → `네이버`

The number-reading model was trained on AIHub's
[숫자가 포함된 패턴 발화 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=484).
See the [data and model notice](NOTICE) for details.

## Installation

Python 3.10 or later and [uv](https://docs.astral.sh/uv/) are required.

```powershell
git clone https://github.com/smbslt3/g2p-ko.git
cd g2p-ko
uv sync --dev --locked
```

PyPI distribution is deferred until an official Kiwi wheel provides `Kiwi.split()` without a local build.

## Usage

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

A JSON user lexicon must be an object in the form `{"source spelling": "complete Korean spoken form"}`. Load it and pass it as `lexicon`.

```python
import json
from pathlib import Path
from g2p_ko import G2P

lexicon = json.loads(Path("lexicon.json").read_text(encoding="utf-8"))
g2p = G2P(lexicon=lexicon)
g2p("NAVER 뉴스")
# >>> '네이버 뉴스'
```

## References

Prior Korean G2P implementations: [g2pK](https://github.com/Kyubyong/g2pK) ·
[g2pK+](https://github.com/harmlessman/g2pkk) ·
[g2pk2](https://github.com/tenebo/g2pk2) ·
[g2pk3](https://github.com/kdrkdrkdr/g2pk3)

[MIT License](LICENSE) · [Data and model notice](NOTICE)
