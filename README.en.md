# g2p-ko

Normalizes Korean TTS input into spoken forms and converts it to surface pronunciation.

[한국어](README.md)

## Features

- Normalizes numbers, units, dates, times, currencies, fractions, and abbreviations
  - `3kg~5kg` → `삼 킬로그램에서 오 킬로그램`
  - `2/5kg` → `오분의 이 킬로그램`
  - `6,000원` → `육천 원`
- Uses a small statistical model to choose context-dependent number readings
  - It considers only numbers that deterministic rules cannot settle, choosing a native Korean, Sino-Korean, or digit-by-digit reading. It keeps the rule result when the rule is conclusive or model confidence is insufficient.
  - The same `3번` is read as `세 번` for a count and `삼 번` for an identifier.
  - `영화를 3번 봤다.` → `영화를 세 번 봤다.`
  - `3번 창구로 가세요.` → `삼 번 창구로 가세요.` (model decision)
  - `우편번호는 54130입니다.` → `우편번호는 오 사 일 삼 공입니다.` (model decision)
- Applies Korean phonological rules
  - `국물` → `궁물`
  - `같이` → `가치`
  - `읽는` → `잉는`
- Prioritizes conventional readings such as NASA, KIA, and TV
  - `NASA` → `나사`
  - `KIA` → `기아`
  - `FIFA와 TV` → `피파와 티비`
- Spells out other English text using Korean letter names
  - `ABC` → `에이비씨`
  - `PYTHON` → `피와이티에이치오엔`
  - `TEST` → `티이에스티`
- Preserves uncertain or unsupported input
  - `A-1` → `A-1`
  - `1-3개` → `1-3개`
  - `192.168.0.1` → `192.168.0.1`
- Supports a user lexicon
  - `NAVER` → `네이버`
  - `RIDI` → `리디`
  - `DAANGN` → `당근`

The number-reading model considers the number's length and value, its unit, and the surface forms and part-of-speech tags of nearby Kiwi morphemes. It was trained on AIHub's [숫자가 포함된 패턴 발화 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&dataSetSn=484). See the [data and model notice](NOTICE) for its structure and detailed data composition.

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

### Already normalized input

Pass output from `KoreanTTSNormalizer` with `normalize=False` to skip duplicate normalization. This path retains input validation and NFC, then applies only English spelling and Korean phonological rules.

```python
normalized = normalizer("사과 3개는 6,000원입니다.")
g2p(normalized, normalize=False)
# >>> '사과 세 개는 육천 워님니다.'
```

Using `normalize=False` on raw text skips number, unit, conventional-reading, and user-lexicon transformations.

### User lexicon

Pass a JSON user lexicon through `lexicon` to replace the default reading with a complete Korean spoken form.

```python
import json
from pathlib import Path
from g2p_ko import G2P

g2p = G2P()
g2p("NAVER")
# >>> '에네이브이이알'

# lexicon.json must be an object in the form
# {"source spelling": "complete Korean spoken form"}.
lexicon = json.loads(Path("lexicon.json").read_text(encoding="utf-8"))
g2p = G2P(lexicon=lexicon)
g2p("NAVER")
# >>> '네이버'
```

## References

Prior Korean G2P implementations: [g2pK](https://github.com/Kyubyong/g2pK) ·
[g2pK+](https://github.com/harmlessman/g2pkk) ·
[g2pk2](https://github.com/tenebo/g2pk2) ·
[g2pk3](https://github.com/kdrkdrkdr/g2pk3)

[MIT License](LICENSE) · [Data and model notice](NOTICE)
