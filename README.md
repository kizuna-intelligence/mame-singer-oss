# mame-singer

短い録音の声で、楽譜を歌わせる歌声合成です。歌わせたい人の歌声は要りません。話し声の録音を数秒用意するだけで歌えます。

ブラウザだけで試せます: https://huggingface.co/spaces/kizuna-intelligence/mame-singer （計算はすべてブラウザの中で行います）

## 使い方

```sh
pip install -r requirements.txt
huggingface-cli login            # モデルは kizuna-intelligence/mame-singer にあります
python app.py                    # GUI（ブラウザで開きます）
python sing.py --score examples/furusato.musicxml --voice my_voice.wav --out out.wav
```

- 楽譜: MusicXML（.musicxml / .mxl）、UTAU（.ust）、OpenUtau（.ustx）、MIDI、VOCALOID（.vsq / .vsqx / .vpr）、Synthesizer V（.svp / .s5p）、CeVIO（.ccs）。歌詞はかなで書いてください。
- 声: 話し声の録音を数秒〜。複数渡すと平均を取ります。
- `--mode 1`（既定）は録音の声に近く、`--mode 0` はより歌声らしくなります。
- GPU があれば `--device cuda` で速くなります。

`web/` はブラウザ版（Pyodide と onnxruntime-web で動く）のソースです。

## クレジット

mame-singer は次のモデルとコードを使っています。作者の方々に感謝します。

| 使っているところ | 元 | 作者 | ライセンス |
|---|---|---|---|
| ボコーダ（メルスペクトログラム → 波形） | [NHVSing](https://github.com/wavtechyukky/NHVSing) | wavtechyukky | MIT |
| F0 モデル、子音長モデル、楽譜の読み込みと音素長の処理 | [PrettyPitch](https://github.com/wavtechyukky/PrettyPitch) | wavtechyukky | MIT |
| 声の特徴の抽出（参照エンコーダ） | [OpenVoice](https://github.com/myshell-ai/OpenVoice) | MyShell.ai | MIT |

音響モデルと、このリポジトリのそれ以外のコードは Kizuna Intelligence が作りました。

## ライセンス

Kizuna Intelligence が作った部分（このリポジトリのコードと音響モデル）は **CC BY 4.0**（`LICENSE`）です。再配布・改変・組み込みは自由ですが、次の表示を付けてください。

> mame-singer（Kizuna Intelligence） https://huggingface.co/spaces/kizuna-intelligence/mame-singer

表示が無いなど条件を守らない場合、CC BY 4.0 第 6 条により利用の許諾は自動的に終了します（違反に気づいてから 30 日以内に直せば元に戻ります）。

ほかの部分は、それぞれの MIT License のままです（`LICENSES/`）。

| 部分 | 元 | ライセンス |
|---|---|---|
| `third_party/PrettyPitch`、F0・子音長モデル、`examples/furusato.musicxml` | PrettyPitch | MIT（`LICENSES/PrettyPitch-MIT.txt`） |
| ボコーダ | NHVSing | MIT（`LICENSES/NHVSing-MIT.txt`） |
| 声の特徴の抽出（`mame_singer/voice.py` の参照エンコーダとその重み） | OpenVoice | MIT（`LICENSES/OpenVoice-MIT.txt`） |

合成した歌声を公開するときは、録音した声の持ち主の許諾と、その録音の利用条件に従ってください。
