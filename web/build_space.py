"""Assemble the static Hugging Face Space: the page, the Python front end as py.zip, the models, an example score.

usage: python web/build_space.py --models DIR (ONNX and data files) --out DIR
"""
import argparse
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PP = ROOT / "third_party" / "PrettyPitch"
PY = ["infer.py", "prettypitch/__init__.py", "prettypitch/baseline.py", "prettypitch/correction.py",
      "prettypitch/pitch.py", "prettypitch/decomp.py", "prettypitch/config.py", "svs/__init__.py", "svs/score_io.py",
      "svs/duration_postprocess.py", "duration_model/phoneset.py", "duration_model/profile.py",
      "duration_model/features.py", "dict/ja.yaml", "dict/ja.phonemes", "dict/ja.mora", "dict/en.yaml",
      "dict/en.phonemes"]
MODELS = ["acoustic.onnx", "voice_encoder.onnx", "f0_cond.onnx", "f0_denoise.onnx", "vocoder.onnx", "f0.json",
          "cons_dur.npz", "cons_dur_config.json", "cons_dur.json", "phonemes.json"]
CARD = """---
title: mame-singer
emoji: 🎤
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
license: cc-by-4.0
---

話し声を数秒録音するだけで、その声で歌わせる歌声合成です。楽譜はブラウザの上で作れて、MusicXML・UTAU・OpenUtau・MIDI・VOCALOID・Synthesizer V・CeVIO の楽譜も開けます。計算はすべてブラウザの中で行います。

- ソースコード: https://github.com/kizuna-intelligence/mame-singer-oss
- モデル: https://huggingface.co/kizuna-intelligence/mame-singer

改造や再配布をする場合は、ライセンスに従ってください。

ライセンス: Kizuna Intelligence が作った部分（音響モデル `models/acoustic.onnx` とページのコード）は CC BY 4.0（`LICENSE`）。再配布・改変時は「mame-singer（Kizuna Intelligence） https://huggingface.co/spaces/kizuna-intelligence/mame-singer」の表示が必要で、守られない場合は CC BY 4.0 第 6 条により許諾が終了します。F0・子音長モデルと楽譜処理（`py.zip` の PrettyPitch 部分）、ボコーダ、声の特徴の抽出は、それぞれの MIT License（`LICENSE-*`）です。
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in ("index.html", "app.js", "editor.js", "worker.js"):
        shutil.copy(HERE / f, out / f)
    with zipfile.ZipFile(out / "py.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(HERE / "mame_web.py", "mame_web.py")
        z.writestr("duration_model/__init__.py", "")
        z.writestr("mame_singer/__init__.py", "")
        z.write(ROOT / "mame_singer" / "formats.py", "mame_singer/formats.py")
        for f in PY:
            z.write(PP / f, f)
    (out / "models").mkdir(exist_ok=True)
    for f in MODELS:
        shutil.copy(Path(a.models) / f, out / "models" / f)
    (out / "examples").mkdir(exist_ok=True)
    shutil.copy(ROOT / "examples" / "furusato.musicxml", out / "examples" / "furusato.musicxml")
    shutil.copy(HERE / "furusato.json", out / "examples" / "furusato.json")
    shutil.copy(ROOT / "LICENSE", out / "LICENSE")
    for name in ("PrettyPitch", "NHVSing", "OpenVoice"):
        shutil.copy(ROOT / "LICENSES" / f"{name}-MIT.txt", out / f"LICENSE-{name}")
    (out / "README.md").write_text(CARD, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
