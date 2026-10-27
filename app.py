"""GUI: pick a score and a short recording of a voice, and the score is sung in that voice.

usage: python app.py [--model-dir DIR] [--device cuda] [--share]
"""
import argparse

import gradio as gr
import numpy as np
import soundfile as sf

from mame_singer import Singer

MODES = {"録音の声に近く（おすすめ）": 1, "歌声らしく": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    singer = Singer(args.model_dir, args.device)

    def run(score, voices, mic, mode, transpose, style):
        recs = [sf.read(f, dtype="float32") for f in (voices or [])]
        if mic is not None:
            sr, w = mic
            w = w.astype(np.float32) / (32768.0 if w.dtype == np.int16 else 1.0)
            recs.append((w, sr))
        if score is None or not recs:
            raise gr.Error("楽譜と声の録音を指定してください")
        y, _ = singer.sing(score, recs, MODES[mode], int(transpose), int(style))
        return 44100, y

    with gr.Blocks(title="mame-singer") as demo:
        gr.Markdown("# mame-singer\n楽譜（MusicXML / .mxl / .ust、歌詞はかな）と、歌わせたい声の短い録音を選んでください。")
        with gr.Row():
            with gr.Column():
                score = gr.File(label="楽譜", file_types=[".musicxml", ".xml", ".mxl", ".ust"], type="filepath")
                gr.Examples([["examples/furusato.musicxml"]], [score])
                voices = gr.File(label="声の録音（数秒〜、複数可）", file_count="multiple", type="filepath")
                mic = gr.Audio(label="またはマイクで録音", sources=["microphone"], type="numpy")
                mode = gr.Radio(list(MODES), value=list(MODES)[0], label="声の出し方")
                transpose = gr.Slider(-12, 12, 0, step=1, label="移調（半音）")
                style = gr.Radio([0, 1, 2], value=0, label="歌い回し")
                go = gr.Button("歌わせる", variant="primary")
            out = gr.Audio(label="歌声", type="numpy")
        go.click(run, [score, voices, mic, mode, transpose, style], out)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
