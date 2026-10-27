"""Sing a score in the voice of a recording.

usage: python sing.py --score examples/furusato.musicxml --voice me.wav [--voice me2.wav] --out out.wav
"""
import argparse

import soundfile as sf

from mame_singer import Singer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True, help="MusicXML, UST, USTX, MIDI, VSQ, VSQX, VPR, SVP, S5P or CCS, with Japanese (kana) lyrics")
    ap.add_argument("--voice", action="append", required=True, help="a recording of the voice (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", type=int, default=1, help="1 close to the recorded voice (default), 0 more singing-like")
    ap.add_argument("--transpose", type=int, default=0, help="semitones")
    ap.add_argument("--style", type=int, default=0, help="timing / pitch style 0-2")
    ap.add_argument("--model-dir", default=None, help="local model files instead of the Hugging Face download")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    singer = Singer(args.model_dir, args.device)
    y, _ = singer.sing(args.score, [sf.read(f, dtype="float32") for f in args.voice], args.mode, args.transpose,
                       args.style)
    sf.write(args.out, y, 44100)
    print(args.out)


if __name__ == "__main__":
    main()
