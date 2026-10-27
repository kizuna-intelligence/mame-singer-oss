"""Zero-shot singing: a score (MusicXML / .mxl / .ust) sung in the voice of a short recording."""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .model import Acoustic
from .voice import VoiceEncoder

SR = 44100
PP = Path(__file__).resolve().parents[1] / "third_party" / "PrettyPitch"
REPO = "kizuna-intelligence/mame-singer"
FILES = ("config.json", "acoustic.pt", "voice_encoder.pt", "vocoder.onnx", "f0.pt", "cons_dur.pt", "cons_dur.json", "phonemes.json")


def download(repo=REPO, token=None):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo, allow_patterns=list(FILES), token=token))


class Singer:
    def __init__(self, model_dir=None, device="cpu"):
        d = Path(model_dir) if model_dir else download()
        for p in (str(PP),):
            if p not in sys.path:
                sys.path.insert(0, p)
        from svs import render
        from svs.cons_dur import load_cons_dur
        self.render, self.device = render, device
        pp = render._load_infer_module("pp_infer", PP)
        f0_model, f0_cfg = pp.load_model(str(d / "f0.pt"), device)
        cons_model, cons_prof = load_cons_dur(d / "cons_dur.pt", profile_path=d / "cons_dur.json", device=device)
        phonemes = json.load(open(d / "phonemes.json"))["phonemes"]
        ac_cfg = {"phonemes": phonemes}
        self.ctx = render.SvsContext(
            pp=pp, ls=None, f0_model=f0_model, f0_cfg=f0_cfg, ac_model=None, ac_cfg=ac_cfg, voc=None,
            ph2id={p: i for i, p in enumerate(phonemes)}, frame_rate=float(f0_cfg["frame_rate"]),
            sample_rate=int(f0_cfg["sample_rate"]), hop=int(f0_cfg.get("hop", 256)), spk_id=0, spk_vec=None,
            device=device, cons_dur_model=cons_model, cons_dur_profile=cons_prof,
            phoneset=render._resolve_phoneset(None, cons_model, ac_cfg, verbose=False))
        self.acoustic = Acoustic.load(d / "acoustic.pt", device)
        self.voices = VoiceEncoder(d / "voice_encoder.pt", device)
        self.vocoder = ort.InferenceSession(str(d / "vocoder.onnx"), providers=["CPUExecutionProvider"])

    def phrases(self, score, transpose=0, style=0):
        """the score's phrases with phonemes, frame counts, F0, voicing and start time"""
        from svs.score_io import load_score
        r = self.render
        self.ctx.spk_id = int(style)
        out = []
        for k, p in enumerate(self.load(score)):
            for n in p.notes:
                n.pitch_midi = int(n.pitch_midi) + int(transpose)
            ph, naturals = r.phrase_cons_durations(self.ctx, p)
            seq, ph_durs = r.phrase_duration_postprocess(self.ctx, ph, naturals)
            f0o = r.phrase_pitch_correction(self.ctx, r.phrase_f0(self.ctx, seq, seed=k))
            f0 = np.asarray(f0o["f0_hz"], np.float32)
            voicing = np.asarray(f0o["uv"], np.float32)
            ph_durs = r._align_durs(ph_durs, len(f0))
            ids = np.asarray([self.ctx.ph2id[x] for x in seq.phonemes], np.int64)
            start, _ = r.phrase_start_sec(ph, seq, self.ctx.phoneset)
            out.append(dict(start=start, ids=ids, durs=ph_durs, f0=f0, voicing=voicing,
                            hard=r.vocoder_uv(self.ctx, ids, ph_durs, voicing, len(f0))))
        return out

    def load(self, score):
        """phrases of any supported score file"""
        from svs.score_io import ScoreNote, load_score, score_to_phrases
        from . import formats
        if Path(score).suffix.lower() not in formats.EXTENSIONS:
            return load_score(Path(score), phoneset=self.ctx.phoneset)
        r = formats.read(score)
        k, sn, t = r["bpm"] / 60.0, [], 0.0
        for n in r["notes"]:
            s, d = n["start"] * k, n["dur"] * k
            if s > t + 1e-6:
                sn.append(ScoreNote(dur_beats=s - t, midi=0, lyric=None, is_rest=True))
            sn.append(ScoreNote(dur_beats=d, midi=n["midi"], lyric=n["lyric"] or None))
            t = s + d
        return score_to_phrases(sn, r["bpm"], phoneset=self.ctx.phoneset)

    def vocode(self, mel, f0, hard):
        y = self.vocoder.run(["waveform"], {"mel": np.ascontiguousarray(mel.T[None], np.float32),
                                            "f0": f0[None, None].astype(np.float32),
                                            "uv": (1.0 - hard)[None, None].astype(np.float32)})[0]
        return np.asarray(y, np.float32).reshape(-1)

    def sing(self, score, recordings, mode=1, transpose=0, style=0):
        """score file, [(wav, sr), ...] of the voice -> (44.1 kHz vocal, offset of the score's time 0 in it)

        mode 1 follows the recorded voice closely (it treats the recording as a speaking voice); mode 0 sings
        in a more singing-like voice."""
        emb = self.voices.voice(recordings)
        parts = self.phrases(score, transpose, style)
        offset = max(0.0, -min(p["start"] for p in parts))
        segs = []
        for p in parts:
            mel = self.acoustic.mel(p["ids"], p["durs"], p["f0"], p["voicing"], emb, mode, self.device)
            segs.append((int(round((p["start"] + offset) * SR)), self.vocode(mel, p["f0"], p["hard"])))
        y = np.zeros(max(a + len(w) for a, w in segs), np.float32)
        for a, w in segs:
            y[a:a + len(w)] += w
        return y, offset
