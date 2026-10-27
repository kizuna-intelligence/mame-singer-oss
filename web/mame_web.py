"""mame-singer in the browser: the same pipeline as mame_singer.Singer, without PyTorch.

Runs in Pyodide (numpy, scipy, pyyaml). The networks run in onnxruntime-web through `run(name, feeds)`, an async
function the page provides; the consonant-duration MLP runs in numpy. For testing outside the browser,
`Bridge.local(dir)` runs the same ONNX files with onnxruntime.
"""
import json
import sys
import types

import numpy as np

sys.modules.setdefault("torch", types.ModuleType("torch"))   # imported (never used) by the vendored helpers

from duration_model import phoneset as _ps_mod  # noqa: E402
from duration_model.features import extract_phrase_records  # noqa: E402
from duration_model.profile import from_dict  # noqa: E402
from svs.duration_postprocess import (FALLBACK_CONS_DUR, PAU, postprocess_durations, split_phrase_syllables,  # noqa: E402
                                      split_syllable, to_frames)
from svs.score_io import (ScoreNote, _as_mora_table, _resolve_phonemes, load_score, parse_musicxml,  # noqa: E402
                          parse_ust, score_to_phrases)
import infer as _pp  # noqa: E402

SR = 44100
TARGET_RMS = 0.111


class Bridge:
    """async run(name, {input: array}) -> {output: array}"""

    def __init__(self, run):
        self._run = run

    async def run(self, name, feeds):
        return await self._run(name, feeds)

    @classmethod
    def local(cls, folder):
        import onnxruntime as ort
        sess = {}

        async def run(name, feeds):
            if name not in sess:
                sess[name] = ort.InferenceSession(f"{folder}/{name}.onnx", providers=["CPUExecutionProvider"])
            outs = sess[name].run(None, feeds)
            return {o.name: v for o, v in zip(sess[name].get_outputs(), outs)}
        return cls(run)


def _layernorm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class ConsDur:
    """the consonant-duration model (embeddings + MLP) in numpy"""

    def __init__(self, weights, config, profile):
        self.w = {k: weights[k] for k in weights.files} if hasattr(weights, "files") else dict(weights)
        self.ps = _ps_mod.load(config.get("lang", "ja"))
        self.profile = from_dict(profile)

    def predict(self, recs):
        w, ps = self.w, self.ps
        col = lambda k, dt: np.asarray([r[k] for r in recs], dt)
        ids = lambda k: np.asarray([ps.to_id(r[k]) for r in recs], np.int64)
        pnd = np.log1p(col("prev_note_dur", np.float32) / 0.3)[:, None]
        nnd = np.log1p(col("next_note_dur", np.float32) / 0.3)[:, None]
        pdf = np.clip(col("pitch_diff", np.float32), -12, 12)[:, None] / 12.0
        h = (w["cons_emb.weight"][ids("cons")] + w["prev_ph_emb.weight"][ids("prev_ph")]
             + w["next_ph_emb.weight"][ids("next_ph")] + w["pitch_emb.weight"][col("pitch_midi", np.int64)]
             + w["prev_exists_emb.weight"][col("prev_note_exists", np.int64)]
             + pnd @ w["prev_note_proj.weight"].T + nnd @ w["next_note_proj.weight"].T
             + pdf @ w["pitch_diff_proj.weight"].T
             + w["spk_emb.weight"][col("spk_id", np.int64)] + w["style_emb.weight"][col("style_id", np.int64)])
        for i in (0, 4):
            h = h @ w[f"mlp.{i}.weight"].T + w[f"mlp.{i}.bias"]
            h = np.maximum(_layernorm(h, w[f"mlp.{i + 1}.weight"], w[f"mlp.{i + 1}.bias"]), 0)
        y = (h @ w["mlp.8.weight"].T + w["mlp.8.bias"])[:, 0]
        return np.maximum(np.exp(y) - 0.001, 0.005)


def _flatten(phrase, ps):
    """as svs.cons_dur._flatten: phonemes, provisional lengths, note pitches, and where each consonant goes"""
    phs, durs, vals, where = [PAU], [float(phrase.leading_pau_sec)], [np.nan], {}
    for j, note in enumerate(phrase.notes):
        onset, nucleus, coda = split_syllable(note.phonemes, ps)
        span = float(note.dur_sec)
        per = span / len(nucleus) if nucleus else span
        for k, c in enumerate(onset):
            where[len(phs)] = (j, "onset", k)
            phs.append(c); durs.append(0.0); vals.append(np.nan)
        core_only = bool(nucleus) and not any(ps.is_nucleus(v) or ps.is_pau(v) for v in nucleus)
        if core_only:
            for k, c in enumerate(nucleus):
                where[len(phs)] = (j, "core", k)
                phs.append(c); durs.append(span if k == 0 else 0.0); vals.append(np.nan)
        else:
            for v in nucleus:
                phs.append(v); durs.append(per)
                vals.append(np.nan if v == PAU else float(note.pitch_midi))
        if not nucleus:
            phs.append(PAU); durs.append(span); vals.append(np.nan)
        for k, c in enumerate(coda):
            where[len(phs)] = (j, "coda", k)
            phs.append(c); durs.append(0.0); vals.append(np.nan)
        if note.gap_after_sec > 0:
            phs.append(PAU); durs.append(float(note.gap_after_sec)); vals.append(np.nan)
    phs.append(PAU); durs.append(float(phrase.trailing_pau_sec)); vals.append(np.nan)
    return phs, np.asarray(durs, np.float64), np.asarray(vals, np.float64), where


def read_score(path):
    """a score file -> (notes, bpm, tempo changes) for the editor: [{start, dur, midi, lyric}] in beats"""
    from mame_singer import formats
    ext = "." + str(path).lower().rsplit(".", 1)[-1]
    if ext in formats.EXTENSIONS:
        r = formats.read(path)
        k = r["bpm"] / 60.0
        return ([{"start": n["start"] * k, "dur": n["dur"] * k, "midi": n["midi"], "lyric": n["lyric"]} for n in r["notes"]],
                float(r["bpm"]), int(r["tempo_changes"]))
    if str(path).lower().endswith(".ust"):
        sn, bpm = parse_ust(path)
    else:
        sn, bpm = parse_musicxml(path, warn=False)
    notes, t = [], 0.0
    for n in sn:
        if not n.is_rest and n.midi > 0:
            notes.append({"start": t, "dur": n.dur_beats, "midi": int(n.midi), "lyric": n.lyric or ""})
        t += n.dur_beats
    return notes, float(bpm), 0


def to_score_notes(notes):
    """editor notes (any order, may overlap) -> ScoreNote list with rests, one voice"""
    notes = sorted(notes, key=lambda n: float(n["start"]))
    out, t = [], 0.0
    for i, n in enumerate(notes):
        start, dur = float(n["start"]), float(n["dur"])
        if i + 1 < len(notes):
            dur = min(dur, float(notes[i + 1]["start"]) - start)
        if dur <= 1e-6:
            continue
        if start > t + 1e-6:
            out.append(ScoreNote(dur_beats=start - t, midi=0, lyric=None, is_rest=True))
        lyric = str(n.get("lyric") or "").strip()
        out.append(ScoreNote(dur_beats=dur, midi=int(n["midi"]), lyric=lyric or None, is_rest=False))
        t = start + dur
    return out


class WebSinger:
    def __init__(self, bridge, files):
        """files: {name: bytes or str} of f0.json, cons_dur.npz, cons_dur_config.json, cons_dur.json, phonemes.json"""
        import io
        self.bridge = bridge
        f0 = json.loads(files["f0.json"])
        self.f0_cfg = f0["config"]
        self.sched = {k: np.asarray(v, np.float64) for k, v in f0["schedule"].items()}
        self.cons = ConsDur(np.load(io.BytesIO(files["cons_dur.npz"])), json.loads(files["cons_dur_config.json"]),
                            json.loads(files["cons_dur.json"]))
        self.ps = self.cons.ps
        phonemes = json.loads(files["phonemes.json"])["phonemes"]
        self.ph2id = {p: i for i, p in enumerate(phonemes)}
        self.frame_rate = float(self.f0_cfg["frame_rate"])

    # ── voice ──────────────────────────────────────────────────────────────
    async def voice(self, recordings22):
        """recordings already at 22.05 kHz (mono float32) -> (256,) voice vector"""
        es = []
        for w in recordings22:
            w = np.asarray(w, np.float32)
            active = np.abs(w) > 1e-2 * np.abs(w).max()
            w = np.clip(w * TARGET_RMS / np.sqrt(np.mean(w[active] ** 2)), -1, 1).astype(np.float64)
            n, hop = 1024, 256
            w = np.pad(w, ((n - hop) // 2, (n - hop) // 2), mode="reflect")
            frames = np.lib.stride_tricks.sliding_window_view(w, n)[::hop]
            win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)              # periodic Hann, as torch
            spec = np.abs(np.fft.rfft(frames * win, axis=-1))
            spec = np.sqrt(spec ** 2 + 1e-6).astype(np.float32)
            out = await self.bridge.run("voice_encoder", {"spec": spec[None]})
            es.append(np.asarray(out["emb"], np.float32).reshape(-1))
        e = np.mean(es, 0)
        return (e / np.linalg.norm(e)).astype(np.float32)

    # ── score front end ────────────────────────────────────────────────────
    def durations(self, phrase, style):
        """as svs.cons_dur.predict_cons_naturals + render.phrase_duration_postprocess"""
        ps = self.ps
        phrase = split_phrase_syllables(phrase, ps)
        onset, coda, core = [], [], []
        for note in phrase.notes:
            on, nuc, cd = split_syllable(note.phonemes, ps)
            core_only = bool(nuc) and not any(ps.is_nucleus(v) or ps.is_pau(v) for v in nuc)
            onset.append([FALLBACK_CONS_DUR] * len(on))
            coda.append([FALLBACK_CONS_DUR] * len(cd))
            core.append([FALLBACK_CONS_DUR] * len(nuc) if core_only else [])
        if phrase.notes:
            phs, durs, vals, where = _flatten(phrase, self.ps)
            recs = [r for r in extract_phrase_records(phs, durs, vals, int(style), 0, phoneset=self.ps)
                    if r["idx"] in where]
            if recs:
                slots = {"onset": onset, "coda": coda, "core": core}
                for r, d in zip(recs, self.cons.predict(recs)):
                    j, slot, k = where[r["idx"]]
                    slots[slot][j][k] = float(d)
        seq = postprocess_durations(phrase, self.cons.profile, onset, coda, core_nat=core, rest_before=1e9,
                                    phoneset=self.ps, spk_id=int(style), style_id=0)
        ph_durs, _ = to_frames(seq.dur_sec, self.frame_rate)
        return phrase, seq, ph_durs

    async def f0(self, seq, style, rng):
        cfg = self.f0_cfg
        score = {"spk_id": int(style), "style_id": 0, "phonemes": [
            {"ph": p, "dur": float(d), **({"midi": int(m)} if int(m) > 0 else {})}
            for p, d, m in zip(seq.phonemes, seq.dur_sec, seq.note_midi)]}
        ph_strs, ph_ids, dur_sec, values = _pp._read_score(score, cfg)
        T, ph_durs = _pp._frame_grid(dur_sec, float(cfg["frame_rate"]))
        baseline, pitch = _pp._baseline_and_model_pitch(ph_strs, values, dur_sec, T, cfg)
        frame_ph = np.repeat(np.arange(len(ph_durs)), ph_durs.astype(np.intp))
        pos = np.concatenate([np.linspace(0, 1, int(d)) for d in ph_durs]).astype(np.float32)
        c = await self.bridge.run("f0_cond", {
            "phoneme_ids": ph_ids[None].astype(np.int64), "pitch": pitch[None].astype(np.float32),
            "durations": ph_durs[None].astype(np.float32), "frame_ph": frame_ph[None].astype(np.int64),
            "pos": pos[None], "spk_id": np.array([int(style)], np.int64), "style_id": np.array([0], np.int64)})
        cond = np.asarray(c["cond"], np.float32).reshape(1, 256, -1)
        uvp = np.asarray(c["uv"], np.float32).reshape(1, 1, -1)
        s = self.sched
        x = rng.standard_normal((1, 2, cond.shape[2])).astype(np.float32)
        n = len(s["betas"])
        for i in reversed(range(n)):
            o = await self.bridge.run("f0_denoise", {"x_t": x, "cond": cond, "t": np.array([i], np.int64)})
            x0 = np.asarray(o["x0"], np.float32).reshape(x.shape)
            ab, ab_prev = s["alpha_bar"][i], (s["alpha_bar"][i - 1] if i > 0 else 1.0)
            a, b = s["alphas"][i], s["betas"][i]
            mu = (np.sqrt(ab_prev) * b / (1 - ab)) * x0 + (np.sqrt(a) * (1 - ab_prev) / (1 - ab)) * x
            if i > 0:
                x = (mu + np.sqrt(b * (1 - ab_prev) / (1 - ab)) * rng.standard_normal(x.shape)).astype(np.float32)
            else:
                x = x0
        out3 = np.concatenate([x, uvp], 1)[0]
        n_frames = min(T, out3.shape[1])
        plain_delta, vib_delta, vuv = _pp._apply_scales(out3, cfg, 1.0, 1.0, n_frames)
        baseline = baseline[:n_frames]
        phonemes_pf, note_pitches_pf, note_bnd_pf = _pp.build_per_frame_arrays(ph_strs, ph_durs, values, n_frames)
        frame_is_pau = np.array([p in _pp._SILENCE_PHS for p in phonemes_pf])
        plain_hz = np.power(2.0, baseline + plain_delta).astype(np.float32)
        out = {"f0_hz": (plain_hz * np.power(2.0, vib_delta)).astype(np.float32),
               "uv": ((vuv > 0.5) & ~frame_is_pau).astype(np.float32), "vib_delta": vib_delta.astype(np.float32),
               "frame_rate": np.float32(cfg["frame_rate"]), "plain_hz": plain_hz,
               "phonemes_pf": np.asarray(phonemes_pf), "note_pitches_pf": np.asarray(note_pitches_pf, np.float64),
               "note_boundaries_pf": np.asarray(note_bnd_pf, bool), "frame_is_pau": np.asarray(frame_is_pau, bool),
               "nucleus_phs": frozenset(cfg["nucleus"])}
        return _pp.correct_f0(out)

    def start_sec(self, phrase, seq):
        n_on0 = len(split_syllable(phrase.notes[0].phonemes, self.ps)[0]) if phrase.notes else 0
        onset0 = float(seq.dur_sec[1:1 + n_on0].sum()) if n_on0 else 0.0
        lead = float(seq.dur_sec[0]) if len(seq.dur_sec) else 0.0
        return phrase.start_sec - lead - onset0

    # ── the whole song ─────────────────────────────────────────────────────
    def check(self, notes):
        """[(index into notes, message)] for lyrics the mora table cannot read"""
        table = _as_mora_table(None, self.ps)
        maxlen = max(len(k) for k in table)
        bad = []
        for i, n in enumerate(notes):
            lyric = str(n.get("lyric") or "").strip()
            if not lyric:
                continue
            try:
                _resolve_phonemes(ScoreNote(dur_beats=1, midi=60, lyric=lyric), table, maxlen, "a", self.ps)
            except KeyError:
                bad.append((i, f"「{lyric}」は読めません"))
        return bad

    async def sing_notes(self, notes, bpm, voice_vec, mode=1, transpose=0, style=0, seed=0, progress=None):
        """editor notes and tempo -> (44.1 kHz vocal, seconds of audio before the score's beat 0)"""
        phrases = score_to_phrases(to_score_notes(notes), float(bpm), phoneset=self.ps)
        return await self.sing_phrases(phrases, voice_vec, mode, transpose, style, seed, progress)

    async def sing(self, score_path, voice_vec, mode=1, transpose=0, style=0, seed=0, progress=None):
        y, _ = await self.sing_phrases(load_score(score_path, phoneset=self.ps), voice_vec, mode, transpose, style,
                                       seed, progress)
        return y

    async def sing_phrases(self, phrases, voice_vec, mode=1, transpose=0, style=0, seed=0, progress=None):
        parts = []
        for k, p in enumerate(phrases):
            for n in p.notes:
                n.pitch_midi = int(n.pitch_midi) + int(transpose)
            phrase, seq, ph_durs = self.durations(p, style)
            f0o = await self.f0(seq, style, np.random.default_rng(seed + k))
            f0 = np.asarray(f0o["f0_hz"], np.float32)
            voicing = np.asarray(f0o["uv"], np.float32)
            T = len(f0)
            ph_durs = np.asarray(ph_durs, np.int64).copy()
            if len(ph_durs) and ph_durs.sum() != T:
                ph_durs[-1] = max(1, ph_durs[-1] + T - int(ph_durs.sum()))
            ids = np.asarray([self.ph2id[x] for x in seq.phonemes], np.int64)
            frame_ph = np.repeat(np.arange(len(ph_durs)), ph_durs.astype(np.intp))[:T]
            frame_ph = np.pad(frame_ph, (0, T - len(frame_ph)), mode="edge")
            start = np.repeat(np.cumsum(ph_durs) - ph_durs, ph_durs.astype(np.intp))[:T]
            start = np.pad(start, (0, T - len(start)), mode="edge")
            pos = ((np.arange(T) - start) / np.maximum(ph_durs[frame_ph], 1)).astype(np.float32)
            m = await self.bridge.run("acoustic", {
                "ph": ids[None], "frame_ph": frame_ph[None].astype(np.int64), "pos": pos[None], "f0": f0[None],
                "unvoiced": (1 - voicing)[None], "emb": voice_vec[None], "mode": np.array([int(mode)], np.int64)})
            mel = np.asarray(m["mel"], np.float32).reshape(1, 128, -1)
            pau = self.ph2id.get("pau", 0)
            fph = np.repeat(ids, np.maximum(ph_durs, 0).astype(np.intp))[:T]
            fph = np.pad(fph, (0, T - len(fph)), constant_values=pau)
            hard = ((voicing > 0.5) & (fph != pau)).astype(np.float32)
            v = await self.bridge.run("vocoder", {"mel": np.ascontiguousarray(mel.transpose(0, 2, 1)),
                                                  "f0": f0[None, None], "uv": (1.0 - hard)[None, None]})
            parts.append((self.start_sec(phrase, seq), np.asarray(v["waveform"], np.float32).reshape(-1)))
            if progress is not None:
                progress(k + 1, len(phrases))
        offset = max(0.0, -min(s for s, _ in parts))
        segs = [(int(round((s + offset) * SR)), w) for s, w in parts]
        y = np.zeros(max(a + len(w) for a, w in segs), np.float32)
        for a, w in segs:
            y[a:a + len(w)] += w
        return y, offset
