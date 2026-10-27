"""PrettyPitch inference: score-free note sheet (JSON) -> F0 curve.

    python -m infer --ckpt checkpoints/f0_3singer.pt --score examples/example_score.json \
        --out out/twinkle.npz --png out/twinkle.png --seed 0

Score JSON:
    {
      "spk_id": 2, "style_id": 0,
      "phonemes": [
        {"ph": "pau", "dur": 0.50},
        {"ph": "k",   "dur": 0.08},
        {"ph": "a",   "dur": 0.42, "midi": 61},      # nucleus: pitch REQUIRED (int or float)
        {"ph": "cl",  "dur": 0.10, "midi": 68},      # non-nucleus pitch OPTIONAL (tuning)
        {"ph": "t",   "dur": 0.08},
        {"ph": "a",   "dur": 0.50, "midi": 65},
        {"ph": "pau", "dur": 0.30}
      ]
    }

Every pitch value (nucleus and non-nucleus alike) shapes the **baseline** and is the
correction **target**. The model input is the nucleus values forward-filled to every
phoneme (a staircase) — so editing a `cl` pitch changes the baseline shape while the
model input (and thus plain/vibrato/vuv) stays identical (fixed seed).

Pipeline (same ordering as training's reconstruction):
    baseline(user values) -> model.sample -> plain/vibrato/vuv
    -> pitch correction on plain only (Blaauw stage 1 + asymmetric Tukey-window stage 2,
       targets = user values)
    -> f0 = 2**(log2(plain_corr) + vib_scale*vibrato) ; voiced = vuv>0.5 & non-pau
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from prettypitch.baseline import build_baseline, phoneme_frame_bounds
from prettypitch.correction import (build_per_frame_arrays, apply_pitch_correction,
                                    apply_hann_pitch_correction, _SILENCE_PHS)


def load_model(ckpt_path: str, device: str = "cpu"):
    """Rebuild the model purely from the checkpoint's self-describing config."""
    from train import build_model
    ck = torch.load(ckpt_path, map_location=device)
    cfg = ck["config"]
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, cfg


# ── generate_f0 の中身を段ごとに ─────────────────────────────────────────────

def _read_score(score: dict, cfg: dict):
    """譜面 dict → 音素と長さと音高。**ここで検証も済ませる。**

    戻り値 (ph_strs, ph_ids, dur_sec, values):
      values は音素ごとの MIDI で、指定の無いところは NaN。核音素には必須、それ以外は
      任意で、書くと baseline の形だけが変わる（モデルの入力は変わらない）。
    """
    ph2id = {p: i for i, p in enumerate(cfg["phonemes"])}
    nucleus_phs = frozenset(cfg["nucleus"])
    slots = score["phonemes"]
    ph_strs = [s["ph"] for s in slots]

    unknown = sorted({p for p in ph_strs if p not in ph2id})
    if unknown:
        raise ValueError(f"phonemes not in the model vocab: {unknown}")
    nuc_missing = [s["ph"] for s in slots
                   if s["ph"] in nucleus_phs and "midi" not in s]
    if nuc_missing:
        raise ValueError(f"nucleus phonemes need a midi value: {nuc_missing}")

    ph_ids = np.array([ph2id[p] for p in ph_strs], dtype=np.int64)
    dur_sec = np.array([float(s["dur"]) for s in slots], dtype=np.float32)
    values = np.array([float(s["midi"]) if "midi" in s else np.nan for s in slots],
                      dtype=np.float64)
    return ph_strs, ph_ids, dur_sec, values


def _frame_grid(dur_sec, frame_rate: float):
    """秒の列 → フレームの格子。戻り値 (T, ph_durs)。

    各音素に最低 1 フレームを保証するので、合計が秒から素直に出る T とは一致しないことが
    ある。以降の T は必ずこちらの合計を使う。
    """
    T = int(np.round(np.sum(dur_sec) * frame_rate))
    fb = phoneme_frame_bounds(dur_sec, frame_rate, T)
    ph_durs = np.maximum(np.diff(fb), 1).astype(np.int64)
    return int(ph_durs.sum()), ph_durs


def _baseline_and_model_pitch(ph_strs, values, dur_sec, T: int, cfg: dict):
    """**このモデルの肝。** 音高から 2 つの別物を作る。学習時と同じ手順でなければならない。

    戻り値 (baseline, pitch):

    baseline … F0 の土台。log2 領域の階段をガウシアン（σ 20ms）で丸めたもの。モデルは
        これを生成せず、学習でも推論でも同じ式で作る。**ここは全音素の値を使う**ので、
        子音に音高を書くと土台の形だけが変わる（調声用の抜け道）。

    pitch … モデルへの入力。**核音素の値だけ**を後ろへ持ち回った階段を正規化したもの
        （dataset.py と同じ）。核以外に書いた値はここには入らないので、モデルの出力は
        変わらない。baseline と入力で使う値が違うのは、この差を作るため。
    """
    baseline = build_baseline(values, dur_sec, T, float(cfg["frame_rate"]),
                              sigma_sec=float(cfg.get("baseline_sigma_sec", 0.02)))
    if baseline is None:
        raise ValueError("score carries no pitch value at all")

    from prettypitch.baseline import fill_values
    center, scale = float(cfg["pitch_center"]), float(cfg["pitch_scale"])
    is_nuc = np.array([p in frozenset(cfg["nucleus"]) for p in ph_strs])
    filled = fill_values(np.where(is_nuc, values, np.nan))
    pitch = np.where(np.isfinite(filled), (filled - center) / scale, 0.0).astype(np.float32)
    return baseline, pitch


def _sample_layers(model, ph_ids, pitch, ph_durs, score: dict, device: str,
                   seed: int | None, infer_steps: int | None, style_mix):
    """モデルを走らせて 3 チャンネルを得る。戻り値 [3, T]（抑揚・ビブラート・v/uv）。"""
    dev = torch.device(device)
    if seed is not None:
        torch.manual_seed(seed)
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=dev).unsqueeze(0)
    args = (t(ph_ids, torch.long), t(pitch, torch.float32), t(ph_durs, torch.long))
    kw = dict(spk_id=torch.tensor([int(score.get("spk_id", 0))], device=dev),
              style_id=torch.tensor([int(score.get("style_id", 0))], device=dev))
    if style_mix is not None:
        kw["style_mix"] = torch.as_tensor(np.asarray(style_mix, np.float32),
                                          device=dev).unsqueeze(0)
    # reflow は infer_steps を Euler ステップ数として受ける。DDPM は n_steps 固定。
    if infer_steps is not None and hasattr(model, "predict_velocity"):
        out3 = model.sample(*args, num_steps=int(infer_steps), **kw)
    else:
        out3 = model.sample(*args, **kw)
    return out3[0].cpu().numpy()


def _apply_scales(out3, cfg: dict, plain_scale, vib_scale, n_frames: int):
    """3 チャンネルに学習時の標準偏差とユーザー指定の倍率を掛ける。

    倍率はスカラーでもフレームごとの配列でもよい。配列ならフレーズより短いと落とす
    （黙って短い方に合わせると、途中から効かなくなって気づけない）。
    """
    def per_frame(v, name):
        a = np.asarray(v, dtype=np.float64)
        if a.ndim == 0:
            return float(a)
        if a.shape[0] < n_frames:
            raise ValueError(
                f"{name}: per-frame array has {a.shape[0]} frames, the phrase has {n_frames}")
        return a[:n_frames]

    plain_delta = out3[0, :n_frames] * float(cfg["plain_std"]) * per_frame(plain_scale, "plain_scale")
    vib_delta = out3[1, :n_frames] * float(cfg["vib_std"]) * per_frame(vib_scale, "vib_scale")
    vuv = out3[2, :n_frames]                              # uv probability in [0,1]
    return plain_delta, vib_delta, vuv


def generate_f0(model, cfg: dict, score: dict, device: str = "cpu",
                seed: int | None = None, plain_scale=1.0, vib_scale=1.0,
                infer_steps: int | None = None, style_mix=None):
    """Generate one phrase's F0 layers — **no pitch correction**.

    This is the model on its own: baseline + expression + vibrato + v/uv, exactly as the
    network produced them. Use ``correct_f0`` afterwards to pull the pitch onto the score,
    or ``synthesize_f0`` to do both in one call.

    Args:
        model, cfg: what ``load_model`` returned.
        score: ``{"spk_id", "style_id", "phonemes": [{"ph", "dur", "midi"?}, ...]}`` —
            see the module docstring. Every nucleus phoneme needs ``midi`` (fractional
            values are fine); a value on any other phoneme only reshapes the baseline.
        seed: fixes the sampling noise (same seed → same plain / vibrato / v-uv).
        plain_scale, vib_scale: multiply the expression (plain) and vibrato deviations.
            A float applies to the whole phrase; an array of length ≥ T (frames)
            applies per frame, which is how per-note or per-phoneme control is done.
        infer_steps: Euler steps for a rectified-flow model (ignored for DDPM).
        style_mix: optional ``[n_styles]`` blend weights (continuous style). One-hot
            equals ``style_id``; when given it takes precedence over ``score["style_id"]``.

    Returns:
        dict of numpy arrays over T frames: ``f0_hz`` (0 where unvoiced is not applied —
        use ``uv``), ``uv`` (1 = voiced), ``baseline_log2``, ``plain_delta`` and
        ``vib_delta`` (log2 deviations, scales already applied), ``note_boundaries_pf``
        (True at each note's first frame), and ``frame_rate``.

        It also carries what ``correct_f0`` needs, so the two stages stay in step without
        rebuilding anything: ``plain_hz`` (the uncorrected curve the correction acts on),
        ``phonemes_pf``, ``note_pitches_pf`` and ``frame_is_pau``.
    """
    frame_rate = float(cfg["frame_rate"])

    # ── ① 譜面を読む（検証込み）──────────────────────────────────────────────
    ph_strs, ph_ids, dur_sec, values = _read_score(score, cfg)

    # ── ② フレーム格子を決める ──────────────────────────────────────────────
    T, ph_durs = _frame_grid(dur_sec, frame_rate)

    # ── ③ baseline とモデル入力の音高を作る（学習時と同じ手順）──────────────
    baseline, pitch = _baseline_and_model_pitch(ph_strs, values, dur_sec, T, cfg)

    # ── ④ モデルを走らせる ──────────────────────────────────────────────────
    out3 = _sample_layers(model, ph_ids, pitch, ph_durs, score, device,
                          seed, infer_steps, style_mix)

    # ── ⑤ 3 チャンネルを取り出して倍率を掛ける ──────────────────────────────
    # モデルが返すフレーム数が格子とずれることがあるので、短い方に合わせる。
    n_frames = min(T, out3.shape[1])
    plain_delta, vib_delta, vuv = _apply_scales(out3, cfg, plain_scale, vib_scale, n_frames)
    baseline = baseline[:n_frames]

    # ── ⑥ 組み立てる ────────────────────────────────────────────────────────
    # 補正と v/uv が要るフレーム展開済みの配列（音素・ノート音高・ノート境界・無音マスク）
    phonemes_pf, note_pitches_pf, note_bnd_pf = build_per_frame_arrays(
        ph_strs, ph_durs, values, n_frames)
    frame_is_pau = np.array([p in _SILENCE_PHS for p in phonemes_pf])

    plain_hz = np.power(2.0, baseline + plain_delta).astype(np.float32)
    f0_hz = (plain_hz * np.power(2.0, vib_delta)).astype(np.float32)
    voiced = ((vuv > 0.5) & ~frame_is_pau).astype(np.float32)

    return {
        "f0_hz": f0_hz, "uv": voiced,
        "baseline_log2": baseline.astype(np.float32),
        "plain_delta": plain_delta.astype(np.float32),
        "vib_delta": vib_delta.astype(np.float32),
        "note_boundaries_pf": np.asarray(note_bnd_pf, dtype=bool),
        "frame_rate": np.float32(frame_rate),
        # ↓ correct_f0 が使う材料。補正は plain（ビブラートを掛け戻す前）に当たるので、
        #   f0_hz ではなく plain_hz を持ち回る。
        "plain_hz": plain_hz,
        "phonemes_pf": np.asarray(phonemes_pf),
        "note_pitches_pf": np.asarray(note_pitches_pf, dtype=np.float64),
        "frame_is_pau": np.asarray(frame_is_pau, dtype=bool),
        "nucleus_phs": frozenset(cfg["nucleus"]),
    }


def correct_f0(out: dict, correction: bool = True, cons_weight: float = 0.0,
               hann_alpha: float = 0.8, hann_window: str = "tukey",
               hann_lead_taper: float = 0.6, hann_trail_taper: float = 0.1) -> dict:
    """Pull the generated pitch onto the score — the two fixed correction stages.

    ``out`` is what ``generate_f0`` returned; a new dict is returned with ``f0_hz`` and
    ``plain_hz`` replaced. Nothing else changes, and ``out`` itself is left alone.

    Both stages act on **plain only**, so the vibrato keeps its shape: the correction moves
    the note's pitch without flattening the oscillation on top of it. The scoop into a note
    and the fall at its end survive for the same reason — stage 1 is one constant per note.

    Args:
        correction, cons_weight: stage 1 — Blaauw's perceived pitch, one constant offset
            per note. ``correction=False`` turns it off.
        hann_alpha, hann_window, hann_lead_taper, hann_trail_taper: stage 2 — a windowed
            pull toward the note pitch (asymmetric Tukey by default).
            ``hann_alpha=0`` turns it off.

    See guide/pitch_correction.md for what each stage does and why.
    """
    plain_hz = out["plain_hz"]
    frame_rate = float(out["frame_rate"])
    voiced = ~out["frame_is_pau"]
    args = (out["phonemes_pf"], out["note_pitches_pf"], out["note_boundaries_pf"])
    if correction:
        plain_hz, _notes = apply_pitch_correction(
            plain_hz, voiced, *args,
            frame_rate=frame_rate, cons_weight=cons_weight,
            nucleus_phs=out["nucleus_phs"])
    if hann_alpha > 0.0:
        plain_hz = apply_hann_pitch_correction(
            plain_hz, voiced, *args,
            frame_rate=frame_rate, alpha=hann_alpha, nucleus_phs=out["nucleus_phs"],
            window=hann_window, lead_taper=hann_lead_taper,
            trail_taper=hann_trail_taper)
    return {**out, "plain_hz": plain_hz,
             "f0_hz": (plain_hz * np.power(2.0, out["vib_delta"])).astype(np.float32)}


def synthesize_f0(model, cfg: dict, score: dict, device: str = "cpu",
                  seed: int | None = None, plain_scale=1.0,
                  vib_scale=1.0, correction: bool = True,
                  cons_weight: float = 0.0, hann_alpha: float = 0.8,
                  hann_window: str = "tukey", hann_lead_taper: float = 0.6,
                  hann_trail_taper: float = 0.1,
                  infer_steps: int | None = None, style_mix=None):
    """``generate_f0`` then ``correct_f0`` — generation and pitch correction in one call.

    Kept because most callers want both. When you need the model's raw output, or want to
    put something of your own between the two, call the two functions yourself.
    """
    out = generate_f0(model, cfg, score, device=device, seed=seed,
                      plain_scale=plain_scale, vib_scale=vib_scale,
                      infer_steps=infer_steps, style_mix=style_mix)
    return correct_f0(out, correction=correction, cons_weight=cons_weight,
                      hann_alpha=hann_alpha, hann_window=hann_window,
                      hann_lead_taper=hann_lead_taper, hann_trail_taper=hann_trail_taper)


def _plot(out: dict, path: str, title: str):
    """--png: svs.plot.plot_f0 と同じ図（baseline 破線 + 有声 F0）を 1 枚保存する。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from svs.plot import plot_f0
    fig, ax = plt.subplots(figsize=(12, 4))
    plot_f0(out, ax=ax, show_delta=False)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="PrettyPitch F0 inference from a JSON score.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--score", required=True, help="score JSON (see module docstring)")
    ap.add_argument("--out", default=None, help="output .npz path")
    ap.add_argument("--png", default=None, help="optional plot path")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--plain_scale", type=float, default=1.0)
    ap.add_argument("--vib_scale", type=float, default=1.0)
    ap.add_argument("--no_correction", action="store_true", help="disable Blaauw stage 1")
    ap.add_argument("--hann_alpha", type=float, default=0.8, help="stage 2 strength (0=off)")
    ap.add_argument("--cons_weight", type=float, default=0.0)
    ap.add_argument("--infer_steps", type=int, default=None,
                    help="reflow の Euler ステップ数（既定 = ckpt の値）。DDPM では無視される")
    args = ap.parse_args()

    model, cfg = load_model(args.ckpt, args.device)
    score = json.load(open(args.score, encoding="utf-8"))
    out = synthesize_f0(model, cfg, score, device=args.device, seed=args.seed,
                        plain_scale=args.plain_scale, vib_scale=args.vib_scale,
                        correction=not args.no_correction,
                        cons_weight=args.cons_weight, hann_alpha=args.hann_alpha,
                        infer_steps=args.infer_steps)
    v = out["uv"] > 0.5
    print(f"frames={len(out['f0_hz'])}  voiced={int(v.sum())}  "
          f"f0 range=[{out['f0_hz'][v].min():.1f}, {out['f0_hz'][v].max():.1f}] Hz"
          if v.any() else "all unvoiced")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, **out)
        print(f"saved {args.out}")
    if args.png:
        Path(args.png).parent.mkdir(parents=True, exist_ok=True)
        _plot(out, args.png, Path(args.score).stem)
        print(f"saved {args.png}")


if __name__ == "__main__":
    main()
