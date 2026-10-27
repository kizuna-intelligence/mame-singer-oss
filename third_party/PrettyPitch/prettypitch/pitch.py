"""Nucleus representative pitch values — the score-free replacement for note MIDI.

For every nucleus-phoneme instance (a melisma "a a a" = three separate values) we
compute a **continuous MIDI value** from the extracted F0, using the weighted mean of
Blaauw & Bonada 2017's perceived-pitch estimator (the same weights as the inference-time
pitch correction in `correction.py`):

    perceived = average(f0_st, weights = w_e * w_d * w_t * mask)

    w_e : Tukey window (edge de-emphasis)                       — asymmetric Tukey by default
    w_d : stability, 1 / min(1 + k·|d f0_st/dt|, 15)             — slope of the SG-smoothed curve
    w_t : target proximity, exp(−0.5·((f0_st − target)/σ)²)      — σ = 1.0 st

The paper's w_t needs a target pitch; with no score we bootstrap it: target₀ = median of
the segment, then iterate target ← perceived (2 fixed-point steps). w_t is what rejects
scoop/portamento frames, so it is kept rather than dropped.

Two departures from the paper, both **on by default** (the bundled weights were trained with
them; `stability='blaauw'` and `crest_max_ms=0` give the paper's estimator instead):

    stability='slope'   w_d from the SG first derivative instead of the SG residual —
                        the residual does not react to a straight ramp, so a scoop got
                        the highest weight.
    crest_max_ms>0      short arch-shaped nuclei take the mean of the top of their arch
                        instead of the weighted mean, which lands between valley and peak.

Devoiced nuclei (A, I, U, E, O — often zero voiced frames) still get a value by default
(`devoiced_fill='interp'`): the stored `f0_interp` is gap-less, so the estimator simply
runs on the interpolated curve without the voiced mask. That makes devoiced vowels
first-class score symbols — the model learns to generate the interpolated F0 through
them, and the user can write them into a score at inference.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .baseline import phoneme_frame_bounds

TUKEY_ALPHA     = 0.5    # Tukey 窓 α（Blaauw 2017 の論文値）
SG_WINDOW_MS    = 55.0   # Savitzky-Golay 窓幅 [ms]
SG_POLY_ORDER   = 3      # SG 多項式次数
TARGET_SIGMA_ST = 1.0    # w_t の σ [半音]（スクープ/ポルタメント排除の強さ。slope と組むと 1.0 が良い）
N_ITER          = 2      # target ← perceived の固定点反復回数
MIN_VOICED      = 3      # 有声フレームがこれ未満の核は「無声化」扱い(devoiced_fill へ)

# ── w_d の定義 'slope'（既定）────────────────────────────────────────────────
# Blaauw の w_d は「SG 平滑からの残差」＝ギザギザさの指標で、まっすぐな傾斜には反応しない
# （3 次多項式が当たってしまう）。結果、しゃくりの最中がいちばん重くなる逆転が起きる。
# 'slope' は SG の 1 階微分（半音/秒）の大きさで測るので、平らなところほど重くなる。
#   w_d = 1 / min(1 + k·|d f0/dt|, cap)
SLOPE_K   = 0.5          # 傾き 2 半音/秒 で重み 1/2
SLOPE_CAP = 15.0         # 下限 1/15（Blaauw と同じ）

# ── 短いノートの山検出（既定 ON）──────────────────────────────────────────────
# 短い母音（子音由来の深い谷 → 山 → 次音への下降）では、w_t が中央値からブートストラップ
# するため頂点が外れ値として捨てられ、代表値が谷と山の間に落ちる。方針は
# 「短いノートの山では頂点付近に乗る／それ以外は一切触らない」。
#   ・核が crest_max_ms 未満（長い核の山は行き過ぎ・ビブラート・しゃくりなので触らない）
#   ・頂点 − 核内 F0 の中央値 ≥ crest_depth_st（ノートの大半が頂点より下＝山形。行き過ぎの
#     尖りは短く、中央値が頂点近くに来るので発火しない）
#   ・極大の卓立 ≥ crest_prom_st
#   → 頂点を含む連続領域で「頂点 − crest_band_st 以内」のフレームを平均した値に全量置換
#     （どんぴしゃの頂点だと行き過ぎるので、頂点のやや下に乗せる）
CREST_MAX_MS      = 300.0  # これ以上長い核は触らない。0 = 山検出オフ
CREST_DEPTH_ST    = 1.0
CREST_PROM_ST     = 1.0
CREST_SG_MS       = 25.0 # 頂点探索の平滑窓（55ms だと短い山を潰す）
CREST_BAND_ST     = 0.75
CREST_OFFSET_CENT = 0.0


def hz_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    """F0 [Hz] → 連続 MIDI 半音 (float64)。1 Hz 未満は 1 Hz に丸める。"""
    return 69.0 + 12.0 * np.log2(np.maximum(np.asarray(f0_hz, dtype=np.float64), 1.0) / 440.0)


def midi_to_hz(midi: np.ndarray) -> np.ndarray:
    """連続 MIDI 半音 → F0 [Hz] (float64)。hz_to_midi の逆。"""
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)


def _asym_tukey_window(n: int, lead: float, trail: float) -> np.ndarray:
    """非対称 tapered-cosine（Tukey）窓 [n]。先頭 lead 割合を 0→1、末尾 trail 割合を 1→0、
    中央は平坦=1（lead/trail は区間長比 [0,1]）。lead=trail=0.5 で Hann 相当、0 で矩形。"""
    w = np.ones(int(n), dtype=np.float64)
    if n <= 1:
        return w
    lead = max(0.0, float(lead)); trail = max(0.0, float(trail))
    if lead + trail > 1.0:
        s = lead + trail; lead, trail = lead / s, trail / s
    nl = min(int(round(lead * n)), n)
    nt = min(int(round(trail * n)), n - nl)
    if nl > 0:
        w[:nl] = np.sin(np.linspace(0.0, np.pi / 2, nl)) ** 2
    if nt > 0:
        w[n - nt:] = np.sin(np.linspace(np.pi / 2, 0.0, nt)) ** 2
    return w


def _edge_weight(n: int, *, edge_window: str, tukey_alpha: float,
                 edge_lead_frac: float | None, edge_trail_frac: float | None,
                 frame_rate: float | None = None,
                 edge_adapt_lo_ms: float = 70.0,
                 edge_adapt_hi_ms: float = 150.0) -> np.ndarray:
    """端テーパー重み w_e。'tukey'(対称α・既定)/'hann'(=Tukey α=1)/'asym'(左右独立)/
    'asym_adaptive'(母音長でテーパー比を線形スケール: lo_ms以下=矩形 → hi_ms以上=asym全開)。"""
    if edge_window == 'hann':
        return signal.windows.tukey(n, alpha=1.0)
    if edge_window in ('asym', 'asym_adaptive'):
        lead = tukey_alpha / 2.0 if edge_lead_frac is None else edge_lead_frac
        trail = tukey_alpha / 2.0 if edge_trail_frac is None else edge_trail_frac
        if edge_window == 'asym_adaptive' and frame_rate:
            dur_ms = n / float(frame_rate) * 1000.0        # 短ノートはテーパーを緩める(削りすぎ回避)
            t = (dur_ms - edge_adapt_lo_ms) / max(1e-6, edge_adapt_hi_ms - edge_adapt_lo_ms)
            t = min(1.0, max(0.0, t))
            lead *= t
            trail *= t
        return _asym_tukey_window(n, lead, trail)
    return signal.windows.tukey(n, alpha=tukey_alpha)


def _representative(f0_st: np.ndarray, mask: np.ndarray, *,
                    n_iter: int, sigma_st: float, sg_window: int,
                    tukey_alpha: float,
                    edge_window: str = 'tukey',
                    edge_lead_frac: float | None = None,
                    edge_trail_frac: float | None = None,
                    frame_rate: float | None = None,
                    edge_adapt_lo_ms: float = 70.0,
                    edge_adapt_hi_ms: float = 150.0,
                    stability: str = 'slope',
                    slope_k: float = SLOPE_K,
                    slope_cap: float = SLOPE_CAP,
                    return_weights: bool = False):
    """1セグメントの代表値 [連続MIDI]。mask: 使うフレーム(0/1)。None = 重みが立たない。

    stability: w_d の定義。'slope'（既定・SG 1 階微分）| 'blaauw'（SG 残差。モジュール冒頭参照）。
    return_weights=True で `(値, 最終反復の重み[n])` を返す。重みは加重平均に実際に入った
    ものそのもの（w_e·w_d·w_l·mask·w_t）で、どのフレームが値を決めたかの可視化に使う。
    """
    n = len(f0_st)
    if n == 0 or mask.sum() <= 0:
        return (None, np.zeros(n)) if return_weights else None
    w_e = _edge_weight(n, edge_window=edge_window, tukey_alpha=tukey_alpha,
                       edge_lead_frac=edge_lead_frac, edge_trail_frac=edge_trail_frac,
                       frame_rate=frame_rate, edge_adapt_lo_ms=edge_adapt_lo_ms,
                       edge_adapt_hi_ms=edge_adapt_hi_ms)
    win = min(sg_window, n)
    if win % 2 == 0:
        win -= 1
    poly = min(SG_POLY_ORDER, win - 1)
    if win < 3 or poly < 1:
        w_d = np.ones(n)
    elif stability == 'slope':
        delta = 1.0 / float(frame_rate) if frame_rate else 1.0          # → 半音/秒
        slope = signal.savgol_filter(f0_st, win, poly, deriv=1, delta=delta)
        w_d = 1.0 / np.minimum(1.0 + float(slope_k) * np.abs(slope), float(slope_cap))
    else:
        d_f0 = f0_st - signal.savgol_filter(f0_st, win, poly)
        w_d = 1.0 / np.minimum(1.0 + 27.0 * np.abs(d_f0), 15.0)
    base = w_e * w_d * mask
    if base.sum() <= 0.0:
        # Tukey が両端0のため n が極小だと base が消えることがある → 素の平均で代替
        val = float(np.average(f0_st, weights=mask))
        return (val, np.asarray(mask, dtype=np.float64)) if return_weights else val
    target = float(np.median(f0_st[mask > 0]))     # bootstrap init (robust to scoops)
    w = base
    for _ in range(n_iter):
        w = base * np.exp(-0.5 * ((f0_st - target) / sigma_st) ** 2)
        tw = float(w.sum())
        if tw <= 1e-12:
            break                                   # w_t が全フレームを棄却 → 現在値で確定
        target = float(np.average(f0_st, weights=w))
    return (target, w) if return_weights else target


def nucleus_values(
    ph_ids: np.ndarray,        # int [L]
    dur_sec: np.ndarray,       # float [L]
    f0_interp: np.ndarray,     # float [T] Hz, gap-less
    uv: np.ndarray,            # float [T] 1=voiced
    frame_rate: float,
    nucleus_ids: frozenset | set,
    *,
    n_iter: int = N_ITER,
    sigma_st: float = TARGET_SIGMA_ST,
    sg_window_ms: float = SG_WINDOW_MS,
    tukey_alpha: float = TUKEY_ALPHA,
    edge_window: str = 'tukey',        # 端テーパー窓: 'tukey'(対称α)/'hann'/'asym'/'asym_adaptive'
    edge_lead_frac: float | None = None,   # 'asym' 先頭テーパー比 [0,1]
    edge_trail_frac: float | None = None,  # 'asym' 末尾テーパー比 [0,1]
    edge_adapt_lo_ms: float = 70.0,        # 'asym_adaptive': この長さ以下は矩形(削らない)
    edge_adapt_hi_ms: float = 150.0,       # 'asym_adaptive': この長さ以上は asym 全開
    min_voiced: int = MIN_VOICED,
    devoiced_fill: str = 'interp',   # 'interp' | 'neighbor' | 'none'
    stability: str = 'slope',              # w_d の定義: 'slope'（既定）| 'blaauw'（冒頭参照）
    slope_k: float = SLOPE_K,
    slope_cap: float = SLOPE_CAP,
    crest_max_ms: float = CREST_MAX_MS,    # 短いノートの山検出。0 でオフ。冒頭参照
    crest_depth_st: float = CREST_DEPTH_ST,
    crest_prom_st: float = CREST_PROM_ST,
    crest_sg_ms: float = CREST_SG_MS,
    crest_band_st: float = CREST_BAND_ST,
    crest_offset_cent: float = CREST_OFFSET_CENT,
    return_weights: bool = False,          # True で (値, 重み[T]) を返す（可視化用）
) -> np.ndarray:
    """One representative pitch value per nucleus phoneme — the score-free note pitch.

    Args:
        ph_ids, dur_sec: the phrase's phonemes (ids into the phoneme table) and their
            lengths in seconds, both of length L.
        f0_interp, uv: the extracted F0 in Hz over T frames, gap-less (interpolated
            through unvoiced stretches), and the voicing flag (1 = voiced).
        frame_rate: frames per second (256/44100 → 172.27 by default).
        nucleus_ids: ids of the nucleus phonemes (vowels and N).
        n_iter, sigma_st, sg_window_ms, tukey_alpha, edge_*: the Blaauw & Bonada
            weights (module docstring); ``sigma_st`` is w_t's σ in semitones.
        min_voiced, devoiced_fill: what to do when a nucleus has fewer than
            ``min_voiced`` voiced frames — ``'interp'`` (default) estimates on the
            interpolated F0 so devoiced vowels still get a value, ``'neighbor'`` copies
            the nearest voiced nucleus, ``'none'`` leaves NaN.
        stability, slope_k, slope_cap: the stability weight w_d — ``'slope'`` (default)
            uses the slope of the SG-smoothed curve, ``'blaauw'`` the paper's residual.
        crest_*: the short-note crest detection (module docstring); ``crest_max_ms=0``
            turns it off.
        return_weights: also return the per-frame weights that entered each mean,
            normalised to a maximum of 1 within each nucleus (0 elsewhere) — useful for
            plotting which frames decided the value.

    Returns:
        float32 array of length L: a continuous MIDI value for every nucleus phoneme,
        NaN for the other phonemes and for nuclei without a value. With
        ``return_weights=True``: ``(values, weights[T])``.
    """
    if devoiced_fill not in ('interp', 'neighbor', 'none'):
        raise ValueError(devoiced_fill)
    L = len(ph_ids)
    T = len(f0_interp)
    fb = phoneme_frame_bounds(dur_sec, frame_rate, T)
    f0_st = hz_to_midi(f0_interp)
    uv = np.asarray(uv)
    sg_window = max(3, int(round(sg_window_ms * 1e-3 * frame_rate)))
    if sg_window % 2 == 0:
        sg_window += 1

    values = np.full(L, np.nan, dtype=np.float32)
    w_all = np.zeros(T, dtype=np.float32) if return_weights else None
    for k in range(L):
        if int(ph_ids[k]) not in nucleus_ids:
            continue
        a, b = fb[k], fb[k + 1]
        if b <= a:
            continue
        s = f0_st[a:b]
        voiced = (uv[a:b] > 0.5).astype(np.float64)
        if voiced.sum() >= min_voiced:
            mask = voiced
        elif devoiced_fill == 'interp':
            mask = np.ones(b - a, dtype=np.float64)   # 補間F0をそのまま使う
        else:
            continue                                   # NaN → neighbor/none で後処理
        out = _representative(s, mask, n_iter=n_iter, sigma_st=sigma_st,
                              sg_window=sg_window, tukey_alpha=tukey_alpha,
                              edge_window=edge_window, edge_lead_frac=edge_lead_frac,
                              edge_trail_frac=edge_trail_frac, frame_rate=frame_rate,
                              edge_adapt_lo_ms=edge_adapt_lo_ms, edge_adapt_hi_ms=edge_adapt_hi_ms,
                              stability=stability, slope_k=slope_k, slope_cap=slope_cap,
                              return_weights=return_weights)
        if return_weights:
            val, w_seg = out
            mx = float(np.max(w_seg)) if len(w_seg) else 0.0
            if mx > 0.0:
                w_all[a:b] = (w_seg / mx).astype(np.float32)
        else:
            val = out
        if val is not None:
            values[k] = val

    # ── 短いノートの山検出（冒頭のコメント参照。crest_max_ms=0 で完全にスキップ）────
    if crest_max_ms > 0.0:
        max_fr = crest_max_ms * 1e-3 * frame_rate
        wc = max(3, int(round(crest_sg_ms * 1e-3 * frame_rate)) | 1)
        sm = (signal.savgol_filter(f0_st, wc, min(SG_POLY_ORDER, wc - 1))
              if T >= wc and wc >= 3 else f0_st)
        peaks, _ = signal.find_peaks(sm, prominence=float(crest_prom_st))
        for k in range(L):
            if int(ph_ids[k]) not in nucleus_ids or not np.isfinite(values[k]):
                continue
            a, b = fb[k], fb[k + 1]
            n = b - a
            if n < 2 or n >= max_fr:
                continue
            inr = peaks[(peaks >= a) & (peaks < b)]
            if inr.size == 0:
                continue                                    # 核の中に山が無い
            c = int(inr[np.argmax(sm[inr])])                # いちばん高い極大
            crest = float(sm[c])
            if crest - float(np.median(sm[a:b])) < crest_depth_st:
                continue                                    # 山形でない（尖り／平ら）
            lo_b = hi_b = c
            if crest_band_st > 0.0:                         # 頂点に滞在した時間の平均
                thr = crest - float(crest_band_st)
                while lo_b - 1 >= a and sm[lo_b - 1] >= thr:
                    lo_b -= 1
                while hi_b + 1 < b and sm[hi_b + 1] >= thr:
                    hi_b += 1
                crest = float(np.mean(sm[lo_b:hi_b + 1]))
            values[k] = np.float32(crest - float(crest_offset_cent) / 100.0)
            if return_weights:                              # 可視化: 使った領域だけ 1
                w_all[a:b] = 0.0
                w_all[lo_b:hi_b + 1] = 1.0

    if devoiced_fill == 'neighbor':
        nuc = np.array([int(p) in nucleus_ids for p in ph_ids])
        need = nuc & ~np.isfinite(values)
        have = np.flatnonzero(nuc & np.isfinite(values))
        if have.size:
            for k in np.flatnonzero(need):
                values[k] = values[have[np.argmin(np.abs(have - k))]]
    return (values, w_all) if return_weights else values
