"""ピッチ補正 — Gaussian 目標近接重み法（Blaauw 2017 改良版）+ 非対称 Tukey 窓の第2段。

**ノート境界は「値付き音素の開始イベント」から作る**（整数MIDIの差分検出ではなく —
連続値では毎フレーム境界化してしまうため）。ターゲット音高も連続MIDI値で受ける。

第1段(apply_pitch_correction): ノートごとに定数の補正量 c = target − perceived
（perceived = 重み付き平均の知覚代表値）を決め、休符は c=0、連結ベクトルを
σ=30ms でガウシアン平滑。ノート内では一定 → ビブラート/しゃくりの形状を完全保存。

第2段(apply_hann_pitch_correction): 各核音素フレーム列に 窓 × α × (target − f0) の
可変補正。窓は既定で非対称 Tukey（入口のしゃくりを残し、終わり際のオーバーシュートを抑える）、
'hann' も選べる。中央ほど強く引き寄せ、端（トランジション）は窓≈0 で触らない。

どちらも plain（vibrato 加算前）にのみ適用すること — ビブラートの振動は
代表値推定を汚す。
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from .baseline import fill_values
from .pitch import _asym_tukey_window, hz_to_midi, midi_to_hz

# ── デフォルトパラメータ ──────────────────────────────────────────────────────
_TUKEY_ALPHA     = 0.5    # Tukey 窓 α（Blaauw 2017 の論文値）
_SG_WINDOW_MS    = 55.0   # Savitzky-Golay 窓幅 [ms]
_SG_POLY_ORDER   = 3      # SG 多項式次数
_TARGET_SIGMA_ST = 0.5    # Gaussian w_t の σ [半音]（小さいほどしゃくれ抑制が強い）
# 補正量平滑化 σ [ms]。ノートごとの定数補正量+休符0を連結したベクトルを平滑化して
# ノート境界をなめらかに繋ぐ(Blaauw 2017 Appendix B: 実効 σ ≈ 20〜30ms)。
_SMOOTH_SIGMA_MS = 30.0

# 無声扱い音素 (w_p = 0) と既定の核音素集合は **言語パックから読む**（ここに直書きしない）。
# dict/<lang>.yaml の silence: / nucleus: が情報源。
# 実運用では infer.synthesize_f0 が ckpt の nucleus を nucleus_phs= で明示的に渡すので、
# ここは「呼び出し側が何も渡さなかったとき」の既定でしかない。
def _pack_defaults():
    try:
        from duration_model import phoneset as _ps_mod
        ps = _ps_mod.active()
        return frozenset(ps.silence), frozenset(ps.nuclei)
    except Exception:                                   # noqa: BLE001
        # 言語パックを読めない環境向けの最終フォールバック
        return (frozenset({'pau', 'PAU', 'sil', 'br', 'sp', 'sli'}),
                frozenset({'a', 'i', 'u', 'e', 'o', 'N', 'A', 'I', 'U', 'E', 'O'}))


_SILENCE_PHS, DEFAULT_NUCLEUS_PHS = _pack_defaults()


def _hann_window(n: int) -> np.ndarray:
    """中央ピークの Hann 窓（中央=1・両端→0）。短い列(n<=2)は全1。
    apply_hann_pitch_correction(window='hann') のときだけ使われる（既定は 'tukey'）。"""
    if n <= 2:
        return np.ones(max(n, 1))
    x = (np.arange(n) - (n - 1) / 2.0) / ((n - 1) / 2.0)   # -1..1
    return np.cos(np.pi * x / 2.0) ** 2


def _ph_weight(ph: str, nucleus_phs, cons_weight: float = 0.0) -> float:
    if ph in _SILENCE_PHS:
        return 0.0
    if ph in nucleus_phs:
        return 1.0
    return cons_weight   # 子音・cl


def _segment_correction_gaussian(
    f0_st: np.ndarray,
    w_p: np.ndarray,
    target_st: float,
    sg_window: int,
    sigma_st: float,
    tukey_alpha: float = _TUKEY_ALPHA,
):
    """ガウシアン目標近接重みによる知覚ピッチ推定と補正量 [半音]。
    返り: (correction_st, w_combined [n])。"""
    n = len(f0_st)
    if n == 0:
        return 0.0, np.zeros(0)

    w_e = signal.windows.tukey(n, alpha=tukey_alpha)

    win = min(sg_window, n)
    if win % 2 == 0:
        win -= 1
    poly = min(_SG_POLY_ORDER, win - 1)
    if win >= 3 and poly >= 1:
        d_f0 = f0_st - signal.savgol_filter(f0_st, win, poly)
        w_d = 1.0 / np.minimum(1.0 + 27.0 * np.abs(d_f0), 15.0)
    else:
        w_d = np.ones(n)

    w_t = np.exp(-0.5 * ((f0_st - target_st) / sigma_st) ** 2)

    w_combined = w_e * w_d * w_p * w_t
    total_w = float(np.sum(w_combined))
    if total_w == 0.0:
        return 0.0, w_combined

    perceived = float(np.average(f0_st, weights=w_combined))
    return target_st - perceived, w_combined


def build_per_frame_arrays(
    ph_strs: list,               # (L,) 音素名 per slot
    ph_durs: np.ndarray,         # (L,) フレーム数 per slot
    values_midi: np.ndarray,     # (L,) 連続MIDI値, NaN = 値なし
    T: int,
    silence_phs=_SILENCE_PHS,
):
    """音素列情報からフレーム単位配列を構築する。

    ノート境界 = ①値付き音素の開始 ②静音 run の開始 ③先頭フレーム。
    ターゲット音高 = forward-fill した値（静音フレームは 0 = 休符）。

    Returns:
        phonemes_pf:        (T,) 各フレームの音素文字列
        note_pitches_pf:    (T,) 各フレームのターゲット連続MIDI (float64, 0=休符)
        note_boundaries_pf: (T,) ノート先頭フレームで True (bool)
    """
    filled = fill_values(np.asarray(values_midi, dtype=np.float64))
    phonemes_pf = ['pau'] * T
    note_pitches_pf = np.zeros(T, dtype=np.float64)
    note_boundaries_pf = np.zeros(T, dtype=bool)

    cursor = 0
    prev_sil = None
    for k in range(len(ph_strs)):
        dur = int(ph_durs[k])
        end = min(cursor + dur, T)
        if end <= cursor:
            continue
        ph = str(ph_strs[k])
        is_sil = ph in silence_phs
        for f in range(cursor, end):
            phonemes_pf[f] = ph
        if not is_sil and np.isfinite(filled[k]):
            note_pitches_pf[cursor:end] = filled[k]
        if cursor == 0 or np.isfinite(values_midi[k]) or (is_sil and prev_sil is not True):
            note_boundaries_pf[cursor] = True
        prev_sil = is_sil
        cursor = end
        if cursor >= T:
            break
    return phonemes_pf, note_pitches_pf, note_boundaries_pf


def apply_pitch_correction(
    f0_hz: np.ndarray,            # (T,) 連続 F0 [Hz]（全フレーム正値）
    voiced_mask: np.ndarray,      # (T,) bool, True = 有声（非 PAU）フレーム
    phonemes_pf: list,            # (T,) フレームごとの音素文字列
    note_pitches_pf: np.ndarray,  # (T,) フレームごとのターゲット連続MIDI（0=休符）
    note_boundaries_pf: np.ndarray,  # (T,) ノート境界フラグ
    frame_rate: float = 172.265625,
    tukey_alpha: float = _TUKEY_ALPHA,
    sigma_st: float = _TARGET_SIGMA_ST,
    smooth_sigma_ms: float = _SMOOTH_SIGMA_MS,
    cons_weight: float = 0.0,     # 子音・cl フレームの重み (0=除外, 1=核と同等)
    nucleus_phs=DEFAULT_NUCLEUS_PHS,
    return_details: bool = False,
):
    """第1段: 知覚代表値からの定数オフセット補正（形状保存）。

    return_details=True のとき (f0_corr, perceived_notes, details) を返す。
    perceived_notes = [(seg_start, seg_end, perceived_midi, target_midi), ...]
    details = {'correction_st': 平滑後補正量[半音], 'weight': 合成重み(ノート内max=1)}
    """
    n = len(f0_hz)

    sg_window = max(3, int(round(_SG_WINDOW_MS * 1e-3 * frame_rate)))
    if sg_window % 2 == 0:
        sg_window += 1
    smooth_sigma = smooth_sigma_ms * 1e-3 * frame_rate

    f0_st = np.zeros(n, dtype=np.float64)
    f0_st[voiced_mask] = hz_to_midi(f0_hz[voiced_mask])

    wp_all = np.array([_ph_weight(p, nucleus_phs, cons_weight) for p in phonemes_pf],
                      dtype=np.float64)
    wp_all *= voiced_mask.astype(np.float64)

    correction = np.zeros(n, dtype=np.float64)
    note_starts = np.where(note_boundaries_pf)[0]
    note_ends = np.append(note_starts[1:], n)

    perceived_notes: list = []
    weight_vis = np.zeros(n, dtype=np.float64)

    for seg_start, seg_end in zip(note_starts, note_ends):
        pitch = float(note_pitches_pf[seg_start])
        if pitch == 0:
            continue  # 休符: 補正しない = correction 0 のまま(論文 Eq.A21)

        c, w_comb = _segment_correction_gaussian(
            f0_st[seg_start:seg_end], wp_all[seg_start:seg_end],
            pitch, sg_window, sigma_st, tukey_alpha)
        perceived_notes.append((int(seg_start), int(seg_end), pitch - c, pitch))
        wmax = float(np.max(w_comb)) if len(w_comb) else 0.0
        if wmax > 0:
            weight_vis[seg_start:seg_end] = w_comb / wmax

        # ノート全体に定数 c（端テーパは掛けない=論文どおり）
        correction[seg_start:seg_end] = c

    # ノート=定数c / 休符=0 の連結ベクトルを全体平滑化（境界をなめらかに繋ぐ）
    correction_smooth = gaussian_filter1d(correction, sigma=smooth_sigma)

    f0_st_corr = f0_st + correction_smooth
    f0_corr = f0_hz.copy()
    f0_corr[voiced_mask] = midi_to_hz(f0_st_corr[voiced_mask]).astype(np.float32)
    if return_details:
        details = {'correction_st': correction_smooth.astype(np.float32),
                   'weight': weight_vis.astype(np.float32)}
        return f0_corr, perceived_notes, details
    return f0_corr, perceived_notes


def apply_hann_pitch_correction(
    f0_hz: np.ndarray,            # (T,) F0 [Hz]（第1段補正後）
    voiced_mask: np.ndarray,      # (T,) bool
    phonemes_pf: list,            # (T,) フレームごとの音素
    note_pitches_pf: np.ndarray,  # (T,) ターゲット連続MIDI（0=休符）
    note_boundaries_pf: np.ndarray,
    frame_rate: float = 172.265625,
    alpha: float = 0.8,           # 補正強度（0=無効, 1=完全補正）
    smooth_sigma_ms: float = 11.0,
    nucleus_phs=DEFAULT_NUCLEUS_PHS,
    window: str = 'tukey',        # 'tukey'(中央平坦+端を別々にテーパー) | 'hann'(中央ピーク)
    lead_taper: float = 0.6,      # tukey: 先頭スロープ長（母音長比）。大=入口のしゃくれ温存
    trail_taper: float = 0.1,     # tukey: 末尾スロープ長（母音長比）。小=後半オーバーシュート抑制
    return_details: bool = False,
):
    """第2段: 各核音素フレームに 窓(hann|tukey) × α × (target − f0) の可変補正。

    第1段が「代表値からの定数オフセット」なのに対し、こちらはフレームごとの実際の
    ズレに比例した補正。window='hann' は中央だけ引き寄せ端は残す。'tukey' は中央平坦(=1)
    ＋先頭/末尾を別々の長さでコサインテーパー（入口のしゃくれを温存しつつ後半の跳躍
    オーバーシュートを平坦部で全補正できる）。
    """
    n = len(f0_hz)
    smooth_sigma = smooth_sigma_ms * 1e-3 * frame_rate

    f0_st = np.zeros(n, dtype=np.float64)
    f0_st[voiced_mask] = hz_to_midi(f0_hz[voiced_mask])

    correction = np.zeros(n, dtype=np.float64)
    note_starts = np.where(note_boundaries_pf)[0]
    note_ends = np.append(note_starts[1:], n)

    for seg_start, seg_end in zip(note_starts, note_ends):
        pitch = float(note_pitches_pf[seg_start])
        if pitch == 0:
            continue

        vowel_idx = np.array([
            j for j, f in enumerate(range(seg_start, seg_end))
            if phonemes_pf[f] in nucleus_phs and voiced_mask[f]])
        n_vow = len(vowel_idx)
        if n_vow == 0:
            continue

        if window == 'tukey':
            win = _asym_tukey_window(n_vow, lead_taper, trail_taper)
        else:
            win = _hann_window(n_vow)
        for j, rel_idx in enumerate(vowel_idx):
            frame = seg_start + rel_idx
            correction[frame] = alpha * win[j] * (pitch - f0_st[frame])

    correction_smooth = gaussian_filter1d(correction, sigma=smooth_sigma)

    f0_st_corr = f0_st + correction_smooth
    f0_corr = f0_hz.copy()
    f0_corr[voiced_mask] = midi_to_hz(f0_st_corr[voiced_mask]).astype(np.float32)
    if return_details:
        return f0_corr, correction_smooth.astype(np.float32)
    return f0_corr
