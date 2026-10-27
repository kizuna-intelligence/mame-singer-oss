"""F0 残差の plain/vibrato 分解ユーティリティ（フレームレート一般化）。

residual = log2(f0) − baseline を次の確定方式で分解する:

    vibrato = make_vibrato_gain(...) × bandpass( compress_residual(residual) )
    plain   = residual − vibrato

  1. compress_residual: 残差を閾値 T 半音まで恒等、超過分のみ圧縮（既定 tanh 飽和）。
     母音冒頭のスクープ（低F0→ノート音高の急上昇、-6半音級）がビブラート帯にエネルギーを
     持ち、フィルタがリンギングして偽ビブラートを生むのを振幅側から抑える。
     実ビブラート（±2半音以内）は無傷。
  2. extract_vibrato: ビブラート帯（既定 4.5-7.5 Hz）のバンドパス FIR（窓 0.3 秒 →
     172.265625 fps で 53 tap）を filtfilt（零位相）で適用。
  3. make_vibrato_gain: 非核音素区間±0.1s と音素境界±0.1s を 0 にした 0/1 カーブを
     ガウシアン σ=0.05s で平滑した 0-1 ゲインを乗算。
  除去分は plain = residual − vibrato が自動的に吸収するので、
  **baseline + plain + vibrato ≡ log2(f0)** の再構成不変量は常に保存される。

FIR はタップ数でなく「秒」で設計する（帯域は物理量でフレームレート非依存）。
filtfilt には最小系列長（3×タップ数+1）があり、短いフレーズは vibrato 全ゼロになる
（vibrato は plain 側へ残る）— データセットの min_sec はこの制約から決まる。
モジュール既定値は configs/ja_example.yaml の decomp: と同じ（学習・推論は config の値を渡す）。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import filtfilt, firwin

FIR_TIME_LEN_SEC = 0.3
VIB_BAND_HZ = (4.5, 7.5)

VIB_COMP_METHOD    = 'tanh'  # ソフトクリップ方式 'tanh' | 'log' | 'linear'
VIB_COMP_THRESH_ST = 2.0     # ここまで恒等 [半音]（実ビブラートの最大想定振幅）
VIB_COMP_PARAM     = {'tanh': 0.5, 'log': 1.0, 'linear': 4.0}  # method別の既定パラメータ
GAIN_MARGIN_SEC    = 0.10    # 非核音素区間・音素境界の周囲をゲイン0にする幅
GAIN_SIGMA_SEC     = 0.05    # ゲイン曲線のガウシアン平滑 σ


def make_vibrato_fir(frame_rate: float,
                     band: tuple[float, float] = VIB_BAND_HZ,
                     time_len_sec: float = FIR_TIME_LEN_SEC) -> np.ndarray:
    """ビブラート帯バンドパス FIR を「秒」で設計する（奇数タップに丸め）。"""
    n_taps = int(round(frame_rate * time_len_sec)) | 1   # 奇数タップ
    return firwin(n_taps, list(band), pass_zero=False,
                  fs=frame_rate, window='hamming')


def filtfilt_min_len(fir: np.ndarray) -> int:
    """filtfilt が受け付ける最小系列長（scipy 既定 padlen = 3*len(b)、len(x) > padlen）。"""
    return 3 * len(fir) + 1


def compress_residual(x: np.ndarray,
                      method: str = VIB_COMP_METHOD,
                      thresh_st: float = VIB_COMP_THRESH_ST,
                      param: float | None = None) -> np.ndarray:
    """差分F0(log2)の振幅抑制。|x|≤T は恒等、超過分のみ圧縮（奇関数）。

    method='tanh'  : y = T + E·tanh((|x|−T)/E)。出力上限 T+E で有界。C1連続。既定。
    method='log'   : y = T + S·ln(1+(|x|−T)/S)。無界だが緩やか。C1連続。
    method='linear': y = T + (|x|−T)/ratio。最も単純（knee で C1 でない）。
    param の意味: tanh→飽和幅E[半音] / log→スケールS[半音] / linear→比 ratio。
    """
    if param is None:
        param = VIB_COMP_PARAM[method]
    T = thresh_st / 12.0
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    ex = np.maximum(ax - T, 0.0)
    if method == 'tanh':
        E = param / 12.0
        comp = T + E * np.tanh(ex / E)
    elif method == 'log':
        S = param / 12.0
        comp = T + S * np.log1p(ex / S)
    elif method == 'linear':
        comp = T + ex / param
    else:
        raise ValueError(method)
    return np.sign(x) * np.where(ax <= T, ax, comp)


def make_vibrato_gain(phoneme_id_per_frame: np.ndarray,
                      is_nucleus_frame: np.ndarray,
                      dur_sec: np.ndarray,
                      frame_rate: float,
                      n_frames: int,
                      margin_sec: float = GAIN_MARGIN_SEC,
                      sigma_sec: float = GAIN_SIGMA_SEC,
                      cons_margin_pre_sec: float | None = None,
                      cons_margin_post_sec: float | None = None,
                      boundary_margin_pre_sec: float | None = None,
                      boundary_margin_post_sec: float | None = None,
                      boundary_mode: str = 'all',
                      limit_margin_to_adjacent: bool = True,
                      min_open_sec: float = 0.0) -> np.ndarray:
    """vibrato 用ゲイン曲線 [T] float64（0-1）。

    ①非核音素区間（非pau・非核）の前 cons_margin_pre_sec / 後 cons_margin_post_sec を 0 →
    ②音素境界の前 boundary_margin_pre_sec / 後 boundary_margin_post_sec を 0 →
    ③ガウシアンフィルタ(σ=sigma_sec)で平滑化（急峻なゲイン切替を避ける）。
    境界は dur_sec 累積からフレーム展開と同じ丸めで復元する。

    前後マージンは None なら margin_sec を使う。
    「子音の後」= しゃくりの立ち上がり、「境界の後」= 次音素(母音)の頭で、
    歌唱ではここに非ビブラートの音程遷移が乗るため後ろ側を長めに取るのが有効。
    boundary_mode: 'all'=全境界(従来) / 'nonnucleus'=核→核境界(メリスマ等)はゼロ化
    しない / 'none'=境界ゼロ化なし(子音マージンのみ)。

    limit_margin_to_adjacent=True(既定): 各マージンは**隣接する1音素の中まで**で強制
    打ち切り（前側=直前音素の開始、後側=直後音素の終了）。短い音素が続いても
    「次の次の音素」へゼロ化が持ち越されない。False で無制限に延長する。

    min_open_sec>0: 平滑化の**前**に、マージンに挟まれて残った開口（gain=1 の連続区間）
    が この幅未満なら完全に閉じる。細い隙間がガウシアン平滑で「小さなコブ」として
    生き残るのを防ぎ、ゲインは十分な広さの白色空間（とその平滑すそ）にだけ立つ。
    """
    T = n_frames
    mc_pre = int(round((margin_sec if cons_margin_pre_sec is None
                        else cons_margin_pre_sec) * frame_rate))
    mc_post = int(round((margin_sec if cons_margin_post_sec is None
                         else cons_margin_post_sec) * frame_rate))
    mb_pre = int(round((margin_sec if boundary_margin_pre_sec is None
                        else boundary_margin_pre_sec) * frame_rate))
    mb_post = int(round((margin_sec if boundary_margin_post_sec is None
                         else boundary_margin_post_sec) * frame_rate))
    gain = np.ones(T, dtype=np.float64)
    phf = np.asarray(phoneme_id_per_frame)
    is_nuc = np.asarray(is_nucleus_frame, bool)
    cons = (phf != 0) & (~is_nuc)
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(dur_sec, dtype=np.float64))])
    fb = np.round(cum * frame_rate).astype(int)

    def _prev_bound(x: int) -> int:
        """x より小さい最大の音素境界（= x を含む/直前の音素の開始）。"""
        i = int(np.searchsorted(fb, x, side='left')) - 1
        return int(fb[i]) if i >= 0 else 0

    def _next_bound(x: int) -> int:
        """x より大きい最小の音素境界（= x から始まる音素の終了）。"""
        i = int(np.searchsorted(fb, x, side='right'))
        return int(fb[i]) if i < len(fb) else T

    t = 0
    while t < T:
        if cons[t]:
            s = t
            while t < T and cons[t]:
                t += 1
            lo, hi = s - mc_pre, t + mc_post
            if limit_margin_to_adjacent:
                lo = max(lo, _prev_bound(s))     # 直前音素の開始まで
                hi = min(hi, _next_bound(t))     # 直後音素の終了まで
            gain[max(0, lo):min(T, hi)] = 0.0
        else:
            t += 1
    if boundary_mode != 'none':
        for k in range(1, len(fb) - 1):
            b = int(fb[k])
            if boundary_mode == 'nonnucleus' and 0 < b < T and \
               is_nuc[b - 1] and is_nuc[b]:
                continue                     # 核→核（メリスマ等）の境界は生かす
            lo, hi = b - mb_pre, b + mb_post
            if limit_margin_to_adjacent:
                lo = max(lo, int(fb[k - 1]))     # 直前音素の開始まで
                hi = min(hi, int(fb[k + 1]))     # 直後音素の終了まで
            gain[max(0, lo):min(T, hi)] = 0.0
    if min_open_sec > 0:                          # 細い開口は平滑前に閉じる（コブ防止）
        mo = int(round(min_open_sec * frame_rate))
        t = 0
        while t < T:
            if gain[t] > 0.0:
                s = t
                while t < T and gain[t] > 0.0:
                    t += 1
                if t - s < mo:
                    gain[s:t] = 0.0
            else:
                t += 1
    return gaussian_filter1d(gain, sigma=max(sigma_sec * frame_rate, 1e-6))


def extract_vibrato(residual: np.ndarray,
                    uv: np.ndarray,
                    fir: np.ndarray) -> np.ndarray:
    """log2 残差からビブラート成分を抽出する。

    filtfilt（零位相）バンドパス → 無声フレーム 0。
    系列が filtfilt_min_len 未満なら全ゼロ（vibrato は plain 側へ残る）。
    """
    residual = np.asarray(residual, dtype=np.float64)
    if len(residual) < filtfilt_min_len(fir):
        return np.zeros_like(residual, dtype=np.float32)
    vib = filtfilt(fir, 1.0, residual)
    return np.where(np.asarray(uv) > 0.5, vib, 0.0).astype(np.float32)
