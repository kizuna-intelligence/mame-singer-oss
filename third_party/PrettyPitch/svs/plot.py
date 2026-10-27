"""SVS の可視化ヘルパ（notebook 用）。

PrettyPitch 本体は mel を扱わないため、mel の描画と「出力音声の再分析」はここに置く。
matplotlib / librosa は pyproject の依存に含まれる。svs/__init__.py からは import しない
（svs パッケージ本体を torch/matplotlib 非依存に保つため）。

    from svs.plot import plot_alignment, plot_f0, plot_mel, plot_wave, mel_from_wav
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from prettypitch.pitch import hz_to_midi

# ボコーダ（NHVSing）と厳密一致する mel レシピ。LeapSinger/leapsinger/mel.py:33
# wav_to_mel_nhv と同一だが、LeapSinger を sys.path に載せずに使えるようここで再実装する。
MEL_SR, MEL_N_FFT, MEL_WIN, MEL_HOP = 44100, 2048, 2048, 256
MEL_N_MELS, MEL_FMIN, MEL_FMAX = 128, 40.0, 16000.0
MEL_VMIN, MEL_VMAX = -11.5, 2.0          # 音響モデル config の既定（描画レンジ）

_MEL_BASIS_CACHE: dict = {}

# 図中のラベルはすべて英語。matplotlib 既定の DejaVu Sans に日本語グリフが無く、
# 環境によって豆腐になるため（notebook の地の文は日本語のまま）。


def mel_from_wav(wav, sr: int = MEL_SR, hop: int = MEL_HOP, n_fft: int = MEL_N_FFT,
                 win: int = MEL_WIN, n_mels: int = MEL_N_MELS,
                 fmin: float = MEL_FMIN, fmax: float = MEL_FMAX) -> np.ndarray:
    """波形 → ln-mel [n_mels, T]（ボコーダ学習時と同一レシピ）。

    予測 mel とフレーム単位で揃うので、そのまま差分を取って「ボコーダが要求された mel を
    再現できているか」を見られる。
    """
    import librosa
    key = (sr, n_fft, n_mels, fmin, fmax)
    if key not in _MEL_BASIS_CACHE:
        _MEL_BASIS_CACHE[key] = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).astype(np.float64)
    pad = (n_fft - hop) // 2
    yp = np.pad(np.asarray(wav, dtype=np.float64), (pad, pad), mode='reflect')
    stft = librosa.stft(yp, n_fft=n_fft, hop_length=hop, win_length=win,
                        window='hann', center=False)
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        mel = _MEL_BASIS_CACHE[key] @ np.abs(stft)
    return np.log(np.maximum(1e-5, mel)).astype(np.float32)


def _ax(ax, figsize):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt
    return plt.subplots(figsize=figsize)[1]


def plot_alignment(seq, ax=None, max_label_sec: float = 0.04, figsize=(14, 2.4)):
    """PhonemeSequence の音素区間を帯で描く（母音は濃色・ノート先頭に▼）。"""
    ax = _ax(ax, figsize)
    t = np.concatenate([[0.0], np.cumsum(np.asarray(seq.dur_sec, float))])
    for i, ph in enumerate(seq.phonemes):
        x0, x1 = t[i], t[i + 1]
        vowel = bool(seq.is_vowel[i])
        pau = ph == 'pau'
        ax.axvspan(x0, x1, color=('#cfd8dc' if pau else ('#7e9cc4' if vowel else '#e8c39e')),
                   ec='white', lw=0.8)
        if (x1 - x0) >= max_label_sec and not pau:
            ax.text((x0 + x1) / 2, 0.5, ph, ha='center', va='center', fontsize=9)
        if seq.note_event_starts[i]:
            ax.plot([x0], [0.93], marker='v', ms=6, color='#c62828')
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 1)
    ax.set_yticks([]); ax.set_xlabel('time [s]')
    ax.set_title('phoneme durations  (blue=vowel / orange=consonant / grey=pau, \u25bc=note onset)')
    return ax


def plot_f0(out: dict, ax=None, seq=None, show_delta: bool = True, figsize=(14, 3.2)):
    """synthesize_f0 の出力を MIDI 目盛で描く（baseline 破線＋補正後 F0 実線）。

    show_delta=True なら plain / vibrato の寄与を cent で第2軸に重ねる。
    """
    ax = _ax(ax, figsize)
    fr = float(out.get('frame_rate', 172.265625))
    f0 = np.asarray(out['f0_hz'], float)
    uv = np.asarray(out['uv'], float)
    t = np.arange(len(f0)) / fr
    midi = np.where(uv > 0, hz_to_midi(f0), np.nan)
    base = np.asarray(out['baseline_log2'], float)
    base_midi = 69 + 12 * (base - np.log2(440.0))
    ax.plot(t, base_midi, '--', lw=1.0, color='#888', label='baseline (score pitch)')
    ax.plot(t, midi, lw=1.4, color='#1565c0', label='generated F0 (after 2-stage correction)')
    if seq is not None:
        b = np.concatenate([[0.0], np.cumsum(np.asarray(seq.dur_sec, float))])
        for x, st in zip(b[:-1], seq.note_event_starts):
            if st:
                ax.axvline(x, color='#c62828', lw=0.5, alpha=0.35)
    ax.set_xlim(t[0], t[-1]); ax.set_ylabel('pitch [MIDI]'); ax.set_xlabel('time [s]')
    ax.legend(loc='upper right', fontsize=8)
    if show_delta and 'plain_delta' in out:
        ax2 = ax.twinx()
        ax2.plot(t, np.asarray(out['plain_delta'], float) * 1200, lw=0.8,
                 color='#2e7d32', alpha=0.7, label='plain (expression)')
        ax2.plot(t, np.asarray(out['vib_delta'], float) * 1200, lw=0.8,
                 color='#ef6c00', alpha=0.8, label='vibrato')
        ax2.set_ylabel('deviation [cent]'); ax2.legend(loc='lower right', fontsize=8)
    ax.set_title('F0')
    return ax


def plot_mel(mel, frame_rate: float = MEL_SR / MEL_HOP, ax=None,
             vmin: Optional[float] = MEL_VMIN, vmax: Optional[float] = MEL_VMAX,
             title: str = 'mel', figsize=(14, 3.0)):
    """ln-mel [n_mels, T] を描く。vmin/vmax は音響モデル config の既定に合わせてある。"""
    ax = _ax(ax, figsize)
    mel = np.asarray(mel)
    im = ax.imshow(mel, origin='lower', aspect='auto', cmap='magma',
                   vmin=vmin, vmax=vmax,
                   extent=[0, mel.shape[1] / frame_rate, 0, mel.shape[0]])
    ax.set_xlabel('time [s]'); ax.set_ylabel('mel bin'); ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, pad=0.01)
    return ax


def plot_wave(wav, sr: int = MEL_SR, ax=None, figsize=(14, 2.0)):
    """波形。"""
    ax = _ax(ax, figsize)
    wav = np.asarray(wav, float)
    ax.plot(np.arange(len(wav)) / sr, wav, lw=0.4, color='#37474f')
    ax.set_xlim(0, len(wav) / sr); ax.set_xlabel('time [s]'); ax.set_ylabel('amplitude')
    ax.set_title(f'waveform  (peak={np.abs(wav).max():.3f})')
    return ax
