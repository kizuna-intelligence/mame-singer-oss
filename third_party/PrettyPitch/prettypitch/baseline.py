"""Baseline construction — shared by training (dataset.py) and inference (infer.py).

The baseline is the smoothed "note pitch" curve the model's plain/vibrato residuals ride
on. In PrettyPitch there is no score: each *valued* phoneme (training: nucleus phonemes
with a representative value from `pitch.py`; inference: any phoneme the user gives a
pitch to, `cl` included) contributes one step of a staircase, values propagate forward
to the following unvalued phonemes, the leading pau/consonants take the first value,
and the staircase is Gaussian-smoothed.

One rule covers onsets and rests: with forward-fill an onset consonant keeps the previous
note's pitch until the nucleus, and a user value on a non-nucleus phoneme at inference
simply inserts an extra step — the model inputs stay nucleus-only, so plain/vibrato/vuv
are unchanged.

σ = 20 ms (≈3.4 frames @172.27fps). Not larger: a 50 ms σ absorbs portamento into the
baseline and strips intonation from the plain channel.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

GAUSSIAN_SIGMA_SEC = 0.02   # 20 ms

LOG2_440 = float(np.log2(440.0))


def phoneme_frame_bounds(dur_sec: np.ndarray, frame_rate: float, T: int) -> np.ndarray:
    """Cumulative phoneme frame boundaries [L+1], last pinned to T.

    The single rounding rule shared by preprocessing, the loader, the baseline and the
    vibrato gain, so every consumer sees identical phoneme spans.
    """
    cum = np.concatenate([[0.0], np.cumsum(np.asarray(dur_sec, dtype=np.float64))])
    fb = np.round(cum * frame_rate).astype(int)
    fb[-1] = int(T)
    return fb


def fill_values(values_midi: np.ndarray) -> np.ndarray:
    """Forward-fill finite values over the phoneme axis, back-fill the head.

    values_midi: float [L], NaN = no value. Returns float64 [L] with no NaN, or all-NaN
    if no phoneme carries a value (caller decides the fallback).
    """
    v = np.asarray(values_midi, dtype=np.float64).copy()
    finite = np.isfinite(v)
    if not finite.any():
        return v
    idx = np.arange(len(v))
    last = np.maximum.accumulate(np.where(finite, idx, -1))       # nearest valued slot to the left
    filled = np.where(last >= 0, v[np.maximum(last, 0)], np.nan)  # forward fill
    first = idx[finite][0]
    filled[:first] = v[first]                                     # back-fill the head
    return filled


def build_baseline(values_midi: np.ndarray,     # float [L], NaN = no value
                   dur_sec: np.ndarray,         # float [L] phoneme durations [s]
                   T: int,
                   frame_rate: float,
                   sigma_sec: float = GAUSSIAN_SIGMA_SEC) -> np.ndarray | None:
    """Valued-phoneme staircase → per-frame → Gaussian smooth → log2(Hz) float32 [T].

    Returns None when no phoneme carries a value (the caller falls back / skips).
    Smoothing runs in MIDI space; the MIDI→log2Hz map is affine, so this is exact.
    """
    ph_val = fill_values(values_midi)
    if not np.isfinite(ph_val).all():
        return None
    fb = phoneme_frame_bounds(dur_sec, frame_rate, T)
    step = np.empty(int(T), dtype=np.float64)
    for k in range(len(ph_val)):
        a, b = fb[k], fb[k + 1]
        if b > a:
            step[a:b] = ph_val[k]
    step = gaussian_filter1d(step, sigma=max(sigma_sec * frame_rate, 1e-6))
    return (LOG2_440 + (step - 69.0) / 12.0).astype(np.float32)
