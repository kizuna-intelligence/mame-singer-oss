"""Voice vector of a few seconds of someone speaking (or singing): a reference encoder on a linear spectrogram.

The encoder is OpenVoice v2's reference encoder (MIT License, Copyright 2024 MyShell.ai; see
LICENSES/OpenVoice-MIT.txt), and so is this file.
"""
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET_RMS = 0.111


class ReferenceEncoder(nn.Module):
    def __init__(self, spec_channels, gin_channels=256):
        super().__init__()
        self.spec_channels = spec_channels
        filters = [1, 32, 32, 64, 64, 128, 128]
        self.convs = nn.ModuleList(nn.Conv2d(filters[i], filters[i + 1], 3, 2, 1) for i in range(6))
        out = spec_channels
        for _ in range(6):
            out = (out - 3 + 2) // 2 + 1
        self.gru = nn.GRU(128 * out, 128, batch_first=True)
        self.proj = nn.Linear(128, gin_channels)
        self.layernorm = nn.LayerNorm(spec_channels)

    def forward(self, spec):                               # (N, T, F)
        x = self.layernorm(spec[:, None])
        for c in self.convs:
            x = F.relu(c(x))
        x = x.transpose(1, 2).flatten(2)
        _, h = self.gru(x)
        return self.proj(h[0])


class VoiceEncoder:
    def __init__(self, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        self.cfg = {k: ck[k] for k in ("sampling_rate", "filter_length", "hop_length", "win_length")}
        self.enc = ReferenceEncoder(ck["filter_length"] // 2 + 1, ck["gin_channels"])
        self.enc.load_state_dict(ck["model"])
        self.enc.to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, wav, sr):
        """(256,) unit vector of one recording"""
        wav = np.asarray(wav, np.float32)
        if wav.ndim > 1:
            wav = wav.mean(1)
        active = np.abs(wav) > 1e-2 * np.abs(wav).max()
        wav = np.clip(wav * TARGET_RMS / np.sqrt(np.mean(wav[active] ** 2)), -1, 1)
        c = self.cfg
        y = torch.from_numpy(librosa.resample(wav, orig_sr=sr, target_sr=c["sampling_rate"]))[None].to(self.device)
        n, hop = c["filter_length"], c["hop_length"]
        y = F.pad(y[:, None], ((n - hop) // 2, (n - hop) // 2), mode="reflect")[:, 0]
        spec = torch.stft(y, n, hop, c["win_length"], torch.hann_window(c["win_length"], device=self.device),
                          center=False, return_complex=True)
        spec = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-6)
        return self.enc(spec.transpose(1, 2))[0].cpu().numpy()

    def voice(self, recordings):
        """voice vector of several (wav, sr) recordings of the same person"""
        e = np.mean([self.embed(w, sr) for w, sr in recordings], 0)
        return (e / np.linalg.norm(e)).astype(np.float32)
