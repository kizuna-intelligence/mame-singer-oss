"""Acoustic model: phonemes, durations and F0 of a score, plus a voice vector, to a 128-bin log-mel."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

F0_REF = 440.0
EMB = 256


class Block(nn.Module):
    def __init__(self, d, k=7, mult=3, dilation=1):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=dilation * (k // 2), dilation=dilation, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, mult * d)
        self.pw2 = nn.Linear(mult * d, d)

    def forward(self, x):                                   # x: (B, d, T)
        h = self.dw(x).transpose(1, 2)
        h = self.pw2(F.gelu(self.pw1(self.norm(h))))
        return x + h.transpose(1, 2)


class Acoustic(nn.Module):
    def __init__(self, dim=128, layers=6, n_ph=72, ph_layers=3, mel=128):
        super().__init__()
        self.ph_emb = nn.Embedding(n_ph, dim)
        self.voice = nn.Sequential(nn.Linear(EMB, dim), nn.GELU(), nn.Linear(dim, dim))
        self.ph_net = nn.Sequential(*[Block(dim, k=5) for _ in range(ph_layers)])
        self.frame_in = nn.Linear(4, dim)
        self.film = nn.Linear(dim, 2 * dim)
        self.net = nn.Sequential(*[Block(dim, dilation=(1, 2, 4)[i % 3]) for i in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, mel)
        self.task = nn.Embedding(2, dim)                    # 0 singing voice, 1 speaking voice

    def forward(self, ph, frame_ph, pos, f0, unvoiced, emb, mode):
        v = self.voice(emb) + self.task(mode)
        h = self.ph_emb(ph) + v[:, None]
        h = self.ph_net(h.transpose(1, 2)).transpose(1, 2)
        h = torch.gather(h, 1, frame_ph[..., None].expand(-1, -1, h.size(-1)))
        lf0 = torch.log2(f0.clamp(min=20.0) / F0_REF)
        h = h + self.frame_in(torch.stack([lf0, unvoiced, pos, 1 - pos], -1))
        g, b = self.film(v).chunk(2, -1)
        h = h * (1 + g[:, None]) + b[:, None]
        h = self.net(h.transpose(1, 2)).transpose(1, 2)
        return self.out(self.norm(h)).transpose(1, 2)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        m = cls(ck["dim"], ck["layers"], ck["n_ph"])
        m.load_state_dict(ck["model"])
        return m.to(device).eval()

    @torch.no_grad()
    def mel(self, ph_ids, ph_durs, f0, voicing, emb, mode, device="cpu"):
        """one phrase: phoneme ids and frame counts, per-frame F0 (Hz) and voicing (0..1), voice vector (256,)"""
        durs = np.asarray(ph_durs, np.int64)
        T = len(f0)
        frame_ph = np.repeat(np.arange(len(durs)), durs)[:T]
        frame_ph = np.pad(frame_ph, (0, T - len(frame_ph)), mode="edge")
        start = np.repeat(np.cumsum(durs) - durs, durs)[:T]
        start = np.pad(start, (0, T - len(start)), mode="edge")
        pos = (np.arange(T) - start) / np.maximum(durs[frame_ph], 1)
        t = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)[None]
        out = self(t(ph_ids, torch.long), t(frame_ph, torch.long), t(pos), t(f0), 1 - t(voicing), t(emb),
                   torch.full((1,), int(mode), dtype=torch.long, device=device))
        return out[0].cpu().numpy()
