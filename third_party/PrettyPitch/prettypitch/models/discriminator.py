"""Multi-scale 1D PatchGAN discriminator + hinge/FM losses.

Input 4ch: [plain+vibrato (sum), plain, vibrato, vuv] — judges the naturalness of the
sum, whether plain/vibrato each keep their character (smooth / oscillatory), and the
realism of the v/uv pattern. v/uv comes from a separate BCE head (not diffused), but is
still fed here so the GAN shapes its voiced/unvoiced temporal pattern; the head's ±1 value
is aligned to the real 0/1 target as 2·sigmoid−1.
Adversarial terms are weighted per item by SNR (= alpha_bar[t]) so high-noise diffusion
steps contribute little (reflow uses uniform weight).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as WN

DISC_IN = 4


def disc_input(x):                                   # x:[B,3,T] (plain_n, vib_n, vuv±1)
    pl, vb, uv = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return torch.cat([pl + vb, pl, vb, uv], dim=1)   # [B,4,T]


class Disc1D(nn.Module):
    """One-scale PatchGAN (unconditional). Input 4ch."""

    def __init__(self, in_ch=DISC_IN, ch=64):
        super().__init__()
        self.convs = nn.ModuleList([
            WN(nn.Conv1d(in_ch,  ch,     15, 1, 7)),
            WN(nn.Conv1d(ch,     ch,     21, 4, 10, groups=4)),
            WN(nn.Conv1d(ch,     ch * 2, 21, 4, 10, groups=8)),
            WN(nn.Conv1d(ch * 2, ch * 2,  5, 1, 2)),
        ])
        self.post = WN(nn.Conv1d(ch * 2, 1, 3, 1, 1))

    def forward(self, x):
        h, feats = x, []
        for c in self.convs:
            h = F.leaky_relu(c(h), 0.2)
            feats.append(h)
        return self.post(h), feats


class MultiScaleDisc(nn.Module):
    """3 scales (avg-pool 1/1, 1/2, 1/4). Unconditional."""

    def __init__(self, n_scales=3):
        super().__init__()
        self.n_scales = n_scales
        self.D = nn.ModuleList([Disc1D() for _ in range(n_scales)])
        self.pool = nn.AvgPool1d(2, 2, count_include_pad=False)

    def forward(self, x):                         # x:[B,4,T]
        outs = []
        for i, D in enumerate(self.D):
            xi = x
            for _ in range(i):
                xi = self.pool(xi)
            outs.append(D(xi))
        return outs                               # [(logit[B,1,T'], feats), ...]


def d_loss_fn(D, aux):
    """Hinge D loss (real≥1 / fake≤−1), SNR(alpha_bar[t])-weighted.
    Real intermediate features are cached (detached) for G's feature matching."""
    vmask = aux['valid'][:, None, :].float()               # [B,1,T]
    real = disc_input(aux['x0'].detach() * vmask)          # [B,4,T]
    fake = disc_input(aux['x0_pred'].detach() * vmask)
    wt = aux['snr'][:, None, None] if aux.get('snr') is not None else 1.0  # reflow: uniform

    ro = D(real)
    ro_feats_cache = [[f.detach() for f in feats] for _, feats in ro]
    fo = D(fake)
    loss_d = sum(
        (F.relu(1.0 - rl) * wt).mean() + (F.relu(1.0 + fl) * wt).mean()
        for (rl, _), (fl, _) in zip(ro, fo)
    ) / D.n_scales
    return loss_d, ro_feats_cache


def g_adv_fm_fn(D, aux, ro_feats_cache):
    """G adversarial + feature-matching losses, SNR-weighted."""
    vmask = aux['valid'][:, None, :].float()
    fake = disc_input(aux['x0_pred'] * vmask)              # grads flow to G
    wt = aux['snr'][:, None, None] if aux.get('snr') is not None else 1.0  # reflow: uniform

    fo = D(fake)
    loss_adv = sum((-fl * wt).mean() for (fl, _) in fo) / D.n_scales
    loss_fm = sum(
        (f_fake - f_real).abs().mean()
        for (_, fake_feats), real_feats in zip(fo, ro_feats_cache)
        for f_fake, f_real in zip(fake_feats, real_feats)
    ) / D.n_scales
    return loss_adv, loss_fm
