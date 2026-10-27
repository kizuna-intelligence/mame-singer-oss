"""ConsonantDurModel — 子音長推定 MLP（neighbor 方式・話者/スタイル対応）。

入力: cons_id + prev_ph_id + next_ph_id（直前・直後の音素を子音/母音の区別なく統一音素
      空間で）+ prev/next_note_dur, pitch_midi, pitch_diff, prev_note_exists
      (+ use_cluster_len なら cluster_len, n_speakers>1 なら spk_id, n_styles>0 なら style_id)。
出力: log_dur_pred [B]（log ドメイン）。推論: dur = exp(pred) - LOG_OFFSET, clamp ≥ 0.005。

話者/スタイルは埋め込みを加算（F0 モデルと同じ意味論）。n_speakers<=1 / n_styles<=0 なら
その埋め込みを持たず、従来（無条件）と完全同一。予測対象は子音長のみ。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

import warnings

from . import phoneset as _ps_mod

LOG_OFFSET = 0.001    # log(dur + LOG_OFFSET) の offset
DUR_SCALE = 1 / 0.3   # 典型ノート長 ~0.3s が log1p で ~1 程度にスケール
MAX_CLUSTER = 8       # cluster_len の埋め込み上限（英語で実測 5、日本語は 2。余裕を取る）


class ConsonantDurModel(nn.Module):
    """Args: hidden E（既定 64）, dropout, n_speakers（>1 で話者埋め込み）, n_styles（>0 でスタイル）,
    n_phonemes（省略時は既定 PhoneSet。言語で変わるので ckpt から復元する）。"""

    def __init__(self, hidden: int = 64, dropout: float = 0.1,
                 n_speakers: int = 1, n_styles: int = 0,
                 n_phonemes: int | None = None, use_cluster_len: bool = False,
                 use_wide_context: bool = False):
        super().__init__()
        E = hidden
        self.n_speakers = int(n_speakers)
        self.n_styles = int(n_styles)
        self.use_cluster_len = bool(use_cluster_len)
        self.use_wide_context = bool(use_wide_context)

        n_ph = int(n_phonemes) if n_phonemes else _ps_mod.active().n_phonemes
        self.n_phonemes = n_ph
        self.cons_emb    = nn.Embedding(n_ph, E)
        self.prev_ph_emb = nn.Embedding(n_ph, E)   # 位置の区別のみ（各テーブルは全音素を張る）
        self.next_ph_emb = nn.Embedding(n_ph, E)
        if use_wide_context:
            # ±2 の音素。クラスタの 3 番目以降の正体を知る唯一の入口。
            self.prev2_ph_emb = nn.Embedding(n_ph, E)
            self.next2_ph_emb = nn.Embedding(n_ph, E)
        self.pitch_emb       = nn.Embedding(128, E)
        self.prev_exists_emb = nn.Embedding(2,   E)
        self.prev_note_proj  = nn.Linear(1, E, bias=False)
        self.next_note_proj  = nn.Linear(1, E, bias=False)
        self.pitch_diff_proj = nn.Linear(1, E, bias=False)
        if self.use_cluster_len:
            # 自分が属する子音連結の長さ。前後 1 音素の特徴では表現できない量で、
            # 「あと何個続くか」を知らせる唯一の入口。日本語では 0/1 にしかならず
            # 実質不活性、英語やスラブ語で効く（言語非依存に定義されている）。
            self.cluster_emb = nn.Embedding(MAX_CLUSTER + 1, E)
        if self.n_speakers > 1:
            self.spk_emb = nn.Embedding(self.n_speakers, E)
        if self.n_styles > 0:
            self.style_emb = nn.Embedding(self.n_styles, E)
        self.mlp = nn.Sequential(
            nn.Linear(E, E), nn.LayerNorm(E), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(E, E), nn.LayerNorm(E), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(E, 1),
        )

    def encode(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        prev_nd = torch.log1p(b['prev_note_dur'].unsqueeze(-1) * DUR_SCALE)
        next_nd = torch.log1p(b['next_note_dur'].unsqueeze(-1) * DUR_SCALE)
        pdiff   = b['pitch_diff'].unsqueeze(-1).clamp(-12, 12) / 12.0
        h = (self.cons_emb(b['cons_id'])
             + self.prev_ph_emb(b['prev_ph_id'])
             + self.next_ph_emb(b['next_ph_id'])
             + self.pitch_emb(b['pitch_midi'])
             + self.prev_exists_emb(b['prev_note_exists'])
             + self.prev_note_proj(prev_nd)
             + self.next_note_proj(next_nd)
             + self.pitch_diff_proj(pdiff))
        if self.use_wide_context:
            h = h + self.prev2_ph_emb(b['prev2_ph_id']) + self.next2_ph_emb(b['next2_ph_id'])
        if self.use_cluster_len:
            h = h + self.cluster_emb(b['cluster_len'].clamp(0, MAX_CLUSTER))
        if self.n_speakers > 1:
            h = h + self.spk_emb(b['spk_id'])
        if self.n_styles > 0:
            if b.get('style_mix') is not None:
                # 連続スタイル: [N, n_styles] の混合比 × 埋め込み表（one-hot なら style_id と同値）
                h = h + b['style_mix'] @ self.style_emb.weight
            else:
                h = h + self.style_emb(b['style_id'])
        return h

    def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.mlp(self.encode(b)).squeeze(-1)

    @staticmethod
    def to_log_target(dur_sec: torch.Tensor) -> torch.Tensor:
        return torch.log(dur_sec + LOG_OFFSET)

    def loss(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        return nn.functional.mse_loss(self.forward(b), self.to_log_target(b['target_dur']))

    @torch.no_grad()
    def predict(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        """→ 予測持続長 [sec], clamp ≥ 0.005"""
        return (torch.exp(self.forward(b)) - LOG_OFFSET).clamp(min=0.005)


def load_cons_dur_model(ckpt, device: str | torch.device = 'cpu',
                        dropout: float = 0.1) -> ConsonantDurModel:
    """ckpt（パス or dict）から構築・load。

    話者/スタイル数・音素数・hidden は **state_dict の形状を最優先**で判定するので、
    config が欠けた古い ckpt も正しく再構築できる。
    音素表は ckpt の config['lang'] / ['phonemes'] / ['nucleus'] から PhoneSet を
    復元して `model.phoneset` に付ける（推論で言語を取り違えないため）。
    その情報が無い古い ckpt は既定の PhoneSet にフォールバックし、警告を出す。
    """
    if isinstance(ckpt, (str, Path)):
        ckpt = torch.load(ckpt, map_location=device)
    sd = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    cfg = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
    hidden = int(cfg.get('hidden') or (sd['mlp.0.weight'].shape[0] if 'mlp.0.weight' in sd else 64))
    # state_dict の形状を最優先（config 欠落 ckpt も正しく再構築できる）
    n_speakers = int(sd['spk_emb.weight'].shape[0]) if 'spk_emb.weight' in sd else int(cfg.get('n_speakers', 1))
    n_styles = int(sd['style_emb.weight'].shape[0]) if 'style_emb.weight' in sd else int(cfg.get('n_styles', 0))
    n_phonemes = int(sd['cons_emb.weight'].shape[0]) if 'cons_emb.weight' in sd else None
    # cluster_len 埋め込みの有無は **state_dict で判定**する。config を持たない古い ckpt を
    # そのまま strict=True で読むため。
    use_cluster_len = 'cluster_emb.weight' in sd
    use_wide_context = 'prev2_ph_emb.weight' in sd

    ps = _phoneset_from_cfg(cfg, n_phonemes)
    model = ConsonantDurModel(hidden=hidden, dropout=dropout,
                              n_speakers=n_speakers, n_styles=n_styles,
                              n_phonemes=n_phonemes or ps.n_phonemes,
                              use_cluster_len=use_cluster_len,
                              use_wide_context=use_wide_context).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    model.phoneset = ps
    # spk_map（DB名 -> 話者 id）を記録しているのはこの ckpt だけなので、
    # 表示用の話者名の唯一の情報源として持ち回る（svs.render.speaker_names が使う）。
    model.spk_map = dict(cfg.get('spk_map') or {})
    return model


def _phoneset_from_cfg(cfg: dict, n_phonemes: int | None):
    """ckpt の config から PhoneSet を復元する。情報が無ければ既定へフォールバック。"""
    lang = cfg.get('lang')
    if lang:
        ps = None
        try:
            ps = _ps_mod.load(lang, phonemes=cfg.get('phonemes_path'),
                              nucleus=cfg.get('nucleus_path'))
        except Exception as e:                       # noqa: BLE001
            # 記録された音素表のパスが配布先に無いのはよくある（作者のローカルパスが
            # 焼かれた古い ckpt など）。ここで既定言語に落ちると英語 ckpt が黙って
            # 日本語になるので、まず言語パック自身の音素表で読み直す。
            try:
                ps = _ps_mod.load(lang)
                warnings.warn(
                    f"ckpt に記録された音素表を読めませんでした（{e}）。"
                    f"言語パック '{lang}' 同梱の音素表を使います。")
            except Exception as e2:                  # noqa: BLE001
                warnings.warn(f"ckpt の言語パック '{lang}' を読めませんでした（{e2}）。既定を使います。")
        if ps is not None:
            # ckpt は音素の**並び**も記録している。id は行順なので、並びが違えば
            # 埋め込みの行と音素の対応がずれる。数だけでなく中身を照合する。
            rec = list(cfg.get('phonemes') or [])
            cur = list(ps.phonemes)
            if rec and cur[:len(rec)] == rec:
                # パックが ckpt の音素表を**前方一致で拡張**しただけ（末尾に足した）。
                # 既存 id は動いていないので推論は正しい。ckpt が知らない音素だけ使えない。
                if len(cur) > len(rec):
                    warnings.warn(
                        f"言語パック '{lang}' は ckpt より新しい音素を持ちます"
                        f"（ckpt {len(rec)} / パック {len(cur)}）。既存 id は不変なので推論は"
                        f"正しく動きますが、追加分 {cur[len(rec):]} はこの ckpt では扱えません。")
            elif rec and cur != rec:
                warnings.warn(
                    f"ckpt の音素表が言語パック '{lang}' と一致しません"
                    f"（ckpt {len(rec)} / パック {ps.n_phonemes}）。音素IDがずれるため"
                    f"推論結果が壊れます。正しい音素表を --phonemes で指定してください。")
            elif n_phonemes and ps.n_phonemes != n_phonemes:
                warnings.warn(
                    f"ckpt の音素数 {n_phonemes} が言語パック '{lang}' の {ps.n_phonemes} と"
                    f"一致しません。ckpt 側を信用して埋め込みを作りますが、音素表を確認してください。")
            return ps
    ps = _ps_mod.active()
    if n_phonemes and ps.n_phonemes != n_phonemes:
        warnings.warn(
            f"この ckpt は音素表を記録していません（古い形式）。既定 '{ps.lang}' は "
            f"{ps.n_phonemes} 音素ですが ckpt は {n_phonemes} です。"
            f"--lang で正しい言語パックを指定してください。")
    return ps
