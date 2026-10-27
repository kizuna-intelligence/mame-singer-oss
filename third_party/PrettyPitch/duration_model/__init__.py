"""duration_model — 子音長推定モデル（neighbor 方式・学習/プロファイル作成つき）。

PrettyPitch 単体で完結（外部依存なし）。統一音素 id は dict/ja.phonemes 準拠。
直前・直後の音素を子音/母音の区別なく入れる方式（子音連結＝英語等に対応）。
"""
from .model import ConsonantDurModel, load_cons_dur_model

__all__ = ['ConsonantDurModel', 'load_cons_dur_model']
