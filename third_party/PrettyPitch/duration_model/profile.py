"""音源別「子音長の縮小規則」プロファイル（自己完結・外部依存なし）。

早口時の子音長 floor（子音別 c_min + 母音 v_min）を JSON で管理する。
ファイルが無ければ組込み既定にフォールバック。

スキーマ:
    { "enabled": true,
      "vowel_floor_sec": 0.075,                   # 全話者共通の母音 floor
      "cons_floor_sec":  {"k":0.01, ...},
      "lead_cons_floor_sec": {"w":0.055, ...},    # フレーズ先頭子音の floor
      "phoneme_weight": {"k":1.0, "a":1.2, ...},  # 区間が足りないときの取り分の重み（音素別）
      "phoneme_weight_default": 1.0,              # 上の表に無い音素の重み
      "per_speaker": {                            # "<spk>" / "<spk>:<style>" 別の上書き
        "1": { "vowel_floor": 0.075, "phoneme_weight_default": 1.0,
               "cons_floor": {...}, "lead_cons_floor": {...}, "phoneme_weight": {...} } } }

母音 floor は `svs/duration_postprocess._compress_region` が次の規則で使う:
「なるべく floor を保証（子音は自然長、母音に v_min を残す所まで）／母音 floor + 子音 floor が
ノート長に入らないときだけ重み比で按分」。

**値の出どころ**: 話者別の値はレシピ（configs/recipes/<db>.yaml の `duration:`）が持ち、
`build_profile` がそれをこの JSON の `per_speaker` へ書き出す。推論はこの JSON だけを読む。
モジュール定数（75ms / 50ms / 1.0）は **プロファイルもレシピも無いときの最終フォールバック**で、
話者の性質はそこに書かない。

`phoneme_weight` は `svs/duration_postprocess._compress_region` が使う。区間が足りないとき、
その区間に居る音素どうしで長さを分ける比で、**子音にも母音にも引ける**。既定 1.0 で、
全音素が既定なら等分になる。音素別・話者別・スタイル別に上書きできる。
旧キー `cons_weight`（子音のみ）と `short_note_ratio`（割合 r。r/(1-r) で換算）も読める。
floor の比 `c_min : v_min` を配分に使わないのは、`c_min` が分布の下限であって配分の重みでは
ないため（摩擦音は下限が高く破裂音は低いので、破裂音だけが構造的に潰れる）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from . import phoneset as _ps_mod

# ── 最終フォールバック（プロファイル JSON もレシピも無いときだけ効く）───────────────
# 話者ごとの値はレシピ `duration:` → build_profile → JSON の per_speaker で持つ。ここは
# 「何も無いときに壊れない」ための値であって、話者の性質を表す場所ではない。
VOWEL_FLOOR_SEC = 0.075            # 母音の最低保証（早口時にこれ以下へ潰れない。話者別はレシピ）
_DEFAULT_VOWEL_FLOOR = VOWEL_FLOOR_SEC
_DEFAULT_CONS_FLOOR = 0.05         # 統計の無い子音の floor
_DEFAULT_PHONEME_WEIGHT = 1.0      # 区間が足りないときの音素ごとの取り分の重み（子音も母音も）
#   全音素が 1.0 なら等分。ある音素を 2.0 にするとその音素だけ 2 倍の長さを取る。
#   旧キー short_note_ratio（子音側が受け取る「割合」r）は r/(1-r) で重みへ読み替える
#   （子音 1 本なら r:(1-r) = 重み r/(1-r) : 1 で同じ配分。旧既定 0.5 → 重み 1.0）。

_PROFILE_DIR = Path(__file__).resolve().parent / 'profiles'


class ConsDurProfile:
    """子音別 floor + 単一母音 floor。欠損子音キーは既定で補完。"""

    def __init__(self, vowel_floor: float,
                 cons_floor: Dict[str, float],
                 enabled: bool = True,
                 lead_cons_floor: Dict[str, float] | None = None,
                 per_speaker: Dict[str, dict] | None = None,
                 phoneme_weight: Dict[str, float] | None = None,
                 phoneme_weight_default: float | None = None):
        self.vowel_floor = float(vowel_floor)
        self.cons_floor = dict(cons_floor)
        self.lead_cons_floor = dict(lead_cons_floor or {})
        # 音素別の取り分の重み。表に無い音素は phoneme_weight_default（未指定なら定数）。
        self.phoneme_weight = dict(phoneme_weight or {})
        self.phoneme_weight_default = (None if phoneme_weight_default is None
                                       else float(phoneme_weight_default))
        self.enabled = bool(enabled)
        # 話者（またはスタイル）別の上書き。キーは "<spk_id>" か "<spk_id>:<style_id>"。
        # 統計が足りた音素だけが入るので、残りは全話者共通値へ落ちる（下の _floor 参照）。
        self.per_speaker = {str(k): dict(v) for k, v in (per_speaker or {}).items()}

    def _speaker_keys(self, spk_id, style_id):
        if spk_id is None:
            return []
        keys = []
        if style_id is not None:
            keys.append(f'{int(spk_id)}:{int(style_id)}')
        keys.append(str(int(spk_id)))
        return keys

    def _scalar(self, name: str, spk_id=None, style_id=None):
        """話者別スカラー（vowel_floor / phoneme_weight_default）を 3 段で引く。無ければ None。"""
        for k in self._speaker_keys(spk_id, style_id):
            v = (self.per_speaker.get(k) or {}).get(name)
            if v is not None:
                return float(v)
        return None

    def v_min(self, spk_id=None, style_id=None) -> float:
        """母音の最低保証 [s]。"<spk>:<style>" → "<spk>" → 全話者共通（vowel_floor）。"""
        v = self._scalar('vowel_floor', spk_id, style_id)
        return self.vowel_floor if v is None else v

    def _floor(self, table: str, cons: str, spk_id=None, style_id=None):
        """話者別 → 全話者共通 の順で floor を引く。見つからなければ None。

        優先順は 3 段: (1) "<spk>:<style>" (2) "<spk>" (3) 全話者共通。
        話者別テーブルには統計が足りた音素しか入っていないので、
        足りない音素は自動的に (3) へ落ちる。
        """
        for k in self._speaker_keys(spk_id, style_id):
            v = (self.per_speaker.get(k) or {}).get(table, {}).get(cons)
            if v is not None:
                return float(v)
        v = getattr(self, table).get(cons)
        return None if v is None else float(v)

    def c_min(self, cons: str, spk_id=None, style_id=None) -> float:
        v = self._floor('cons_floor', cons, spk_id, style_id)
        return _DEFAULT_CONS_FLOOR if v is None else v

    def lead_c_min(self, cons: str, spk_id=None, style_id=None) -> float:
        """フレーズ先頭子音の floor。未指定なら medial floor(c_min)にフォールバック。"""
        v = self._floor('lead_cons_floor', cons, spk_id, style_id)
        return self.c_min(cons, spk_id, style_id) if v is None else v

    def weight(self, ph: str, spk_id=None, style_id=None) -> float:
        """区間が足りないときの、この音素の取り分の重み。子音・母音のどちらにも引ける。既定 1.0。

        重みの比がそのまま長さの比になる（全部 1.0 なら等分、2.0 の音素だけ 2 倍）。
        floor と同じ 3 段の優先順（"<spk>:<style>" → "<spk>" → 全話者共通）で引く。
        0 に張り付くとその音素が消えるので 0.05〜20 へ収める。
        """
        v = self._floor('phoneme_weight', ph, spk_id, style_id)
        if v is None:                                   # 音素別が無ければ話者別既定 → 共通既定 → 定数
            v = self._scalar('phoneme_weight_default', spk_id, style_id)
        if v is None:
            v = self.phoneme_weight_default
        w = _DEFAULT_PHONEME_WEIGHT if v is None else v
        return min(20.0, max(0.05, float(w)))

    def to_dict(self) -> dict:
        d = {'enabled': self.enabled,
             'vowel_floor_sec': self.vowel_floor,
             'cons_floor_sec': self.cons_floor,
             'lead_cons_floor_sec': self.lead_cons_floor}
        if self.phoneme_weight:
            d['phoneme_weight'] = self.phoneme_weight
        if self.phoneme_weight_default is not None:
            d['phoneme_weight_default'] = self.phoneme_weight_default
        if self.per_speaker:
            d['per_speaker'] = self.per_speaker
        return d


def default_profile(phoneset=None) -> ConsDurProfile:
    """組込み既定プロファイル（全 31 子音を既定値で埋める）。"""
    return ConsDurProfile(
        _DEFAULT_VOWEL_FLOOR,
        {c: _DEFAULT_CONS_FLOOR for c in (phoneset or _ps_mod.active()).consonants},
    )


def _ratio_to_weight(r: float) -> float:
    """旧 short_note_ratio（子音側の割合）→ 新 cons_weight（母音 1.0 に対する重み）。"""
    r = min(0.95, max(0.05, float(r)))
    return r / (1.0 - r)


def from_dict(d: dict, phoneset=None) -> ConsDurProfile:
    """JSON 由来 dict → プロファイル（既定に上書き。未知子音キーは無視、欠損は既定補完）。"""
    p = default_profile(phoneset)
    p.enabled = bool(d.get('enabled', True))
    p.vowel_floor = float(d.get('vowel_floor_sec', _DEFAULT_VOWEL_FLOOR))
    for c, v in (d.get('cons_floor_sec') or {}).items():
        if c in p.cons_floor:
            p.cons_floor[c] = float(v)
    for c, v in (d.get('lead_cons_floor_sec') or {}).items():
        p.lead_cons_floor[c] = float(v)
    # 取り分の重み（音素別・任意）。書かれていない音素は _DEFAULT_PHONEME_WEIGHT。
    for c, v in (d.get('phoneme_weight') or {}).items():
        p.phoneme_weight[c] = float(v)
    if d.get('phoneme_weight_default') is not None:
        p.phoneme_weight_default = float(d['phoneme_weight_default'])
    # 旧キー cons_weight（子音のみの表）も同じ表へ読み込む。
    for c, v in (d.get('cons_weight') or {}).items():
        p.phoneme_weight.setdefault(c, float(v))
    if (d.get('cons_weight_default') is not None
            and p.phoneme_weight_default is None):
        p.phoneme_weight_default = float(d['cons_weight_default'])
    # 旧キー short_note_ratio（割合 r）は重み r/(1-r) へ読み替える。
    for c, v in (d.get('short_note_ratio') or {}).items():
        p.phoneme_weight.setdefault(c, _ratio_to_weight(float(v)))
    if (d.get('short_note_ratio_default') is not None
            and p.phoneme_weight_default is None):
        p.phoneme_weight_default = _ratio_to_weight(float(d['short_note_ratio_default']))
    # 話者別上書き。無ければ空 = 全話者共通のみ（従来のプロファイルはこの形）。
    # 値は 音素別の表（dict）とスカラー（vowel_floor / phoneme_weight_default）の両方。
    p.per_speaker = {}
    for k, v in (d.get('per_speaker') or {}).items():
        e = {}
        for kk, vv in (v or {}).items():
            e[kk] = ({c: float(x) for c, x in vv.items()} if isinstance(vv, dict) else float(vv))
        if 'cons_weight' in e and 'phoneme_weight' not in e:
            e['phoneme_weight'] = e.pop('cons_weight')
        if 'cons_weight_default' in e and 'phoneme_weight_default' not in e:
            e['phoneme_weight_default'] = e.pop('cons_weight_default')
        if 'short_note_ratio' in e and 'phoneme_weight' not in e:
            e['phoneme_weight'] = {c: _ratio_to_weight(x)
                                   for c, x in e.pop('short_note_ratio').items()}
        if 'short_note_ratio_default' in e and 'phoneme_weight_default' not in e:
            e['phoneme_weight_default'] = _ratio_to_weight(e.pop('short_note_ratio_default'))
        p.per_speaker[str(k)] = e
    return p


def _profile_path(model_key: str) -> Path:
    key = os.path.splitext(os.path.basename(model_key))[0]
    return _PROFILE_DIR / f'{key}.json'


def load_profile(model_key: str) -> ConsDurProfile:
    """model_key（ckpt パス/basename）に対応する duration_model/profiles/<stem>.json。
    無ければ組込み既定。壊れた JSON は黙って既定に落とさず、そのまま例外にする。"""
    if not model_key:
        return default_profile()
    p = _profile_path(model_key)
    if p.exists():
        return from_dict(json.loads(p.read_text(encoding='utf-8')))
    return default_profile()
