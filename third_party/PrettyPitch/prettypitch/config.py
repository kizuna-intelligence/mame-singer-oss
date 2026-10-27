"""Configuration objects, populated from YAML.

PrettyPitch is mel-free, so the only frame settings are the sample rate and the hop
size. The default is **hop 256 @ 44100 Hz** (frame rate 172.265625 fps) — the same
frame grid as the sibling LeapSinger acoustic model, so a pitch curve predicted here
drops straight onto its frames. Every field is configurable; the same FrameSpec is
shared by preprocessing (F0 frame rate), the dataset loader (duration → frame
expansion) and inference so they always agree.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import yaml


@dataclass
class FrameSpec:
    sr: int = 44_100
    hop: int = 256          # default 256; set 512 (etc.) to train at another frame size

    @property
    def frame_rate(self) -> float:
        return self.sr / self.hop

    @classmethod
    def from_dict(cls, d: dict | None) -> "FrameSpec":
        if not d:
            return cls()
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


# 推論時に **1 組しか持てない** decomp キー。ckpt は plain_std / vib_std /
# baseline_sigma_sec を平坦な単一値として書き出し、infer.py はそれで復元する
# （ckpt_config 参照）。DB 別に変えると、その DB の学習教師と推論の復元がずれる。
_GLOBAL_ONLY_DECOMP_KEYS = ("plain_std", "vib_std", "baseline_sigma_sec")


def merge_per_db_from_recipes(cfg: dict, data_dirs) -> None:
    """レシピ configs/recipes/{db}.yaml の `decomp:` / `pitch:` / `reject_f0_below:` を
    config の `decomp.per_db` / `pitch.per_db` / `data.reject_f0_per_db` へ畳み込む（in-place）。

    **DB 別の値はレシピが持つ**。config が持つのは全 DB 共通の既定だけ。
    同じキーが両方にあればレシピが勝つ（レシピの方が話者に近いため）。
    """
    dec_per_db = cfg["decomp"].setdefault("per_db", {}) or {}
    pit_per_db = cfg["pitch"].setdefault("per_db", {}) or {}
    rej_per_db = dict(cfg["data"].get("reject_f0_per_db") or {})
    cfg["decomp"]["per_db"], cfg["pitch"]["per_db"] = dec_per_db, pit_per_db
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for d in data_dirs:
        db = os.path.basename(os.path.normpath(d))
        rp = os.path.join(repo_root, "configs", "recipes", f"{db}.yaml")   # レシピ名 = data_dir の名前
        if not os.path.exists(rp):
            print(f"[recipes] {db}: configs/recipes/{db}.yaml が無いので DB 別の設定なし（共通設定のまま）")
            continue
        r = yaml.safe_load(open(rp, encoding="utf-8")) or {}
        if r.get("decomp"):
            bad = [k for k in _GLOBAL_ONLY_DECOMP_KEYS if k in r["decomp"]]
            if bad:
                raise ValueError(
                    f"{rp}: decomp.{bad} は DB 別にできません。ckpt は 1 組しか持たず "
                    f"推論側が全話者に同じ値を使うため、その DB だけ復元がずれます。"
                    f"変えるなら config の decomp: に書いて全 DB 共通にしてください")
            dec_per_db[db] = {**(dec_per_db.get(db) or {}), **r["decomp"]}
        if r.get("pitch"):
            pit_per_db[db] = {**(pit_per_db.get(db) or {}), **r["pitch"]}
        if r.get("reject_f0_below") is not None:
            rej_per_db[db] = float(r["reject_f0_below"])
    cfg["data"]["reject_f0_per_db"] = rej_per_db
    if dec_per_db or pit_per_db or rej_per_db:
        print(f"[recipe] decomp.per_db={ {k: sorted(v) for k, v in dec_per_db.items()} }\n"
              f"[recipe] pitch.per_db={ {k: sorted(v) for k, v in pit_per_db.items()} }\n"
              f"[recipe] reject_f0_per_db={rej_per_db}")
