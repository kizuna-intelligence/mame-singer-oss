"""Phoneme <-> id vocabulary + the nucleus phoneme set, loaded from files (default: Japanese).

The phoneme set is NOT hard-coded here: it is read from a plain text file (one phoneme per
line, order = id) so PrettyPitch can train on any language by supplying a different list. The
bundled default is ``dict/ja.phonemes`` — byte-identical to the sibling LeapSinger repo (the
shared phoneme-set contract between the acoustic and F0 models). Pass ``--phonemes <path>``
to ``preprocess.run`` and ``train`` to use another set; the list is also stored in the
checkpoint, so inference reconstructs it automatically.

The **nucleus set** is PrettyPitch's one addition: the phonemes that carry a pitch value
(vowels + moraic nasal for Japanese). It is declared by the language pack
(``nucleus:`` in ``dict/<lang>.yaml``) and must be a subset of the phoneme set.
Multilingual support = add a ``<lang>.yaml`` pack and point ``lang:`` at it.

``DEFAULT_VOCAB`` is the default (Japanese) vocab; code that must honour a custom set takes
a ``Vocab``.
"""
from __future__ import annotations

import os

_DICT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dict")
DEFAULT_PHONEMES_PATH = os.path.join(_DICT_DIR, "ja.phonemes")
# 核音素は言語パック（dict/<lang>.yaml の nucleus:）が持つ。
# 外部データセット向けにファイル指定も受ける（load_nucleus(path)）。


def _pack_nuclei() -> list[str]:
    """既定の核音素を言語パックから取る（--nucleus 未指定時の入口）。"""
    from duration_model import phoneset as _ps_mod      # 遅延 import（循環回避）
    return sorted(_ps_mod.active().nuclei)

# tokens treated as silence for cut/fade (cl is voiced -> not silence). Language-independent.
SILENCE_TOKENS = frozenset({"pau"})


def _read_list(path: str) -> list[str]:
    """Read a phoneme-list file -> ordered list. One phoneme per line; blank lines and text
    after '#' are ignored."""
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.split("#", 1)[0].strip()
            if ln:
                out.append(ln)
    return out


def load_phonemes(path: str) -> list[str]:
    """Read a phoneme-list file -> ordered list. The order defines the id (first = id 0)."""
    out = _read_list(path)
    if not out:
        raise ValueError(f"no phonemes found in {path}")
    if out[0] != "pau":
        raise ValueError(f"{path}: id 0 must be 'pau' (silence/pad), got {out[0]!r}")
    if len(set(out)) != len(out):
        dup = sorted({p for p in out if out.count(p) > 1})
        raise ValueError(f"{path}: duplicate phonemes {dup}")
    return out


def load_nucleus(path: str | None = None, vocab: "Vocab | None" = None,
                 validate: bool = True) -> frozenset[str]:
    """Nucleus phoneme set. ``path`` 未指定なら言語パックの nucleus: から取る
    （dict/<lang>.yaml）。By default validated to be a subset
    of ``vocab`` (default: the bundled Japanese vocab) so a typo fails loudly at load
    time. Pass ``validate=False`` when the matching phoneme vocab is not known yet —
    e.g. the tuner reads the vocab per database from ``metadata.json`` and checks there."""
    src = path or "言語パック（dict/<lang>.yaml の nucleus:）"
    names = _read_list(path) if path else _pack_nuclei()
    if not names:
        raise ValueError(f"no nucleus phonemes found in {src}")
    if validate:
        vocab = vocab or DEFAULT_VOCAB
        unknown = sorted(set(names) - set(vocab.phonemes))
        if unknown:
            raise ValueError(f"{src}: nucleus phonemes not in the "
                             f"phoneme vocab: {unknown}")
    return frozenset(names)


class Vocab:
    """A phoneme set: name<->id maps + the silence/pad id. Build from a list or a file."""

    def __init__(self, phonemes: list[str]):
        self.phonemes: list[str] = list(phonemes)
        self.phoneme2id: dict[str, int] = {ph: i for i, ph in enumerate(self.phonemes)}
        self.id2phoneme: dict[int, str] = {i: ph for i, ph in enumerate(self.phonemes)}
        self.n_phonemes: int = len(self.phonemes)
        self.pau_id: int = self.phoneme2id.get("pau", 0)
        self.silence: frozenset[str] = frozenset(SILENCE_TOKENS & set(self.phonemes))

    @classmethod
    def load(cls, path: str | None = None) -> "Vocab":
        """Load from a phoneme-list file (default: the bundled Japanese ``dict/ja.phonemes``)."""
        return cls(load_phonemes(path or DEFAULT_PHONEMES_PATH))

    def __contains__(self, ph: str) -> bool:
        return ph in self.phoneme2id

    def get(self, ph: str, default=None):
        return self.phoneme2id.get(ph, default)

    def __len__(self) -> int:
        return self.n_phonemes


# ── default (Japanese) vocab ─────────────────────────────────────────────────────────────
DEFAULT_VOCAB = Vocab.load()
