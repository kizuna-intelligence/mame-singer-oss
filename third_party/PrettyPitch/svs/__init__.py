"""PrettyPitch SVS: 楽譜（UST/MusicXML）→ 音素列・持続長・F0・音響・波形の統合。

モジュール:
  score_io             : 楽譜（UST/MusicXML）→ 共通 Note → フレーズ列。
  cons_dur             : 子音長モデルの読み込みと自然長の予測。
  duration_postprocess : 持続長の後処理（floor/圧縮・多モーラノートの等分）。
  render               : F0 → 音響（LeapSinger）→ 波形（NHVSing）の統合と CLI。
  plot                 : notebook 用の描画。

多言語: 音素表は dict/<lang>.phonemes、核音素は dict/<lang>.yaml の nucleus:、歌詞→音素は
dict/<lang>.mora（load_score(mora_table=...) で差し替え）。
"""
from .duration_postprocess import (
    Note,
    Phrase,
    PhonemeSequence,
    is_nucleus_ph,
    postprocess_durations,
    split_phrase_syllables,
    split_syllable,
    to_frames,
)
from .score_io import (
    DEFAULT_MORA_TABLE,
    ScoreNote,
    load_mora_table,
    load_score,
    parse_musicxml,
    parse_ust,
    score_to_phrases,
)

__all__ = [
    'Note',
    'Phrase',
    'PhonemeSequence',
    'is_nucleus_ph',
    'postprocess_durations',
    'split_phrase_syllables',
    'split_syllable',
    'to_frames',
    'DEFAULT_MORA_TABLE',
    'ScoreNote',
    'load_mora_table',
    'load_score',
    'parse_musicxml',
    'parse_ust',
    'score_to_phrases',
]
