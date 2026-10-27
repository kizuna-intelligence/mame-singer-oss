"""Score files of other singing-synthesis tools, read into one shape: notes with start and length in seconds, a
MIDI pitch and a lyric, plus the tempo at the start.

  .mid .midi   Standard MIDI File (lyric meta events; UTF-8 or Shift_JIS)
  .vsq         VOCALOID2 (MIDI with the sequence as text)
  .vsqx        VOCALOID3 / 4
  .vpr         VOCALOID5 / 6
  .svp .s5p    Synthesizer V Studio / Synthesizer V (2018)
  .ccs         CeVIO (song tracks)
  .ustx        OpenUtau

Tempo changes are kept: every position goes through the file's tempo map to seconds. When a file has several
voice tracks, the one with the most notes is read. Overlapping notes are cut so one voice remains.
"""
import json
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

EXTENSIONS = (".mid", ".midi", ".vsq", ".vsqx", ".vpr", ".svp", ".s5p", ".ccs", ".ustx")
MELISMA = {"-": "ー", "+": "ー", "+~": "ー", "+*": "ー", "~": "ー"}


class TempoMap:
    """ticks -> seconds through tempo changes [(tick, bpm)] at `ppq` ticks per quarter note"""

    def __init__(self, ppq, events):
        ev = sorted((float(t), float(b)) for t, b in events if b and b > 0) or [(0.0, 120.0)]
        if ev[0][0] > 0:
            ev.insert(0, (0.0, ev[0][1]))
        self.ppq, self.ev = float(ppq), ev
        self.acc = [0.0]
        for (t0, b0), (t1, _) in zip(ev, ev[1:]):
            self.acc.append(self.acc[-1] + (t1 - t0) / self.ppq * 60.0 / b0)

    def sec(self, tick):
        i = 0
        while i + 1 < len(self.ev) and self.ev[i + 1][0] <= tick:
            i += 1
        t, b = self.ev[i]
        return self.acc[i] + (tick - t) / self.ppq * 60.0 / b

    @property
    def bpm(self):
        return self.ev[0][1]

    @property
    def changes(self):
        return len({round(b, 3) for _, b in self.ev}) - 1


def _finish(notes, tempo):
    """[(start_s, dur_s, midi, lyric)] -> sorted, one voice, melisma marks unified"""
    notes = sorted((n for n in notes if n[1] > 0), key=lambda n: (n[0], -n[2]))
    out = []
    for s, d, m, ly in notes:
        ly = (ly or "").strip()
        ly = MELISMA.get(ly, ly)
        if out and out[-1][0] + out[-1][1] > s + 1e-6:
            if abs(out[-1][0] - s) < 1e-6:          # a chord: keep the top note
                continue
            p = out[-1]
            out[-1] = (p[0], s - p[0], p[2], p[3])
        out.append((s, d, int(m), ly))
    return {"notes": [{"start": s, "dur": d, "midi": m, "lyric": ly} for s, d, m, ly in out], "bpm": tempo.bpm,
            "tempo_changes": tempo.changes}


def _text(b):
    for enc in ("utf-8", "cp932"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("latin-1")


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _child(el, *names):
    for c in el:
        if _local(c.tag) in names:
            return c
    return None


def _children(el, *names):
    return [c for c in el if _local(c.tag) in names]


def _val(el, *names, default=None, cast=float):
    c = _child(el, *names)
    return cast(c.text) if c is not None and c.text is not None else default


def _json(path):
    raw = Path(path).read_bytes().rstrip(b"\x00")
    txt = raw.decode("utf-8-sig")
    dec = json.JSONDecoder()
    try:
        return dec.raw_decode(txt.strip())[0]
    except json.JSONDecodeError:                     # hand-edited files: trailing commas
        return dec.raw_decode(re.sub(r",(\s*[}\]])", r"\1", txt).strip())[0]


# ── MIDI ──────────────────────────────────────────────────────────────
def _smf(data):
    """-> (ppq, [track events [(tick, kind, payload)]])"""
    if data[:4] != b"MThd":
        raise ValueError("not a MIDI file")
    hl = struct.unpack(">I", data[4:8])[0]
    _, ntr, div = struct.unpack(">HHH", data[8:14])
    if div & 0x8000:
        raise ValueError("SMPTE time division is not supported")
    pos, tracks = 8 + hl, []
    while pos + 8 <= len(data) and len(tracks) < ntr:
        cid, ln = data[pos:pos + 4], struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body, pos = data[pos + 8:pos + 8 + ln], pos + 8 + ln
        if cid != b"MTrk":
            continue
        ev, i, tick, run = [], 0, 0, None

        def vlq():
            nonlocal i
            v = 0
            while True:
                b = body[i]
                i += 1
                v = (v << 7) | (b & 0x7F)
                if not b & 0x80:
                    return v
        while i < len(body):
            tick += vlq()
            st = body[i]
            if st == 0xFF:
                typ = body[i + 1]
                i += 2
                ln2 = vlq()
                ev.append((tick, ("meta", typ), body[i:i + ln2]))
                i += ln2
            elif st in (0xF0, 0xF7):
                i += 1
                i += vlq()
            else:
                if st & 0x80:
                    run, i = st, i + 1
                hi = run & 0xF0
                n = 1 if hi in (0xC0, 0xD0) else 2
                d = body[i:i + n]
                i += n
                if hi == 0x90 and d[1] > 0:
                    ev.append((tick, "on", d[0]))
                elif hi == 0x80 or (hi == 0x90 and d[1] == 0):
                    ev.append((tick, "off", d[0]))
        tracks.append(ev)
    return div, tracks


def read_midi(path):
    ppq, tracks = _smf(Path(path).read_bytes())
    tempo = TempoMap(ppq, [(t, 60e6 / int.from_bytes(p, "big")) for tr in tracks for t, k, p in tr
                           if k == ("meta", 0x51) and len(p) == 3])
    best = None
    for tr in tracks:
        notes, open_ = [], {}
        lyr = {}
        for t, k, p in tr:
            if k == "on":
                if p in open_:
                    notes.append((open_.pop(p), t, p))
                open_[p] = t
            elif k == "off" and p in open_:
                notes.append((open_.pop(p), t, p))
            elif k == ("meta", 0x05) or (k == ("meta", 0x01) and t not in lyr):
                lyr[t] = _text(bytes(p))
        if not notes:
            continue
        n_lyr = sum(1 for s, _, _ in notes if s in lyr)
        key = (n_lyr, len(notes))
        if best is None or key > best[0]:
            best = (key, notes, lyr)
    if best is None:
        return read_vsq(path)                         # VOCALOID2 files are MIDI with the notes as text
    _, notes, lyr = best
    return _finish([(tempo.sec(s), tempo.sec(e) - tempo.sec(s), m, lyr.get(s, "")) for s, e, m in notes], tempo)


def read_vsq(path):
    """VOCALOID2: the sequence is ini text split over text events of each track ("DM:nnnn:...")"""
    ppq, tracks = _smf(Path(path).read_bytes())
    tempo_ev = [(t, 60e6 / int.from_bytes(p, "big")) for tr in tracks for t, k, p in tr if k == ("meta", 0x51)]
    best = None
    for tr in tracks:
        chunks = [bytes(p) for t, k, p in tr if k == ("meta", 0x01)]
        txt = "".join(re.sub(rb"^DM:\d{4}:", b"", c).decode("cp932", errors="replace") for c in chunks if c.startswith(b"DM:"))
        if "[EventList]" not in txt:
            continue
        sec, cur = {}, None
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("[") and line.endswith("]"):
                cur = line[1:-1]
                sec[cur] = {} if cur != "EventList" else []
            elif cur == "EventList" and "=" in line:
                t, ids = line.split("=", 1)
                sec[cur].append((int(t), ids.split(",")))
            elif cur and "=" in line and cur != "EventList":
                k, v = line.split("=", 1)
                sec[cur][k] = v
        pre = int(sec.get("Master", {}).get("PreMeasure", "1"))
        notes = []
        for t, ids in sec.get("EventList", []):
            for idd in ids:
                e = sec.get(idd)
                if not e or e.get("Type") != "Anote":
                    continue
                h = sec.get(e.get("LyricHandle", "").replace("h#", "h#"), {})
                m = re.match(r'"([^"]*)"', h.get("L0", ""))
                notes.append((t, t + int(e.get("Length", 0)), int(e.get("Note#", 60)), m.group(1) if m else ""))
        if notes and (best is None or len(notes) > len(best[0])):
            best = (notes, pre)
    if best is None:
        raise ValueError("no VOCALOID2 sequence")
    notes, pre = best
    tempo = TempoMap(ppq, tempo_ev)
    t0 = tempo.sec(pre * 4 * ppq)
    return _finish([(tempo.sec(s) - t0, tempo.sec(e) - tempo.sec(s), m, ly) for s, e, m, ly in notes], tempo)


# ── VOCALOID3/4/5 ─────────────────────────────────────────────────────
def read_vsqx(path):
    root = ET.parse(str(path)).getroot()
    master = _child(root, "masterTrack")
    res = _val(master, "resolution", default=480)
    pre = _val(master, "preMeasure", default=1)
    ts = _child(master, "timeSig")
    nu = _val(ts, "nu", "nume", default=4) if ts is not None else 4
    de = _val(ts, "de", "denomi", default=4) if ts is not None else 4
    tempo_ev = []
    for t in _children(master, "tempo"):
        tempo_ev.append((_val(t, "t", "posTick", default=0), _val(t, "v", "bpm", default=12000) / 100))
    tempo = TempoMap(res, tempo_ev)
    t0 = tempo.sec(pre * nu * res * 4 / de)
    best = []
    for tr in _children(root, "vsTrack"):
        notes = []
        for part in _children(tr, "vsPart", "musicalPart"):
            pt = _val(part, "t", "posTick", default=0)
            for n in _children(part, "note"):
                s = pt + _val(n, "t", "posTick", default=0)
                d = _val(n, "dur", "durTick", default=0)
                y = _child(n, "y", "lyric")
                notes.append((tempo.sec(s) - t0, tempo.sec(s + d) - tempo.sec(s), int(_val(n, "n", "noteNum", default=60)),
                              y.text if y is not None and y.text else ""))
        if len(notes) > len(best):
            best = notes
    if not best:
        raise ValueError("no notes")
    return _finish(best, tempo)


def read_vpr(path):
    z = zipfile.ZipFile(str(path))
    name = next(n for n in z.namelist() if n.replace("\\", "/").lower() == "project/sequence.json")
    d = json.loads(z.read(name).decode("utf-8-sig"))
    ev = d.get("masterTrack", {}).get("tempo", {}).get("events", [])
    tempo = TempoMap(480, [(e.get("pos", 0), e.get("value", 12000) / 100) for e in ev])
    best = []
    for tr in d.get("tracks", []):
        notes = []
        for part in tr.get("parts", []):
            pp = part.get("pos", 0)
            for n in part.get("notes", []):
                s, du = pp + n.get("pos", 0), n.get("duration", 0)
                notes.append((tempo.sec(s), tempo.sec(s + du) - tempo.sec(s), n.get("number", 60), n.get("lyric", "")))
        if len(notes) > len(best):
            best = notes
    if not best:
        raise ValueError("no notes")
    return _finish(best, tempo)


# ── Synthesizer V ─────────────────────────────────────────────────────
BLICK = 705600000


def read_svp(path):
    d = _json(path)
    if "time" in d:                                   # Synthesizer V Studio (.svp)
        tempo = TempoMap(BLICK, [(t.get("position", 0), t.get("bpm", 120)) for t in d["time"].get("tempo", [])])
        lib = {g.get("uuid"): g for g in d.get("library", [])}
        best = []
        for tr in d.get("tracks", []):
            raw = [(n, 0, 0) for n in tr.get("mainGroup", {}).get("notes", [])]
            for ref in tr.get("groups", []):
                g = lib.get(ref.get("groupID"))
                if g:
                    raw += [(n, ref.get("blickOffset", 0), ref.get("pitchOffset", 0)) for n in g.get("notes", [])]
            notes = [(tempo.sec(n["onset"] + o), tempo.sec(n["onset"] + o + n["duration"]) - tempo.sec(n["onset"] + o),
                      n.get("pitch", 60) + po, n.get("lyrics", n.get("lyric", ""))) for n, o, po in raw]
            if len(notes) > len(best):
                best = notes
    else:                                             # Synthesizer V (2018, .s5p)
        tempo = TempoMap(BLICK, [(t.get("position", 0), t.get("beatPerMinute", 120)) for t in d.get("tempo", [])])
        best = []
        for tr in d.get("tracks", []):
            notes = [(tempo.sec(n["onset"]), tempo.sec(n["onset"] + n["duration"]) - tempo.sec(n["onset"]),
                      n.get("pitch", 60), n.get("lyric", "")) for n in tr.get("notes", [])]
            if len(notes) > len(best):
                best = notes
    if not best:
        raise ValueError("no notes")
    return _finish(best, tempo)


# ── CeVIO ─────────────────────────────────────────────────────────────
def read_ccs(path):
    root = ET.parse(str(path)).getroot()
    best, best_tempo = [], None
    for unit in root.iter():
        if _local(unit.tag) != "Unit" or unit.get("Category") != "SingerSong":
            continue
        song = _child(unit, "Song")
        if song is None:
            continue
        tp = _child(song, "Tempo")
        tempo = TempoMap(960, [(float(s.get("Clock", 0)), float(s.get("Tempo", 120)))
                               for s in (_children(tp, "Sound") if tp is not None else [])])
        sc = _child(song, "Score")
        notes = []
        for n in (_children(sc, "Note") if sc is not None else []):
            s, du = float(n.get("Clock", 0)), float(n.get("Duration", 0))
            m = (int(n.get("PitchOctave", 4)) + 1) * 12 + int(n.get("PitchStep", 0))
            notes.append((tempo.sec(s), tempo.sec(s + du) - tempo.sec(s), m, n.get("Lyric", "")))
        if len(notes) > len(best):
            best, best_tempo = notes, tempo
    if not best:
        raise ValueError("no song notes")
    return _finish(best, best_tempo)


# ── OpenUtau ──────────────────────────────────────────────────────────
def read_ustx(path):
    import yaml
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    res = d.get("resolution", 480)
    ev = [(t.get("position", 0), t.get("bpm", 120)) for t in d.get("tempos") or []] or [(0, d.get("bpm", 120))]
    tempo = TempoMap(res, ev)
    by_track = {}
    for part in d.get("voice_parts") or []:
        pp = part.get("position", 0)
        for n in part.get("notes") or []:
            s, du = pp + n.get("position", 0), n.get("duration", 0)
            by_track.setdefault(part.get("track_no", 0), []).append(
                (tempo.sec(s), tempo.sec(s + du) - tempo.sec(s), n.get("tone", 60), str(n.get("lyric", ""))))
    if not by_track:
        raise ValueError("no notes")
    return _finish(max(by_track.values(), key=len), tempo)


READERS = {".mid": read_midi, ".midi": read_midi, ".vsq": read_vsq, ".vsqx": read_vsqx, ".vpr": read_vpr,
           ".svp": read_svp, ".s5p": read_svp, ".ccs": read_ccs, ".ustx": read_ustx}


def read(path):
    """-> {"notes": [{start, dur (seconds), midi, lyric}], "bpm": first tempo, "tempo_changes": n}"""
    ext = Path(path).suffix.lower()
    if ext not in READERS:
        raise ValueError(f"unsupported score file: {ext}")
    return READERS[ext](path)
