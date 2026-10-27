// Piano-roll score editor: notes with pitch, timing and a lyric each, edited with mouse, touch and keyboard.
// Times are in ticks (TPB per beat). One voice: when an edit makes notes overlap, the others are trimmed.
(function () {
  const TPB = 480;
  const KEYS_W = 56;
  const RULER_H = 28;
  const NOTE_NAMES = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
  const BLACK = new Set([1, 3, 6, 8, 10]);
  const EDGE = 7;           // px: grab the right edge to resize

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  class PianoRoll {
    constructor(root, { onChange, onPreview, onSeek, normalize } = {}) {
      this.root = root;
      this.onChange = onChange || (() => {});
      this.onPreview = onPreview || (() => {});
      this.onSeek = onSeek || (() => {});
      this.normalize = normalize || ((v) => v);
      this.notes = [];
      this.nextId = 1;
      this.sel = new Set();
      this.bad = new Map();             // id -> message
      this.snap = TPB / 2;
      this.ppb = 72;
      this.rowH = 18;
      this.low = 36;                    // C2
      this.high = 96;                   // C7
      this.beatsPerBar = 4;
      this.cursor = 0;                  // ticks: where playback starts
      this.playhead = null;             // ticks while playing
      this.defaultLen = TPB;
      this.undoStack = [];
      this.redoStack = [];
      this.drag = null;
      this.hover = null;

      root.innerHTML = `<div class="roll-scroll" tabindex="0" aria-label="ピアノロール"><div class="roll-size"></div><canvas></canvas></div>
        <input class="lyric-edit" type="text" autocomplete="off" spellcheck="false" hidden>`;
      this.scroll = root.querySelector(".roll-scroll");
      this.sizer = root.querySelector(".roll-size");
      this.canvas = root.querySelector("canvas");
      this.ctx = this.canvas.getContext("2d");
      this.input = root.querySelector(".lyric-edit");
      this.editing = null;

      new ResizeObserver(() => this.resize()).observe(this.scroll);
      this.scroll.addEventListener("scroll", () => { this.hideEditor(true); this.draw(); });
      this.canvas.addEventListener("pointerdown", (e) => this.down(e));
      this.canvas.addEventListener("pointermove", (e) => this.move(e));
      this.canvas.addEventListener("pointerup", (e) => this.up(e));
      this.canvas.addEventListener("pointercancel", (e) => this.up(e));
      this.canvas.addEventListener("pointerleave", () => { this.hover = null; this.draw(); });
      this.canvas.addEventListener("dblclick", (e) => this.dbl(e));
      this.canvas.addEventListener("contextmenu", (e) => this.context(e));
      this.scroll.addEventListener("keydown", (e) => this.key(e));
      this.scroll.addEventListener("wheel", (e) => {
        if (e.ctrlKey || e.metaKey) { e.preventDefault(); this.zoom(e.deltaY < 0 ? 1.15 : 1 / 1.15, e); }
      }, { passive: false });
      this.input.addEventListener("keydown", (e) => this.inputKey(e));
      this.input.addEventListener("blur", () => this.hideEditor(true));
      matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => this.draw());
      this.resize();
    }

    // ── model ──────────────────────────────────────────────────────────
    snapshot() { return JSON.stringify({ n: this.notes, i: this.nextId }); }
    restore(s) { const o = JSON.parse(s); this.notes = o.n; this.nextId = o.i; this.sel.clear(); this.changed(); }
    checkpoint() { this.undoStack.push(this.snapshot()); if (this.undoStack.length > 200) this.undoStack.shift(); this.redoStack = []; }
    undo() { if (!this.undoStack.length) return; this.redoStack.push(this.snapshot()); this.restore(this.undoStack.pop()); }
    redo() { if (!this.redoStack.length) return; this.undoStack.push(this.snapshot()); this.restore(this.redoStack.pop()); }
    changed() { this.bad.clear(); this.updateSize(); this.draw(); this.onChange(); }

    setNotes(notes, { keepHistory = false } = {}) {
      if (keepHistory) this.checkpoint(); else { this.undoStack = []; this.redoStack = []; }
      this.notes = notes.map((n) => ({ id: this.nextId++, start: Math.round(n.start), dur: Math.max(1, Math.round(n.dur)), midi: n.midi, lyric: n.lyric || "" }));
      this.sel.clear();
      this.cursor = 0;
      this.changed();
      this.focusMelody();
    }

    sorted() { return [...this.notes].sort((a, b) => a.start - b.start || a.midi - b.midi); }
    byId(id) { return this.notes.find((n) => n.id === id); }
    endTick() { return this.notes.reduce((m, n) => Math.max(m, n.start + n.dur), 0); }

    // keep one voice: the notes in `ids` win, the others are trimmed or removed where they overlap
    resolve(ids) {
      const win = this.notes.filter((n) => ids.has(n.id));
      for (const w of win) {
        const a = w.start, b = w.start + w.dur;
        for (const n of this.notes) {
          if (ids.has(n.id)) continue;
          const s = n.start, e = n.start + n.dur;
          if (e <= a || s >= b) continue;
          if (s < a) n.dur = a - s;                          // starts before: shorten
          else if (e > b) { n.dur = e - b; n.start = b; }   // ends after: move its start
          else n.dur = 0;                                    // covered: remove
        }
      }
      // overlaps among the winners themselves: earlier one is shortened
      const ws = win.sort((x, y) => x.start - y.start);
      for (let i = 0; i + 1 < ws.length; i++) if (ws[i].start + ws[i].dur > ws[i + 1].start) ws[i].dur = Math.max(0, ws[i + 1].start - ws[i].start);
      this.notes = this.notes.filter((n) => n.dur > 0);
      for (const id of [...this.sel]) if (!this.byId(id)) this.sel.delete(id);
    }

    // ── geometry ───────────────────────────────────────────────────────
    totalBeats() { return Math.max(64, Math.ceil(this.endTick() / TPB) + 32); }
    updateSize() {
      this.sizer.style.width = `${KEYS_W + this.totalBeats() * this.ppb}px`;
      this.sizer.style.height = `${RULER_H + (this.high - this.low + 1) * this.rowH}px`;
    }
    resize() {
      const w = this.scroll.clientWidth, h = this.scroll.clientHeight, dpr = devicePixelRatio || 1;
      this.vw = w; this.vh = h;
      this.canvas.style.width = `${w}px`; this.canvas.style.height = `${h}px`;
      this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
      this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      this.updateSize();
      this.draw();
    }
    x(tick) { return KEYS_W + (tick / TPB) * this.ppb - this.scroll.scrollLeft; }
    y(midi) { return RULER_H + (this.high - midi) * this.rowH - this.scroll.scrollTop; }
    at(e) {
      const r = this.canvas.getBoundingClientRect();
      const vx = e.clientX - r.left, vy = e.clientY - r.top;
      const tick = ((vx - KEYS_W + this.scroll.scrollLeft) / this.ppb) * TPB;
      const midi = this.high - Math.floor((vy - RULER_H + this.scroll.scrollTop) / this.rowH);
      return { vx, vy, tick, midi, area: vx < KEYS_W ? (vy < RULER_H ? "corner" : "keys") : vy < RULER_H ? "ruler" : "grid" };
    }
    snapT(t) { return Math.round(t / this.snap) * this.snap; }
    floorT(t) { return Math.floor(t / this.snap) * this.snap; }
    hit(p) {
      for (const n of this.sorted().reverse()) {
        if (n.midi !== p.midi || p.tick < n.start || p.tick > n.start + n.dur) continue;
        const right = this.x(n.start + n.dur);
        return { note: n, edge: right - p.vx <= EDGE && (n.dur / TPB) * this.ppb > EDGE * 2 };
      }
      return null;
    }

    // ── pointer ────────────────────────────────────────────────────────
    down(e) {
      this.scroll.focus({ preventScroll: true });
      if (e.button === 2) return;
      const p = this.at(e);
      if (p.area === "keys") { this.onPreview(p.midi); return; }
      if (p.area === "ruler") { this.setCursor(this.snapT(Math.max(0, p.tick))); this.onSeek(this.cursor); this.drag = { kind: "seek" }; this.canvas.setPointerCapture(e.pointerId); return; }
      if (p.area !== "grid") return;
      this.canvas.setPointerCapture(e.pointerId);
      const h = this.hit(p);
      if (h) {
        const n = h.note;
        if (e.shiftKey || e.metaKey || e.ctrlKey) { this.sel.has(n.id) ? this.sel.delete(n.id) : this.sel.add(n.id); this.draw(); this.onChange(); return; }
        if (!this.sel.has(n.id)) { this.sel.clear(); this.sel.add(n.id); }
        this.checkpoint();
        const orig = new Map([...this.sel].map((id) => { const m = this.byId(id); return [id, { start: m.start, dur: m.dur, midi: m.midi }]; }));
        this.drag = h.edge ? { kind: "resize", p0: p, orig, note: n } : { kind: "move", p0: p, orig, note: n, moved: false };
        this.onPreview(n.midi);
      } else if (e.shiftKey) {
        this.drag = { kind: "box", p0: p, p1: p, base: new Set(this.sel) };
      } else if (e.pointerType === "touch") {
        this.drag = { kind: "pan", p0: p, cx: e.clientX, cy: e.clientY, sl: this.scroll.scrollLeft, st: this.scroll.scrollTop, panned: false };
      } else {
        this.checkpoint();
        const start = Math.max(0, this.floorT(p.tick));
        const n = { id: this.nextId++, start, dur: this.defaultLen, midi: p.midi, lyric: this.nextLyric(start) };
        this.notes.push(n);
        this.sel.clear(); this.sel.add(n.id);
        this.drag = { kind: "create", p0: p, note: n };
        this.onPreview(n.midi);
        this.updateSize(); this.draw();
      }
      this.draw();
    }

    move(e) {
      const p = this.at(e);
      if (!this.drag) {
        const h = p.area === "grid" ? this.hit(p) : null;
        this.canvas.style.cursor = p.area === "ruler" ? "pointer" : p.area === "keys" ? "pointer" : h ? (h.edge ? "ew-resize" : "grab") : "crosshair";
        const hv = h ? h.note.id : null;
        if (hv !== this.hover) { this.hover = hv; this.draw(); }
        return;
      }
      const d = this.drag;
      if (d.kind === "seek") { this.setCursor(this.snapT(Math.max(0, p.tick))); this.onSeek(this.cursor); return; }
      if (d.kind === "pan") {
        const dx = e.clientX - d.cx, dy = e.clientY - d.cy;
        if (Math.abs(dx) + Math.abs(dy) > 6) d.panned = true;
        if (d.panned) { this.scroll.scrollLeft = d.sl - dx; this.scroll.scrollTop = d.st - dy; }
        return;
      }
      if (d.kind === "create") {
        const end = Math.max(d.note.start + this.snap, this.snapT(p.tick));
        d.note.dur = end - d.note.start;
      } else if (d.kind === "resize") {
        const delta = this.snapT(p.tick - d.p0.tick);
        for (const [id, o] of d.orig) { const n = this.byId(id); n.dur = Math.max(this.snap, o.dur + delta); }
      } else if (d.kind === "move") {
        let dt = this.snapT(p.tick - d.p0.tick), dm = p.midi - d.p0.midi;
        const minStart = Math.min(...[...d.orig.values()].map((o) => o.start));
        dt = Math.max(dt, -minStart);
        for (const [id, o] of d.orig) {
          const n = this.byId(id);
          n.start = o.start + dt;
          n.midi = Math.min(this.high, Math.max(this.low, o.midi + dm));
        }
        if (dt || dm) d.moved = true;
        if (dm !== d.lastDm) { d.lastDm = dm; if (dm) this.onPreview(d.note.midi); }
        this.canvas.style.cursor = "grabbing";
      } else if (d.kind === "box") {
        d.p1 = p;
        const t0 = Math.min(d.p0.tick, p.tick), t1 = Math.max(d.p0.tick, p.tick);
        const m0 = Math.min(d.p0.midi, p.midi), m1 = Math.max(d.p0.midi, p.midi);
        this.sel = new Set(d.base);
        for (const n of this.notes) if (n.start < t1 && n.start + n.dur > t0 && n.midi >= m0 && n.midi <= m1) this.sel.add(n.id);
      }
      this.updateSize(); this.draw();
    }

    up(e) {
      const d = this.drag;
      this.drag = null;
      if (!d) return;
      if (d.kind === "pan") {
        if (!d.panned) {                                     // a tap on empty grid: place a note there
          this.checkpoint();
          const n = { id: this.nextId++, start: Math.max(0, this.floorT(d.p0.tick)), dur: this.defaultLen, midi: d.p0.midi, lyric: "ら" };
          this.notes.push(n); this.sel = new Set([n.id]); this.onPreview(n.midi);
          this.resolve(new Set([n.id])); this.changed();
        }
        return;
      }
      if (d.kind === "create") { this.defaultLen = d.note.dur; this.resolve(new Set([d.note.id])); this.changed(); }
      else if (d.kind === "resize") { this.defaultLen = d.note.dur; this.resolve(new Set(d.orig.keys())); this.changed(); }
      else if (d.kind === "move") {
        if (d.moved) { this.resolve(new Set(d.orig.keys())); this.changed(); return; }
        this.undoStack.pop(); this.draw(); this.onChange();
        if (e.pointerType === "touch") {                     // double tap on a note: edit its lyric
          const now = performance.now();
          if (this.lastTap && this.lastTap.id === d.note.id && now - this.lastTap.t < 400) { this.lastTap = null; this.editLyric(d.note); }
          else this.lastTap = { id: d.note.id, t: now };
        }
      }
      else { this.draw(); this.onChange(); }
    }

    dbl(e) {
      const p = this.at(e);
      if (p.area !== "grid") return;
      const h = this.hit(p);
      if (h) this.editLyric(h.note);
    }

    context(e) {
      e.preventDefault();
      const p = this.at(e);
      const h = p.area === "grid" ? this.hit(p) : null;
      if (!h) return;
      this.checkpoint();
      const ids = this.sel.has(h.note.id) ? new Set(this.sel) : new Set([h.note.id]);
      this.notes = this.notes.filter((n) => !ids.has(n.id));
      this.sel.clear();
      this.changed();
    }

    nextLyric(start) {
      // a new note between others gets "ら" — a placeholder that is easy to sing and easy to spot
      return "ら";
    }

    // ── keyboard ───────────────────────────────────────────────────────
    key(e) {
      if (this.editing) return;
      const mod = e.ctrlKey || e.metaKey;
      const k = e.key;
      if (mod && k.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? this.redo() : this.undo(); return; }
      if (mod && k.toLowerCase() === "y") { e.preventDefault(); this.redo(); return; }
      if (mod && k.toLowerCase() === "a") { e.preventDefault(); this.sel = new Set(this.notes.map((n) => n.id)); this.draw(); this.onChange(); return; }
      if (mod && k.toLowerCase() === "c") { this.copy(); return; }
      if (mod && k.toLowerCase() === "v") { this.paste(); return; }
      if (mod && k.toLowerCase() === "d") { e.preventDefault(); this.copy(); this.paste(this.selEnd()); return; }
      if (k === "Delete" || k === "Backspace") { if (this.sel.size) { e.preventDefault(); this.deleteSelected(); } return; }
      if (k === "Enter" && this.sel.size === 1) { e.preventDefault(); this.editLyric(this.byId([...this.sel][0])); return; }
      if (k === "Escape") { this.sel.clear(); this.draw(); this.onChange(); return; }
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(k)) {
        e.preventDefault();
        if (!this.sel.size) return;
        if (e.altKey && (k === "ArrowLeft" || k === "ArrowRight")) { this.selectNeighbour(k === "ArrowRight" ? 1 : -1); return; }
        this.checkpoint();
        for (const id of this.sel) {
          const n = this.byId(id);
          if (k === "ArrowUp") n.midi = Math.min(this.high, n.midi + (e.shiftKey ? 12 : 1));
          if (k === "ArrowDown") n.midi = Math.max(this.low, n.midi - (e.shiftKey ? 12 : 1));
          if (k === "ArrowLeft") { if (e.shiftKey) n.dur = Math.max(this.snap, n.dur - this.snap); else n.start = Math.max(0, n.start - this.snap); }
          if (k === "ArrowRight") { if (e.shiftKey) n.dur += this.snap; else n.start += this.snap; }
        }
        this.resolve(new Set(this.sel));
        if (k === "ArrowUp" || k === "ArrowDown") this.onPreview(this.byId([...this.sel][0]).midi);
        this.changed();
        this.reveal(this.byId([...this.sel][0]));
      }
    }

    selectNeighbour(dir) {
      const s = this.sorted();
      const cur = this.sel.size ? s.findIndex((n) => this.sel.has(n.id)) : -1;
      const j = Math.max(0, Math.min(s.length - 1, cur + dir));
      if (!s[j]) return;
      this.sel = new Set([s[j].id]);
      this.reveal(s[j]); this.draw(); this.onChange();
    }

    selEnd() { return Math.max(...this.notes.filter((n) => this.sel.has(n.id)).map((n) => n.start + n.dur)); }
    copy() {
      const s = this.notes.filter((n) => this.sel.has(n.id));
      if (!s.length) return;
      const t0 = Math.min(...s.map((n) => n.start));
      this.clip = s.map((n) => ({ start: n.start - t0, dur: n.dur, midi: n.midi, lyric: n.lyric }));
    }
    paste(at) {
      if (!this.clip || !this.clip.length) return;
      const t = at ?? this.cursor;
      this.checkpoint();
      const added = this.clip.map((c) => ({ ...c, id: this.nextId++, start: c.start + t }));
      this.notes.push(...added);
      this.sel = new Set(added.map((n) => n.id));
      this.resolve(new Set(this.sel));
      this.changed();
      this.reveal(added[0]);
    }
    deleteSelected() {
      if (!this.sel.size) return;
      this.checkpoint();
      this.notes = this.notes.filter((n) => !this.sel.has(n.id));
      this.sel.clear();
      this.changed();
    }
    transposeSelected(d) {
      const ids = this.sel.size ? this.sel : new Set(this.notes.map((n) => n.id));
      this.checkpoint();
      for (const n of this.notes) if (ids.has(n.id)) n.midi = Math.min(this.high, Math.max(this.low, n.midi + d));
      this.changed();
    }

    // ── lyric editing ──────────────────────────────────────────────────
    editLyric(n) {
      if (!n) return;
      this.reveal(n);
      this.editing = n.id;
      this.sel = new Set([n.id]);
      this.draw();
      const x = this.x(n.start), y = this.y(n.midi);
      const w = Math.max(64, (n.dur / TPB) * this.ppb);
      Object.assign(this.input.style, { left: `${Math.max(KEYS_W, x)}px`, top: `${y - 4}px`, width: `${w}px`, height: `${this.rowH + 8}px` });
      this.input.value = n.lyric;
      this.input.hidden = false;
      this.input.focus();
      this.input.select();
    }
    inputKey(e) {
      if (e.isComposing || e.keyCode === 229) return;       // IME: Enter confirms the conversion, not the lyric
      if (e.key === "Escape") { e.preventDefault(); this.hideEditor(false); this.scroll.focus(); return; }
      if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        const id = this.editing;
        this.hideEditor(true);
        const s = this.sorted();
        const i = s.findIndex((n) => n.id === id);
        const next = s[i + (e.shiftKey ? -1 : 1)];
        if (next) this.editLyric(next); else this.scroll.focus();
      }
    }
    hideEditor(commit) {
      if (this.editing === null) return;
      const n = this.byId(this.editing);
      const v = this.normalize(this.input.value.trim());
      this.editing = null;
      this.input.hidden = true;
      if (commit && n && v !== n.lyric) { this.checkpoint(); n.lyric = v; this.changed(); } else this.draw();
    }

    // ── view ───────────────────────────────────────────────────────────
    reveal(n) {
      if (!n) return;
      const s = this.scroll;
      const x0 = KEYS_W + (n.start / TPB) * this.ppb, x1 = x0 + (n.dur / TPB) * this.ppb;
      if (x0 - KEYS_W < s.scrollLeft + 8) s.scrollLeft = Math.max(0, x0 - KEYS_W - 40);
      else if (x1 > s.scrollLeft + this.vw - 8) s.scrollLeft = x1 - this.vw + 40;
      const y0 = RULER_H + (this.high - n.midi) * this.rowH;
      if (y0 - RULER_H < s.scrollTop || y0 + this.rowH > s.scrollTop + this.vh) s.scrollTop = y0 - RULER_H - this.vh / 2;
    }
    focusMelody() {
      const ms = this.notes.map((n) => n.midi);
      const mid = ms.length ? (Math.min(...ms) + Math.max(...ms)) / 2 : 67;
      requestAnimationFrame(() => {
        this.scroll.scrollTop = RULER_H + (this.high - mid) * this.rowH - this.vh / 2;
        const first = this.notes.length ? Math.min(...this.notes.map((n) => n.start)) : 0;
        this.scroll.scrollLeft = Math.max(0, (first / TPB) * this.ppb - 24);
        this.draw();
      });
    }
    zoom(f, e) {
      const s = this.scroll;
      const anchorV = e ? e.clientX - this.canvas.getBoundingClientRect().left - KEYS_W : (this.vw - KEYS_W) / 2;
      const beat = (anchorV + s.scrollLeft) / this.ppb;
      this.ppb = Math.min(320, Math.max(16, this.ppb * f));
      this.updateSize();
      s.scrollLeft = beat * this.ppb - anchorV;
      this.draw();
    }
    setCursor(t) { this.cursor = t; this.draw(); }
    setPlayhead(t) {
      this.playhead = t;
      if (t !== null) {
        const x = KEYS_W + (t / TPB) * this.ppb - this.scroll.scrollLeft;
        if (x > this.vw - 40 || x < KEYS_W) this.scroll.scrollLeft += x - KEYS_W - 60;
      }
      this.draw();
    }
    setBad(map) { this.bad = map; this.draw(); }
    // accompaniment waveform under the ruler: peaks every `hop` seconds; secOf(tick) = its time in the accompaniment
    setTrack(track) { this.track = track; this.draw(); }

    draw() {
      if (!this.vw) return;
      if (this._raf) return;
      this._raf = requestAnimationFrame(() => { this._raf = null; this.paint(); });
    }

    paint() {
      const c = this.ctx, W = this.vw, H = this.vh, sx = this.scroll.scrollLeft, sy = this.scroll.scrollTop;
      const col = {
        bg: css("--roll-bg"), black: css("--roll-black"), line: css("--roll-line"), beat: css("--roll-beat"),
        bar: css("--roll-bar"), note: css("--accent"), noteSel: css("--note-sel"), noteText: css("--note-text"),
        bad: css("--bad"), fg: css("--fg"), muted: css("--muted"), panel: css("--panel"), keyW: css("--key-white"),
        keyB: css("--key-black"), play: css("--play"), cursor: css("--cursor"),
      };
      c.clearRect(0, 0, W, H);
      c.fillStyle = col.bg; c.fillRect(0, 0, W, H);
      const r0 = Math.max(0, Math.floor(sy / this.rowH)), r1 = Math.min(this.high - this.low, Math.ceil((sy + H) / this.rowH));
      for (let r = r0; r <= r1; r++) {
        const m = this.high - r, y = RULER_H + r * this.rowH - sy;
        if (BLACK.has(m % 12)) { c.fillStyle = col.black; c.fillRect(KEYS_W, y, W, this.rowH); }
        if (m % 12 === 0) { c.fillStyle = col.bar; c.fillRect(KEYS_W, y + this.rowH - 1, W, 1); }
        else if (m % 12 === 5) { c.fillStyle = col.line; c.fillRect(KEYS_W, y + this.rowH - 1, W, 1); }
      }
      // vertical grid
      const t0 = Math.floor(((sx) / this.ppb) * TPB / this.snap) * this.snap, t1 = ((sx + W) / this.ppb) * TPB;
      for (let t = t0; t <= t1; t += this.snap) {
        const x = Math.round(this.x(t)) + 0.5;
        if (x < KEYS_W) continue;
        const bar = t % (TPB * this.beatsPerBar) === 0, beat = t % TPB === 0;
        c.fillStyle = bar ? col.bar : beat ? col.beat : col.line;
        c.fillRect(Math.round(x), RULER_H, 1, H);
      }
      // notes
      c.font = `${Math.max(11, this.rowH - 6)}px system-ui, "Hiragino Sans", "Noto Sans JP", sans-serif`;
      c.textBaseline = "middle";
      for (const n of this.notes) {
        const x = this.x(n.start), y = this.y(n.midi), w = (n.dur / TPB) * this.ppb;
        if (x + w < KEYS_W || x > W || y + this.rowH < RULER_H || y > H) continue;
        const sel = this.sel.has(n.id), bad = this.bad.has(n.id);
        c.fillStyle = bad ? col.bad : sel ? col.noteSel : col.note;
        roundRect(c, x + 0.5, y + 1, Math.max(2, w - 1), this.rowH - 2, 4); c.fill();
        if (sel) { c.strokeStyle = col.fg; c.lineWidth = 1.5; roundRect(c, x + 0.5, y + 1, Math.max(2, w - 1), this.rowH - 2, 4); c.stroke(); }
        if (this.hover === n.id && !this.drag) { c.fillStyle = "rgba(255,255,255,.35)"; c.fillRect(x + w - 5, y + 4, 2, this.rowH - 8); }
        if (w > 14) {
          c.save(); c.beginPath(); c.rect(x + 3, y, w - 6, this.rowH); c.clip();
          c.fillStyle = col.noteText;
          c.fillText(n.lyric || "・", x + 5, y + this.rowH / 2 + 0.5);
          c.restore();
        }
      }
      // box selection
      if (this.drag && this.drag.kind === "box") {
        const a = this.drag.p0, b = this.drag.p1;
        c.fillStyle = "rgba(120,150,255,.15)"; c.strokeStyle = col.note; c.lineWidth = 1;
        const x = Math.min(a.vx, b.vx), y = Math.min(a.vy, b.vy);
        c.fillRect(x, y, Math.abs(a.vx - b.vx), Math.abs(a.vy - b.vy)); c.strokeRect(x + .5, y + .5, Math.abs(a.vx - b.vx), Math.abs(a.vy - b.vy));
      }
      // cursor and playhead
      const cx = this.x(this.cursor);
      if (cx >= KEYS_W) { c.fillStyle = col.cursor; c.fillRect(Math.round(cx), RULER_H, 1.5, H); }
      if (this.playhead !== null) { const px = this.x(this.playhead); if (px >= KEYS_W) { c.fillStyle = col.play; c.fillRect(Math.round(px), 0, 2, H); } }
      // ruler
      c.fillStyle = col.panel; c.fillRect(KEYS_W, 0, W, RULER_H);
      if (this.track) {
        const { peaks, hop, secOf } = this.track;
        c.fillStyle = col.beat;
        for (let x = KEYS_W; x < W; x += 2) {
          const tick = ((x - KEYS_W + sx) / this.ppb) * TPB;
          const i = Math.floor(secOf(tick) / hop);
          if (i < 0 || i >= peaks.length) continue;
          const h = Math.max(1, peaks[i] * (RULER_H - 4));
          c.fillRect(x, RULER_H - 2 - h, 1.5, h);
        }
      }
      c.fillStyle = col.bar; c.fillRect(KEYS_W, RULER_H - 1, W, 1);
      c.font = `12px system-ui, sans-serif`; c.textBaseline = "middle";
      const bt = TPB * this.beatsPerBar;
      for (let t = Math.floor(t0 / bt) * bt; t <= t1; t += TPB) {
        const x = Math.round(this.x(t));
        if (x < KEYS_W - 30) continue;
        const isBar = t % bt === 0;
        c.fillStyle = isBar ? col.fg : col.muted;
        c.fillRect(x, isBar ? 8 : 18, 1, isBar ? RULER_H - 8 : RULER_H - 18);
        if (isBar && x >= KEYS_W) c.fillText(String(t / bt + 1), x + 4, 12);
      }
      if (cx >= KEYS_W) { c.fillStyle = col.cursor; c.beginPath(); c.moveTo(cx - 6, RULER_H - 9); c.lineTo(cx + 6, RULER_H - 9); c.lineTo(cx, RULER_H - 1); c.fill(); }
      // keys
      for (let r = r0; r <= r1; r++) {
        const m = this.high - r, y = RULER_H + r * this.rowH - sy, black = BLACK.has(m % 12);
        c.fillStyle = black ? col.keyB : col.keyW; c.fillRect(0, y, KEYS_W, this.rowH);
        c.fillStyle = col.line; c.fillRect(0, y + this.rowH - 1, KEYS_W, 1);
        if (m % 12 === 0 || this.rowH >= 18 && !black) {
          c.fillStyle = m % 12 === 0 ? col.fg : col.muted;
          c.font = `${m % 12 === 0 ? 600 : 400} 11px system-ui, sans-serif`;
          c.fillText(`${NOTE_NAMES[m % 12]}${Math.floor(m / 12) - 1}`, 8, y + this.rowH / 2);
        }
      }
      c.fillStyle = col.bar; c.fillRect(KEYS_W - 1, 0, 1, H);
      c.fillStyle = col.panel; c.fillRect(0, 0, KEYS_W, RULER_H);
    }
  }

  function roundRect(c, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    c.beginPath();
    c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath();
  }

  PianoRoll.TPB = TPB;
  PianoRoll.NOTE_NAMES = NOTE_NAMES;
  window.PianoRoll = PianoRoll;
})();
