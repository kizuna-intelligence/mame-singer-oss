// mame-singer in the browser: the score editor (editor.js), the voice recordings, and the singer — Pyodide runs the
// score front end (mame_web.py), onnxruntime-web runs the networks. Nothing leaves the browser.
const TPB = PianoRoll.TPB;
const MODELS = ["acoustic", "voice_encoder", "f0_cond", "f0_denoise", "vocoder"];
const STORE = "mame-singer/score/v1";
const SCORE_RE = /\.(musicxml|xml|mxl|ust|ustx|mid|midi|vsq|vsqx|vpr|svp|s5p|ccs|json)$/i;
const $ = (id) => document.getElementById(id);

const app = { py: null, ready: false, busy: false, voices: [], mode: 1, style: 0, result: null, pendingOpen: null, acc: null, vocal: null };
// the singer runs in a worker (worker.js); call(cmd, args) -> Promise of its result
const worker = new Worker("worker.js");
const pending = new Map();
let nextCall = 1;
worker.onmessage = (e) => {
  const m = e.data;
  if (m.progress) { onProgress(...m.progress); return; }
  if (m.loading) { onLoading(m.loading); return; }
  const p = pending.get(m.id);
  if (!p) return;
  pending.delete(m.id);
  m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
};
function call(cmd, args = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = nextCall++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, cmd, args }, transfer);
  });
}

// ── status ─────────────────────────────────────────────────────────────
function status(text, kind = "", frac = null) {
  $("statusText").textContent = text;
  $("status").className = `status ${kind}`;
  if (frac !== null) $("loadBar").style.width = `${Math.round(frac * 100)}%`;
}
function message(text, bad = false, html = false) {
  const m = $("msg");
  m.className = `msg${bad ? " bad" : ""}`;
  if (html) m.innerHTML = text; else m.textContent = text;
}

// ── lyrics ─────────────────────────────────────────────────────────────
const SMALL = new Set("ぁぃぅぇぉゃゅょゎゕゖァィゥェォャュョヮヵヶ");
function hira(s) { return s.replace(/[ァ-ヶ]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0x60)); }
function morae(text) {
  const out = [];
  for (const ch of hira(text).replace(/[\s、。，．,.!?！？「」『』（）()・…〜~]/g, "")) {
    if (SMALL.has(ch) && out.length) out[out.length - 1] += ch; else out.push(ch);
  }
  return out;
}

// ── editor ─────────────────────────────────────────────────────────────
const roll = new PianoRoll($("roll"), {
  onChange: () => { save(); updateCounts(); scheduleCheck(); refresh(); },
  onPreview: (m) => blip(m),
  onSeek: () => { if (player.playing) startMelody(); },
  normalize: (v) => hira(v),
});

function scoreNotes() {
  return roll.sorted().map((n) => ({ start: n.start / TPB, dur: n.dur / TPB, midi: n.midi, lyric: n.lyric }));
}
function save() {
  try { localStorage.setItem(STORE, JSON.stringify({ bpm: bpm(), meter: roll.beatsPerBar, accOffset: accOffset(), notes: scoreNotes() })); } catch (e) { /* storage may be off */ }
}
function load(obj, keepHistory = false) {
  $("bpm").value = Math.round((obj.bpm || 120) * 100) / 100;
  roll.beatsPerBar = obj.meter || 4; $("meter").value = String(roll.beatsPerBar);
  if (obj.accOffset !== undefined) $("accOffset").value = obj.accOffset;
  roll.setNotes((obj.notes || []).map((n) => ({ start: n.start * TPB, dur: n.dur * TPB, midi: n.midi, lyric: n.lyric || "" })), { keepHistory });
}
const bpm = () => Math.min(300, Math.max(30, Number($("bpm").value) || 120));

function updateCounts() {
  const m = morae($("lyricsText").value).length;
  const s = roll.sorted();
  const from = roll.sel.size ? s.findIndex((n) => roll.sel.has(n.id)) : 0;
  $("moraCount").textContent = m ? `${m} 音 → ノート ${Math.max(0, s.length - from)} 個` : `ノート ${s.length} 個`;
}
$("lyricsText").addEventListener("input", updateCounts);
$("flowBtn").addEventListener("click", () => {
  const ms = morae($("lyricsText").value);
  const s = roll.sorted();
  if (!ms.length || !s.length) return;
  const from = roll.sel.size ? s.findIndex((n) => roll.sel.has(n.id)) : 0;
  roll.checkpoint();
  ms.forEach((m, i) => { if (s[from + i]) s[from + i].lyric = m; });
  roll.changed();
  const left = ms.length - (s.length - from);
  message(left > 0 ? `ノートが ${left} 個足りません。残りの歌詞は割り当てていません。` : `${Math.min(ms.length, s.length - from)} 個のノートに歌詞を入れました。`, left > 0);
});
$("pullBtn").addEventListener("click", () => { $("lyricsText").value = roll.sorted().map((n) => n.lyric).join(""); updateCounts(); });

// toolbar
$("newBtn").addEventListener("click", () => { if (!roll.notes.length || confirm("今の楽譜を消して新しく始めますか？（元に戻す で戻せます）")) { load({ bpm: bpm(), meter: roll.beatsPerBar, notes: [] }, true); } });
$("exampleBtn").addEventListener("click", async () => {
  if (roll.notes.length && !confirm("例の楽譜に置き換えますか？（元に戻す で戻せます）")) return;
  load(await (await fetch("examples/furusato.json")).json(), true);
  $("lyricsText").value = "";
  updateCounts();
});
$("openBtn").addEventListener("click", () => $("openFile").click());
$("openFile").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (f) openScore(f.name, new Uint8Array(await f.arrayBuffer()));
});
async function openScore(name, bytes) {
  const ext = name.split(".").pop().toLowerCase();
  if (ext === "json") {
    try { load(JSON.parse(new TextDecoder().decode(bytes)), true); message(`${name} を開きました。`); } catch (e) { message("楽譜を読めませんでした。", true); }
    return;
  }
  if (!app.ready) { app.pendingOpen = [name, bytes]; message(`準備ができたら ${name} を開きます…`); return; }
  try {
    const res = await call("read", { name, bytes });
    if (!res.notes.length) { message("歌詞の付いたノートが見つかりませんでした。", true); return; }
    load(res, true);
    message(`${name} を開きました（ノート ${res.notes.length} 個、テンポ ${Math.round(res.bpm * 100) / 100}）。` +
      (res.changes ? `途中のテンポ変化（${res.changes} 回）は、タイミングを保ったまま最初のテンポの目盛りに直しています。` : ""));
  } catch (err) {
    console.error(err);
    message(`${name} を読めませんでした。対応形式: MusicXML / UST / USTX / MIDI / VSQ / VSQX / VPR / SVP / S5P / CCS`, true);
  }
}
$("saveBtn").addEventListener("click", () => {
  const xml = musicXML(scoreNotes(), bpm(), roll.beatsPerBar);
  download(new Blob([xml], { type: "application/vnd.recordare.musicxml+xml" }), "mame-singer.musicxml");
});
$("bpm").addEventListener("change", () => { $("bpm").value = bpm(); save(); });
$("meter").addEventListener("change", (e) => { roll.beatsPerBar = Number(e.target.value); roll.draw(); save(); });
$("snap").addEventListener("change", (e) => { roll.snap = Number(e.target.value); roll.draw(); });
$("zoomIn").addEventListener("click", () => roll.zoom(1.25));
$("zoomOut").addEventListener("click", () => roll.zoom(0.8));
$("undoBtn").addEventListener("click", () => roll.undo());
$("redoBtn").addEventListener("click", () => roll.redo());
$("lyricBtn").addEventListener("click", () => { const s = roll.sorted().filter((n) => roll.sel.has(n.id)); roll.editLyric(s[0] || roll.sorted()[0]); });
$("delBtn").addEventListener("click", () => roll.deleteSelected());
$("homeBtn").addEventListener("click", () => { roll.setCursor(0); roll.scroll.scrollLeft = 0; if (player.playing) startMelody(); });
$("helpBtn").addEventListener("click", () => $("help").showModal());
$("helpClose").addEventListener("click", () => $("help").close());

// MusicXML: 1 part, divisions = TPB, rests between notes, notes split at barlines with ties
function musicXML(notes, tempo, beats) {
  const steps = [["C", 0], ["C", 1], ["D", 0], ["D", 1], ["E", 0], ["F", 0], ["F", 1], ["G", 0], ["G", 1], ["A", 0], ["A", 1], ["B", 0]];
  const types = [[4, "whole"], [3, "half", 1], [2, "half"], [1.5, "quarter", 1], [1, "quarter"], [0.75, "eighth", 1], [0.5, "eighth"], [0.25, "16th"], [0.125, "32nd"]];
  const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
  const bar = beats * TPB;
  const ev = [];                                       // [start, dur, note|null]
  let t = 0;
  for (const n of notes) {
    const s = Math.round(n.start * TPB), d = Math.round(n.dur * TPB);
    if (s > t) ev.push([t, s - t, null]);
    if (s < t) continue;
    ev.push([s, d, n]); t = s + d;
  }
  const end = Math.max(bar, Math.ceil(t / bar) * bar);
  if (t < end) ev.push([t, end - t, null]);
  const measures = Array.from({ length: end / bar }, () => []);
  for (const [s, d, n] of ev) {
    let a = s, first = true;
    while (a < s + d) {
      const m = Math.floor(a / bar), b = Math.min(s + d, (m + 1) * bar);
      measures[m].push({ dur: b - a, n, tieStart: n && b < s + d, tieStop: n && !first });
      first = false; a = b;
    }
  }
  const typeOf = (d) => { const q = d / TPB; const f = types.find((x) => Math.abs(x[0] - q) < 1e-6); return f ? `<type>${f[1]}</type>${f[2] ? "<dot/>" : ""}` : ""; };
  let body = "";
  measures.forEach((ms, i) => {
    body += `<measure number="${i + 1}">`;
    if (i === 0) body += `<attributes><divisions>${TPB}</divisions><key><fifths>0</fifths></key><time><beats>${beats}</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>` +
      `<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>${tempo}</per-minute></metronome></direction-type><sound tempo="${tempo}"/></direction>`;
    for (const e of ms) {
      if (!e.n) { body += `<note><rest/><duration>${e.dur}</duration><voice>1</voice>${typeOf(e.dur)}</note>`; continue; }
      const [step, alter] = steps[e.n.midi % 12], oct = Math.floor(e.n.midi / 12) - 1;
      const ties = (e.tieStop ? '<tie type="stop"/>' : "") + (e.tieStart ? '<tie type="start"/>' : "");
      const nots = (e.tieStop ? '<tied type="stop"/>' : "") + (e.tieStart ? '<tied type="start"/>' : "");
      body += `<note><pitch><step>${step}</step>${alter ? "<alter>1</alter>" : ""}<octave>${oct}</octave></pitch><duration>${e.dur}</duration>${ties}<voice>1</voice>${typeOf(e.dur)}` +
        (alter ? "<accidental>sharp</accidental>" : "") + (nots ? `<notations>${nots}</notations>` : "") +
        (!e.tieStop && e.n.lyric ? `<lyric number="1"><syllabic>single</syllabic><text>${esc(e.n.lyric)}</text></lyric>` : "") + "</note>";
    }
    body += "</measure>";
  });
  return `<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n` +
    `<score-partwise version="4.0"><part-list><score-part id="P1"><part-name>Voice</part-name></score-part></part-list><part id="P1">${body}</part></score-partwise>\n`;
}

function download(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

// ── melody preview ─────────────────────────────────────────────────────
let audioCtx = null;
const actx = () => (audioCtx ||= new AudioContext());
const hz = (m) => 440 * Math.pow(2, (m - 69) / 12);
function tone(ctx, dest, m, t0, dur, gain = 0.18) {
  const o = ctx.createOscillator(), g = ctx.createGain(), f = ctx.createBiquadFilter();
  o.type = "triangle"; o.frequency.value = hz(m);
  f.type = "lowpass"; f.frequency.value = 2400;
  g.gain.setValueAtTime(0, t0); g.gain.linearRampToValueAtTime(gain, t0 + 0.012);
  g.gain.setTargetAtTime(gain * 0.6, t0 + 0.05, 0.1);
  g.gain.setTargetAtTime(0, t0 + Math.max(0.03, dur - 0.03), 0.02);
  o.connect(f); f.connect(g); g.connect(dest);
  o.start(t0); o.stop(t0 + dur + 0.2);
  return o;
}
function blip(m) { const c = actx(); tone(c, c.destination, m, c.currentTime, 0.22, 0.14); }

const player = { playing: false, nodes: [], raf: 0, kind: null };
function stopMelody() {
  for (const o of player.nodes) { try { o.stop(); } catch (e) { /* already stopped */ } }
  player.nodes = []; player.playing = false; cancelAnimationFrame(player.raf);
  if (player.kind === "melody") roll.setPlayhead(null);
  player.kind = null;
  $("playBtn").classList.remove("on");
  $("playBtn").firstChild.textContent = "▶ ";
}
function startMelody() {
  stopMelody();
  const out = $("out"); if (!out.paused) out.pause();
  const c = actx(), sec = 60 / bpm(), from = roll.cursor, t0 = c.currentTime + 0.05;
  const gain = c.createGain(); gain.gain.value = 1; gain.connect(c.destination);
  for (const n of roll.sorted()) {
    const s = n.start, e = n.start + n.dur;
    if (e <= from) continue;
    const a = Math.max(s, from);
    player.nodes.push(tone(c, gain, n.midi, t0 + ((a - from) / TPB) * sec, ((e - a) / TPB) * sec));
  }
  if (app.acc) {
    const src = c.createBufferSource(), g = c.createGain();
    src.buffer = app.acc.buf; g.gain.value = dbGain(accGainDb()) * app.acc.norm;
    src.connect(g); g.connect(c.destination);
    const at = (from / TPB) * sec + accOffset();
    if (at >= 0) src.start(t0, at); else src.start(t0 - at, 0);
    player.nodes.push(src);
  }
  const end = Math.max(roll.endTick(), app.acc ? ((app.acc.buf.duration - accOffset()) / sec) * TPB : 0);
  player.playing = true; player.kind = "melody";
  $("playBtn").classList.add("on"); $("playBtn").firstChild.textContent = "■ ";
  const tick = () => {
    const t = from + ((c.currentTime - t0) / sec) * TPB;
    if (t > end + TPB) { stopMelody(); return; }
    roll.setPlayhead(Math.max(from, t));
    player.raf = requestAnimationFrame(tick);
  };
  tick();
}
$("playBtn").addEventListener("click", () => (player.playing ? stopMelody() : startMelody()));
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.target.closest("input, textarea, select, button, dialog") || roll.editing) return;
  e.preventDefault();
  player.playing ? stopMelody() : startMelody();
});

// the sung result follows the roll's playhead while it plays
(function followResult() {
  const out = $("out");
  let raf = 0;
  const loop = () => {
    if (!app.result) return;
    const t = (out.currentTime - app.result.offset) * (app.result.bpm / 60) * TPB;
    roll.setPlayhead(t >= 0 ? t : null);
    raf = requestAnimationFrame(loop);
  };
  out.addEventListener("play", () => { stopMelody(); cancelAnimationFrame(raf); loop(); });
  out.addEventListener("pause", () => { cancelAnimationFrame(raf); roll.setPlayhead(null); });
  out.addEventListener("ended", () => { cancelAnimationFrame(raf); roll.setPlayhead(null); });
})();

// ── accompaniment ──────────────────────────────────────────────────────
const accOffset = () => Number($("accOffset").value) || 0;
const accGainDb = () => Number($("accGain").value);
const dbGain = (db) => Math.pow(10, db / 20);
function activeRms(chs) {
  let mx = 0;
  for (const ch of chs) for (let i = 0; i < ch.length; i += 4) mx = Math.max(mx, Math.abs(ch[i]));
  const th = mx * 0.01;
  let s = 0, n = 0;
  for (const ch of chs) for (let i = 0; i < ch.length; i += 4) { const v = ch[i]; if (Math.abs(v) > th) { s += v * v; n++; } }
  return n ? Math.sqrt(s / n) : 1;
}
async function setAccompaniment(name, arrayBuffer) {
  try {
    const ctx = new AudioContext({ sampleRate: 44100 });
    const buf = await ctx.decodeAudioData(arrayBuffer.slice(0));
    ctx.close();
    const chs = [...Array(buf.numberOfChannels).keys()].map((i) => buf.getChannelData(i));
    const hop = 0.02, step = Math.round(hop * buf.sampleRate), peaks = new Float32Array(Math.ceil(buf.length / step));
    let mx = 1e-6;
    for (let k = 0; k < peaks.length; k++) {
      let p = 0;
      for (const ch of chs) for (let i = k * step, e = Math.min(ch.length, i + step); i < e; i += 3) p = Math.max(p, Math.abs(ch[i]));
      peaks[k] = p; mx = Math.max(mx, p);
    }
    for (let k = 0; k < peaks.length; k++) peaks[k] /= mx;
    app.acc = { name, buf, peaks, hop, norm: 0.12 / activeRms(chs), url: URL.createObjectURL(new Blob([arrayBuffer])) };
    $("accName").textContent = name; $("accSec").textContent = fmtTime(buf.duration);
    $("accInfo").hidden = false; $("accDrop").hidden = true;
    updateTrack(); remix(); save();
  } catch (e) {
    message(`${name} を読めませんでした（音声ファイルを選んでください）。`, true);
  }
}
function fmtTime(s) { return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`; }
function updateTrack() {
  roll.setTrack(app.acc ? { peaks: app.acc.peaks, hop: app.acc.hop, secOf: (tick) => (tick / TPB) * (60 / bpm()) + accOffset() } : null);
}
$("accPick").addEventListener("click", () => $("accFile").click());
$("accFile").addEventListener("change", async (e) => { const f = e.target.files[0]; e.target.value = ""; if (f) setAccompaniment(f.name, await f.arrayBuffer()); });
const accDrop = $("accDrop");
["dragenter", "dragover"].forEach((ev) => accDrop.addEventListener(ev, (e) => { e.preventDefault(); accDrop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => accDrop.addEventListener(ev, (e) => { e.preventDefault(); accDrop.classList.remove("over"); }));
accDrop.addEventListener("drop", async (e) => { const f = e.dataTransfer.files[0]; if (f) setAccompaniment(f.name, await f.arrayBuffer()); });
$("accDel").addEventListener("click", () => {
  if (player.playing) stopMelody();
  URL.revokeObjectURL(app.acc.url); app.acc = null;
  $("accInfo").hidden = true; $("accDrop").hidden = false;
  updateTrack(); remix();
});
$("accPlay").addEventListener("click", () => new Audio(app.acc.url).play());
$("accOffset").addEventListener("input", () => { updateTrack(); remixSoon(); save(); });
$("accGain").addEventListener("input", (e) => { const v = Number(e.target.value); $("accGainVal").textContent = `${v > 0 ? "+" : v < 0 ? "−" : "±"}${Math.abs(v)} dB`; remixSoon(); });
$("bpm").addEventListener("input", updateTrack);

// the outputs: the vocal alone, and (with an accompaniment) the mix on the accompaniment's timeline
let remixTimer = 0;
function remixSoon() { clearTimeout(remixTimer); remixTimer = setTimeout(remix, 250); }
function remix() {
  const v = app.vocal;
  if (!v) return;
  const out = $("out"), wasPlaying = !out.paused, at = out.currentTime;
  const scoreAt = app.result ? at - app.result.offset : 0;
  if (app.result) { URL.revokeObjectURL(app.result.url); if (app.result.mixUrl) URL.revokeObjectURL(app.result.mixUrl); }
  const vocalUrl = URL.createObjectURL(wav([v.f32], 44100));
  let url = vocalUrl, mixUrl = null, offset = v.offset;
  if (app.acc) {
    const acc = app.acc.buf, ga = dbGain(accGainDb()) * app.acc.norm, gv = 0.12 / activeRms([v.f32]);
    const vs = Math.round((accOffset() - v.offset) * 44100);          // vocal sample 0 on the accompaniment timeline
    const n = Math.max(acc.length, vs + v.f32.length);
    const chs = [0, 1].map((c) => {
      const a = acc.getChannelData(Math.min(c, acc.numberOfChannels - 1)), y = new Float32Array(n);
      for (let i = 0; i < a.length; i++) y[i] = a[i] * ga;
      for (let i = Math.max(0, -vs); i < v.f32.length; i++) y[i + vs] += v.f32[i] * gv;
      return y;
    });
    let peak = 0;
    for (const y of chs) for (let i = 0; i < y.length; i++) peak = Math.max(peak, Math.abs(y[i]));
    const k = peak > 0.95 ? 0.95 / peak : 1;
    if (k < 1) for (const y of chs) for (let i = 0; i < y.length; i++) y[i] *= k;
    mixUrl = url = URL.createObjectURL(wav(chs, 44100));
    offset = accOffset();
  }
  app.result = { url: vocalUrl, mixUrl, offset, bpm: v.bpm };
  out.src = url;
  $("dl").href = vocalUrl;
  $("dlMix").href = mixUrl || "#"; $("dlMix").hidden = !mixUrl;
  $("result").style.display = "block";
  out.currentTime = Math.max(0, scoreAt + offset);
  if (wasPlaying) out.play().catch(() => {});
}

// ── voices ─────────────────────────────────────────────────────────────
async function to22k(arrayBuffer) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await ctx.decodeAudioData(arrayBuffer.slice(0));
  ctx.close();
  const n = Math.ceil(buf.duration * 22050);
  const off = new OfflineAudioContext(1, n, 22050);
  const src = off.createBufferSource();
  src.buffer = buf; src.connect(off.destination); src.start();
  return (await off.startRendering()).getChannelData(0).slice();
}
async function addVoice(name, arrayBuffer, blob) {
  try {
    const pcm = await to22k(arrayBuffer);
    if (pcm.length < 22050) { message(`${name} は短すぎます（1 秒以上にしてください）。`, true); return; }
    app.voices.push({ name, pcm, sec: pcm.length / 22050, url: URL.createObjectURL(blob || new Blob([arrayBuffer])) });
    renderClips(); refresh();
  } catch (e) {
    message(`${name} を読めませんでした（音声ファイルを選んでください）。`, true);
  }
}
function renderClips() {
  const ul = $("clips");
  ul.innerHTML = "";
  app.voices.forEach((v, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<button title="聞く">▶</button><span class="name"></span><span class="sec">${v.sec.toFixed(1)} 秒</span><button title="外す">✕</button><i class="bar"></i>`;
    li.querySelector(".name").textContent = v.name;
    const [play, del] = li.querySelectorAll("button");
    play.addEventListener("click", () => previewClip(v.url, li, play));
    del.addEventListener("click", () => { URL.revokeObjectURL(v.url); app.voices.splice(i, 1); renderClips(); refresh(); });
    ul.appendChild(li);
  });
}
let clipAudio = null;
function previewClip(url, li, btn) {
  const was = clipAudio && clipAudio.li === li;
  if (clipAudio) { clipAudio.a.pause(); clipAudio.done(); }
  if (was) return;
  const a = new Audio(url), bar = li.querySelector(".bar");
  const done = () => { li.classList.remove("playing"); btn.textContent = "▶"; bar.style.width = "0"; cancelAnimationFrame(raf); clipAudio = null; };
  let raf = 0;
  const tick = () => { if (a.duration) bar.style.width = `${(a.currentTime / a.duration) * 100}%`; raf = requestAnimationFrame(tick); };
  a.addEventListener("ended", done);
  clipAudio = { a, li, done };
  li.classList.add("playing"); btn.textContent = "■";
  a.play(); tick();
}
$("pickBtn").addEventListener("click", () => $("voiceFile").click());
$("voiceFile").addEventListener("change", async (e) => {
  for (const f of e.target.files) await addVoice(f.name, await f.arrayBuffer(), f);
  e.target.value = "";
});
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async (e) => { for (const f of e.dataTransfer.files) await addVoice(f.name, await f.arrayBuffer(), f); });
document.addEventListener("dragover", (e) => e.preventDefault());
document.addEventListener("drop", async (e) => {
  if (e.target.closest("#drop")) return;
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f && SCORE_RE.test(f.name)) openScore(f.name, new Uint8Array(await f.arrayBuffer()));
  else if (f) for (const g of e.dataTransfer.files) await addVoice(g.name, await g.arrayBuffer(), g);
});

const rec = { r: null, t0: 0, timer: 0, raf: 0 };
$("recBtn").addEventListener("click", async () => {
  if (rec.r) { rec.r.stop(); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: true, autoGainControl: true } }); }
  catch (e) { message("マイクを使えませんでした（ブラウザの許可を確認してください）。", true); return; }
  const chunks = [];
  const r = new MediaRecorder(stream);
  const c = actx(), src = c.createMediaStreamSource(stream), an = c.createAnalyser();
  an.fftSize = 1024; src.connect(an);
  const buf = new Float32Array(an.fftSize);
  $("level").style.display = "block";
  const meter = () => { an.getFloatTimeDomainData(buf); let p = 0; for (const v of buf) p = Math.max(p, Math.abs(v)); $("level").firstChild.style.width = `${Math.min(100, p * 140)}%`; rec.raf = requestAnimationFrame(meter); };
  meter();
  r.ondataavailable = (ev) => chunks.push(ev.data);
  r.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop()); src.disconnect();
    cancelAnimationFrame(rec.raf); clearInterval(rec.timer);
    $("level").style.display = "none"; $("recBtn").classList.remove("live"); $("recLabel").textContent = "録音";
    rec.r = null;
    const blob = new Blob(chunks, { type: chunks[0]?.type || "audio/webm" });
    await addVoice(`録音 ${app.voices.length + 1}`, await blob.arrayBuffer(), blob);
  };
  r.start();
  rec.r = r; rec.t0 = performance.now();
  $("recBtn").classList.add("live");
  const tick = () => { const s = (performance.now() - rec.t0) / 1000; $("recLabel").textContent = `止める ${s.toFixed(0)} 秒`; if (s > 30) r.stop(); };
  tick(); rec.timer = setInterval(tick, 250);
});

// ── options ────────────────────────────────────────────────────────────
function segmented(id, key) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of $(id).children) x.classList.toggle("on", x === b);
    app[key] = Number(b.dataset.v);
  });
}
segmented("mode", "mode");
segmented("style", "style");
$("key").addEventListener("input", (e) => { const v = Number(e.target.value); $("keyVal").textContent = v > 0 ? `+${v}` : v < 0 ? `${v}` : "±0"; });

// ── lyric check (red notes) ────────────────────────────────────────────
let checkTimer = 0;
function scheduleCheck() { clearTimeout(checkTimer); checkTimer = setTimeout(check, 300); }
async function check() {
  if (!app.ready) return [];
  const s = roll.sorted();
  const bad = await call("check", { notes: JSON.stringify(s.map((n) => ({ lyric: n.lyric }))) });
  const map = new Map(bad.filter(([i]) => s[i]).map(([i, msg]) => [s[i].id, msg]));
  roll.setBad(map);
  return [...map.entries()];
}

function refresh() {
  const go = $("goBtn");
  go.disabled = !app.ready || app.busy || !roll.notes.length || !app.voices.length;
  go.classList.toggle("working", app.busy || !app.ready);
  go.innerHTML = (app.busy || !app.ready ? '<span class="spin"></span>' : "") + (!app.ready ? "準備しています…" : app.busy ? "歌っています…" : !roll.notes.length ? "楽譜にノートを置いてください" : !app.voices.length ? "声を入れてください" : "歌わせる");
  $("undoBtn").disabled = !roll.undoStack.length;
  $("redoBtn").disabled = !roll.redoStack.length;
  $("delBtn").disabled = !roll.sel.size;
}

// ── singing ────────────────────────────────────────────────────────────
function onProgress(k, n) {
  $("progress").firstChild.style.width = `${Math.round((k / n) * 100)}%`;
  message(`歌っています… ${k} / ${n} フレーズ`);
}

$("goBtn").addEventListener("click", async () => {
  const bad = await check();
  if (bad.length) {
    message(`読めない歌詞が ${bad.length} か所あります（赤いノート）。ひらがなで書いてください。 <button id="showBad">最初のノートへ</button>`, true, true);
    $("showBad").addEventListener("click", () => roll.editLyric(roll.byId(bad[0][0])));
    return;
  }
  stopMelody();
  app.busy = true; refresh();
  $("progress").style.display = "block"; $("progress").firstChild.style.width = "0";
  message("声の特徴を取り出しています…");
  const t0 = performance.now();
  try {
    const tempo = bpm();
    const { buffer, offset } = await call("sing", { notes: JSON.stringify(scoreNotes()), bpm: tempo, voices: app.voices.map((v) => v.pcm),
      mode: app.mode, transpose: Number($("key").value), style: app.style });
    const f32 = new Float32Array(buffer);
    app.vocal = { f32, offset, bpm: tempo };
    app.result = null;
    remix();
    message(`できました（${((performance.now() - t0) / 1000).toFixed(0)} 秒）。再生すると楽譜の上に位置が出ます。`);
    $("out").currentTime = Math.max(0, app.result.offset + (roll.cursor / TPB) * (60 / tempo) - 0.3);
    $("out").play().catch(() => {});
  } catch (e) {
    console.error(e);
    message("歌わせられませんでした: " + String(e.message || e).split("\n").slice(-2).join(" "), true);
  } finally {
    app.busy = false; refresh();
    $("progress").style.display = "none";
  }
});

function wav(chs, sr) {
  const nc = chs.length, n = chs[0].length, bytes = n * nc * 2;
  const b = new ArrayBuffer(44 + bytes), v = new DataView(b);
  const s = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  s(0, "RIFF"); v.setUint32(4, 36 + bytes, true); s(8, "WAVE"); s(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, nc, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * nc * 2, true); v.setUint16(32, nc * 2, true); v.setUint16(34, 16, true); s(36, "data"); v.setUint32(40, bytes, true);
  let o = 44;
  for (let i = 0; i < n; i++) for (let c = 0; c < nc; c++) { v.setInt16(o, Math.max(-1, Math.min(1, chs[c][i])) * 32767, true); o += 2; }
  return new Blob([b], { type: "audio/wav" });
}

// ── start ──────────────────────────────────────────────────────────────
let loaded = 0;
function onLoading(kind) {
  loaded++;
  status(kind === "python" ? "Python を準備しています…" : "モデルを読み込んでいます…", "", Math.min(0.95, loaded / (MODELS.length + 2)));
}
async function fetchBytes(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

async function init() {
  let stored = null;
  try { stored = JSON.parse(localStorage.getItem(STORE) || "null"); } catch (e) { stored = null; }
  if (stored && stored.notes && stored.notes.length) load(stored);
  else load(await (await fetch("examples/furusato.json")).json());
  updateCounts(); refresh();
  try {
    status("モデルを読み込んでいます…", "", 0.02);
    await call("init", { base: new URL(".", location.href).href });
    app.ready = true;
    status("準備ができました", "ready", 1);
    if (app.pendingOpen) { const [n, b] = app.pendingOpen; app.pendingOpen = null; openScore(n, b); }
    scheduleCheck(); refresh();
  } catch (e) {
    console.error(e);
    status("読み込みに失敗しました。ページを再読み込みしてください。", "error");
  }
}

init();
