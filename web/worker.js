// The singer in a worker, so the page stays responsive while it computes: Pyodide runs the score front end
// (mame_web.py), onnxruntime-web runs the networks. Messages: {id, cmd, args} -> {id, ok, result | error}, and
// {progress: [k, n]} while singing.
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js");
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.2/full/pyodide.js");

const MODELS = ["acoustic", "voice_encoder", "f0_cond", "f0_denoise", "vocoder"];
const DATA = ["f0.json", "cons_dur.npz", "cons_dur_config.json", "cons_dur.json", "phonemes.json"];
const sessions = {};
let py = null;

async function fetchBytes(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

self.mameRun = async (name, spec) => {
  const feeds = {};
  for (const [key, dtype, shape, bytes] of spec) {
    const buf = bytes.slice().buffer;
    feeds[key] = dtype === "int64" ? new ort.Tensor("int64", new BigInt64Array(buf), Array.from(shape))
      : new ort.Tensor("float32", new Float32Array(buf), Array.from(shape));
  }
  const res = await sessions[name].run(feeds);
  return Object.entries(res).map(([key, t]) => [key, Array.from(t.dims), new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength)]);
};
self.mameProgress = (k, n) => postMessage({ progress: [k, n] });

const step = (text) => postMessage({ loading: text });

const MODEL_REPO = "https://huggingface.co/kizuna-intelligence/mame-singer-oss/resolve/main/";
const REPO_PATH = { "phonemes.json": "phonemes.json", "cons_dur.json": "cons_dur.json" };   // the rest are under onnx/

// the models come from the model repository (its config.json is what counts a download there); if that cannot be
// reached, from the copy inside this Space
async function modelSource(base) {
  try {
    const r = await fetch(`${MODEL_REPO}config.json`, { cache: "no-store" });
    if (r.ok) return (f) => `${MODEL_REPO}${REPO_PATH[f] || `onnx/${f}`}`;
  } catch (e) { /* offline or blocked: use the Space's copy */ }
  return (f) => `${base}models/${f}`;
}

async function init(base) {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const pyP = (async () => {
    const p = await loadPyodide();
    step("python");
    await p.loadPackage(["numpy", "scipy", "pyyaml"]);
    step("python");
    return p;
  })();
  const src = await modelSource(base);
  await Promise.all(MODELS.map(async (m) => {
    sessions[m] = await ort.InferenceSession.create(await fetchBytes(src(`${m}.onnx`)), { executionProviders: ["wasm"] });
    step("model");
  }));
  py = await pyP;
  py.unpackArchive((await fetchBytes(`${base}py.zip`)).buffer, "zip", { extractDir: "/home/pyodide/mame" });
  const files = {};
  for (const f of DATA) files[f] = await fetchBytes(src(f));
  py.globals.set("_files", py.toPy(files));
  await py.runPythonAsync(`
import sys, json
sys.path[:0] = ["/home/pyodide/mame"]
import numpy as np
from pyodide.ffi import to_js
import js
from mame_web import WebSinger, Bridge, read_score

def _b(v):
    return v.to_bytes() if hasattr(v, "to_bytes") else bytes(v)

async def _run(name, feeds):
    spec = [[k, str(v.dtype), list(v.shape), to_js(np.ascontiguousarray(v).tobytes())] for k, v in feeds.items()]
    res = await js.mameRun(name, to_js(spec))
    out = {}
    for item in res:
        key, dims, data = item[0], list(item[1].to_py()), item[2]
        out[key] = np.frombuffer(data.to_bytes(), np.float32).reshape(dims)
    return out

singer = WebSinger(Bridge(_run), {k: _b(v) for k, v in _files.items()})
del _files

def read_js(path):
    notes, bpm, changes = read_score(path)
    return json.dumps({"notes": notes, "bpm": bpm, "changes": changes}, ensure_ascii=False)

def check_js(notes):
    return json.dumps(singer.check(json.loads(notes)), ensure_ascii=False)

async def sing_js(notes, bpm, voices, mode, transpose, style):
    emb = await singer.voice([np.frombuffer(_b(v), np.float32) for v in voices])
    y, offset = await singer.sing_notes(json.loads(notes), float(bpm), emb, int(mode), int(transpose), int(style),
                                        progress=js.mameProgress)
    return to_js([to_js(np.clip(y, -1, 1).astype(np.float32).tobytes()), float(offset)])
`);
}

const handlers = {
  init: ({ base }) => init(base),
  read: ({ name, bytes }) => {
    const ext = name.split(".").pop().toLowerCase();
    const path = `/tmp/open.${ext}`;
    py.FS.writeFile(path, bytes);
    return JSON.parse(py.globals.get("read_js")(path));
  },
  check: ({ notes }) => JSON.parse(py.globals.get("check_js")(notes)),
  sing: async ({ notes, bpm, voices, mode, transpose, style }) => {
    const pv = py.toPy(voices);
    const res = await py.globals.get("sing_js")(notes, bpm, pv, mode, transpose, style);
    pv.destroy?.();
    const [bytes, offset] = res.toJs ? res.toJs() : res;
    const buf = bytes.slice().buffer;
    return { buffer: buf, offset };
  },
};

let queue = Promise.resolve();
self.onmessage = (e) => {
  const { id, cmd, args } = e.data;
  queue = queue.then(async () => {
    try {
      const result = await handlers[cmd](args);
      postMessage({ id, ok: true, result }, result && result.buffer instanceof ArrayBuffer ? [result.buffer] : []);
    } catch (err) {
      postMessage({ id, ok: false, error: String(err && err.message || err) });
    }
  });
};
