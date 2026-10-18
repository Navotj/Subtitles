"""How quickly does the engine notice that someone else started talking?  (no Whisper or GPU needed)
Runs the test calls twice (first to learn the voices, then with them known) and, for every change of speaker, prints how
long after the new person really started the right name was shown, plus how many splits were made in the wrong place.
    python tools/speaker_latency.py
Tunables can be tried without editing the engine, e.g.:  SC_SCAN_EXTRA=-0.04 python tools/speaker_latency.py
"""
import sys, os, json, tempfile, statistics, importlib
import numpy as np, soundfile as sf, soxr
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.environ.get("ENGINE_DIR", ROOT))
import engine
from engine import SR

def apply_env():
    P = engine.Pipeline
    if "SC_CUTS" in os.environ: P.SC_CUTS = tuple((float(a.split(":")[0]), float(a.split(":")[1])) for a in os.environ["SC_CUTS"].split(","))
    if "SC_QUICK" in os.environ: P.SC_QUICK = tuple(float(v) for v in os.environ["SC_QUICK"].split(","))
    if "SC_MIN" in os.environ: P.SC_MIN = float(os.environ["SC_MIN"])
    if "SC_LEFT" in os.environ: P.SC_LEFT = float(os.environ["SC_LEFT"])
    if "SC_FAST_EVERY" in os.environ: P.SC_FAST_EVERY = float(os.environ["SC_FAST_EVERY"])
    if "SC_SCAN_EXTRA" in os.environ: P.SC_SCAN_EXTRA = float(os.environ["SC_SCAN_EXTRA"])
    if "SC_W" in os.environ: P.SC_W = float(os.environ["SC_W"])
    if "SC_EVERY" in os.environ: P.SC_EVERY = float(os.environ["SC_EVERY"])
CUTS = []
_orig_split = engine.Segmenter.split_at
def _rec_split(self, cut):
    r = _orig_split(self, cut); CUTS.append((r[0] + len(r[1])) / SR); return r
engine.Segmenter.split_at = _rec_split

class FakeASR:
    name, device = "fake", "cpu"
    def partial(self, audio): return "..."
    def final(self, audio):
        d = len(audio) / SR
        return [(t, t + 0.25, " w") for t in np.arange(0.3, max(d - 0.3, 0.31), 0.3)]
    def vocab_terms(self): return []

def load(name):
    x, sr = sf.read(os.path.join(ROOT, "test", f"{name}.flac"), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR: x = soxr.resample(x, sr, SR).astype(np.float32)
    return x, json.load(open(os.path.join(ROOT, "test", f"{name}.json")))["utts"]

def run(x, bank, cfg, emb, collect):
    events = []
    clock = [0.0]
    def on_event(e): events.append((clock[0], e))
    p = engine.Pipeline(cfg, bank, FakeASR(), emb, on_event)
    step = int(0.03 * SR)
    for i in range(0, len(x), step):
        clock[0] = (i + step) / SR
        p.feed(x[i:i + step])
    clock[0] = len(x) / SR; p.flush()
    return events, p.stats

def evaluate(name, overrides, emb, verbose=False):
    x, utts = load(name)
    cfg = engine.load_config(); cfg.update(overrides)
    bank = engine.VoiceBank(path=os.path.join(tempfile.mkdtemp(), "v.json"), model=cfg["voice_model"])
    bank.voices.clear(); bank.next_id = 1
    c1 = dict(cfg, auto_new_voices=True)
    run(x, bank, c1, emb, None)                                    # pass 1: learn the voices
    CUTS.clear()
    events, stats = run(x, bank, dict(cfg, auto_new_voices=False), emb, None)   # pass 2: voices known
    cuts = list(CUTS)
    def who(tm):
        for u in utts:
            if u["start"] - 0.05 <= tm <= u["end"] + 0.05: return u["speaker"]
    pairs = []
    for t, e in events:
        if e["kind"] != "final": continue
        for v, s0, s1 in e["spans"]:
            for tm in np.arange(s0, s1, 0.1):
                w = who(tm)
                if w: pairs.append((v, w))
    m = {v: Counter(w for vv, w in pairs if vv == v).most_common(1)[0][0] for v in set(p_[0] for p_ in pairs) if v is not None}
    acc = sum(1 for v, w in pairs if v is not None and m.get(v) == w) / max(len(pairs), 1)
    lat_tight, lat_all = [], []
    for k, u in enumerate(utts):
        if k == 0 or utts[k-1]["speaker"] == u["speaker"]: continue
        gap = u["start"] - utts[k-1]["end"]
        found = None
        for t, e in events:
            if t < u["start"]: continue
            if e["kind"] == "partial": vs = [e["voice"]]
            else: vs = [v for v, s0, s1 in e["spans"] if s1 > u["start"] + 0.1 and s0 < u["end"]]
            if any(v is not None and m.get(v) == u["speaker"] for v in vs): found = t - u["start"]; break
        if found is None: found = 99.0
        if found > 2.0: print(f'   slow: {utts[k-1]["speaker"]}->{u["speaker"]} at {u["start"]:.2f}s gap {gap:.2f}s dur {u["end"]-u["start"]:.2f}s took {found:.1f}s')
        (lat_tight if gap < 0.5 else lat_all).append(found)
        lat_all.append(found) if gap < 0.5 else None
    bounds = [u["start"] for k, u in enumerate(utts) if k and utts[k-1]["speaker"] != u["speaker"]]
    true_c = sum(1 for c in cuts if any(abs(c - b) < 0.9 for b in bounds))
    return dict(cuts=len(cuts), true_cuts=true_c, acc=acc, tight=lat_tight, all=lat_all, early=stats.get("early_splits", 0), utts=stats["utts"])

if __name__ == "__main__":
    apply_env()
    overrides = {}
    for kv in sys.argv[1:]:
        k, v = kv.split("=", 1); overrides[k] = float(v)
    emb = engine.Embedder(engine.load_config()["voice_model"])
    for name in ("conversation", "conv2"):
        r = evaluate(name, overrides, emb)
        t = sorted(r["tight"]); a = sorted(r["all"])
        f = lambda l: f"n={len(l)} median={statistics.median(l):.2f}s mean={statistics.mean(l):.2f}s p90={l[int(len(l)*0.9)-1]:.2f}s max={l[-1]:.2f}s" if l else "n=0"
        print(f"{name}: who-said-it {r['acc']*100:.1f}%  cuts {r['cuts']} (near a real change: {r['true_cuts']}, false: {r['cuts']-r['true_cuts']})  utts {r['utts']}\n   tight turns (<0.5 s gap): {f(t)}\n   all speaker changes:      {f(a)}")
