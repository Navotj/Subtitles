"""
Runs a recorded call through the subtitle engine (no audio device, no overlay) and prints what it would show.
  selftest.py <file.wav> [--truth file.json] [--fake-asr] [--fresh] [--realtime]
With --truth it also scores who-said-what against the real speakers.
"""
import sys, os, json, time, argparse, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import SR

ap = argparse.ArgumentParser()
ap.add_argument("wav"); ap.add_argument("--truth"); ap.add_argument("--fake-asr", action="store_true")
ap.add_argument("--fresh", action="store_true", help="start with no known voices (default)", default=True)
ap.add_argument("--realtime", action="store_true", help="feed at real speed (tests partial text timing)")
ap.add_argument("--set", action="append", default=[], help="config override key=value")
a = ap.parse_args()

cfg = engine.load_config()
for kv in a.set:
    k, v = kv.split("=", 1); cfg[k] = type(engine.DEFAULT_CONFIG.get(k, 0.0))(v) if k in engine.DEFAULT_CONFIG else v

class FakeASR:
    name, device = "fake", "cpu"
    def partial(self, audio): return "..."
    def final(self, audio):
        d = len(audio) / SR
        return [(t, t + 0.25, " w") for t in np.arange(0.3, max(d - 0.3, 0.31), 0.3)]

# decode
try:
    import soundfile as sf
    x, sr = sf.read(a.wav, dtype="float32", always_2d=True)
except Exception:
    from faster_whisper.audio import decode_audio
    x, sr = decode_audio(a.wav, sampling_rate=SR)[:, None], SR
x = x.mean(axis=1)
if sr != SR:
    import soxr; x = soxr.resample(x, sr, SR).astype(np.float32)
print(f"audio: {len(x)/SR:.1f} s", flush=True)

bank = engine.VoiceBank(path=os.path.join(tempfile.gettempdir(), "dndsubs_selftest_voices.json"), model=cfg["voice_model"])
bank.voices.clear(); bank.next_id = 1
t = time.time()
asr = FakeASR() if a.fake_asr else engine.Transcriber(cfg)
print(f"whisper load {time.time()-t:.1f}s on {asr.device}", flush=True)
emb = engine.Embedder(cfg['voice_model'])
events = []
def on_event(e):
    events.append(e)
    if e["kind"] == "final":
        for (v, txt), (_, s0, s1) in zip(e["parts"], e["spans"]):
            print(f"  [{s0:7.2f}-{s1:7.2f}] {bank.get(v)[0]:>8}: {txt}", flush=True)
    elif a.realtime:
        print(f"    (partial) {bank.get(e['voice'])[0]}: {e['text']}", flush=True)
p = engine.Pipeline(cfg, bank, asr, emb, on_event)
t = time.time()
step = int(0.03 * SR)
for i in range(0, len(x), step):
    p.feed(x[i:i + step])
    if a.realtime:
        lag = (i + step) / SR - (time.time() - t)
        if lag > 0: time.sleep(lag)
p.flush()
el = time.time() - t
st = p.stats
print(f"\nprocessed {len(x)/SR:.1f}s of audio in {el:.1f}s; {st['utts']} utterances, whisper time {st['asr_s']:.1f}s "
      f"({st['asr_s']/max(st['utts'],1)*1000:.0f} ms per utterance)")
print("voices:", [(v['name'], v['count']) for v in bank.snapshot()])
terms = asr.vocab_terms() if hasattr(asr, "vocab_terms") else []
if terms:
    import re
    texts = [("partial", e["text"]) for e in events if e["kind"] == "partial"] + \
            [("final", t) for e in events if e["kind"] == "final" for _, t in e["parts"]]
    bad = [(k, t) for k, t in texts if any(re.search(r"(?<![\w'])" + re.escape(w.lower()) + r"(?![\w'])", t.lower()) for w in terms)]
    print(f"\nwords-to-expect check ({', '.join(terms)}): {len(texts)} texts shown, {len(bad)} contain list words "
          f"(none are actually said in this test); list re-run {getattr(asr, 'leaks', 0)} times")
    for k, t in bad[:20]: print(f"   {k}: {t}")

if a.truth:
    tr = json.load(open(a.truth))
    utts = tr["utts"]
    def who(tm):
        for u in utts:
            if u["start"] - 0.05 <= tm <= u["end"] + 0.05: return u["speaker"]
        return None
    # word-level: each final span sampled every 0.1 s
    pairs = []
    for e in events:
        if e["kind"] != "final": continue
        for v, s0, s1 in e["spans"]:
            for tm in np.arange(s0, s1, 0.1):
                w = who(tm)
                if w: pairs.append((v, w))
    from collections import Counter
    # map each voice to the real person it mostly covers
    m = {}
    for v in set(p_[0] for p_ in pairs):
        m[v] = Counter(w for vv, w in pairs if vv == v).most_common(1)[0][0]
    ok = sum(1 for v, w in pairs if v is not None and m[v] == w)
    unk = sum(1 for v, w in pairs if v is None)
    print(f"\nwho-said-it accuracy: {ok/len(pairs)*100:.1f}%  (unknown {unk/len(pairs)*100:.1f}%)")
    for v in sorted(m, key=lambda z: (z is None, z or 0)):
        n = sum(1 for vv, _ in pairs if vv == v)
        print(f"   {bank.get(v)[0]:>8} -> {m[v]:<6} ({n} samples, purity {sum(1 for vv,w in pairs if vv==v and w==m[v])/n*100:.0f}%)")
    # per utterance: how many truth utterances got fully the right colour
    good = 0
    for u in utts:
        vs = [v for v, w in pairs if False]
    # utterance level
    ut_ok = 0
    for u in utts:
        labs = []
        for e in events:
            if e["kind"] != "final": continue
            for v, s0, s1 in e["spans"]:
                ov = min(s1, u["end"]) - max(s0, u["start"])
                if ov > 0.05: labs.append((ov, v))
        if labs:
            v = max(labs)[1]
            if v is not None and m.get(v) == u["speaker"]: ut_ok += 1
    print(f"utterances shown in the right person's colour: {ut_ok}/{len(utts)}")
