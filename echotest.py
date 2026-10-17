"""Plays a recorded call and a mic recording (your voice + the call leaking in) side by side through the app's
engine, and counts how many of your real lines come through vs how much echo slips in.
  echotest.py test\\conversation.flac test\\mic.flac test\\mic.json [--fake-asr]"""
import sys, os, json, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine
from engine import SR

def load(p):
    import soundfile as sf, soxr
    x, sr = sf.read(p, dtype="float32", always_2d=True); x = x.mean(axis=1)
    return soxr.resample(x, sr, SR).astype(np.float32) if sr != SR else x

call_wav, mic_wav, truth = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3]))
fake = "--fake-asr" in sys.argv
cfg = engine.load_config()
bank = engine.VoiceBank(path=os.path.join(tempfile.gettempdir(), "dndsubs_echotest.json"), model=cfg["voice_model"])
bank.voices.clear(); bank.me_forget(); bank.next_id = 1

class FakeASR:
    name, device = "fake", "cpu"
    def partial(self, a): return "x"
    def final(self, a):
        d = len(a) / SR
        return [(t, t + 0.25, f" w{np.random.randint(1e9)}") for t in np.arange(0.1, max(d - 0.2, 0.2), 0.3)]
asr = FakeASR() if fake else engine.Transcriber(cfg)
emb = engine.Embedder(cfg["voice_model"])
t = [0.0]; clock = lambda: t[0]
mine = []
call = engine.Pipeline(cfg, bank, asr, emb, lambda e: None); call.clock = clock
mic = engine.MicPipeline(cfg, bank, asr, emb, lambda e: mine.append((t[0], e))); mic.clock = clock; mic.call = call; mic.debug = "--debug" in sys.argv
a, b = load(call_wav), load(mic_wav)
n = min(len(a), len(b)); step = int(0.03 * SR)
for i in range(0, n, step):
    t[0] = (i + step) / SR
    call.feed(a[i:i + step], t[0]); mic.feed(b[i:i + step], t[0])
call.flush(); mic.flush()
got = [(when, e["parts"][0][1], e["t0"], e["t0"] + e["dur"]) for when, e in mine if e["kind"] == "final"]
real = 0; leaked = []
hit = set()
for when, txt, s0, s1 in got:
    m = [k for k, u in enumerate(truth) if min(s1, u["end"]) - max(s0, u["start"]) > 0.3]
    if m: real += 1; hit.update(m); print(f"  YOU   [{s0:6.1f}] {txt}")
    else: leaked.append(txt); print(f"  ECHO  [{s0:6.1f}] {txt}")
print(f"\nyour lines caught: {len(hit)}/{len(truth)}   echo lines that slipped through: {len(leaked)}   "
      f"thrown away as echo: {mic.stats.get('echo', 0)}   your voice learned: {len(bank.me)} fingerprint(s)")
