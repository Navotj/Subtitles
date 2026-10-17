"""Makes a test microphone track: your lines (a 7th computer voice) mixed with the call leaking in from your
speakers/headphones, plus a matching call track with quiet moments. Used by echotest.py to check that echo of
the call is ignored and your own lines are kept.  Needs the same tools as make_test_call.py and test/conversation.*
"""
import sherpa_onnx, numpy as np, soundfile as sf, json, random, soxr
from scipy.signal import butter, sosfilt
np.random.seed(3)
d = "tts/vits-piper-en_US-libritts_r-medium"
tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
    vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=f"{d}/en_US-libritts_r-medium.onnx", tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data"), num_threads=4)))
call, sr = sf.read("test/conversation.wav", dtype="float32", always_2d=True); call = soxr.resample(call.mean(1), sr, 16000).astype(np.float32)
tr = json.load(open("test/conversation.json"))
def say(t): x = np.asarray(tts.generate(t, sid=250).samples, np.float32); return soxr.resample(x, 22050, 16000).astype(np.float32) * 0.9
quiet_lines = ["I cast shield as a reaction.", "My wizard checks the bookshelf for anything magical.", "How much gold do we have left?",
               "I run to the door and try to lock it.", "I think we should rest here tonight."]
over_lines = ["Can I see the map again?", "Twenty three to hit.", "I'll hold my action until the goblin comes closer."]
# insert a pause in the call after utterances 6, 18, 30, 42, 54 for my quiet lines
cuts = [tr["utts"][k]["end"] + 0.15 for k in (6, 18, 30, 42, 54)]
pieces_call, pieces_me, me, newtr, shift, prev = [], [], [], [], 0.0, 0
for k, c in enumerate(cuts):
    s0 = int(c * 16000); pieces_call.append(call[prev:s0]); pieces_me.append(np.zeros(s0 - prev, np.float32)); prev = s0
    x = say(quiet_lines[k]); pad = np.zeros(int(0.6 * 16000), np.float32)
    seg = np.concatenate([pad, x, pad])
    me.append({"start": round(c + shift + 0.6, 2), "end": round(c + shift + 0.6 + len(x) / 16000, 2), "text": quiet_lines[k], "kind": "quiet"})
    pieces_call.append(np.zeros(len(seg), np.float32)); pieces_me.append(seg); shift += len(seg) / 16000
pieces_call.append(call[prev:]); pieces_me.append(np.zeros(len(call) - prev, np.float32))
call2 = np.concatenate(pieces_call); mine = np.concatenate(pieces_me)
# lines over other people
for t, text in zip((40.0, 95.0, 150.0), over_lines):
    x = say(text); s0 = int(t * 16000); mine[s0:s0 + len(x)] += x
    me.append({"start": t, "end": round(t + len(x) / 16000, 2), "text": text, "kind": "over"})
sos = butter(4, [200, 5000], btype="band", fs=16000, output="sos")
echo = np.concatenate([np.zeros(480, np.float32), sosfilt(sos, call2).astype(np.float32) * 0.35])[:len(call2)]
mic = echo + mine + np.random.randn(len(call2)).astype(np.float32) * 0.003
sf.write("test/call2.flac", np.clip(call2, -1, 1), 16000); sf.write("test/mic.flac", np.clip(mic, -1, 1), 16000)
json.dump(me, open("test/mic.json", "w"), indent=1); print(len(call2) / 16000, me)
