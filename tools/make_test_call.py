"""Makes a synthetic 6-person test call (computer voices, different "bad mic" effects, Discord-like Opus codec)
with a truth file of who said what. Used to tune and check voice recognition without real recordings.
Needs: pip install sherpa-onnx soundfile scipy  +  ffmpeg  +  the Piper LibriTTS voice model unpacked in ./tts/
  (https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2)
Usage: python make_test_call.py <out name> [seed] [number of lines]   ->  <out>.wav + <out>.json
"""
import sherpa_onnx, numpy as np, soundfile as sf, json, random, subprocess, os, sys
from scipy.signal import butter, sosfilt, resample_poly
random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 7); np.random.seed(int(sys.argv[2]) if len(sys.argv) > 2 else 7)
out = sys.argv[1]
d = "tts/vits-piper-en_US-libritts_r-medium"
tts = sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=sherpa_onnx.OfflineTtsModelConfig(
    vits=sherpa_onnx.OfflineTtsVitsModelConfig(model=f"{d}/en_US-libritts_r-medium.onnx", tokens=f"{d}/tokens.txt", data_dir=f"{d}/espeak-ng-data"),
    num_threads=4)))
names = ["Dana", "Omer", "Yael", "Itai", "Noa", "Gil"]
sids = random.sample(range(0, 900), 6)
lines = [
 "Okay, so you walk into the tavern and the barkeep looks up at you.",
 "I want to roll for perception.", "That's a seventeen.", "Can I cast detect magic on the chest?",
 "Wait, are we sure this isn't a trap?", "I attack the goblin with my longsword.", "Natural twenty!",
 "The dragon breathes fire across the bridge. Everyone make a dexterity saving throw.",
 "I failed. I take how much damage?", "Twenty eight points of fire damage.", "Yeah.", "No way.",
 "I'll use my bonus action to drink a healing potion.", "Does anyone have a rope?",
 "I think we should go back to the village and talk to the old wizard first.",
 "Sure, let's do that.", "Hold on, I need to check my spell slots.", "Roll initiative.",
 "I got a twelve.", "Nineteen.", "My character would never trust that merchant.",
 "The door creaks open and you see a long hallway lit by torches.",
 "I sneak ahead and look for traps.", "Okay.", "I'm casting fireball at the group of cultists.",
 "How many are in the radius?", "Four of them, and they're all burning now.",
 "Can I persuade the guard to let us through?", "Give me a charisma check.", "Eight. Not great.",
 "The guard laughs at you and points his spear at your chest.", "Alright, I punch him.",
]
dm = 0  # Dana talks most (DM)
utts = []
t = 1.0
sr_out = 16000
mics = []
for k in range(6):  # different mic per person: band limit + gain + hum/noise level
    lo = random.choice([80, 150, 300]); hi = random.choice([3400, 5000, 7000, 7600])
    mics.append((lo, hi, random.uniform(0.5, 1.3), random.uniform(0.002, 0.012)))
N = int(sys.argv[3]) if len(sys.argv) > 3 else 60
total = np.zeros(int(16000 * (N * 6 + 30)), dtype=np.float32)
for i in range(N):
    spk = dm if random.random() < 0.3 else random.randrange(6)
    text = random.choice(lines)
    a = tts.generate(text, sid=sids[spk], speed=random.uniform(0.95, 1.15))
    x = np.asarray(a.samples, dtype=np.float32); sr = a.sample_rate
    x = resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x
    lo, hi, g, nz = mics[spk]
    sos = butter(4, [lo, min(hi, 7900)], btype="band", fs=16000, output="sos")
    x = sosfilt(sos, x).astype(np.float32) * g
    x = x + np.random.randn(len(x)).astype(np.float32) * nz
    gap = random.choice([0.15, 0.25, 0.4, 0.7, 1.0, 1.5])
    s0 = int(t * 16000)
    total[s0:s0 + len(x)] += x
    utts.append({"start": round(t, 3), "end": round(t + len(x) / 16000, 3), "speaker": names[spk], "text": text})
    t += len(x) / 16000 + gap
total = total[:int((t + 1) * 16000)]
total = np.clip(total, -1, 1)
sf.write(out + ".raw.wav", total, 16000)
# Discord-like codec: opus 32 kbps
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".raw.wav", "-c:a", "libopus", "-b:a", "32k", out + ".opus"], check=True)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out + ".opus", "-ar", "48000", "-ac", "2", out + ".wav"], check=True)
os.remove(out + ".raw.wav"); os.remove(out + ".opus")
json.dump({"names": names, "sids": sids, "utts": utts}, open(out + ".json", "w"), indent=1)
print("done", round(t, 1), "s", len(utts), "utts", sids)
