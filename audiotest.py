"""Lists what the app can listen to and records a few seconds from one of them.
  audiotest.py                      -> the device chosen in the app (or automatic)
  audiotest.py "part of a name"     -> a playback device (what it plays)
  audiotest.py "rec:part of a name" -> a recording device (e.g. rec:CABLE Output)
  audiotest.py mic                  -> your microphone"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, engine
print("Things the app can listen to:")
for value, label in engine.list_listen_sources(): print(f"  {label}")
arg = sys.argv[1] if len(sys.argv) > 1 else engine.load_config().get("audio_device", "auto")
src = engine.AudioSource(engine.load_config().get("mic_device", "auto"), mic=True) if arg == "mic" else engine.AudioSource(arg)
t = time.time(); got = []
while time.time() - t < 6:
    try: got.append(src.q.get(timeout=0.5)[1])
    except Exception: pass
src.close()
x = np.concatenate(got) if got else np.zeros(0)
loud = 20 * np.log10(np.abs(x).max() + 1e-9) if len(x) else -999
print(f"captured {len(x)/16000:.1f} s from '{src.name}', loudest {loud:.1f} dBFS"
      + ("   <- silent: nothing is playing through this one" if len(x) == 0 or loud < -60 else ""))
