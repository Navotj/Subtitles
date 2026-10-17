"""Lists the playback devices that can be listened to and records a few seconds from the chosen one."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, engine
print("Devices that can be listened to:")
for i, n in engine.list_output_devices(): print(f"  [{i}] {n}")
mic = len(sys.argv) > 1 and sys.argv[1] == "mic"
if mic:
    print("Microphones:")
    for i, n in engine.list_input_devices(): print(f"  [{i}] {n}")
    src = engine.AudioSource(engine.load_config().get("mic_device", "auto"), mic=True)
else:
    want = sys.argv[1] if len(sys.argv) > 1 else engine.load_config().get("audio_device", "auto")
    src = engine.AudioSource(want)
t = time.time(); got = []
while time.time() - t < 6:
    try: got.append(src.q.get(timeout=0.5))
    except Exception: pass
src.close()
x = np.concatenate(got) if got else np.zeros(0)
print(f"captured {len(x)/16000:.1f} s at 16 kHz from '{src.name}', loudest {20*np.log10(np.abs(x).max()+1e-9) if len(x) else -999:.1f} dBFS")
