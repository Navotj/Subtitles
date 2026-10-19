"""Which AI models the app uses, what each one needs, and how to download and remove them.
No Qt in here, so the installer and the tests can use it too.

    python models.py                 show this computer and the recommended choice
    python models.py install         download what config.json asks for (or the recommended models on a first install)
    python models.py check           check that every download link in the list still works (needs internet)"""
import os, sys, json, glob, shutil, subprocess, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"

# Speech to text (Whisper, run by faster-whisper). langs: "en" = English only, "all" = every language Whisper knows.
SPEECH = [
    dict(id="small.en", label="Small, English only", langs="en", gpu=False, mb=500,
         desc="English only. Runs on any computer, no graphics card needed. About 0.5 GB. Fastest, least accurate."),
    dict(id="small", label="Small, all languages", langs="all", gpu=False, mb=500,
         desc="All languages. Runs on any computer, no graphics card needed. About 0.5 GB. Basic accuracy."),
    dict(id="large-v3-turbo", label="Large v3 turbo", langs="all", gpu=True, mb=1600,
         desc="All languages. Needs an NVIDIA graphics card (about 3 GB of its memory). About 1.6 GB. "
              "High GPU use. Very accurate and fast: the normal choice with an NVIDIA card."),
    dict(id="large-v3", label="Large v3", langs="all", gpu=True, mb=3100,
         desc="All languages. Needs an NVIDIA graphics card (about 5 GB of its memory). About 3.1 GB. "
              "Highest GPU use, about twice as slow as turbo. Most accurate."),
]

# Voice fingerprints, to tell who is talking. Runs on the processor.
SPEAKER = [
    dict(id="nemo_en_titanet_small.onnx", label="TitaNet small", mb=40,
         desc="Any computer, very light. About 40 MB. Good for most calls."),
    dict(id="nemo_en_titanet_large.onnx", label="TitaNet large", mb=100,
         desc="Any computer, a little more processor use. About 100 MB. Tells similar voices apart better."),
]

# Whisper language codes (Latin alphabet only for now)
LANGUAGES = [("en", "English"), ("de", "German"), ("es", "Spanish"), ("fr", "French"),
             ("it", "Italian"), ("pt", "Portuguese"), ("nl", "Dutch")]


def by_id(table, id):
    return next((m for m in table if m["id"] == id), None)


def detect_gpu():
    """-> (name, memory in MB) of the first NVIDIA graphics card, or None"""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if r.returncode == 0 and r.stdout.strip():
            name, mem = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")[:2]]
            return name, int(float(mem))
    except Exception:
        pass
    return None


def recommended_speech(gpu, languages=("en",)):
    english = list(languages) == ["en"]
    if gpu and gpu[1] >= 6000: return "large-v3-turbo"
    return "small.en" if english else "small"


def supports(model, languages):
    """can this speech model transcribe all of these languages?"""
    return model["langs"] == "all" or list(languages) == ["en"]


# ------------------------------------------------------------------------------------------ files on disk
def speech_dirs(id):
    """the Whisper download folder(s) for a model, e.g. models/models--Systran--faster-whisper-small"""
    return [d for d in glob.glob(os.path.join(MODELS, "models--*--*faster-whisper-" + id)) if os.path.isdir(d)]


def speech_installed(id):
    return any(glob.glob(os.path.join(d, "snapshots", "*", "model.bin")) for d in speech_dirs(id))


def speaker_installed(id):
    return os.path.isfile(os.path.join(MODELS, id))


def _size(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try: total += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return total


def download_speech(id, progress=None):
    """progress(fraction 0..1) is called while it downloads"""
    from faster_whisper.utils import download_model
    os.makedirs(MODELS, exist_ok=True)
    m = by_id(SPEECH, id); done = threading.Event()
    total = (m["mb"] if m else 1000) * 1e6
    base = sum(_size(d) for d in glob.glob(os.path.join(MODELS, "models--*")))
    def watch():
        while not done.wait(0.7):
            got = sum(_size(d) for d in glob.glob(os.path.join(MODELS, "models--*"))) - base
            progress(max(0.0, min(0.99, got / total)))
    if progress: threading.Thread(target=watch, daemon=True).start()
    try:
        download_model(id, cache_dir=MODELS)
    finally:
        done.set()
    if progress: progress(1.0)


def download_speaker(id, progress=None):
    os.makedirs(MODELS, exist_ok=True)
    dst = os.path.join(MODELS, id); tmp = dst + ".part"
    req = urllib.request.Request(SHERPA + id, headers={"User-Agent": "subtitles-models"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0); got = 0
        while True:
            chunk = r.read(1 << 16)
            if not chunk: break
            f.write(chunk); got += len(chunk)
            if progress and total: progress(got / total)
    os.replace(tmp, dst)
    if progress: progress(1.0)


def delete_speech(id):
    for d in speech_dirs(id): shutil.rmtree(d, ignore_errors=True)


def delete_speaker(id):
    try: os.remove(os.path.join(MODELS, id))
    except OSError: pass


def ensure(cfg, progress=None, say=None):
    """download whatever the settings ask for and is missing"""
    say = say or (lambda t: None)
    s = cfg.get("whisper_model", "large-v3-turbo"); v = cfg.get("voice_model", "nemo_en_titanet_small.onnx")
    if not speech_installed(s):
        say(f"Downloading the speech model ({s})..."); download_speech(s, progress)
    if not speaker_installed(v):
        say(f"Downloading the voice model ({v})..."); download_speaker(v, progress)


def download_changes(old, new, progress=None, say=None):
    """Download every model that changed and is not on disk yet.
    old / new: dicts with whisper_model and voice_model. -> [(kind, id)] of the old models to remove later.
    (A model that is open in the running app cannot be deleted on Windows, so cleanup() removes them at the next start.)"""
    say = say or (lambda t: None); stale = []
    for kind, key, have, get in (("speech", "whisper_model", speech_installed, download_speech),
                                 ("speaker", "voice_model", speaker_installed, download_speaker)):
        if old.get(key) != new.get(key):
            if not have(new[key]):
                say(f"Downloading {new[key]}..."); get(new[key], progress)
            if old.get(key): stale.append((kind, old[key]))
    return stale


def cleanup(cfg):
    """delete the models listed in cfg["models_to_delete"] (unless they are the ones in use again); -> what was removed"""
    keep = {cfg.get("whisper_model"), cfg.get("voice_model")}; gone = []
    for kind, id in cfg.get("models_to_delete") or []:
        if id in keep: continue
        (delete_speech if kind == "speech" else delete_speaker)(id); gone.append(id)
    cfg["models_to_delete"] = []
    return gone


def check_links():
    """HEAD every speaker model link, to catch a renamed file; returns [(id, ok, size_mb or error)]"""
    out = []
    for m in SPEAKER:
        try:
            req = urllib.request.Request(SHERPA + m["id"], method="HEAD", headers={"User-Agent": "subtitles-models"})
            with urllib.request.urlopen(req, timeout=30) as r: out.append((m["id"], True, round(int(r.headers.get("Content-Length", 0)) / 1e6, 1)))
        except Exception as e:
            out.append((m["id"], False, str(e)))
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    gpu = detect_gpu()
    cfgp = os.path.join(HERE, "config.json")
    try: cfg = json.load(open(cfgp, encoding="utf-8"))
    except Exception: cfg = {}
    langs = cfg.get("languages") or ["en"]
    if cmd == "check":
        for r in check_links(): print(*r)
    elif cmd == "install":
        if "whisper_model" not in cfg:
            cfg["whisper_model"] = recommended_speech(gpu, langs)
            json.dump(cfg, open(cfgp, "w", encoding="utf-8"), indent=2)       # remembered; the app adds the other settings
            print("chosen for this computer:", cfg["whisper_model"])
        ensure(cfg, say=print); print("models ready")
    else:
        print("graphics card:", f"{gpu[0]} ({gpu[1] // 1024} GB)" if gpu else "no NVIDIA card found")
        print("recommended speech model:", recommended_speech(gpu, langs))
