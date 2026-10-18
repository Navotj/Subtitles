"""
Core of D&D Subtitles: voice detection, speech-to-text (Whisper on the GPU) and telling the voices apart.
No UI code here, so it can be tested on its own (selftest.py).
"""
import os, sys, json, time, threading, queue, re, datetime, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = os.path.join(HERE, "groups")
MY_VOICE = os.path.join(HERE, "my_voice.json")


def group_path(name):
    return os.path.join(GROUPS, re.sub(r'[<>:"/\\|?*]', "_", name).strip() + ".json")


def list_groups():
    os.makedirs(GROUPS, exist_ok=True)
    return sorted(os.path.splitext(f)[0] for f in os.listdir(GROUPS) if f.endswith(".json"))
MODELS = os.path.join(HERE, "models")
SR = 16000
FRAME = 512                     # 32 ms, what Silero VAD wants

# ----------------------------------------------------------------------------------------------- config
DEFAULT_CONFIG = {
    "audio_device": "auto",            # "auto" = NVIDIA Broadcast speakers if present, else Windows default output
    "whisper_model": "large-v3-turbo",
    "language": "en",
    "vocabulary": "",                  # names and words Whisper should expect, e.g. "Thalindra, Neverwinter, Eldritch Blast"
    "show_partial": True,              # show text while someone is still talking
    # new voices are only ever added by you (right-click a "?" line); "auto_new_voices" is for the self-tests only
    "max_voices": 8,
    "font_size": 32,
    "max_lines": 3,
    "line_seconds": 9,
    "overlay_width_pct": 75,
    "overlay_x_px": 0,                 # 0 = centred on the screen; + moves right, - moves left
    "overlay_y_px": 0,                 # 0 = near the bottom of the screen; + moves down, - moves up
    "keep_transcript": True,           # False: nothing is written to files; the transcript window only shows the last minute
    "transient_seconds": 60,           # how long lines stay in the window when "keep transcript" is off
    "session_auto_resume_min": 5,      # closed for less than this: carry on in the same transcript automatically
    "session_ask_resume_min": 60,      # closed for less than this (but longer than the above): ask; longer: new transcript
    "screen": 0,
    "background_opacity": 0.55,
    # voice matching (cosine similarity of voice fingerprints)
    "voice_model": "nemo_en_titanet_small.onnx",
    "match_threshold": 0.58,           # at least this similar -> same person (for 1.2 s+ of speech)
    "short_match_threshold": 0.42,     # same, for short replies ("yeah", "okay") which are harder to tell apart
    "new_voice_threshold": 0.50,       # less similar than this to everyone -> a new person
    "update_threshold": 0.62,          # this sure -> also refine that person's fingerprint
    "min_new_voice_seconds": 1.4,
    "change_threshold": 0.35,          # voice before/after a point this different -> a different person took over
    "max_segment_seconds": 10,
    "group": "Group 1",                # which D&D group's voices are in use (groups\\<name>.json)
    "word_fixes": [],                  # [["heard as", "replace with"], ...] (kept per group)
    "hide_short_unknown": True,
    "echo_threshold": 0.5,             # mic audio that follows the call this closely is echo, not you        # transcript: hide 1-2 word lines nobody was recognised for
    "my_voice": True,                  # add your own microphone to the transcript (not the subtitles)
    "my_name": "",                     # empty = your Windows user name
    "my_color": "#FF6B6B",
    "mic_device": "auto",              # "auto" = NVIDIA Broadcast microphone if present, else Windows default
}

def apply_fixes(text, fixes):
    """word fixes: every [heard, replace] pair, whole words/phrases, any capitalisation"""
    if not text or not fixes: return text
    for a, b in sorted((f for f in fixes if f and f[0].strip()), key=lambda f: -len(f[0])):
        text = re.sub(r"(?<![\w'])" + re.escape(a.strip()) + r"(?!\w)", b.strip(), text, flags=re.I)   # "…'s" still matches
    return text


def load_config():
    p = os.path.join(HERE, "config.json")
    cfg = dict(DEFAULT_CONFIG)
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f: cfg.update(json.load(f))
        except Exception as e:
            print(f"[config] could not read config.json ({e}); using defaults", flush=True)
    if not cfg.get("my_name"): cfg["my_name"] = default_my_name()
    return cfg

def default_my_name():
    n = (os.environ.get("USERNAME") or os.environ.get("USER") or "").strip()
    return n[:1].upper() + n[1:] if n else "Me"


def save_config(cfg):
    p = os.path.join(HERE, "config.json"); tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(cfg, f, indent=2)
    os.replace(tmp, p)

def log(*a):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}]", *a, flush=True)

# ----------------------------------------------------------------------------------------------- CUDA DLLs (Windows)
def add_cuda_dll_dirs():
    """faster-whisper (CTranslate2) needs cuBLAS / cuDNN; they come from the nvidia-* pip packages."""
    if os.name != "nt": return
    import site
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for r in roots:
        for b in glob.glob(os.path.join(r, "nvidia", "*", "bin")):
            try: os.add_dll_directory(b)
            except Exception: pass
            os.environ["PATH"] = b + os.pathsep + os.environ.get("PATH", "")

# ----------------------------------------------------------------------------------------------- voices
PALETTE = ["#FFD54F", "#4FC3F7", "#81C784", "#F48FB1", "#CE93D8", "#FF8A65", "#80CBC4", "#E6EE9C",
           "#90CAF9", "#FFAB91", "#B39DDB", "#A5D6A7"]
UNKNOWN_COLOR = "#D0D0D0"

class VoiceBank:
    """The known voices: name, colour and a voice fingerprint (average speaker embedding)."""
    MAX_PROTOS = 8

    def __init__(self, path=None, model=""):
        self.path = path or os.path.join(HERE, "voices.json")
        self.model = model
        self.lock = threading.RLock()
        self.voices = {}            # id -> dict(id, name, color, centroid np, count, last_text, last_heard)
        self.next_id = 1
        self.aliases = {}           # merged-away id -> id it was merged into (for lines already on screen)
        self.me = []                # your own voice fingerprints (from your mic, used to ignore echo of the call)
        self.me_counts = []
        self.extra = {}             # group settings kept with the voices: words to expect, word fixes
        self.on_change = None       # callback(kind) kind in {"added","removed","updated"}
        self._dirty = False
        self.load()

    def load_me(self):
        try:
            with open(MY_VOICE, encoding="utf-8") as f: d = json.load(f)
            if d.get("model") == self.model and d.get("me_version") == 3:
                self.me = [np.asarray(p, dtype=np.float32) for p in d.get("me", [])]
                self.me_counts = d.get("me_counts", [1] * len(self.me))
        except Exception:
            pass

    def save_me(self):
        tmp = MY_VOICE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"model": self.model, "me_version": 3, "me": [p.tolist() for p in self.me],
                       "me_counts": self.me_counts}, f)
        os.replace(tmp, MY_VOICE)

    def switch(self, path):
        """use another group's voices (your own voice stays)"""
        self.save(force=True)
        with self.lock:
            self.voices = {}; self.aliases = {}; self.next_id = 1; self.path = path; self.extra = {}
        self.load()
        self._changed("removed")

    def load(self):
        self.load_me()
        if not os.path.exists(self.path): return
        try:
            with open(self.path, encoding="utf-8") as f: d = json.load(f)
            if self.model and d.get("model") and d["model"] != self.model:
                bak = self.path + "." + d["model"] + ".bak"
                os.replace(self.path, bak)
                log(f"[voices] voices.json was made with another voice model; moved it to {os.path.basename(bak)}")
                return
            if d.get("me_version") == 3 and not os.path.exists(MY_VOICE):   # (older files kept you in here)
                self.me = [np.asarray(p, dtype=np.float32) for p in d.get("me", [])]
                self.me_counts = d.get("me_counts", [1] * len(self.me))
            self.extra = {k: d[k] for k in ("vocabulary", "word_fixes") if k in d}
            for v in d.get("voices", []):
                if "protos" in v:
                    v["protos"] = [np.asarray(p, dtype=np.float32) for p in v["protos"]]
                else:
                    v["protos"] = [np.asarray(v.pop("centroid"), dtype=np.float32)]
                v.setdefault("pcounts", [v.get("count", 1)] * len(v["protos"]))
                v.setdefault("last_text", ""); v.setdefault("last_heard", 0)
                self.voices[v["id"]] = v
            self.next_id = d.get("next_id", max(self.voices, default=0) + 1)
        except Exception as e:
            log(f"[voices] could not read voices.json: {e}")

    def save(self, force=False):
        with self.lock:
            if not (self._dirty or force): return
            self.save_me()
            d = {"model": self.model, "next_id": self.next_id, **self.extra, "voices": [
                {k: ([p.tolist() for p in v[k]] if k == "protos" else v[k]) for k in
                 ("id", "name", "color", "protos", "pcounts", "count", "last_text", "last_heard")}
                for v in self.voices.values()]}
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f: json.dump(d, f)
            os.replace(tmp, self.path); self._dirty = False

    def _changed(self, kind):
        self._dirty = True
        if self.on_change:
            try: self.on_change(kind)
            except Exception: pass

    def _free_color(self):
        used = {v["color"].upper() for v in self.voices.values()}
        for c in PALETTE:
            if c.upper() not in used: return c
        return PALETTE[len(self.voices) % len(PALETTE)]

    def add(self, emb, name=None, count=1):
        with self.lock:
            vid = self.next_id; self.next_id += 1
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if name is None:
                used = {v["name"] for v in self.voices.values()}
                n = 0
                while f"Voice {letters[n % 26]}{'' if n < 26 else n // 26}" in used: n += 1
                name = f"Voice {letters[n % 26]}{'' if n < 26 else n // 26}"
            self.voices[vid] = {"id": vid, "name": name, "color": self._free_color(),
                                "protos": [(emb / (np.linalg.norm(emb) + 1e-9)).astype(np.float32)], "pcounts": [count],
                                "count": count, "last_text": "", "last_heard": time.time()}
            log(f"[voices] new voice: {name}")
        self._changed("added")
        return vid

    def classify(self, emb):
        """-> list of (similarity, id) best first; a voice's similarity is its best-matching fingerprint"""
        with self.lock:
            out = [(max(float(p @ emb) for p in v["protos"]), vid) for vid, v in self.voices.items()]
        return sorted(out, reverse=True)

    def refine(self, vid, emb, weight=1.0, cap=150):
        with self.lock:
            v = self.voices.get(vid)
            if v is None: return
            k = int(np.argmax([p @ emb for p in v["protos"]]))
            n = min(v["pcounts"][k], cap)
            c = v["protos"][k] * n + emb * weight
            v["protos"][k] = (c / (np.linalg.norm(c) + 1e-9)).astype(np.float32)
            v["pcounts"][k] += 1; v["count"] += 1
            self._dirty = True

    def teach(self, vid, emb):
        """you said this line was <vid>: fold it into their fingerprint, or keep it as another way they sound"""
        emb = np.asarray(emb, dtype=np.float32); emb = emb / (np.linalg.norm(emb) + 1e-9)
        with self.lock:
            v = self.voices.get(vid)
            if v is None: return
            best = max(float(p @ emb) for p in v["protos"])
        if best >= 0.55: self.refine(vid, emb, weight=1.5)
        else: self.add_proto(vid, emb)
        self._changed("updated")

    def add_proto(self, vid, emb):
        """Same person, but sounding different (e.g. the DM doing a character voice)."""
        with self.lock:
            v = self.voices.get(vid)
            if v is None: return
            v["protos"].append(emb.astype(np.float32)); v["pcounts"].append(1)
            if len(v["protos"]) > self.MAX_PROTOS:          # drop the least used
                k = int(np.argmin(v["pcounts"])); v["protos"].pop(k); v["pcounts"].pop(k)
            self._dirty = True

    def heard(self, vid, text):
        with self.lock:
            v = self.voices.get(vid)
            if v is None: return
            v["last_text"] = text[-120:]; v["last_heard"] = time.time(); self._dirty = True
        self._changed("updated")

    def rename(self, vid, name):
        with self.lock:
            if vid in self.voices and name.strip(): self.voices[vid]["name"] = name.strip()
        self._changed("updated")

    def recolor(self, vid, color):
        with self.lock:
            if vid in self.voices: self.voices[vid]["color"] = color
        self._changed("updated")

    def remove(self, vid):
        with self.lock: self.voices.pop(vid, None)
        self._changed("removed")

    def merge(self, src, dst):
        """src becomes part of dst (dst keeps its name/colour, and now also recognises src's way of talking)."""
        with self.lock:
            a, b = self.voices.get(src), self.voices.get(dst)
            if not a or not b or src == dst: return
            b["protos"] += a["protos"]; b["pcounts"] += a["pcounts"]; b["count"] += a["count"]
            while len(b["protos"]) > self.MAX_PROTOS:
                k = int(np.argmin(b["pcounts"])); b["protos"].pop(k); b["pcounts"].pop(k)
            self.voices.pop(src)
            self.aliases[src] = dst
            for k, t in list(self.aliases.items()):
                if t == src: self.aliases[k] = dst
        self._changed("removed")

    def me_sim(self, emb):
        with self.lock:
            return max((float(p @ emb) for p in self.me), default=-1.0)

    def me_learn(self, emb, cap=150):
        with self.lock:
            if self.me and max(float(p @ emb) for p in self.me) >= 0.55:
                k = int(np.argmax([p @ emb for p in self.me])); n = min(self.me_counts[k], cap)
                c = self.me[k] * n + emb; self.me[k] = (c / (np.linalg.norm(c) + 1e-9)).astype(np.float32)
                self.me_counts[k] += 1
            elif len(self.me) < 4:
                self.me.append(emb.astype(np.float32)); self.me_counts.append(1)
            self._dirty = True

    def me_forget(self):
        with self.lock: self.me, self.me_counts = [], []; self._dirty = True

    def closest_other(self, vid):
        """-> (similarity, id) of the other voice that sounds most like this one (merge hint)"""
        with self.lock:
            v = self.voices.get(vid)
            if not v: return (0.0, None)
            best = (0.0, None)
            for oid, o in self.voices.items():
                if oid == vid: continue
                s = max(float(p @ q) for p in v["protos"] for q in o["protos"])
                if s > best[0]: best = (s, oid)
            return best

    def resolve(self, vid):
        return self.aliases.get(vid, vid)

    def get(self, vid):
        with self.lock:
            v = self.voices.get(self.aliases.get(vid, vid))
            return (v["name"], v["color"]) if v else ("?", UNKNOWN_COLOR)

    def snapshot(self):
        with self.lock:
            return [{k: v[k] for k in ("id", "name", "color", "count", "last_text", "last_heard")}
                    for v in sorted(self.voices.values(), key=lambda v: v["id"])]

# ----------------------------------------------------------------------------------------------- models
class Embedder:
    """Voice fingerprints (NVIDIA TitaNet, runs on the CPU via sherpa-onnx)."""
    def __init__(self, name="nemo_en_titanet_small.onnx"):
        import sherpa_onnx
        path = name if os.path.isabs(name) else os.path.join(MODELS, name)
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=path, num_threads=2, provider="cpu")
        if not cfg.validate(): raise RuntimeError(f"bad speaker model {path}")
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def __call__(self, audio):
        s = self.ex.create_stream()
        s.accept_waveform(SR, np.ascontiguousarray(audio, dtype=np.float32))
        s.input_finished()
        e = np.asarray(self.ex.compute(s), dtype=np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


class Transcriber:
    """Whisper (faster-whisper / CTranslate2) on the GPU, CPU fallback."""
    HALLUCINATIONS = {"thank you.", "thanks for watching!", "thanks for watching.", "thank you for watching.",
                      "thank you for watching!", "please subscribe.", "bye.", "you", "you.", "subtitles by the amara.org community",
                      "thank you very much.", "i'll see you next time.", "see you next time.", "."}

    def __init__(self, cfg):
        add_cuda_dll_dirs()
        from faster_whisper import WhisperModel
        self.cfg = cfg; self.device = "cuda"
        name = cfg.get("whisper_model", "large-v3-turbo")
        try:
            self.model = WhisperModel(name, device="cuda", compute_type="float16", download_root=MODELS)
            list(self.model.transcribe(np.zeros(SR, dtype=np.float32), language=cfg.get("language") or None,
                                       beam_size=1)[0])                       # warm-up; fails here if CUDA is broken
        except Exception as e:
            log(f"[whisper] GPU failed ({e}); falling back to CPU with the 'small.en' model (slower, less accurate)")
            self.device = "cpu"
            self.model = WhisperModel("small.en", device="cpu", compute_type="int8", download_root=MODELS)
        self.name = name if self.device == "cuda" else "small.en"
        log(f"[whisper] {self.name} ready on {self.device.upper()}")

    def vocab_terms(self):
        seen, out = set(), []
        fix_words = [f[1] for f in (self.cfg.get("word_fixes") or []) if len(f) > 1 and f[1].strip()]
        for t in re.split(r"[,;\n]+", self.cfg.get("vocabulary") or "") + fix_words:
            t = t.strip()
            if t and t.lower() not in seen: seen.add(t.lower()); out.append(t)
        return out

    def _opts(self, vocab=False):
        terms = self.vocab_terms() if vocab else []
        return dict(language=self.cfg.get("language") or None, condition_on_previous_text=False, vad_filter=False,
                    no_speech_threshold=0.6, log_prob_threshold=-1.0, compression_ratio_threshold=2.4,
                    hotwords=", ".join(terms) or None)

    def _clean(self, segs, dur):
        out = []
        for s in segs:
            t = s.text.strip()
            if not t: continue
            if s.no_speech_prob > 0.75 and s.avg_logprob < -0.6: continue
            if t.lower() in self.HALLUCINATIONS and (dur < 2.5 or s.no_speech_prob > 0.3): continue
            out.append(s)
        return out

    @staticmethod
    def repetitive(text):
        """same 2-4 words over and over (a typical Whisper failure)"""
        w = re.findall(r"[\w']+", text.lower())
        for n in (1, 2, 3, 4):
            grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
            if grams and max(grams.count(g) for g in set(grams)) >= (5 if n == 1 else 3): return True
        return False

    def leaked(self, text):
        """True when the words-to-expect list ended up in the text instead of what was said."""
        terms = [t.lower() for t in self.vocab_terms()]
        if not terms or not text: return False
        low = text.lower(); n_words = max(len(re.findall(r"[\w']+", low)), 1)
        hits, distinct, hit_words = 0, 0, 0
        for t in terms:
            c = len(re.findall(r"(?<![\w'])" + re.escape(t) + r"(?![\w'])", low))
            hits += c; distinct += c > 0; hit_words += c * len(t.split())
        if hits >= 3 or (distinct >= 2 and hit_words / n_words > 0.35): return True
        for t1 in terms:                           # two list entries back to back: "Mimir, Commendatori"
            for t2 in terms:
                if t1 != t2 and re.search(re.escape(t1) + r"\s*[,.;]\s*" + re.escape(t2), low): return True
        return self.repetitive(text)

    _lock = threading.Lock()

    def _run(self, audio, vocab, **kw):
        with self._lock:                      # the call and your mic share one Whisper
            segs, _ = self.model.transcribe(audio, **kw, **self._opts(vocab))
            segs = list(segs)
        return self._clean(segs, len(audio) / SR)

    def partial(self, audio):
        # never with the word list: short in-progress clips are where Whisper copies it into the text
        segs = self._run(audio, False, beam_size=1, temperature=0.0, without_timestamps=True)
        text = " ".join(s.text.strip() for s in segs).strip()
        return "" if self.repetitive(text) else text

    def final(self, audio):
        """-> list of (start_s, end_s, word_text)"""
        kw = dict(beam_size=5, temperature=[0.0, 0.2, 0.4], word_timestamps=True)
        use_vocab = bool(self.vocab_terms()) and len(audio) / SR >= 1.5
        segs = self._run(audio, use_vocab, **kw)
        if use_vocab and self.leaked(" ".join(s.text for s in segs)):
            self.leaks = getattr(self, "leaks", 0) + 1
            segs = self._run(audio, False, **kw)              # redo without the list
        if self.repetitive(" ".join(s.text for s in segs)): return []
        words = []
        for s in segs:
            if s.words:
                words += [(w.start, w.end, w.word) for w in s.words]
            else:
                words.append((s.start, s.end, " " + s.text.strip()))
        return words

# ----------------------------------------------------------------------------------------------- utterance cutting
class Segmenter:
    """Cuts the audio stream into utterances with Silero VAD. Yields ('start'|'grow'|'end', utt) events."""
    PRE = 0.30          # seconds of audio kept before speech start
    TAIL = 0.25         # seconds kept after speech end

    def __init__(self, max_seconds=14):
        import sherpa_onnx
        c = sherpa_onnx.VadModelConfig()
        c.silero_vad.model = os.path.join(MODELS, "silero_vad.onnx")
        c.silero_vad.threshold = 0.45
        c.silero_vad.min_silence_duration = 0.35
        c.silero_vad.min_speech_duration = 0.20
        c.silero_vad.window_size = FRAME
        c.silero_vad.max_speech_duration = 60
        c.sample_rate = SR; c.num_threads = 1
        self.vad = sherpa_onnx.VadModel.create(c)
        self.max_len = int(max_seconds * SR)
        self.pending = np.zeros(0, dtype=np.float32)
        self.ring = np.zeros(int(1.2 * SR), dtype=np.float32)      # recent history
        self.n = 0                     # samples consumed so far (absolute clock)
        self.active = False
        self.buf = []                  # frames of the current utterance
        self.t0 = 0
        self.last_speech_n = 0

    def _ring_push(self, fr):
        self.ring = np.concatenate([self.ring[len(fr):], fr])

    def feed(self, x):
        ev = []
        self.pending = np.concatenate([self.pending, x])
        while len(self.pending) >= FRAME:
            fr, self.pending = self.pending[:FRAME], self.pending[FRAME:]
            sp = self.vad.is_speech(fr)
            self._ring_push(fr); self.n += FRAME
            if sp: self.last_speech_n = self.n
            if not self.active and sp:
                back = int((0.20 + 0.032 + self.PRE) * SR)           # min_speech + this frame + pre-roll
                back = min(back, len(self.ring), self.n)
                self.buf = [self.ring[-back:].copy()]
                self.t0 = self.n - back; self.active = True
                ev.append(("start", None, False))
            elif self.active:
                self.buf.append(fr)
                if not sp:
                    ev.append(("end", self._finish(trim=int((0.35 - self.TAIL) * SR)), False))
                elif sum(len(b) for b in self.buf) >= self.max_len:
                    ev.append(("end", self._force_split(), True))
        if self.active: ev.append(("grow", None, False))
        return ev

    def current(self):
        return np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)

    def _finish(self, trim=0):
        a = self.current()
        if trim > 0 and len(a) > trim + SR // 4: a = a[:-trim]
        utt = (self.t0, a)
        self.active = False; self.buf = []
        return utt

    def split_at(self, cut):
        """cut the utterance in progress at sample <cut>: returns the first part, the rest carries on"""
        a = self.current()
        utt = (self.t0, a[:cut])
        self.t0 += cut; self.buf = [a[cut:]]
        return utt

    def _force_split(self):
        """Utterance too long (narration): cut at the quietest 0.2 s in its last 3 s and carry on."""
        a = self.current()
        w = int(0.2 * SR); look = min(len(a) - w, int(3 * SR))
        start = len(a) - look
        best, best_e = len(a), 1e9
        for i in range(start, len(a) - w, w // 2):
            e = float(np.mean(a[i:i + w] ** 2))
            if e < best_e: best_e, best = e, i + w // 2
        head, tail = a[:best], a[best:]
        utt = (self.t0, head)
        self.t0 = self.t0 + best; self.buf = [tail]
        return utt

    def flush(self):
        """No audio arriving (stream went silent) -> close the current utterance."""
        if not self.active: return None
        self.vad.reset()
        return self._finish()

# ----------------------------------------------------------------------------------------------- pipeline
class Pipeline:
    """Audio in -> on_event(dict) out.
       {"kind":"partial","utt":id,"voice":vid|None,"text":str}
       {"kind":"final","utt":id,"parts":[(vid|None,text),...],"t":unix_time}
    """
    def __init__(self, cfg, bank, asr, emb, on_event):
        self.cfg, self.bank, self.asr, self.emb = cfg, bank, asr, emb
        def emit(ev):
            fx = self.cfg.get("word_fixes") or []
            if fx:
                if "text" in ev: ev = {**ev, "text": apply_fixes(ev["text"], fx)}
                if "parts" in ev: ev = {**ev, "parts": [(v, apply_fixes(t, fx)) for v, t in ev["parts"]]}
            on_event({**ev, "source": self.source})
        self.on_event = emit
        self.seg = Segmenter(cfg.get("max_segment_seconds", 10))
        self.utt = 0
        self.last_partial_len = 0
        self.last_partial_t = 0
        self.partial_voice = None
        self.behind = False                    # set by the feeder when audio is queuing up
        self.stats = {"utts": 0, "asr_s": 0.0, "audio_s": 0.0}

    # ---------------- voices
    def _assign(self, e, dur, learn=True):
        """fingerprint of dur seconds of speech -> (voice id or None, similarity). May create a new voice."""
        cfg = self.cfg
        ranked = self.bank.classify(e)
        best_s, best = ranked[0] if ranked else (-1.0, None)
        if dur < 1.2:                                       # short: never learn from it
            return (best, best_s) if best is not None and best_s >= cfg["short_match_threshold"] else (None, best_s)
        if best is not None and best_s >= cfg["match_threshold"]:
            if dur >= 1.5 and best_s >= cfg["update_threshold"]:
                self.bank.refine(best, e, weight=min(dur / 3.0, 1.5))
            return best, best_s
        if (learn and cfg.get("auto_new_voices", False) and dur >= cfg["min_new_voice_seconds"]
                and best_s < cfg["new_voice_threshold"] and len(self.bank.voices) < cfg.get("max_voices", 8)):
            return self.bank.add(e), 1.0
        if best is not None and best_s >= cfg["short_match_threshold"]:
            return best, best_s
        return None, best_s

    def _speaker_runs(self, audio):
        """Split one utterance into runs of one speaker each -> [(start_s, end_s, voice)].
        Looks for points where the voice before and after differ, then matches each piece to the known voices."""
        dur = len(audio) / SR
        W, G = 1.5, self.cfg.get("change_step", 0.4)
        cuts = []
        if dur >= 2.0:
            cache = {}
            def E(t0, t1):
                k = (round(t0, 2), round(t1, 2))
                if k not in cache: cache[k] = self.emb(audio[int(t0 * SR):int(t1 * SR)])
                return cache[k]
            cands = []
            for bt in np.arange(0.8, dur - 0.8 + 1e-6, G):
                l, r = E(max(0.0, bt - W), bt), E(bt, min(dur, bt + W))
                cands.append((float(l @ r), float(bt)))
            thr = self.cfg.get("change_threshold", 0.35)
            for sc, bt in sorted(cands):
                if sc >= thr: break
                if all(abs(bt - c) >= 1.0 for c in cuts): cuts.append(bt)
            cuts.sort()
        edges = [0.0] + cuts + [dur]
        out = []
        for t0, t1 in zip(edges[:-1], edges[1:]):
            v, _ = self._assign(self.emb(audio[int(t0 * SR):int(t1 * SR)]), t1 - t0)
            if out and out[-1][2] == v: out[-1] = (out[-1][0], t1, v)
            else: out.append((t0, t1, v))
        return out

    # ---------------- events
    clock = staticmethod(time.time)       # replaceable for tests

    def _note(self, kind, forced=False):
        """wall-clock record of when the call had speech (used to recognise echo on your mic)"""
        if not hasattr(self, "spans"): self.spans = []; self.active_since = None; self.said = []
        now = self.clock()
        if kind == "start": self.active_since = now - 0.5
        elif kind == "end" and self.active_since is not None:
            self.spans.append((self.active_since, now)); self.spans = self.spans[-200:]
            self.active_since = now if forced else None

    def overlap(self, t0, t1):
        """seconds of call speech between t0 and t1"""
        if not hasattr(self, "spans"): return 0.0
        spans = list(self.spans) + ([(self.active_since, self.clock())] if self.active_since else [])
        return sum(max(0.0, min(b, t1) - max(a, t0)) for a, b in spans)

    def echoes(self, text, window=25):
        """does text repeat something said on the call in the last <window> seconds?"""
        import difflib
        if not hasattr(self, "said"): return False
        a = re.sub(r"[^\w' ]", "", text.lower()).strip()
        if not a: return True
        now = self.clock()
        for t, said in reversed(self.said):
            if now - t > window: break
            b = re.sub(r"[^\w' ]", "", said.lower())
            if a in b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.55: return True
            if len(a.split()) >= 3 and difflib.SequenceMatcher(None, a, b[-len(a) - 40:]).ratio() >= 0.6: return True
        return False

    RING_S = 40

    def _keep(self, x, t_end):
        """recent audio + the wall-clock time of its last sample (to line the mic up with the call)"""
        if not hasattr(self, "ring"): self.ring = np.zeros(0, np.float32); self.n_total = 0
        self.ring = np.concatenate([self.ring, x])[-self.RING_S * SR:]
        self.n_total += len(x); self.wall_end = t_end if t_end is not None else self.clock()

    def wall_of(self, sample):
        """wall-clock time of an absolute sample index of this stream"""
        return self.wall_end - (self.n_total - sample) / SR

    def audio_between(self, w0, w1):
        """this stream's audio between two wall-clock times (zeros where we have none)"""
        if not hasattr(self, "ring"): return np.zeros(max(0, int((w1 - w0) * SR)), np.float32)
        n = len(self.ring)
        i1 = n - int(round((self.wall_end - w1) * SR)); i0 = i1 - int(round((w1 - w0) * SR))
        out = np.zeros(i1 - i0, np.float32)
        a, b = max(i0, 0), min(i1, n)
        if b > a: out[a - i0:b - i0] = self.ring[a:b]
        return out

    def feed(self, x, t_end=None):
        self._keep(x, t_end)
        for kind, utt, forced in self.seg.feed(x):
            if kind in ("start", "end"): self._note(kind, forced)
            if kind == "start":
                self.utt += 1; self.last_partial_len = 0; self.partial_voice = None; self.new_speaker = False
                self.sc_len = self.fast_len = 0; self.sc_cache = {}; self.sc_next = 0
            elif kind == "end":
                self._final(*utt)
                if forced:                              # the rest continues as a new utterance
                    self.utt += 1; self.last_partial_len = 0; self.sc_len = self.fast_len = 0
            elif kind == "grow":
                if self._maybe_split(): self.last_partial_len = 0      # show the newcomer's first words right away
                self._maybe_partial()

    def flush(self):
        u = self.seg.flush()
        if u: self._final(*u)

    source = "call"
    partials = True

    def _remember(self, text):
        if not hasattr(self, "said"): self._note("init")
        self.said.append((self.clock(), text)); self.said = self.said[-60:]

    # ---- speaker change while someone is talking (nobody paused, so the voice detector sees one long stretch)
    SC_EVERY = 0.3                     # seconds of new audio between checks
    SC_FAST_EVERY = 0.6                # the fast check (below) is a bit more expensive, so less often
    SC_MIN = 2.8                       # the fast check needs a stretch at least this long
    SC_QUICK = (1.2, 3.2)              # fast check: newest 1.2 s against the 2 s before it -> "did something change?"
    SC_LEFT = 1.4
    SC_CUTS = ((2.0, 0.0), (1.7, 0.0), (1.4, 0.0), (1.1, 0.0))   # (seconds of newcomer audio, extra similarity allowed)
    SC_W, SC_G = 1.6, 0.4              # scan: compare 1.6 s before and after every 0.4 s point in the stretch
    SC_SCAN_EXTRA = 0.0                # added to change_threshold for the scan (negative = needs a clearer difference)

    def _fast_change(self, a):
        """quick: did a different person take over in the last ~2 s? -> cut sample or None"""
        if len(a) < int(self.SC_MIN * SR): return None
        thr = self.cfg.get("change_threshold", 0.35)
        qn, qb = self.SC_QUICK
        quick = float(self.emb(a[-int(qn * SR):]) @ self.emb(a[max(0, len(a) - int(qb * SR)):-int(qn * SR)]))
        if quick >= thr + 0.12: return None
        best = (0.0, None)                       # most negative margin below its threshold wins
        for back, extra in self.SC_CUTS:
            cut = len(a) - int(back * SR)
            if cut < int(1.0 * SR): continue
            l = self.emb(a[max(0, cut - int(self.SC_LEFT * SR)):cut]); r = self.emb(a[cut:])
            margin = float(l @ r) - (thr + extra)
            if margin < best[0]: best = (margin, cut)
        return best[1]

    def _scan_change(self, a):
        """the same idea as the split made once a stretch is over, but live: compare the voice 1.6 s before and
        after every 0.4 s point. Fingerprints are kept between checks, so each check adds only about one new one.
        Finds a change however long ago it happened (the fast check only looks at the last ~2 s)."""
        W, G = int(self.SC_W * SR), int(self.SC_G * SR)
        base = self.seg.t0                          # absolute sample number of a[0]
        cache = self.__dict__.setdefault("sc_cache", {})
        def E(i0, i1):
            k = (base + i0, base + i1)
            if k not in cache: cache[k] = self.emb(a[i0:i1])
            return cache[k]
        thr = self.cfg.get("change_threshold", 0.35) + self.SC_SCAN_EXTRA
        k = max(getattr(self, "sc_next", 0), -(-(base + int(0.8 * SR)) // G))      # first point >= 0.8 s into the stretch
        best = (thr, None)
        while True:
            rel = k * G - base
            if rel + W > len(a): break               # the newcomer's 1.6 s are not all here yet
            sim = float(E(max(0, rel - W), rel) @ E(rel, rel + W))
            if sim < best[0]: best = (sim, rel)
            k += 1
        self.sc_next = k
        return best[1]

    def _speaker_change(self, a):
        if self.source != "call": return None
        n = len(a)
        cut = None
        if n - getattr(self, "fast_len", 0) >= int(self.SC_FAST_EVERY * SR) or n < getattr(self, "fast_len", 0):
            self.fast_len = n
            cut = self._fast_change(a)
        return cut if cut is not None else self._scan_change(a)

    def _maybe_split(self):
        """a few times a second: someone else started talking in the middle of this stretch? close the first
        person's line now. True if it was split."""
        if not self.partials or not self.cfg.get("show_partial", True) or self.behind: return False
        a = self.seg.current()
        n = len(a)
        if 0 <= n - getattr(self, "sc_len", 0) < int(self.SC_EVERY * SR): return False
        self.sc_len = n
        cut = self._speaker_change(a)
        if cut is None: return False
        self._final(*self.seg.split_at(cut))
        self.utt += 1; self.partial_voice = None; self.new_speaker = True
        self.stats["early_splits"] = self.stats.get("early_splits", 0) + 1
        self.sc_len = self.fast_len = len(self.seg.current())
        return True

    def _maybe_partial(self):
        if not self.partials or not self.cfg.get("show_partial", True) or self.behind: return
        a = self.seg.current()
        if len(a) < int(0.7 * SR) or len(a) - self.last_partial_len < int(0.6 * SR): return
        self.last_partial_len = len(a)
        tail = a[-int(6 * SR):]
        if len(a) >= SR and (self.partial_voice is None or len(a) < 4 * SR):
            ranked = self.bank.classify(self.emb(tail))
            if ranked and ranked[0][0] >= self.cfg["short_match_threshold"]: self.partial_voice = ranked[0][1]
        t = time.time()
        text = self.asr.partial(a[-int(28 * SR):])
        self.stats["asr_s"] += time.time() - t
        if text:
            self._remember(text)
            self.on_event({"kind": "partial", "utt": self.utt, "voice": self.partial_voice, "text": text,
                           "new_speaker": getattr(self, "new_speaker", False)})

    def _final(self, t0, audio):
        dur = len(audio) / SR
        if dur < 0.35: return
        t = time.time()
        words = self.asr.final(audio)
        self.stats["asr_s"] += time.time() - t; self.stats["audio_s"] += dur; self.stats["utts"] += 1
        if not words:
            self.on_event({"kind": "final", "utt": self.utt, "parts": [], "spans": [], "t": time.time(), "t0": t0 / SR,
                           "dur": dur}); return
        runs = self._speaker_runs(audio)
        groups = []                                  # [voice, text, start_s, end_s]
        for (ws, we, w) in words:
            mid = (ws + we) / 2
            v = runs[-1][2]
            for (a, b, rv) in runs:
                if a <= mid < b: v = rv; break
            if groups and groups[-1][0] == v: groups[-1][1] += w; groups[-1][3] = we
            else: groups.append([v, w, ws, we])
        groups = [g for g in groups if g[1].strip()]
        parts = [(v, re.sub(r"\s+", " ", txt).strip()) for v, txt, _, _ in groups]
        spans = [(v, t0 / SR + a, t0 / SR + b) for v, _, a, b in groups]
        # each part's voice fingerprint, so a "?" line can later become a new voice or teach an existing one
        embs = [(self.emb(audio[int(max(0, a - 0.1) * SR):int(min(dur, b + 0.1) * SR)]).tolist() if b - a >= 0.6 else None)
                for _, _, a, b in groups] if self.source == "call" else [None] * len(groups)
        for v, txt in parts:
            if v is not None and v != "me": self.bank.heard(v, txt)
        if self.source == "call": self._remember(" ".join(t for _, t in parts))
        self.on_event({"kind": "final", "utt": self.utt, "parts": parts, "spans": spans, "embs": embs,
                       "t": time.time(), "t0": t0 / SR, "dur": dur})

class MicPipeline(Pipeline):
    """Your own microphone -> transcript only.
    The mic also hears the call coming out of your speakers/headphones; those bits are thrown away:
    only speech that sounds like YOU (learned while the call is quiet) and doesn't repeat the call is kept."""
    source = "me"
    partials = False
    call = None              # the call Pipeline (set by the app)

    debug = False

    @staticmethod
    def _env(x, hop=320):
        n = len(x) // hop
        if n == 0: return np.zeros(0)
        return np.log(np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-4)

    def echo_score(self, mic, w0, w1):
        """how much this bit of mic audio rises and falls with what the call played (0 = not at all, 1 = echo)"""
        if self.call is None: return 0.0
        ref = self.call.audio_between(w0 - 0.6, w1 + 0.1)          # the mic hears the call a little later
        if np.sqrt(np.mean(ref ** 2) + 1e-12) < 0.004: return 0.0      # call was silent
        em, er = self._env(mic), self._env(ref)
        if len(em) < 10: return 0.0
        em = em - em.mean()
        best = 0.0
        for lag in range(0, len(er) - len(em) + 1):                    # 20 ms steps over the 0-0.7 s window
            seg = er[lag:lag + len(em)]; seg = seg - seg.mean()
            c = float(em @ seg / (np.linalg.norm(em) * np.linalg.norm(seg) + 1e-9))
            if c > best: best = c
        return best

    def _final(self, t0, audio):
        dur = len(audio) / SR
        if dur < 0.35: return
        now = self.clock()
        w_start = self.wall_of(t0) if hasattr(self, "n_total") else now - dur
        W, H = 1.5, 0.5
        starts = [0.0] if dur <= W else list(np.arange(0, dur - W + 1e-6, H))
        if dur > W and dur - W - starts[-1] > 0.2: starts.append(dur - W)
        me_win = []
        thr = self.cfg.get("echo_threshold", 0.5)
        for ws in starts:
            we = min(dur, ws + W)
            chunk = audio[int(ws * SR):int(we * SR)]
            echo = self.echo_score(chunk, w_start + ws, w_start + we)
            if echo >= thr:                                   # plainly the call leaking in: skip the costly checks
                if self.debug: print(f"    mic [{ws:5.1f}-{we:5.1f}] echo {echo:4.2f} -> -")
                continue
            e = self.emb(chunk)
            ranked = self.bank.classify(e); other = ranked[0][0] if ranked else -1.0
            me = self.bank.me_sim(e)
            ok = echo < thr and other < self.cfg["match_threshold"] and not (self.bank.me and me < 0.25 and other > me)
            if self.debug: print(f"    mic [{ws:5.1f}-{we:5.1f}] echo {echo:4.2f} other {other:5.2f} me {me:5.2f} -> {'ME' if ok else '-'}")
            if ok and echo < thr * 0.5 and (we - ws) >= 1.2 and other < 0.35: self.bank.me_learn(e)
            if ok: me_win.append((ws, we))
        if not me_win:
            self.stats["echo"] = self.stats.get("echo", 0) + 1; return
        SH = 0.4
        core = [(a if a <= 0.01 else a + SH, b if b >= dur - 0.01 else b - SH) for a, b in me_win]
        def is_me_at(t): return any(a <= t <= b for a, b in core)
        words = self.asr.final(audio)
        keep = [w for w in words if is_me_at((w[0] + w[1]) / 2)]
        text = re.sub(r"\s+", " ", "".join(w for _, _, w in keep)).strip()
        if not text: return
        if self.call and self.call.echoes(text):
            self.stats["echo"] = self.stats.get("echo", 0) + 1; return
        self.stats["utts"] += 1
        self.on_event({"kind": "final", "utt": self.utt, "parts": [("me", text)],
                       "spans": [("me", t0 / SR + me_win[0][0], t0 / SR + me_win[-1][1])], "t": now,
                       "t0": t0 / SR + me_win[0][0], "dur": me_win[-1][1] - me_win[0][0]})

# ----------------------------------------------------------------------------------------------- audio input
def list_input_devices():
    """-> list of (index, name) for microphones (WASAPI)"""
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)["index"]
        out = []
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] == wasapi and d["maxInputChannels"] > 0 and not d.get("isLoopbackDevice", False):
                out.append((d["index"], d["name"]))
        return out
    finally:
        p.terminate()

def list_output_devices():
    """-> list of (index, name) for loopback (what-you-hear) devices"""
    import pyaudiowpatch as pyaudio
    p = pyaudio.PyAudio()
    try:
        return [(d["index"], d["name"]) for d in p.get_loopback_device_info_generator()]
    finally:
        p.terminate()

def list_listen_sources():
    """-> [(value, label)] for "Listen to": what playback devices play, plus recording devices
    (a virtual cable's output, e.g. "CABLE Output", carries whatever Discord sends into the cable)"""
    out = [(n, n) for _, n in list_output_devices()]
    try:
        out += [("rec:" + n, n + "  (recording device)") for _, n in list_input_devices()]
    except Exception as e:
        log(f"[audio] could not list recording devices: {e}")
    return out


class AudioSource:
    """Captures what a playback device is playing (WASAPI loopback) -> 16 kHz mono float32 chunks in self.q"""
    def __init__(self, want="auto", mic=False):
        import pyaudiowpatch as pyaudio, soxr
        if want and want.startswith("rec:"): want, mic = want[4:], True        # a recording device as the call
        self.pa = pyaudio.PyAudio(); self.pyaudio = pyaudio; self.mic = mic
        self.q = queue.Queue()
        self.level = -90.0
        self.dev = self._pick(want)
        rate0, ch0 = int(self.dev["defaultSampleRate"]), max(1, int(self.dev["maxInputChannels"]))
        # some drivers refuse the first format; try a few common ones before giving up
        tries = [(pyaudio.paFloat32, ch0, rate0), (pyaudio.paInt16, ch0, rate0), (pyaudio.paFloat32, 2, 48000),
                 (pyaudio.paInt16, 2, 48000), (pyaudio.paFloat32, 2, 44100), (pyaudio.paInt16, 2, 44100),
                 (pyaudio.paFloat32, 1, rate0), (pyaudio.paInt16, 1, rate0)]
        err = None
        for fmt, ch, rate in dict.fromkeys(tries):
            try:
                self.fmt, self.ch, self.rate = fmt, ch, rate
                self.rs = soxr.ResampleStream(rate, SR, 1, dtype="float32")
                self.stream = self.pa.open(format=fmt, channels=ch, rate=rate, input=True,
                                           input_device_index=self.dev["index"], frames_per_buffer=int(rate * 0.03),
                                           stream_callback=self._cb)
                break
            except Exception as e:
                err = e
        else:
            self.pa.terminate()
            raise OSError(f"could not open {self.dev['name']}: {err}")
        self.name = self.dev["name"]
        log(f"[audio] {'microphone' if mic else 'listening to'}: {self.name} ({self.rate} Hz, {self.ch} ch)")

    def _pick(self, want):
        if self.mic:
            wasapi = self.pa.get_host_api_info_by_type(self.pyaudio.paWASAPI)
            devs = [self.pa.get_device_info_by_index(i) for i in range(self.pa.get_device_count())]
            devs = [d for d in devs if d["hostApi"] == wasapi["index"] and d["maxInputChannels"] > 0
                    and not d.get("isLoopbackDevice", False)]
            if want and want != "auto":
                for d in devs:
                    if want.lower() in d["name"].lower(): return d
            for d in devs:
                if "nvidia broadcast" in d["name"].lower(): return d
            return self.pa.get_device_info_by_index(wasapi["defaultInputDevice"])
        devs = list(self.pa.get_loopback_device_info_generator())
        if want and want != "auto":
            for d in devs:
                if want.lower() in d["name"].lower(): return d
            log(f"[audio] device '{want}' not found, using auto")
        for d in devs:
            if "nvidia broadcast" in d["name"].lower(): return d
        return self.pa.get_default_wasapi_loopback()

    def _cb(self, data, frames, info, status):
        if self.fmt == self.pyaudio.paInt16: x = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        else: x = np.frombuffer(data, dtype=np.float32)
        if self.ch > 1: x = x.reshape(-1, self.ch).mean(axis=1)
        y = self.rs.resample_chunk(x.astype(np.float32))
        if len(y):
            self.level = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)
            self.q.put((time.time(), y))
        return (None, self.pyaudio.paContinue)

    def close(self):
        try: self.stream.stop_stream(); self.stream.close()
        except Exception: pass
        try: self.pa.terminate()
        except Exception: pass

# ----------------------------------------------------------------------------------------------- worker
class Worker(threading.Thread):
    """Moves audio from the AudioSource into the Pipeline. Swappable device."""
    def __init__(self, pipeline):
        super().__init__(daemon=True)
        self.p = pipeline; self.src = None; self.stop_flag = False; self.swap_to = None
        self.last_audio = time.time()

    def set_source(self, src):
        old = self.src; self.src = src
        if old: old.close()

    def run(self):
        while not self.stop_flag:
            src = self.src
            if src is None: time.sleep(0.1); continue
            try:
                x = src.q.get(timeout=0.25)
            except queue.Empty:
                if time.time() - self.last_audio > 0.8:
                    try: self.p.flush()
                    except Exception as e: log(f"[worker] {e}")
                continue
            self.last_audio = time.time()
            chunks = [x]
            while not src.q.empty() and len(chunks) < 50: chunks.append(src.q.get_nowait())
            t_end = chunks[-1][0]; chunks = [c[1] for c in chunks]
            self.p.behind = src.q.qsize() * 0.03 > 1.0 or len(chunks) > 30
            try:
                self.p.feed(np.concatenate(chunks), t_end)
            except Exception as e:
                import traceback; traceback.print_exc()
