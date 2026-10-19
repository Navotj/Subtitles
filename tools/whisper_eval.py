"""Check how well the app's Whisper setup does on a recording that has human-made captions.

    python tools\\whisper_eval.py <audio or video file> <captions .vtt or .srt> [options]

It cuts the audio into utterances and transcribes them exactly the way the app does (same voice-activity detector, same
Whisper model and settings), lines the result up against the captions caption by caption, and writes a report of
  * which words Whisper gets wrong most often ("heard" -> "should be"),
  * words it leaves out,
  * how often each of YOUR terms (--terms) came out right, and what it was usually heard as.
That is the raw material for the preset: manual swaps, automatic-swap words and words to expect.

Options
  --expect FILE   words to expect, one per line or comma separated; given to Whisper like the app does (capped by expect_cap)
  --auto FILE     automatic word swap list; applied afterwards (needs wordswap.py next to engine.py)
  --fixes FILE    manual word swaps as JSON: [["heard as", "replace with"], ...]; applied afterwards
  --terms FILE    terms to score (default: the expect and auto lists)
  --gpu 80        keep the GPU about this busy (percent); 100 = no limit
  --minutes N     only use the first N minutes
  --reuse         reuse the saved Whisper output if it was made with the same expect list (so swaps can be tried in seconds)
Reports go to eval\\reports, saved Whisper output to eval\\work.
"""
import os, sys, re, json, time, html, bisect, difflib, argparse, datetime
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
EVAL = os.path.join(ROOT, "eval")
PAD = 0.3                    # seconds of slack around each caption when picking the words Whisper produced for it


# ------------------------------------------------------------------------------------------------ captions
_TS = re.compile(r"(?:(\d+):)?(\d\d):(\d\d)[.,](\d{1,3})\s*-->\s*(?:(\d+):)?(\d\d):(\d\d)[.,](\d{1,3})")


def _sec(h, m, s, ms):
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000


def clean_caption(t):
    t = re.sub(r"<[^>]+>", "", t)
    t = html.unescape(t)
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)|[♪♫]+|>>", " ", t)
    t = re.sub(r"(?m)^\s*[-–]\s+", "", t)
    t = re.sub(r"(?m)^\s*[A-Z][A-Z .'-]{1,20}:\s+", "", t)          # "MATT: " speaker labels
    return re.sub(r"\s+", " ", t).strip()


def parse_captions(text):
    """.vtt or .srt text -> [(start_s, end_s, text)]; empty cues and immediate repeats are dropped"""
    cues, lines, i = [], text.replace("\r", "").split("\n"), 0
    while i < len(lines):
        m = _TS.search(lines[i])
        if not m:
            i += 1; continue
        a = _sec(*m.group(1, 2, 3, 4)); b = _sec(*m.group(5, 6, 7, 8))
        i += 1; body = []
        while i < len(lines) and lines[i].strip():
            body.append(lines[i]); i += 1
        t = clean_caption(" ".join(body))
        if t and not (cues and cues[-1][2] == t):
            cues.append((a, b, t))
    return cues


_ONES = {w: str(n) for n, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                         "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_TENS = {w: str(10 * n) for n, w in enumerate("x x twenty thirty forty fifty sixty seventy eighty ninety".split()) if n > 1}
_NUM = {**_ONES, **_TENS, "hundred": "100"}
# spoken forms the captions and Whisper write differently; both sides are brought to one spelling so they are not "mistakes"
_SAY = {"gonna": "going to", "wanna": "want to", "gotta": "got to", "kinda": "kind of", "sorta": "sort of", "cause": "because",
        "cannot": "can not", "ok": "okay", "yeah": "yes", "yep": "yes", "yup": "yes", "nope": "no", "dunno": "do not know",
        "gimme": "give me", "lemme": "let me", "outta": "out of", "lotta": "lot of", "coulda": "could have", "woulda": "would have",
        "shoulda": "should have", "alright": "all right", "'cause": "because", "y'all": "you all"}
_IS = {"it", "that", "he", "she", "there", "what", "here", "who", "how", "where", "when", "why"}
_FILLERS = {"um", "uh", "uhm", "er", "erm", "hmm", "mm", "mmm", "mhm", "ah", "oh", "huh", "hm"}
_NT = {"can": "can not", "won": "will not", "shan": "shall not", "ain": "is not"}


def _expand(w):
    """one lower-case word -> list of words (contractions opened up, filler dropped)"""
    if w in _FILLERS: return []
    if w in _SAY: return _SAY[w].split()
    if "'" not in w: return [w]
    base, _, tail = w.partition("'")
    if tail == "t" and base.endswith("n"): return (_NT.get(base) or base[:-1] + " not").split()
    if tail == "re": return [base, "are"]
    if tail == "ve": return [base, "have"]
    if tail == "ll": return [base, "will"]
    if tail == "m": return [base, "am"]
    if tail == "d": return [base, "would"]
    if tail == "s" and base in _IS: return [base, "is"]
    if tail == "s" and base == "let": return ["let", "us"]
    return [w]


def toks(text):
    """lower-case words for comparing. Number words become digits ("twenty one" -> 21), contractions are opened up
    (gonna / going to, I'm / I am) and fillers (um, uh) dropped, so none of those count as mistakes"""
    t = text.lower().replace("’", "'").replace("-", " ")
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    out = []
    for w0 in re.findall(r"[a-z0-9]+(?:'[a-z]+)?", t):
        for w in _expand(w0):
            n = _NUM.get(w, w)
            if out and out[-1] in _TENS.values() and len(n) == 1 and n in "123456789":
                out[-1] = str(int(out[-1]) + int(n)); continue
            out.append(n)
    return out


# ------------------------------------------------------------------------------------------------ comparing
def _trivial(a, b):
    """differences that are not worth reporting: plurals and apostrophes (a split word like "fire ball" IS worth reporting)"""
    x, y = a.replace("'", ""), b.replace("'", "")
    if x == y: return True
    if " " in x or " " in y: return False
    return x + "s" == y or y + "s" == x or x + "es" == y or y + "es" == x


def _find(seq, sub):
    n = len(sub)
    return [i for i in range(len(seq) - n + 1) if seq[i:i + n] == sub] if n else []


def _chunks(cues, max_cues=25, max_secs=75.0):
    """consecutive captions grouped into stretches; each stretch is compared as a whole, so a word a little early or late
    never lands in the wrong place"""
    out, cur = [], []
    for c in cues:
        if cur and (len(cur) >= max_cues or c[1] - cur[0][0] > max_secs):
            out.append(cur); cur = []
        cur.append(c)
    if cur: out.append(cur)
    return out


def align(cues, words, terms=(), post=None):
    """words: [(start_s, end_s, text)] from Whisper. post: optional function applied to the text Whisper gave for each
    stretch (word swaps). Each Whisper word is used once; speech more than 2 s outside any caption (music, crosstalk the
    captioner skipped) is not counted. -> dict with the counts the report needs"""
    words = sorted(words)
    mids = [(s + e) / 2 for s, e, _ in words]
    conf, missed, extra = Counter(), Counter(), Counter()
    n_ref = errors = S = D = I = 0
    tt = [(t, toks(t)) for t in terms if toks(t)]
    term_n, term_ok, term_heard = Counter(), Counter(), defaultdict(Counter)
    groups = _chunks([c for c in cues if toks(c[2])])
    for gi, g in enumerate(groups):
        t0, t1 = g[0][0], g[-1][1]
        lo = max((groups[gi - 1][-1][1] + t0) / 2 if gi else -1e9, t0 - 2.0)
        hi = min((t1 + groups[gi + 1][0][0]) / 2 if gi + 1 < len(groups) else 1e9, t1 + 2.0)
        ref = [w for c in g for w in toks(c[2])]
        i0, i1_ = bisect.bisect_left(mids, lo), bisect.bisect_left(mids, hi)
        said = " ".join(w[2].strip() for w in words[i0:i1_])
        if post: said = post(said)
        hyp = toks(said)
        ops = difflib.SequenceMatcher(None, ref, hyp, autojunk=False).get_opcodes()
        n_ref += len(ref)
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal": continue
            errors += max(i2 - i1, j2 - j1)
            if tag == "delete":
                D += i2 - i1; missed[" ".join(ref[i1:i2])] += 1
            elif tag == "insert":
                I += j2 - j1; extra[" ".join(hyp[j1:j2])] += 1
            else:
                S += max(i2 - i1, j2 - j1)
                r, h = ref[i1:i2], hyp[j1:j2]
                if len(r) == len(h):
                    pairs = [(a, b) for a, b in zip(r, h) if a != b]
                elif len(r) <= 3 and len(h) <= 3:
                    pairs = [(" ".join(r), " ".join(h))]
                else:
                    pairs = []
                for a, b in pairs:
                    if not _trivial(a, b): conf[(a, b)] += 1
        for term, tk in tt:
            for p in _find(ref, tk):
                term_n[term] += 1
                bad = [op for op in ops if op[0] != "equal" and op[1] < p + len(tk) and (op[2] > p)
                       and not (op[0] == "insert" and not (p < op[1] < p + len(tk)))]
                if not bad:
                    term_ok[term] += 1
                else:
                    heard = " ".join(" ".join(hyp[o[3]:o[4]]) or "(missed)" for o in bad)
                    term_heard[term][heard] += 1
    return dict(n_ref=n_ref, errors=errors, S=S, D=D, I=I, conf=conf, missed=missed, extra=extra,
                term_n=term_n, term_ok=term_ok, term_heard=term_heard)


# ------------------------------------------------------------------------------------------------ report
def build_report(title, info, r):
    wer = 100 * r["errors"] / max(r["n_ref"], 1)
    L = [f"# Whisper check: {title}", ""]
    L += [f"- {info}", f"- captions: {r['n_ref']} words; word error rate **{wer:.1f}%** "
          f"({r['S']} swapped, {r['D']} left out, {r['I']} added)", ""]
    common = [(c, a, b) for (a, b), c in r["conf"].most_common() if c >= 2]
    L += ["## Most common mistakes (heard → should be)", "", "| times | heard | should be |", "|---|---|---|"]
    L += [f"| {c} | {b} | {a} |" for c, a, b in common[:80]] or ["| | (none seen twice or more) | |"]
    once = sorted(((a, b) for (a, b), c in r["conf"].items() if c == 1 and " " not in a and len(a) >= 5),
                  key=lambda p: -len(p[0]))[:60]
    if once:
        L += ["", "## Seen once (long words only)", "", "| heard | should be |", "|---|---|"] + [f"| {b} | {a} |" for a, b in once]
    out = [(c, w) for w, c in r["missed"].most_common() if c >= 3]
    if out:
        L += ["", "## Left out (often just caption timing)", "", "| times | words |", "|---|---|"]
        L += [f"| {c} | {w} |" for c, w in out[:40]]
    if r["term_n"]:
        L += ["", "## Your terms", "", "| term | in captions | right | usually heard as |", "|---|---|---|---|"]
        for t, n in sorted(r["term_n"].items(), key=lambda kv: (r["term_ok"][kv[0]] / kv[1], -kv[1])):
            ok = r["term_ok"][t]
            heard = ", ".join(f"{h} ×{c}" for h, c in r["term_heard"][t].most_common(3))
            L.append(f"| {t} | {n} | {ok} ({100 * ok // n}%) | {heard} |")
        tot, ok = sum(r["term_n"].values()), sum(r["term_ok"].values())
        L += ["", f"Overall: {ok} of {tot} uses of your terms came out right ({100 * ok // max(tot, 1)}%)."]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ whisper
def read_list(path):
    """a list file: one term per line (or comma separated); lines starting with # are comments"""
    if not path: return ""
    return "\n".join(l for l in open(path, encoding="utf-8").read().split("\n") if not l.lstrip().startswith("#"))


def transcribe(media, cfg, gpu_percent=80, minutes=None):
    """the app's own pipeline: Silero voice detector -> utterances -> Whisper. -> (words, info)"""
    import numpy as np
    import engine
    from engine import SR
    try:
        from faster_whisper.audio import decode_audio
        x = decode_audio(media, sampling_rate=SR)
    except Exception:
        import soundfile as sf, soxr
        x, sr = sf.read(media, dtype="float32", always_2d=True); x = x.mean(axis=1)
        if sr != SR: x = soxr.resample(x, sr, SR).astype(np.float32)
    x = np.asarray(x, dtype=np.float32)
    if minutes: x = x[:int(minutes * 60 * SR)]
    dur = len(x) / SR
    asr = engine.Transcriber(cfg)
    seg = engine.Segmenter(cfg.get("max_segment_seconds", 10))
    utts, step = [], int(0.03 * SR)
    for i in range(0, len(x), step):
        for kind, utt, forced in seg.feed(x[i:i + step]):
            if kind == "end": utts.append(utt)
    u = seg.flush()
    if u: utts.append(u)
    words, t_all, done = [], time.time(), 0.0
    for t0, a in utts:
        if len(a) < 0.35 * SR: continue
        t = time.time()
        for s, e, w in asr.final(a): words.append((t0 / SR + s, t0 / SR + e, w))
        el = time.time() - t
        if gpu_percent < 100 and asr.device == "cuda":
            time.sleep(el * (100 - gpu_percent) / max(gpu_percent, 1))        # busy about gpu_percent of the time
        done = (t0 + len(a)) / SR
        if int(done) // 30 != int(done - len(a) / SR) // 30:
            print(f"  {done / 60:5.1f} / {dur / 60:.1f} min  ({time.time() - t_all:.0f} s)", flush=True)
    info = dict(model=asr.name, device=asr.device.upper(), minutes=round(dur / 60, 1),
                seconds_used=round(time.time() - t_all))
    return words, info


def run(media, captions, name=None, expect="", auto="", fixes=None, terms=None, gpu=80, minutes=None, reuse=False,
        title=None):
    name = name or os.path.splitext(os.path.basename(media))[0]
    for d in ("work", "reports"): os.makedirs(os.path.join(EVAL, d), exist_ok=True)
    cues = parse_captions(open(captions, encoding="utf-8", errors="replace").read())
    if minutes: cues = [c for c in cues if c[0] < minutes * 60]
    if not cues: raise SystemExit(f"no captions found in {captions}")
    cache = os.path.join(EVAL, "work", name + ".words.json")
    words = info = None
    if reuse and os.path.exists(cache):
        d = json.load(open(cache, encoding="utf-8"))
        if d.get("expect") == expect and d.get("minutes_limit") == minutes:
            words, info = [tuple(w) for w in d["words"]], d["info"]; print("using the saved Whisper output")
    if words is None:
        import engine
        cfg = engine.load_config(); cfg["vocabulary"] = expect; cfg["word_fixes"] = []
        print(f"transcribing {os.path.basename(media)} ...", flush=True)
        words, info = transcribe(media, cfg, gpu, minutes)
        json.dump(dict(expect=expect, minutes_limit=minutes, words=words, info=info), open(cache, "w", encoding="utf-8"))
    post = None
    steps = []
    if fixes or auto:
        import engine
        def post(t, fixes=fixes, auto=auto):
            if fixes: t = engine.apply_fixes(t, fixes)
            if auto:
                import wordswap
                t = wordswap.apply_auto(t, auto)
            return t
        steps = ([f"{len(fixes)} manual swaps"] if fixes else []) + (["automatic word swap"] if auto else [])
    if terms is None:
        terms = [t.strip() for t in re.split(r"[,;\n]+", expect + "\n" + auto) if t.strip()]
    seen = set(); terms = [t for t in terms if not (t.lower() in seen or seen.add(t.lower()))]
    r = align(cues, words, terms, post)
    n_exp = len([t for t in re.split(r"[,;\n]+", expect) if t.strip()])
    line = (f"{info['minutes']} min on {info['device']} ({info['model']}); words to expect: {n_exp or 'none'}; "
            f"swaps afterwards: {', '.join(steps) or 'none'}")
    rep = build_report(title or name, line, r)
    path = os.path.join(EVAL, "reports", f"{name}_{datetime.datetime.now():%Y-%m-%d_%H-%M-%S}.md")
    open(path, "w", encoding="utf-8").write(rep)
    print("\n" + "\n".join(rep.split("\n")[:34]))
    print(f"\nreport: {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("media"); ap.add_argument("captions")
    ap.add_argument("--expect"); ap.add_argument("--auto"); ap.add_argument("--fixes"); ap.add_argument("--terms")
    ap.add_argument("--gpu", type=int, default=80); ap.add_argument("--minutes", type=float)
    ap.add_argument("--reuse", action="store_true"); ap.add_argument("--name")
    a = ap.parse_args(argv)
    fixes = json.load(open(a.fixes, encoding="utf-8")) if a.fixes else None
    terms = [t.strip() for t in re.split(r"[,;\n]+", read_list(a.terms)) if t.strip()] if a.terms else None
    run(a.media, a.captions, a.name, read_list(a.expect).strip(), read_list(a.auto).strip(), fixes, terms,
        a.gpu, a.minutes, a.reuse)


if __name__ == "__main__":
    main()
