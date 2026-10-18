"""Automatic word swap: turn a heard word or phrase into a known term when it is spelled, or sounds, nearly the same
("bee holder" -> Beholder, "fire ball" -> fireball, "magic missle" -> magic missile).

Conservative on purpose: a wrong swap is worse than a missed one, so short words and ordinary English words are left
alone and the match has to be close. Anything stranger belongs in the manual word swap list.
Big lists are fine: terms are indexed by first sound and length, so only a handful are compared per heard phrase."""
import re
import difflib

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?")
_JOIN = re.compile(r"[\s\-]+")

# ordinary words we never swap, even when a term in the list looks a bit like them
COMMON = frozenset("""about above after again against almost alone along already also although always among another
answer anyone around asked because become before began behind being below between both bring brought called cannot
certain change child close could course doing during early either enough every everyone everything fight first found
front given going great group guess happen heard heavy house inside large later learn leave light little longer might
money mostly mother never night often other others outside people place plain point power quite really right round
seems shall should since small something sound spell still story strong sure taken thank their there these thing think
those though three through today together toward under until using usually visit water where which while whole whose
within without woman world would write wrong young""".split())


def parse_terms(raw):
    """'Beholder, Eldritch Blast; Tiamat' (commas, semicolons or new lines) -> ['Beholder', 'Eldritch Blast', 'Tiamat']"""
    seen, out = set(), []
    for t in re.split(r"[,;\n]+", raw or ""):
        t = t.strip()
        if t and t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return out


def _norm(s):
    return re.sub(r"[^a-z]", "", s.lower())


def _key(w):
    """rough 'how it sounds' key: similar consonants merged, vowels dropped (except the first letter)"""
    s = w
    for a, b in (("ph", "f"), ("ck", "k"), ("qu", "kw"), ("gh", ""), ("kn", "n"), ("wr", "r"), ("x", "ks")):
        s = s.replace(a, b)
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.replace("c", "k").replace("z", "s").replace("v", "f").replace("q", "k")
    s = s[:1] + re.sub(r"[aeiouyhw]", "", s[1:])
    s = re.sub(r"(.)\1+", r"\1", s)
    return s.translate(str.maketrans("dbg", "tpk"))


class AutoSwap:
    MIN_LEN = 5                     # letters; shorter words are too easy to confuse

    def __init__(self, terms):
        self.by = {}                # (first sound, length) -> [(letters, sound key, term)]
        self.max_words = 1
        for t in terms:
            n = _norm(t)
            if len(n) < self.MIN_LEN: continue
            k = _key(n)
            self.by.setdefault((k[:1], len(n)), []).append((n, k, t))
            self.max_words = max(self.max_words, len(_TOKEN.findall(t)))
        self.size = sum(len(v) for v in self.by.values())

    def _best(self, w, size):
        """best known term for the letters w, as (score, term) or None; (1.0, term) when it is already exact"""
        L = len(w); k = _key(w); best = None
        for l2 in range(L - 2, L + 3):
            for tn, tk, term in self.by.get((k[:1], l2), ()):
                if tn == w:
                    return 1.0, term
                sm = difflib.SequenceMatcher(None, w, tn)
                if sm.real_quick_ratio() < 0.66 or sm.quick_ratio() < 0.66: continue
                r = sm.ratio()
                ok = r >= (0.86 if size == 1 else 0.84) or (tk == k and min(L, len(tn)) >= 6 and r >= 0.66)
                if ok and (best is None or r > best[0]): best = (r, term)
        return best

    def apply(self, text):
        if not text or not self.by: return text
        toks = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
        found = []                                              # (score, size, first token, last token, replacement)
        for i in range(len(toks)):
            for size in range(1, min(self.max_words + 1, 4) + 1):
                j = i + size
                if j > len(toks): break
                if size > 1 and not _JOIN.fullmatch(text[toks[j - 2][1]:toks[j - 1][0]]): break
                phrase = text[toks[i][0]:toks[j - 1][1]]
                tail = ""
                m = re.search(r"(['’]s)$", phrase, re.I)
                if m: phrase, tail = phrase[:m.start()], m.group(1)
                w = _norm(phrase)
                if len(w) < self.MIN_LEN or (size == 1 and w in COMMON): continue
                hit = self._best(w, size)
                if hit is None: continue
                score, term = hit
                tn, plural = _norm(term), ""
                if w != tn and w.startswith(tn) and w[len(tn):] in ("s", "es"):               # "beholders" keeps its s
                    plural = w[len(tn):]; score = 1.0
                term += plural
                if phrase == term: found.append((2.0, size, i, j - 1, None)); continue       # already right: protect it
                if score == 1.0 and size == 1 and term == term.lower(): continue             # only the capital differs
                found.append((score, size, i, j - 1, term + tail))
        used, edits = set(), []
        for score, size, a, b, rep in sorted(found, key=lambda f: (-f[0], -f[1], f[2])):
            if any(x in used for x in range(a, b + 1)): continue
            used.update(range(a, b + 1))
            if rep is not None: edits.append((toks[a][0], toks[b][1], rep))
        for s, e, rep in sorted(edits, reverse=True):
            text = text[:s] + rep + text[e:]
        return text


_cache = {}

def apply_auto(text, raw):
    """swap in known terms; raw is the list as typed (comma / new line separated). The index is built once."""
    if not raw or not text: return text
    sw = _cache.get(raw)
    if sw is None:
        if len(_cache) > 4: _cache.clear()
        sw = _cache[raw] = AutoSwap(parse_terms(raw))
    return sw.apply(text)
