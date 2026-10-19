"""
D&D Subtitles - live subtitles for Discord calls, coloured by who is talking.
Run Subtitles.bat. Close the "D&D Subtitles" window to quit.
"""
import sys, os, re, json, random, time, threading, datetime, html, traceback, subprocess
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import engine
import wordswap
import updater
from engine import log

# started without a CLI window (pythonw): status messages and errors go to app.log instead
if sys.stdout is None or sys.stderr is None or "--log" in sys.argv:
    _logf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log"), "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = _logf
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

HERE = engine.HERE
TRANSCRIPTS = os.path.join(HERE, "transcripts")
BOTTOM_MARGIN = 90            # px between the subtitles and the bottom of the screen when the Y offset is 0


MIN_SENTENCE_WORDS = 4


def sentences(text):
    """split at sentence ends, but a "sentence" under 4 words ("Okay.", "Yeah, sure.") runs on into the next one"""
    out, carry = [], ""
    for s in re.split(r"(?<=[.!?…])\s+", text.strip()):
        if not s: continue
        s = (carry + " " + s).strip() if carry else s
        if len(re.findall(r"[\w']+", s)) < MIN_SENTENCE_WORDS: carry = s; continue
        out.append(s); carry = ""
    if carry:
        if out: out[-1] += " " + carry
        else: out.append(carry)
    return out


def ended(text):
    """does this end a real sentence (. ! ? and at least 4 words since the previous sentence end)?"""
    t = text.strip()
    if not re.search(r"[.!?…]$", t): return False
    last = re.split(r"(?<=[.!?…])\s+", t)[-1]
    return len(re.findall(r"[\w']+", last)) >= MIN_SENTENCE_WORDS


def split_rows(text, max_chars=95, fits=None):
    """Rows for display: a new row after each sentence end; a long sentence is broken after a comma.
    fits(s) -> does s fit on one row (measured); without it, rows are limited to max_chars characters."""
    fits = fits or (lambda x: len(x) <= max_chars)
    rows = []
    for sent in sentences(text):
        if not sent: continue
        if fits(sent): rows.append(sent); continue
        cur = ""; clauses = re.split(r"(?<=,)\s+", sent)
        for i, c in enumerate(clauses):
            rest = len(" ".join(clauses[i:]))
            if cur and not fits(cur + " " + c) and rest >= 30: rows.append(cur); cur = c   # no tiny leftovers
            else: cur = (cur + " " + c).strip()
        if cur: rows.append(cur)
    return rows


HIGHLIGHT = re.compile(r"\bnat(?:ural)?\.?[\s-]*(?:20|twenty)\b", re.I)     # "natural 20", "nat 20", "natural twenty"
GLOW = "#FFE07A"


LINE = re.compile(r"^\[(\d\d):(\d\d):(\d\d)\] (.*?): (.*)$")


def _read_lines(path):
    """[(seconds since the file's day started, line)] - times after midnight count as the next day"""
    out, day, prev = [], 0, None
    for raw in open(path, encoding="utf-8", errors="replace"):
        m = LINE.match(raw.rstrip("\n"))
        if not m: continue
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        if prev is not None and t + day < prev - 12 * 3600: day += 24 * 3600
        prev = t + day; out.append((t + day, raw.rstrip("\n")))
    return out


def merge_transcripts(dest_folder, sources, transcripts_root):
    """Fold transcript files into one file per date in dest_folder (time order, no duplicates).
    The source files are moved to the "old sessions" folder (nothing is deleted)."""
    os.makedirs(dest_folder, exist_ok=True)
    old = os.path.join(transcripts_root, "old sessions"); os.makedirs(old, exist_ok=True)
    by_date = {}
    for src in sources:
        name = os.path.basename(src)
        m = re.search(r"(\d{4}-\d\d-\d\d)", name)
        if not m or not name.endswith(".txt") or not os.path.isfile(src): continue
        by_date.setdefault(m.group(1), []).append(src)
    for date, srcs in by_date.items():
        dest = os.path.join(dest_folder, date + ".txt")
        lines = (_read_lines(dest) if os.path.exists(dest) else [])
        for src in srcs:
            if os.path.abspath(src) != os.path.abspath(dest): lines += _read_lines(src)
        seen, merged = set(), []
        for t, line in sorted(lines, key=lambda x: x[0]):
            if line in seen: continue
            seen.add(line); merged.append(line)
        tmp = dest + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f: f.write("\n".join(merged) + "\n")
        os.replace(tmp, dest)
        for src in srcs:
            if os.path.abspath(src) == os.path.abspath(dest): continue
            target = os.path.join(old, os.path.basename(os.path.dirname(src)) + " - " + os.path.basename(src)
                                  if os.path.dirname(os.path.abspath(src)) != os.path.abspath(transcripts_root)
                                  else os.path.basename(src))
            try: os.replace(src, target)
            except Exception as e: log(f"[transcript] could not move {src}: {e}")
    log(f"[transcript] merged {sum(len(v) for v in by_date.values())} file(s) into {dest_folder}")


class GapDelegate(QtWidgets.QStyledItemDelegate):
    """Transcript rows drawn as rich text: no gap between lines of the same person, a small gap where a new
    person starts, and "natural 20" in bold + underlined with a soft golden glow."""
    GAP = 7
    PAD = 4
    R = 3                    # glow radius (px)

    def _glow(self, option, index):
        """the highlighted words, blurred, as a picture (cached)"""
        fg = index.data(Qt.ForegroundRole)
        key = (index.data(Qt.DisplayRole), option.rect.width(), option.font.pixelSize(), option.font.pointSizeF(),
               fg.color().name() if fg is not None else "")
        if not hasattr(self, "_glows"): self._glows = {}
        img = self._glows.get(key)
        if img is not None: return img
        g = self._doc(option, index, glow_only=True)
        w, h = int(g.size().width()) + 2 * self.R, int(g.size().height()) + 2 * self.R
        src = QtGui.QImage(w, h, QtGui.QImage.Format_ARGB32); src.fill(Qt.transparent)
        p = QtGui.QPainter(src); p.translate(self.R, self.R); g.drawContents(p); p.end()
        a = np.frombuffer(src.constBits(), dtype=np.uint8).reshape(h, src.bytesPerLine() // 4, 4)[:, :w, 3].astype(np.float32)
        k = np.ones(self.R) / self.R
        for _ in range(1):                          # one light blur
            a = np.apply_along_axis(lambda r: np.convolve(r, k, mode="same"), 1, a)
            a = np.apply_along_axis(lambda c: np.convolve(c, k, mode="same"), 0, a)
        a = np.clip(a * 1.5, 0, 255 * 0.6)           # faint, tight
        fg = index.data(Qt.ForegroundRole)
        c = fg.color() if fg is not None else QtGui.QColor(GLOW)
        rgba = np.zeros((h, w, 4), np.uint8)
        rgba[..., 0], rgba[..., 1], rgba[..., 2] = c.blue(), c.green(), c.red()      # BGRA in memory
        rgba[..., 3] = a.astype(np.uint8)
        img = QtGui.QImage(rgba.tobytes(), w, h, 4 * w, QtGui.QImage.Format_ARGB32).copy()
        if len(self._glows) > 300: self._glows.clear()
        self._glows[key] = img
        return img

    def _html(self, text, color, glow_only=False):
        out, last = [], 0
        for m in HIGHLIGHT.finditer(text):
            out.append(html.escape(text[last:m.start()]))
            out.append(f'<b><u>{html.escape(m.group(0))}</u></b>' if not glow_only
                       else f'<span style="color:{GLOW}"><b><u>{html.escape(m.group(0))}</u></b></span>')
            last = m.end()
        out.append(html.escape(text[last:]))
        base = "transparent" if glow_only else color
        return f'<span style="color:{base}">{"".join(out).replace("  ", "&nbsp; ")}</span>'

    def _doc(self, option, index, glow_only=False):
        text = index.data(Qt.DisplayRole) or ""
        fg = index.data(Qt.ForegroundRole)
        color = fg.color().name() if fg is not None else "#dddddd"
        doc = QtGui.QTextDocument(); doc.setDefaultFont(option.font); doc.setDocumentMargin(0)
        doc.setHtml(self._html(text, color, glow_only))
        w = (option.rect.width() if option.rect.width() > 50 else self.parent().viewport().width()) - 2 * self.PAD
        doc.setTextWidth(max(w, 50))
        return doc

    def sizeHint(self, option, index):
        doc = self._doc(option, index)
        gap = self.GAP if index.data(Qt.UserRole + 1) else 0
        return QtCore.QSize(int(doc.idealWidth()) + 2 * self.PAD, int(doc.size().height()) + 2 + gap)

    def paint(self, painter, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option); self.initStyleOption(opt, index)
        if index.data(Qt.UserRole + 1): opt.rect.adjust(0, self.GAP, 0, 0)
        opt.text = ""
        opt.widget.style().drawControl(QtWidgets.QStyle.CE_ItemViewItem, opt, painter, opt.widget)   # selection etc.
        text = index.data(Qt.DisplayRole) or ""
        painter.save()
        painter.translate(opt.rect.left() + self.PAD, opt.rect.top() + 1)
        if HIGHLIGHT.search(text):                  # soft golden glow behind the highlighted words
            painter.drawImage(QtCore.QPointF(-self.R, -self.R), self._glow(opt, index))
        self._doc(opt, index).drawContents(painter)
        painter.restore()


class Bridge(QtCore.QObject):
    """Thread-safe hand-off from the audio/AI threads to the UI thread."""
    event = QtCore.Signal(dict)
    voices_changed = QtCore.Signal(str)
    status = QtCore.Signal(str)
    ready = QtCore.Signal()


# =============================================================================================== overlay
class Overlay(QtWidgets.QWidget):
    """Caption box at the bottom of the screen.
    Calm on purpose: text always starts at the same left edge and grows to the right; new lines go below the
    previous ones; when the box is full everything scrolls up smoothly; words already shown don't change while
    someone is still talking; the whole block fades out together after a pause."""
    def __init__(self, cfg, bank):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool |
                         Qt.WindowTransparentForInput | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setWindowTitle("D&D Subtitles overlay")
        self.cfg, self.bank = cfg, bank
        self.lines = []          # dict(voice, final, putt, ptext, committed, hyp, t)
        self.ypos = {}           # animated y of each box
        self.last_activity = 0.0
        self._docs = {}
        self._last_tick = time.time()
        self.place()
        self.timer = QtCore.QTimer(self, interval=16, timeout=self.tick); self.timer.start()
        self._last_raise = 0

    def place(self):
        screens = QtGui.QGuiApplication.screens()
        sc = screens[min(max(int(self.cfg.get("screen", 0)), 0), len(screens) - 1)]
        g = sc.geometry()
        w = int(g.width() * self.cfg.get("overlay_width_pct", 75) / 100)
        h = int(g.height() * 0.6)
        # 0, 0 = centred, a little above the bottom edge. x: + right, - left.  y: + down, - up.
        x = g.x() + (g.width() - w) // 2 + int(self.cfg.get("overlay_x_px", 0))
        y = g.y() + g.height() - h - BOTTOM_MARGIN + int(self.cfg.get("overlay_y_px", 0))
        self.setGeometry(x, y, w, h)
        self._docs.clear(); self.update()

    # ---- content
    @staticmethod
    def _norm(w): return re.sub(r"[^\w']", "", w.lower())

    def _stable(self, l, text):
        """Words that two guesses in a row agree on are locked in; only the words after them may still change."""
        new = text.split(); prev = l.get("hyp", "").split()
        k = 0
        while k < min(len(new), len(prev)) and self._norm(new[k]) == self._norm(prev[k]): k += 1
        if k > len(l["committed"]): l["committed"] = new[:k]
        l["hyp"] = text
        c = l["committed"]
        l["ptext"] = " ".join(c + new[len(c):]) if len(new) >= len(c) else " ".join(c)

    def _new_line(self, voice, now):
        l = {"voice": voice, "final": "", "putt": None, "ptext": "", "committed": [], "hyp": "", "t": now}
        self.lines.append(l); return l

    def _can_join(self, l, voice, now):
        """same person and their last line stopped mid-sentence -> carry on in that line;
        a new line only starts after a sentence end (. ! ?)"""
        return (l is not None and l["voice"] == voice and voice is not None and l["putt"] is None
                and now - l["t"] < 20 and len(l["final"]) < 600 and not ended(l["final"]))

    def on_event(self, e):
        if e.get("source") == "me": return          # your own voice goes to the transcript only
        now = time.time(); self.last_activity = now
        last = self.lines[-1] if self.lines else None
        if e["kind"] == "partial":
            l = next((x for x in self.lines if x["putt"] == e["utt"]), None)
            if l is None:
                if e.get("new_speaker"):          # the app heard a different person take over: own line
                    join = e["voice"] is not None and self._can_join(last, e["voice"], now)
                else:
                    join = self._can_join(last, e["voice"], now) or (
                        e["voice"] is None and last is not None and self._can_join(last, last["voice"], now))
                l = last if join else self._new_line(e["voice"], now)
                l["putt"] = e["utt"]; l["committed"] = []; l["hyp"] = ""
            if not l["final"] and e["voice"] is not None: l["voice"] = e["voice"]
            self._stable(l, e["text"]); l["t"] = now
        else:
            l = next((x for x in self.lines if x["putt"] == e["utt"]), None)
            parts = list(e["parts"])
            if l is not None:
                l["putt"] = None; l["ptext"] = ""
                if parts:
                    v, txt = parts[0]
                    if not l["final"]: l["voice"] = v
                    if l["voice"] == v:
                        l["final"] = (l["final"] + " " + txt).strip(); l["t"] = now; parts = parts[1:]
                if not l["final"]: self.lines.remove(l)
            for v, txt in parts:
                last = self.lines[-1] if self.lines else None
                if self._can_join(last, v, now): last["final"] = (last["final"] + " " + txt).strip(); last["t"] = now
                else: self._new_line(v, now)["final"] = txt
        self.update()

    def demo(self):
        snap = self.bank.snapshot()
        samples = ["I want to roll for perception.", "The door creaks open and you see a long hallway lit by torches.",
                   "Natural twenty!", "Wait, are we sure this isn't a trap?", "Roll initiative.", "Okay, I'm in."]
        if not snap:
            self.on_event({"kind": "final", "utt": -1, "parts": [(None, "No voices yet - they appear as people talk.")]})
        for i, v in enumerate(snap[:6]):
            self.on_event({"kind": "final", "utt": -1 - i, "parts": [(v["id"], samples[i % len(samples)])]})

    # ---- test run: made-up conversation on the overlay only (nothing reaches the transcript or the voice list)
    TEST_PEOPLE = [("Alex", "#7FD1FF"), ("Sam", "#FFB86B"), ("Jordan", "#9BE37D")]
    TEST_SHORT = ["Okay.", "Yeah, sure.", "Wait, really?", "That works for me.", "Sounds good.", "I think so, yes.",
                  "No way.", "Let's do it.", "Hold on a second.", "Right, makes sense."]
    TEST_MEDIUM = ["I'm not sure that is going to work, but we can try it again tomorrow morning.",
                   "Did anyone actually read the message that came in last night about the schedule?",
                   "Honestly, I would rather finish this part first and then look at the rest together.",
                   "That was a lot louder than I expected, so I turned the volume down a bit."]
    TEST_LONG = ["So what happened was, we waited for almost an hour, and then somebody finally came out and told us "
                 "the whole thing had been moved to the other building, which nobody had thought to mention. After that "
                 "we walked across the square in the rain, found the right door, and of course it was locked.",
                 "The way I see it, there are really two options here. We can keep going the way we have been, which is "
                 "slow but safe, or we can split up, cover twice as much ground, and meet again at the end of the day "
                 "to compare notes, as long as everybody remembers to check in on time."]

    def start_test(self):
        self._test_on = True; self._test_u = None; self._test_wait = 0; self._test_n = getattr(self, "_test_n", 0)
        snap = self.bank.snapshot()[:3]
        self._test_ppl = [v["id"] for v in snap] + [f"test:{i}" for i in range(len(snap), 3)]
        self._test_names = {f"test:{i}": self.TEST_PEOPLE[i] for i in range(3)}
        if not hasattr(self, "_test_timer"):
            self._test_timer = QtCore.QTimer(self, interval=230, timeout=self._test_step)
        self._test_timer.start()

    def stop_test(self):
        self._test_on = False
        if hasattr(self, "_test_timer"): self._test_timer.stop()

    def test_names(self, v):
        """(name, colour) for a made-up test voice, or None for a real one"""
        return getattr(self, "_test_names", {}).get(v)

    def _test_scene(self):
        """-> [(voice, text)]: one line, a quick back-and-forth, one person with several lines, or a long speech"""
        r = random.random(); a, b, c = random.sample(self._test_ppl, 3)
        if r < 0.22: return [(a, random.choice(self.TEST_SHORT))]
        if r < 0.42: return [(a, random.choice(self.TEST_MEDIUM))]
        if r < 0.60: return [(a, random.choice(self.TEST_SHORT)), (b, random.choice(self.TEST_SHORT)),
                             (a, random.choice(self.TEST_MEDIUM))]
        if r < 0.75: return [(a, random.choice(self.TEST_MEDIUM)), (a, random.choice(self.TEST_SHORT)),
                             (b, random.choice(self.TEST_SHORT))]
        if r < 0.90: return [(a, random.choice(self.TEST_LONG))]
        return [(a, random.choice(self.TEST_LONG)), (b, random.choice(self.TEST_SHORT)), (c, random.choice(self.TEST_MEDIUM))]

    def _test_step(self):
        if not getattr(self, "_test_on", False): return
        if self._test_wait > 0: self._test_wait -= 1; return
        if self._test_u is None:
            if not getattr(self, "_test_queue", None): self._test_queue = self._test_scene()
            v, text = self._test_queue.pop(0)
            self._test_n += 1
            self._test_u = dict(utt=-1000 - self._test_n, voice=v, words=text.split(), k=0, new=True)
        u = self._test_u
        u["k"] = min(len(u["words"]), u["k"] + random.randint(1, 3))
        txt = " ".join(u["words"][:u["k"]])
        if u["k"] < len(u["words"]):
            self.on_event({"kind": "partial", "utt": u["utt"], "voice": u["voice"], "text": txt, "new_speaker": u["new"]})
            u["new"] = False
        else:
            self.on_event({"kind": "final", "utt": u["utt"], "parts": [(u["voice"], txt)]})
            self._test_u = None
            self._test_wait = random.choice([0, 1, 1, 2, 3, 6, 12]) if not self._test_queue else random.choice([0, 1, 2])

    # ---- layout
    def _metrics(self):
        fs = int(self.cfg.get("font_size", 32))
        font = QtGui.QFont("Segoe UI"); font.setPixelSize(fs); font.setWeight(QtGui.QFont.DemiBold)
        padx, pady, gap = int(fs * 0.55), int(fs * 0.22), int(fs * 0.25)
        ls = QtGui.QFontMetricsF(font).lineSpacing()
        rows = max(1, int(self.cfg.get("max_lines", 3)))
        region = rows * (ls + 2 * pady + gap) - gap
        return fs, font, padx, pady, gap, region

    def _groups(self):
        """consecutive lines from the same person share one box (name shown once)"""
        out = []
        for l in self.lines:
            v = self.bank.resolve(l["voice"]) if l["voice"] is not None else None
            if out and out[-1][0] == v and v is not None: out[-1][1].append(l)
            else: out.append((v, [l]))
        return out

    def _rows(self, group, font, width):
        """-> name, colour, name width, [(row html, row plain text)] for one person's box"""
        v, ls = group
        name, color = (self.test_names(v) or self.bank.get(v)) if v is not None else ("?", engine.UNKNOWN_COLOR)
        fm = QtGui.QFontMetricsF(font); bold = QtGui.QFont(font); bold.setBold(True)
        nw = QtGui.QFontMetricsF(bold).horizontalAdvance(name + ": ")
        maxc = max(20, int((width - nw) / (fm.averageCharWidth() * 1.08)))     # what fits on one row of the box
        rows = []
        for l in ls:
            room = (width - nw) * 0.98
            r = [[html.escape(x), x] for x in split_rows(l["final"], maxc, fits=lambda t: fm.horizontalAdvance(t) <= room)]
            if l["ptext"]:
                if r and not ended(l["final"]): r[-1][0] += " " + html.escape(l["ptext"]); r[-1][1] += " " + l["ptext"]
                else: r.append([html.escape(l["ptext"]), l["ptext"]])
            rows += r
        return name, color, nw, [tuple(x) for x in rows] or [("", "")]

    def _doc(self, name, color, nw, rows, font, width):
        key = (name, color, tuple(r[0] for r in rows), width, font.pixelSize())
        d = self._docs.get(key)
        if d is None:
            doc = QtGui.QTextDocument(); doc.setDefaultFont(font); doc.setDocumentMargin(0)
            # name in its own column, each line of speech below the previous one, lined up after the name
            doc.setHtml(f'<table cellspacing="0" cellpadding="0" style="color:{color}"><tr>'
                        f'<td valign="top" width="{int(nw) + 1}"><b>{html.escape(name)}:</b></td>'
                        f'<td valign="top">{"<br>".join(r[0] for r in rows)}</td></tr></table>')
            doc.setTextWidth(width)
            probe = QtGui.QTextDocument(); probe.setDefaultFont(font); probe.setDocumentMargin(0)
            probe.setHtml("<br>".join(r[0] for r in rows)); probe.setTextWidth(max(width - nw, 50))
            tw = min(nw + probe.idealWidth() + 2, width)
            d = (doc, tw, doc.size().height(), len(self._row_tops(doc)))
            if len(self._docs) > 600: self._docs.clear()
            self._docs[key] = d
        return d

    @staticmethod
    def _row_tops(doc):
        """y of every line of text in a laid-out document"""
        tops = set(); lay = doc.documentLayout(); b = doc.begin()
        while b.isValid():
            r = lay.blockBoundingRect(b); bl = b.layout()
            for i in range(bl.lineCount()): tops.add(round(r.top() + bl.lineAt(i).y(), 2))
            b = b.next()
        return sorted(tops)

    def _layout(self, font, width):
        """Which boxes (and which of their lines) fit in the caption area, newest first.
        Every box starts with its name. A box that doesn't fit whole keeps only its newest sentences
        (starting right after a . ? !); older boxes are dropped."""
        budget = max(1, int(self.cfg.get("max_lines", 3)))
        out = []
        for g in reversed(self._groups()):
            name, color, nw, rows = self._rows(g, font, width)
            d = self._doc(name, color, nw, rows, font, width)
            if d[3] <= budget:
                out.insert(0, (g, d)); budget -= d[3]; continue
            starts = [i for i in range(len(rows)) if i == 0 or re.search(r"[.!?…]$", rows[i - 1][1].strip())]
            pick = None
            for i in starts:                                  # most text that still fits, from a sentence start
                dd = self._doc(name, color, nw, rows[i:], font, width)
                if dd[3] <= budget: pick = dd; break
            if pick is None and not out:                      # one very long sentence: its newest lines
                for i in range(len(rows)):
                    dd = self._doc(name, color, nw, rows[i:], font, width)
                    if dd[3] <= budget: pick = dd; break
                if pick is None:                              # even the last line is too long: its newest words
                    words_h, words_p = rows[-1][0].split(" "), rows[-1][1].split(" ")
                    owner = g[1][-1]
                    def cut(k):
                        return [("… " + " ".join(words_h[k:]), "… " + " ".join(words_p[k:]))]
                    k = owner.get("trim", 0)
                    if not (0 < k < len(words_p)) or self._doc(name, color, nw, cut(k), font, width)[3] > budget:
                        # cut again, leaving a free line so the text doesn't shift on every new word
                        room = budget - 1 if budget > 1 else 1
                        lo, hi = 1, len(words_p)
                        while lo < hi:
                            mid = (lo + hi + 1) // 2
                            if self._doc(name, color, nw, cut(len(words_p) - mid), font, width)[3] <= room: lo = mid
                            else: hi = mid - 1
                        k = len(words_p) - lo; owner["trim"] = k
                    pick = self._doc(name, color, nw, cut(k), font, width)
            if pick is not None: out.insert(0, (g, pick))
            break
        return out

    def tick(self):
        now = time.time(); dt = min(now - self._last_tick, 0.1); self._last_tick = now
        ttl = float(self.cfg.get("line_seconds", 9))
        changed = False
        if self.lines and now - self.last_activity > ttl + 1.0:       # quiet for a while: clear (after fading)
            self.lines = []; self.ypos = {}; changed = True
        if self.lines:
            fs, font, padx, pady, gap, region = self._metrics()
            width = self.width() - 2 * padx
            lay = self._layout(font, width)
            shown = {id(g[1][0]) for g, _ in lay}
            # forget lines from boxes older than everything shown (unless someone is still talking in them)
            first = lay[0][0][1][0] if lay else None
            if first is not None:
                k = self.lines.index(first)
                if k > 0 and all(l["putt"] is None for l in self.lines[:k]): del self.lines[:k]
            # boxes slide to their place (no fading, nothing cut off at the top)
            if not hasattr(self, "ypos"): self.ypos = {}
            y = 0.0; targets = {}
            for g, d in lay:
                targets[id(g[1][0])] = y; y += d[2] + 2 * pady + gap
            for key, t in targets.items():
                cur = self.ypos.get(key, t)
                if abs(t - cur) > 0.5: cur += (t - cur) * min(1.0, dt * 10); changed = True
                else:
                    if cur != t: changed = True
                    cur = t
                self.ypos[key] = cur
            self.ypos = {k: v for k, v in self.ypos.items() if k in shown}
            self._lay = lay
            if now - self.last_activity > ttl: changed = True                 # fading
        if changed: self.update()
        if now - self._last_raise > 3 and self.isVisible():
            self.raise_(); self._last_raise = now          # games sometimes push us behind

    def paintEvent(self, ev):
        if not self.lines: return
        p = QtGui.QPainter(self); p.setRenderHint(QtGui.QPainter.Antialiasing); p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        fs, font, padx, pady, gap, region = self._metrics()
        W, H = self.width(), self.height()
        width = W - 2 * padx
        top = H - region
        ttl = float(self.cfg.get("line_seconds", 9))
        quiet = time.time() - self.last_activity
        p.setOpacity(1.0 if quiet < ttl else max(0.0, 1.0 - (quiet - ttl)))
        bg = float(self.cfg.get("background_opacity", 0.55))
        lay = self._layout(font, width)            # always the current text; positions come from tick()
        ypos = getattr(self, "ypos", {})
        prev_bottom = None
        for g, (doc, tw, th, _) in lay:
            h = th + 2 * pady
            y = ypos.get(id(g[1][0]))
            if y is None: y = prev_bottom + gap if prev_bottom is not None else 0.0
            if prev_bottom is not None: y = max(y, prev_bottom + gap)      # never overlap the box above
            p.setPen(Qt.NoPen); p.setBrush(QtGui.QColor(0, 0, 0, int(255 * bg)))
            p.drawRoundedRect(QtCore.QRectF(0, top + y, tw + 2 * padx, h), fs * 0.3, fs * 0.3)
            p.save(); p.translate(padx, top + y + pady); doc.drawContents(p); p.restore()
            prev_bottom = y + h
        p.end()


# =============================================================================================== control window
class RescanCombo(QtWidgets.QComboBox):
    """A drop-down that asks for its list to be rebuilt every time it is opened (so a headset plugged in
    a minute ago, or a virtual cable that was started later, shows up without a Refresh button)."""
    about_to_show = QtCore.Signal()
    def showPopup(self):
        self.about_to_show.emit()
        super().showPopup()


class Panel(QtWidgets.QWidget):
    update_found = QtCore.Signal(object)         # result of a background check: a dict, or the exception
    update_done = QtCore.Signal(object)         # result of a background install: (ok, text, changed files)

    def __init__(self, cfg, bank, overlay, bridge):
        super().__init__()
        self.cfg, self.bank, self.overlay, self.bridge = cfg, bank, overlay, bridge
        self.worker = None; self.pipeline = None
        self.setWindowTitle("D&D Subtitles")
        self.resize(1380, 720)
        outer = QtWidgets.QHBoxLayout(self)
        split = QtWidgets.QSplitter(Qt.Horizontal); outer.addWidget(split)
        left = QtWidgets.QWidget(); lay = QtWidgets.QVBoxLayout(left); lay.setContentsMargins(0, 0, 0, 0)
        split.addWidget(left)
        right = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        th = QtWidgets.QHBoxLayout()
        th.addWidget(QtWidgets.QLabel("<b>Transcript</b> <span style='color:#888'>(right-click a line to fix who said it)</span>"))
        th.addStretch(1)
        self.cb_hide = QtWidgets.QCheckBox("Hide short unknown bits")
        self.cb_hide.setToolTip("Hides one- or two-word lines nobody was recognised for (“?: Uh,”)")
        self.cb_hide.setChecked(cfg.get("hide_short_unknown", True))
        self.cb_hide.toggled.connect(lambda v: (self.set("hide_short_unknown", v), self.repaint_transcript()))
        th.addWidget(self.cb_hide)
        rl.addLayout(th)
        self.tlist = QtWidgets.QListWidget()
        self.tlist.setWordWrap(True); self.tlist.setTextElideMode(Qt.ElideNone)
        self.tlist.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.tlist.setStyleSheet("QListWidget{background:#1b1b1f; border:1px solid #333; font-size:14px}"
                                 "QListWidget::item{padding:0px 4px; margin:0px}"
                                 "QListWidget::item:selected{background:#3a3a46}")
        self.tlist.setItemDelegate(GapDelegate(self.tlist))
        self.tlist.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)   # ctrl/shift-click for several
        QtGui.QShortcut(QtGui.QKeySequence.Copy, self.tlist, activated=self.copy_lines)
        self.tlist.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tlist.customContextMenuRequested.connect(self.transcript_menu)
        rl.addWidget(self.tlist)
        split.addWidget(right); split.setSizes([820, 560])

        grow = QtWidgets.QHBoxLayout()
        grow.addWidget(QtWidgets.QLabel("<b>Group:</b>"))
        self.group = QtWidgets.QComboBox(); self.group.setMinimumWidth(200); grow.addWidget(self.group)
        b = QtWidgets.QPushButton("New group…"); b.clicked.connect(self.new_group); grow.addWidget(b)
        b = QtWidgets.QPushButton("Rename group…"); b.clicked.connect(self.rename_group); grow.addWidget(b)
        grow.addStretch(1)
        lay.addLayout(grow)
        self.fill_groups()
        self.group.currentIndexChanged.connect(lambda _: self.switch_group(self.group.currentText()))
        self.status = QtWidgets.QLabel("Loading Whisper on the GPU… (first start downloads ~1.6 GB)")
        self.status.setWordWrap(True); lay.addWidget(self.status)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("System audio:"))
        self.dev = RescanCombo(); self.dev.setMinimumWidth(260)
        self.dev.setToolTip("What you hear: the output the call plays through. Automatic picks NVIDIA Broadcast speakers if you have them, "
                          "otherwise the Windows default. The list refreshes each time you open it.")
        self.dev.about_to_show.connect(self.fill_devices); row.addWidget(self.dev, 1)
        row.addSpacing(16)
        self.cb_me = QtWidgets.QCheckBox("My mic:")
        self.cb_me.setToolTip("Add your own voice to the transcript. Untick it when you only listen (a film, a stream, a recording) "
                              "and there is nobody on your side to record.")
        self.cb_me.setChecked(cfg.get("my_voice", True)); row.addWidget(self.cb_me)
        self.mic_dev = RescanCombo(); self.mic_dev.setMinimumWidth(260)
        self.mic_dev.setToolTip("Your own microphone (added to the transcript only). Automatic picks the NVIDIA Broadcast mic if you have one, "
                              "otherwise the Windows default. The list refreshes each time you open it.")
        self.mic_dev.about_to_show.connect(self.fill_mics); row.addWidget(self.mic_dev, 1)
        self.mic_status = QtWidgets.QLabel(""); self.mic_status.setStyleSheet("color:#888"); row.addWidget(self.mic_status)
        row.addSpacing(16)
        row.addWidget(QtWidgets.QLabel("Sound:"))
        self.meter = QtWidgets.QProgressBar(); self.meter.setRange(0, 60); self.meter.setTextVisible(False)
        self.meter.setFixedWidth(120); row.addWidget(self.meter)
        lay.addLayout(row)

        box = QtWidgets.QGroupBox("Voices"); bl = QtWidgets.QVBoxLayout(box)
        hint = QtWidgets.QLabel("People the app doesn't know show as “?” in the transcript: right-click a line to add them "
                                "or say who it is. If one person is here twice, use “Merge into” to join them.")
        hint.setWordWrap(True); hint.setStyleSheet("color:#888"); bl.addWidget(hint)
        self.grid_host = QtWidgets.QWidget(); self.grid = QtWidgets.QGridLayout(self.grid_host)
        self.grid.setColumnStretch(3, 1)
        sc = QtWidgets.QScrollArea(); sc.setWidgetResizable(True); sc.setWidget(self.grid_host)
        bl.addWidget(sc); lay.addWidget(box, 1)
        self.rows = {}

        opt = QtWidgets.QGroupBox("Options"); g = QtWidgets.QGridLayout(opt)
        self.cb_part = QtWidgets.QCheckBox("Show text while people are still talking"); self.cb_part.setChecked(cfg["show_partial"])
        self.cb_part.toggled.connect(lambda v: self.set("show_partial", v))
        self.cb_show = QtWidgets.QCheckBox("Show subtitles"); self.cb_show.setChecked(True)
        self.cb_show.toggled.connect(lambda v: self.overlay.setVisible(v))
        g.addWidget(self.cb_show, 0, 0); g.addWidget(self.cb_part, 0, 1, 1, 3)
        g.setColumnStretch(1, 1); g.setColumnStretch(3, 1)

        def spin(key, lo, hi, step=1, suffix=""):
            s = QtWidgets.QSpinBox(); s.setRange(lo, hi); s.setSingleStep(step); s.setValue(int(cfg[key])); s.setSuffix(suffix)
            s.valueChanged.connect(lambda v: (self.set(key, v), self.overlay.place())); return s
        g.addWidget(QtWidgets.QLabel("Text size"), 1, 0); g.addWidget(spin("font_size", 12, 72, 2, " px"), 1, 1)
        g.addWidget(QtWidgets.QLabel("Width"), 1, 2); g.addWidget(spin("overlay_width_pct", 30, 100, 5, " %"), 1, 3)
        g.addWidget(QtWidgets.QLabel("Lines on screen"), 2, 0); g.addWidget(spin("max_lines", 1, 8), 2, 1)
        g.addWidget(QtWidgets.QLabel("Seconds on screen"), 2, 2); g.addWidget(spin("line_seconds", 2, 60), 2, 3)
        g.addWidget(QtWidgets.QLabel("Screen"), 3, 0)
        self.scr = QtWidgets.QComboBox()
        for i, s in enumerate(QtGui.QGuiApplication.screens()):
            self.scr.addItem(f"{i + 1}: {s.name()} ({s.geometry().width()}×{s.geometry().height()})")
        self.scr.setCurrentIndex(min(int(cfg.get("screen", 0)), self.scr.count() - 1))
        self.scr.currentIndexChanged.connect(lambda i: (self.set("screen", i), self.overlay.place()))
        g.addWidget(self.scr, 3, 1, 1, 3)

        screens = QtGui.QGuiApplication.screens()
        geo = screens[min(int(cfg.get("screen", 0)), len(screens) - 1)].geometry()
        def offset(label, tip, key, span, row):
            """slider + typed pixel box that stay in sync; 0 = default position, negatives allowed"""
            g.addWidget(QtWidgets.QLabel(label), row, 0)
            h = QtWidgets.QHBoxLayout()
            sl = QtWidgets.QSlider(Qt.Horizontal); sl.setRange(-span, span)
            sp = QtWidgets.QSpinBox(); sp.setRange(-5000, 5000); sp.setSuffix(" px"); sp.setFixedWidth(95)
            sp.setValue(int(cfg.get(key, 0))); sl.setValue(max(-span, min(span, sp.value())))
            sl.setToolTip(tip); sp.setToolTip(tip)
            def apply(v): self.set(key, v); self.overlay.place()
            def from_slider(v):
                sp.blockSignals(True); sp.setValue(v); sp.blockSignals(False); apply(v)
            def from_spin(v):
                sl.blockSignals(True); sl.setValue(max(-span, min(span, v))); sl.blockSignals(False); apply(v)
            sl.valueChanged.connect(from_slider); sp.valueChanged.connect(from_spin)
            sl.sliderReleased.connect(self.overlay.demo)
            rb = QtWidgets.QPushButton("Reset"); rb.setFixedWidth(60); rb.clicked.connect(lambda: sp.setValue(0))
            h.addWidget(sl, 1); h.addWidget(sp); h.addWidget(rb)
            g.addLayout(h, row, 1, 1, 3)
        offset("Horizontal offset", "0 = centred. Negative moves left, positive moves right.", "overlay_x_px", geo.width() // 2, 4)
        offset("Vertical offset", "0 = near the bottom. Negative moves up, positive moves down.", "overlay_y_px", geo.height() // 2, 5)

        g.addWidget(QtWidgets.QLabel("Background darkness"), 7, 0)
        bgs = QtWidgets.QSlider(Qt.Horizontal); bgs.setRange(0, 100)
        bgs.setValue(int(round(float(cfg.get("background_opacity", 0.55)) * 100)))
        bgl = QtWidgets.QLabel(f"{bgs.value()} %"); bgl.setFixedWidth(45)
        def bg_changed(v):
            bgl.setText(f"{v} %"); self.set("background_opacity", v / 100); self.overlay.update()
        bgs.valueChanged.connect(bg_changed)
        bgs.sliderReleased.connect(self.overlay.demo)
        g.addWidget(bgs, 7, 1, 1, 2); g.addWidget(bgl, 7, 3)
        self.me_name_label = QtWidgets.QLabel("My name")
        g.addWidget(self.me_name_label, 8, 2)
        self.me_name = QtWidgets.QLineEdit(cfg.get("my_name", "Me"))
        self.me_name.editingFinished.connect(lambda: self.set_my_name(self.me_name.text()))
        g.addWidget(self.me_name, 8, 3)
        lay.addWidget(opt)

        wf = QtWidgets.QGroupBox("Word fixing"); wl = QtWidgets.QGridLayout(wf)
        def heading(t):
            l = QtWidgets.QLabel(f"<b>{t}</b>"); return l
        def hint(t):
            l = QtWidgets.QLabel(t); l.setWordWrap(True); l.setStyleSheet("color:#888"); return l
        self.lbl_manual = QtWidgets.QLabel(); self.lbl_manual.setStyleSheet("color:#888")
        eb = QtWidgets.QPushButton("Edit…"); eb.clicked.connect(self.edit_word_fixes)
        wl.addWidget(heading("Manual word swap (low impact)"), 0, 0); wl.addWidget(self.lbl_manual, 0, 1); wl.addWidget(eb, 0, 2)
        wl.addWidget(hint("Always change one exact word or phrase into another, e.g. straw → Strahd. Runs first."), 1, 0, 1, 3)
        self.lbl_auto = QtWidgets.QLabel(); self.lbl_auto.setStyleSheet("color:#888")
        wl.addWidget(heading("Automatic word swap (medium impact)"), 2, 0); wl.addWidget(self.lbl_auto, 2, 1, 1, 2)
        self.auto_edit = QtWidgets.QPlainTextEdit(); self.auto_edit.setFixedHeight(62)
        self.auto_edit.setPlaceholderText("Only the correct words, e.g.  Strahd, Waterdeep, Beholder")
        wl.addWidget(self.auto_edit, 3, 0, 1, 3)
        wl.addWidget(hint("Just the correct word. The app finds the wrong versions itself: “bee holder” → Beholder."), 4, 0, 1, 3)
        self.lbl_expect_h = QtWidgets.QLabel(); self.lbl_expect = QtWidgets.QLabel()
        wl.addWidget(self.lbl_expect_h, 5, 0); wl.addWidget(self.lbl_expect, 5, 1, 1, 2)
        self.voc = QtWidgets.QLineEdit()
        self.voc.setPlaceholderText("The names and terms that matter most, e.g. Thalindra, Neverwinter")
        wl.addWidget(self.voc, 6, 0, 1, 3)
        self.lbl_expect_hint = hint(""); wl.addWidget(self.lbl_expect_hint, 7, 0, 1, 3)
        wl.setColumnStretch(0, 1)
        lay.addWidget(wf)
        self.load_words_into_ui()
        self.voc.textChanged.connect(lambda _: self.refresh_word_counts())
        self.voc.editingFinished.connect(lambda: self.set_group_value("vocabulary", self.voc.text().strip()))
        self._auto_t = QtCore.QTimer(self, singleShot=True, interval=700, timeout=self.save_auto_words)
        self.auto_edit.textChanged.connect(lambda: (self.refresh_word_counts(), self._auto_t.start()))

        br = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Show test lines"); b.clicked.connect(self.overlay.demo); br.addWidget(b)
        self.btn_test = QtWidgets.QPushButton("Play a test run"); self.btn_test.setCheckable(True)
        self.btn_test.setToolTip("Plays a made-up conversation on the subtitles (short replies, back-and-forth, long speeches) "
                                 "so you can set the size, width, position and darkness. Nothing is saved to the transcript.")
        self.btn_test.toggled.connect(self.toggle_test_run); br.addWidget(self.btn_test)
        self.cb_keep = QtWidgets.QCheckBox("Keep transcript")
        self.cb_keep.setToolTip("Off: nothing is saved to a file, and the transcript only shows the last minute.")
        self.cb_keep.setChecked(cfg.get("keep_transcript", True)); self.cb_keep.toggled.connect(self.toggle_keep)
        br.addWidget(self.cb_keep)
        b = QtWidgets.QPushButton("Open transcripts folder"); b.clicked.connect(self.open_transcripts); br.addWidget(b)
        self.btn_update = QtWidgets.QPushButton("Check for updates")
        self.btn_update.setToolTip("Looks on GitHub for a newer version and asks before downloading it. "
                                   "Your settings, voices, transcripts and models are never touched.")
        self.btn_update.clicked.connect(self.check_updates)
        self.ver_label = QtWidgets.QLabel(); self.ver_label.setStyleSheet("color:#888; font-size:11px")
        self.ver_label.setAlignment(Qt.AlignHCenter); self.refresh_version()
        vb = QtWidgets.QVBoxLayout(); vb.setSpacing(1); vb.addWidget(self.btn_update); vb.addWidget(self.ver_label)
        br.addLayout(vb)
        self.update_found.connect(self._update_checked); self.update_done.connect(self._update_installed)
        br.addStretch(1)
        self.info = QtWidgets.QLabel(""); self.info.setStyleSheet("color:#888"); br.addWidget(self.info)
        lay.addLayout(br)

        bridge.voices_changed.connect(self.on_voices)
        bridge.status.connect(self.status.setText)
        bridge.ready.connect(self.on_ready)
        self.fill_devices(initial=True)
        self.fill_mics()
        self.mic_dev.currentIndexChanged.connect(self.change_mic)
        self.cb_me.toggled.connect(self.toggle_mic); self.show_my_mic(cfg.get("my_voice", True))
        self.rebuild()
        self.meter_t = QtCore.QTimer(self, interval=120, timeout=self.update_meter); self.meter_t.start()
        self.save_t = QtCore.QTimer(self, interval=5000, timeout=lambda: self.bank.save()); self.save_t.start()
        self.alive_t = QtCore.QTimer(self, interval=30000, timeout=self._write_state); self.alive_t.start()
        self.trim_t = QtCore.QTimer(self, interval=5000, timeout=self._trim_transcript); self.trim_t.start()
        self.dev.currentIndexChanged.connect(self.change_device)

    # ---- groups (each D&D group has its own voices)
    def fill_groups(self):
        self.group.blockSignals(True); self.group.clear()
        names = engine.list_groups()
        cur = self.cfg.get("group", "Group 1")
        if cur not in names: names = sorted(names + [cur])
        self.group.addItems(names); self.group.setCurrentText(cur)
        self.group.blockSignals(False)

    def set_group_value(self, key, value):
        """words to expect / word fixes belong to the current group (saved in its file)"""
        self.cfg[key] = value
        self.bank.extra[key] = value; self.bank.save(force=True)

    def load_group_values(self):
        migrate_words(self.bank)
        self.cfg["vocabulary"] = self.bank.extra.get("vocabulary", "")
        self.cfg["word_fixes"] = self.bank.extra.get("word_fixes", [])
        self.cfg["auto_words"] = self.bank.extra.get("auto_words", "")
        if hasattr(self, "voc"): self.load_words_into_ui()

    def load_words_into_ui(self):
        for w, text in ((self.voc, self.cfg.get("vocabulary", "")), (self.auto_edit, self.cfg.get("auto_words", ""))):
            w.blockSignals(True)
            w.setText(text) if w is self.voc else w.setPlainText(text)
            w.blockSignals(False)
        self.refresh_word_counts()

    def refresh_word_counts(self):
        cap = int(self.cfg.get("expect_cap", 40))
        n_fix = len(self.cfg.get("word_fixes") or [])
        self.lbl_manual.setText(f"{n_fix} swap{'s' if n_fix != 1 else ''}")
        n_auto = len(wordswap.parse_terms(self.auto_edit.toPlainText()))
        self.lbl_auto.setText(f"{n_auto} word{'s' if n_auto != 1 else ''}")
        n = len(wordswap.parse_terms(self.voc.text()))
        self.lbl_expect_h.setText(f"<b>Words to expect (high impact, capped at {cap})</b>")
        over = n > cap
        self.lbl_expect.setText(f"{n} / {cap}" + (f" — only the first {cap} are used" if over else ""))
        self.lbl_expect.setStyleSheet("color:#e5534b" if over else "color:#888")
        self.lbl_expect_hint.setText(f"The strongest help: Whisper listens for these while it transcribes. Only about {cap} fit, "
                                     f"so put the most important first.")

    def save_auto_words(self):
        self.set_group_value("auto_words", self.auto_edit.toPlainText().strip())

    def switch_group(self, name):
        if not name or name == self.cfg.get("group"): return
        self.save_session(); self._write_state()          # the group we are leaving
        self.set("group", name)
        self.bank.switch(engine.group_path(name))
        self.load_group_values()
        self.save_session()
        self.overlay.lines = []; self.overlay.update()
        self.open_session()

    def new_group(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "New group", "Name of the group (e.g. Thursday campaign):")
        name = name.strip()
        if not ok or not name: return
        if name in engine.list_groups(): self.group.setCurrentText(name); return
        self.bank.save(force=True)
        with open(engine.group_path(name), "w", encoding="utf-8") as f:
            json.dump({"model": self.cfg["voice_model"], "next_id": 1, "voices": []}, f)
        self.fill_groups(); self.group.setCurrentText(name)

    def rename_group(self):
        old = self.cfg.get("group", "Group 1")
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename group", "New name:", text=old)
        name = name.strip()
        if not ok or not name or name == old or name in engine.list_groups(): return
        self.bank.save(force=True)
        os.replace(engine.group_path(old), engine.group_path(name))
        self.bank.path = engine.group_path(name)
        oldf = self.group_folder()
        self.set("group", name); self.fill_groups()
        newf = os.path.join(TRANSCRIPTS, re.sub(r'[<>:"/\\|?*]', "_", name).strip())
        try:
            if os.path.isdir(newf) and not os.listdir(newf): os.rmdir(newf)
            if not os.path.exists(newf): os.replace(oldf, newf)
            else: merge_transcripts(newf, [os.path.join(oldf, f) for f in os.listdir(oldf)], TRANSCRIPTS)
        except Exception as e: log(f"[transcript] could not move the folder: {e}")
        self.open_session()

    def set(self, k, v):
        self.cfg[k] = v
        try: engine.save_config(self.cfg)
        except Exception as e: log(f"[config] save failed: {e}")

    # ---- audio devices
    def fill_devices(self, initial=False):
        self.dev.blockSignals(True); self.dev.clear()
        self.dev.addItem("Automatic", "auto")
        try:
            for value, label in engine.list_listen_sources(): self.dev.addItem(label, value)
        except Exception as e:
            log(f"[audio] could not list devices: {e}")
        want = self.cfg.get("audio_device", "auto")
        i = self.dev.findData(want); self.dev.setCurrentIndex(max(i, 0))
        self.dev.blockSignals(False)

    def fill_mics(self):
        self.mic_dev.blockSignals(True); self.mic_dev.clear()
        self.mic_dev.addItem("Automatic", "auto")
        try:
            for _, n in engine.list_input_devices(): self.mic_dev.addItem(n, n)
        except Exception as e: log(f"[audio] could not list mics: {e}")
        self.mic_dev.setCurrentIndex(max(self.mic_dev.findData(self.cfg.get("mic_device", "auto")), 0))
        self.mic_dev.blockSignals(False)

    def change_device(self, *_):
        want = self.dev.currentData() or "auto"
        self.set("audio_device", want)
        if self.worker: self.start_audio()

    def start_audio(self):
        try:
            src = engine.AudioSource(self.cfg.get("audio_device", "auto"))
            self.worker.set_source(src)
            self.src_name = src.name
            self.status.setText(f"Whisper {self.asr_name} on {self.asr_dev} · listening to: {src.name}")
        except Exception as e:
            traceback.print_exc()
            self.status.setText(f"Could not open the audio device: {e}")

    def update_meter(self):
        src = self.worker.src if self.worker else None
        lvl = src.level if src else -90
        if src: src.level = max(src.level - 3, -90)      # decays when no audio arrives
        self.meter.setValue(int(max(0, min(60, lvl + 60))))
        if self.pipeline:
            st = self.pipeline.stats
            if st["utts"]:
                self.info.setText(f"{st['utts']} lines · Whisper {st['asr_s'] / st['utts'] * 1000:.0f} ms per line")

    def on_ready(self):
        self.asr_name, self.asr_dev = self.pipeline.asr.name, self.pipeline.asr.device.upper()
        if getattr(self, "snapshot_mode", False):
            self.status.setText(f"Whisper {self.asr_name} on {self.asr_dev} · (screen check, not listening)"); return
        self.start_audio()
        self.worker.start()
        self.mic_worker.start()
        if self.cfg.get("my_voice", True): self.start_mic()

    def start_mic(self):
        if not getattr(self, "mic_worker", None): return
        try:
            src = engine.AudioSource(self.cfg.get("mic_device", "auto"), mic=True)
            self.mic_worker.set_source(src)
            self.mic_status.setText(f"mic: {src.name}")
        except Exception as e:
            traceback.print_exc(); self.mic_status.setText(f"could not open the mic: {e}")

    def stop_mic(self):
        if getattr(self, "mic_worker", None): self.mic_worker.set_source(None)
        self.mic_status.setText("")

    # ---- updates
    def refresh_version(self):
        v = updater.local_version()
        self.ver_label.setText("ver: " + (v[:7] if v else "unknown"))
        self.ver_label.setToolTip("The version (GitHub commit) of the program files in this folder." if v else
                                  "This folder has no version record yet; it is written after the first update.")

    def check_updates(self):
        self.btn_update.setEnabled(False); self.btn_update.setText("Checking...")
        def work():
            try: r = updater.check()
            except Exception as e: r = e
            self.update_found.emit(r)
        threading.Thread(target=work, daemon=True).start()

    def _update_checked(self, r):
        self.btn_update.setEnabled(True); self.btn_update.setText("Check for updates")
        box = QtWidgets.QMessageBox
        if isinstance(r, Exception):
            box.warning(self, "Check for updates", f"Could not reach GitHub:\n{r}"); return
        new = r["latest"]
        what = f"Newest version: {new['date']} - {new['message']}"
        if r["state"] == "current":
            box.information(self, "Check for updates", "You have the newest version.\n\n" + what); return
        head = ("An update is available" + (f" ({r['behind']} change{'s' if r['behind'] != 1 else ''})" if r["behind"] else "") + "."
                if r["state"] == "update" else "Could not tell which version you have, but the newest is available.")
        if box.question(self, "Check for updates", f"{head}\n\n{what}\n\nDownload and install it now? Your settings, voices, "
                        "transcripts and models are kept.") != box.Yes: return
        self.btn_update.setEnabled(False); self.btn_update.setText("Updating...")
        def work():
            self.update_done.emit(updater.install(new["sha"]))
        threading.Thread(target=work, daemon=True).start()

    def _update_installed(self, res):
        self.btn_update.setEnabled(True); self.btn_update.setText("Check for updates")
        ok, text, changed = res; box = QtWidgets.QMessageBox; self.refresh_version()
        if not ok: box.warning(self, "Update", text); return
        msg = f"Updated {len(changed)} file{'s' if len(changed) != 1 else ''}. Close and reopen the app to use the new version."
        if any(f.replace("\\", "/") in ("install.ps1", "install.bat") for f in changed):
            msg += "\n\nThe setup changed too: run install.bat once more to get any new packages or models."
        box.information(self, "Update", msg)

    def toggle_test_run(self, on):
        self.btn_test.setText("Stop test run" if on else "Play a test run")
        self.overlay.start_test() if on else self.overlay.stop_test()

    def show_my_mic(self, on):
        """with "My mic" off, nothing about you is shown: no mic picker, no name box, no row in the voice list"""
        for w in (self.mic_dev, self.mic_status, self.me_name, self.me_name_label): w.setVisible(on)
        for w in (getattr(self, "my_row_widgets", None) or []): w.setVisible(on)

    def toggle_mic(self, on):
        self.set("my_voice", on)
        self.show_my_mic(on)
        self.start_mic() if on else self.stop_mic()

    def change_mic(self, *_):
        self.set("mic_device", self.mic_dev.currentData() or "auto")
        if self.cfg.get("my_voice", True): self.start_mic()

    # ---- voice rows
    # ---- side transcript
    def add_transcript(self, e):
        if e["kind"] != "final" or not e["parts"]: return
        sb = self.tlist.verticalScrollBar(); at_bottom = sb.value() >= sb.maximum() - 4
        now = datetime.datetime.now(); ts = now.strftime("%H:%M"); full = now.strftime("%H:%M:%S")
        embs = e.get("embs") or [None] * len(e["parts"])
        for (v, txt), emb in zip(e["parts"], embs):
            eid = None
            if emb is not None:
                self._eid = getattr(self, "_eid", 0) + 1; eid = self._eid
                self.line_embs = getattr(self, "line_embs", {}); self.line_embs[eid] = emb
                if len(self.line_embs) > 4000: self.line_embs.pop(next(iter(self.line_embs)))
            pieces = split_rows(txt)
            # the same person's previous line stopped mid-sentence: carry on in that line
            row = self.tlist.count() - 1
            while row >= 0 and self.tlist.item(row).isHidden(): row -= 1
            last = self.tlist.item(row) if row >= 0 else None
            if pieces and last is not None:
                d = last.data(Qt.UserRole)
                if (d["voice"] == v and d.get("fixed") is None and v is not None
                        and not ended(d["text"]) and not d["text"].strip().endswith(",")):
                    rows = split_rows(d["text"].rstrip() + " " + " ".join(pieces))
                    d["text"] = rows[0]; d["ts"] = time.time()
                    if d.get("eid") is None: d["eid"] = eid
                    last.setData(Qt.UserRole, d); self.paint_item(last)
                    pieces = rows[1:]
            for sent in pieces:
                it = QtWidgets.QListWidgetItem()
                it.setData(Qt.UserRole, {"voice": v, "text": sent, "time": ts, "full": full, "eid": eid, "ts": time.time()})
                self.tlist.addItem(it); self.paint_item(it)
        self.save_session_soon()
        if at_bottom: self.tlist.scrollToBottom()

    def _who(self, d):
        """-> (key, name, colour) of who said a transcript line, after merges and fixes"""
        if d.get("fixed") is not None:
            v = self.bank.resolve(d["fixed"]); return (v,) + self.bank.get(v)
        if d["voice"] == "me": return ("me", self.cfg.get("my_name", "Me"), self.cfg.get("my_color", "#FF6B6B"))
        if d.get("label"): return (("label", d["label"]), d["label"], engine.UNKNOWN_COLOR)   # from an older run
        if d["voice"] is None: return (None, "?", engine.UNKNOWN_COLOR)
        v = self.bank.resolve(d["voice"]); return (v,) + self.bank.get(v)

    def paint_item(self, it):
        d = it.data(Qt.UserRole)
        key, name, color = self._who(d)
        hide = (key is None and self.cfg.get("hide_short_unknown", True)
                and len(re.findall(r"[\w']+", d["text"])) <= 2)
        it.setHidden(hide)
        row = self.tlist.row(it) - 1
        while row >= 0 and self.tlist.item(row).isHidden(): row -= 1     # group with the nearest visible line
        prev = self.tlist.item(row) if row >= 0 else None
        same = prev is not None and key is not None and self._who(prev.data(Qt.UserRole))[0] == key
        it.setData(Qt.UserRole + 1, (not same) and row >= 0)
        if same:     # same person as the line above: no name, just the text
            it.setText(d["text"])
        else:
            it.setText(f"{d['time']}  {name}:  {d['text']}")
        it.setForeground(QtGui.QColor(color))

    def repaint_transcript(self):
        self.save_session_soon()
        for i in range(self.tlist.count()): self.paint_item(self.tlist.item(i))

    def transcript_menu(self, pos):
        it = self.tlist.itemAt(pos)
        if it is None: return
        d = it.data(Qt.UserRole)
        if not it.isSelected():
            self.tlist.clearSelection(); it.setSelected(True)
        n = len(self.tlist.selectedItems())
        copy_label = "Copy" if n <= 1 else f"Copy {n} lines"
        if d["voice"] == "me" and d.get("fixed") is None:
            m = QtWidgets.QMenu(self)
            m.addAction(copy_label).triggered.connect(self.copy_lines); m.addSeparator()
            m.addAction("Change my colour…").triggered.connect(self.pick_my_color)
            m.exec(self.tlist.viewport().mapToGlobal(pos)); return
        vid = self.bank.resolve(d["voice"]) if d["voice"] is not None else None
        if d.get("fixed") is not None: vid = self.bank.resolve(d["fixed"])
        emb = getattr(self, "line_embs", {}).get(d.get("eid"))
        others = [o for o in self.bank.snapshot() if o["id"] != vid]
        m = QtWidgets.QMenu(self)
        m.addAction(copy_label).triggered.connect(self.copy_lines); m.addSeparator()
        sub = m.addMenu("This is")
        for o in others:
            sub.addAction(o["name"]).triggered.connect(lambda _=False, it=it, dst=o["id"]: self.fix_line(it, dst))
        if not others: sub.addAction("(no voices yet)").setEnabled(False)
        add = m.addAction("Add as new voice…"); add.triggered.connect(lambda: self.add_voice_from_line(it))
        if emb is None: add.setEnabled(False); add.setText("Add as new voice… (line too short)")
        if vid is not None and vid in self.bank.voices:
            name = self.bank.get(vid)[0]
            m.addSeparator()
            sub = m.addMenu(f"Merge “{name}” into")
            for o in others:
                sub.addAction(o["name"]).triggered.connect(lambda _=False, dst=o["id"]: self.do_merge(vid, dst))
            if not others: sub.addAction("(no other voices)").setEnabled(False)
            m.addAction(f"Rename “{name}”…").triggered.connect(lambda: self.rename_dialog(vid))
        m.exec(self.tlist.viewport().mapToGlobal(pos))

    def copy_lines(self):
        """selected transcript lines -> clipboard, as "Name: text" (in transcript order)"""
        items = sorted(self.tlist.selectedItems(), key=self.tlist.row)
        if not items: return
        out = []
        for it in items:
            d = it.data(Qt.UserRole)
            out.append(f"[{d.get('full', d['time'])}] {self._who(d)[1]}: {d['text']}")
        QtWidgets.QApplication.clipboard().setText("\n".join(out))

    def add_voice_from_line(self, it):
        d = it.data(Qt.UserRole); emb = getattr(self, "line_embs", {}).get(d.get("eid"))
        if emb is None: return
        name, ok = QtWidgets.QInputDialog.getText(self, "Add voice", "Who is this?")
        if not ok or not name.strip(): return
        vid = self.bank.add(np.asarray(emb, dtype=np.float32), name=name.strip())
        self.bank.heard(vid, d["text"]); self.bank.save(force=True)
        d["fixed"] = vid; it.setData(Qt.UserRole, d); self.repaint_transcript()

    def fix_line(self, it, vid):
        """this line was said by vid: relabel it, and teach the app what vid sounds like"""
        d = it.data(Qt.UserRole); d["fixed"] = vid; it.setData(Qt.UserRole, d)
        emb = getattr(self, "line_embs", {}).get(d.get("eid"))
        if emb is not None: self.bank.teach(vid, emb); self.bank.save(force=True)
        self.repaint_transcript()

    def rename_dialog(self, vid):
        name, ok = QtWidgets.QInputDialog.getText(self, "Rename voice", "Name:", text=self.bank.get(vid)[0])
        if ok and name.strip(): self.bank.rename(vid, name.strip()); self.bank.save(force=True)

    def on_voices(self, kind):
        if kind in ("added", "removed"): self.rebuild()
        else: self.refresh_rows()
        self.repaint_transcript(); self.overlay.update()

    def rebuild(self):
        while self.grid.count():
            w = self.grid.takeAt(0).widget()
            if w: w.hide(); w.deleteLater()
        self.rows = {}
        snap = self.bank.snapshot()
        hdr = ["", "Name", "Heard", "Last said", "Sounds a bit like", "", ""]
        for c, h in enumerate(hdr):
            l = QtWidgets.QLabel(f"<b>{h}</b>"); self.grid.addWidget(l, 0, c)
        # you: only a colour and a name (your voice is never matched, merged or learned)
        mcol = QtWidgets.QPushButton(); mcol.setFixedSize(28, 22); mcol.clicked.connect(self.pick_my_color)
        mcol.setStyleSheet(f"background:{self.cfg.get('my_color', '#FF6B6B')}; border:1px solid #444; border-radius:3px")
        mname = QtWidgets.QLineEdit(self.cfg.get("my_name", "Me")); mname.setFixedWidth(150)
        mname.editingFinished.connect(lambda w=mname: self.set_my_name(w.text()))
        mnote = QtWidgets.QLabel("you · transcript only"); mnote.setStyleSheet("color:#888")
        relearn = QtWidgets.QPushButton("Relearn my voice")
        relearn.setToolTip("Forget your voice fingerprint; it is learned again from your mic while the call is quiet")
        relearn.clicked.connect(lambda: (self.bank.me_forget(), self.bank.save(force=True)))
        self.grid.addWidget(mcol, 1, 0); self.grid.addWidget(mname, 1, 1); self.grid.addWidget(mnote, 1, 2, 1, 2)
        self.grid.addWidget(relearn, 1, 5, 1, 2)
        self.my_row = dict(col=mcol, name=mname)
        self.my_row_widgets = [mcol, mname, mnote, relearn]
        for w in self.my_row_widgets: w.setVisible(bool(self.cfg.get("my_voice", True)))
        for r, v in enumerate(snap, start=2):
            vid = v["id"]
            col = QtWidgets.QPushButton(); col.setFixedSize(28, 22)
            col.clicked.connect(lambda _=False, vid=vid: self.pick_color(vid))
            name = QtWidgets.QLineEdit(v["name"]); name.setFixedWidth(150)
            name.editingFinished.connect(lambda vid=vid, w=name: self.bank.rename(vid, w.text()))
            heard = QtWidgets.QLabel(); last = QtWidgets.QLabel(); last.setMinimumWidth(200)
            like = QtWidgets.QLabel()
            merge = QtWidgets.QPushButton("Merge into…")       # a menu, so the mouse wheel can't trigger it
            merge.clicked.connect(lambda _=False, vid=vid, w=merge: self.merge_menu(vid, w))
            dele = QtWidgets.QPushButton("Delete"); dele.clicked.connect(lambda _=False, vid=vid: self.do_delete(vid))
            for c, w in enumerate([col, name, heard, last, like, merge, dele]): self.grid.addWidget(w, r, c)
            self.rows[vid] = dict(col=col, name=name, heard=heard, last=last, like=like)
        self.grid.setRowStretch(len(snap) + 2, 1)
        self.refresh_rows()

    def refresh_rows(self):
        snap = {v["id"]: v for v in self.bank.snapshot()}
        if set(snap) != set(self.rows): return self.rebuild()
        for vid, w in self.rows.items():
            v = snap[vid]
            w["col"].setStyleSheet(f"background:{v['color']}; border:1px solid #444; border-radius:3px")
            if not w["name"].hasFocus() and w["name"].text() != v["name"]: w["name"].setText(v["name"])
            w["heard"].setText(str(v["count"]))
            fm = w["last"].fontMetrics()
            w["last"].setText(fm.elidedText(f"“{v['last_text']}”" if v["last_text"] else "", Qt.ElideRight, 300))
            w["last"].setToolTip(v["last_text"])
            sim, other = self.bank.closest_other(vid)
            w["like"].setText(f"{self.bank.get(other)[0]} ({sim:.2f})" if other is not None and sim > 0.35 else "")

    def pick_my_color(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.cfg.get("my_color", "#FF6B6B")), self, "My colour")
        if not c.isValid(): return
        self.set("my_color", c.name())
        if getattr(self, "my_row", None):
            self.my_row["col"].setStyleSheet(f"background:{c.name()}; border:1px solid #444; border-radius:3px")
        self.repaint_transcript()

    def set_my_name(self, name):
        name = name.strip() or "Me"
        self.set("my_name", name)
        for w in (getattr(self, "me_name", None), (getattr(self, "my_row", None) or {}).get("name")):
            if w is not None and w.text() != name: w.setText(name)
        self.repaint_transcript()

    def pick_color(self, vid):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.bank.get(vid)[1]), self, "Subtitle colour")
        if c.isValid(): self.bank.recolor(vid, c.name())

    def merge_menu(self, vid, button):
        m = QtWidgets.QMenu(self)
        for o in self.bank.snapshot():
            if o["id"] != vid:
                m.addAction(o["name"]).triggered.connect(lambda _=False, dst=o["id"]: self.do_merge(vid, dst))
        if m.isEmpty(): m.addAction("(no other voices)").setEnabled(False)
        m.exec(button.mapToGlobal(QtCore.QPoint(0, button.height())))

    def do_merge(self, vid, dst):
        self.bank.merge(vid, dst); self.bank.save(force=True)

    def do_delete(self, vid):
        if QtWidgets.QMessageBox.question(self, "Delete voice",
                f"Forget “{self.bank.get(vid)[0]}”? If they talk again they come back as a new voice.") == QtWidgets.QMessageBox.Yes:
            self.bank.remove(vid); self.bank.save(force=True)

    # ---- saved session: transcripts\<group>\<date>.txt, carried on across restarts, kept in sync with fixes
    def group_folder(self):
        safe = re.sub(r'[<>:"/\\|?*]', "_", self.cfg.get("group", "Group 1")).strip()
        d = os.path.join(TRANSCRIPTS, safe); os.makedirs(d, exist_ok=True); return d

    def open_session(self):
        """Which transcript are we writing to?
        - the app was closed only a few minutes ago (session_auto_resume_min): carry on in the same file;
        - closed a little longer ago, up to an hour (session_ask_resume_min): ask whether it is the same session;
        - longer than that: a new file, named by date and time (2026-10-19_01-45.txt)."""
        if not self.cfg.get("keep_transcript", True):          # nothing is ever saved or reloaded
            self.session_path = None; self.tlist.clear(); return
        folder = self.group_folder()
        prev, gap = self._previous_session(folder)
        mins = gap / 60
        self.session_path = None
        if prev is not None:
            if mins <= float(self.cfg.get("session_auto_resume_min", 5)):
                self.session_path = prev
            elif mins <= float(self.cfg.get("session_ask_resume_min", 60)):
                if self._ask_continue(prev, mins): self.session_path = prev
        if self.session_path is None:
            self.session_path = self._new_session_path(folder)
        self._write_state()
        self.tlist.clear()
        if not os.path.exists(self.session_path): return
        names = {v["name"]: v["id"] for v in self.bank.snapshot()}
        me = self.cfg.get("my_name", "Me")
        try:
            for line in open(self.session_path, encoding="utf-8"):
                m = re.match(r"^\[(\d\d:\d\d:\d\d)\] (.*?): (.*)$", line.rstrip("\n"))
                if not m: continue
                full, name, text = m.groups()
                d = {"voice": None, "text": text, "time": full[:5], "full": full, "eid": None}
                if name == me: d["voice"] = "me"
                elif name in names: d["voice"] = names[name]
                elif name != "?": d["label"] = name
                it = QtWidgets.QListWidgetItem(); it.setData(Qt.UserRole, d)
                self.tlist.addItem(it); self.paint_item(it)
            self.tlist.scrollToBottom()
        except Exception as e:
            log(f"[transcript] could not read {self.session_path}: {e}")

    # ---- which transcript was the app writing to, and when was it last running
    def _state_path(self): return os.path.join(self.group_folder(), ".session.json")

    def _write_state(self):
        """called every 30 s while the app runs and when it closes: the file in use and 'last alive' time"""
        if not getattr(self, "session_path", None) or not self.cfg.get("keep_transcript", True): return
        try:
            with open(self._state_path(), "w", encoding="utf-8") as f:
                json.dump({"file": os.path.basename(self.session_path), "alive_at": time.time()}, f)
        except Exception as e:
            log(f"[transcript] could not write session state: {e}")

    def _previous_session(self, folder):
        """-> (path of the group's last transcript, seconds since the app was last running) or (None, 0)"""
        try:
            with open(os.path.join(folder, ".session.json"), encoding="utf-8") as f: st = json.load(f)
            path = os.path.join(folder, st["file"])
            if os.path.isfile(path): return path, max(0.0, time.time() - float(st["alive_at"]))
        except Exception: pass
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]     # no state yet (older versions)
        if not files: return None, 0
        latest = max(files, key=os.path.getmtime)
        return latest, max(0.0, time.time() - os.path.getmtime(latest))

    @staticmethod
    def _new_session_path(folder):
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        path, n = os.path.join(folder, stamp + ".txt"), 2
        while os.path.exists(path):
            path = os.path.join(folder, f"{stamp} ({n}).txt"); n += 1
        return path

    @staticmethod
    def _session_label(path):
        """'19 Oct, 01:45' from a transcript's file name (date only for files from older versions)"""
        m = re.match(r"^(\d{4})-(\d\d)-(\d\d)(?:[_ ](\d\d)-(\d\d))?", os.path.basename(path))
        if not m: return os.path.basename(path)
        y, mo, d, hh, mm = m.groups()
        day = datetime.date(int(y), int(mo), int(d)).strftime("%d %b").lstrip("0")
        return f"{day}, {hh}:{mm}" if hh else day

    def _ask_continue(self, path, mins):
        ago = f"{int(round(mins))} minute{'s' if round(mins) != 1 else ''} ago"
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Continue the previous session?")
        box.setText(f"Is this a continuation of the previous session from {self._session_label(path)}?")
        box.setInformativeText(f"The app was last running {ago}. Yes keeps adding to that transcript; "
                               f"No starts a new one.")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.Yes)
        return box.exec() == QtWidgets.QMessageBox.Yes

    def toggle_keep(self, on):
        """off: nothing more is written to files and the window only shows the last minute.
        on again: a fresh file (the window only holds a minute, so the old file must not be overwritten)"""
        if not on: self.save_session()
        self.set("keep_transcript", on)
        if on:
            self.session_path = self._new_session_path(self.group_folder()); self._write_state(); self.save_session()
        else:
            self.session_path = None; self._trim_transcript()

    def _trim_transcript(self):
        if self.cfg.get("keep_transcript", True): return
        limit = time.time() - float(self.cfg.get("transient_seconds", 60))
        for i in range(self.tlist.count() - 1, -1, -1):
            if self.tlist.item(i).data(Qt.UserRole).get("ts", 0) < limit: self.tlist.takeItem(i)

    def save_session_soon(self):
        if not hasattr(self, "_save_t"):
            self._save_t = QtCore.QTimer(self, singleShot=True, interval=800, timeout=self.save_session)
        self._save_t.start()

    def save_session(self):
        if (not getattr(self, "session_path", None) or self.tlist.count() == 0
                or not self.cfg.get("keep_transcript", True)): return
        rows = []
        for i in range(self.tlist.count()):
            d = self.tlist.item(i).data(Qt.UserRole)
            rows.append(f"[{d.get('full', d['time'])}] {self._who(d)[1]}: {d['text']}")
        try:
            tmp = self.session_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f: f.write("\n".join(rows) + "\n")
            os.replace(tmp, self.session_path)
        except Exception as e:
            log(f"[transcript] could not save: {e}")

    def edit_word_fixes(self):
        dlg = QtWidgets.QDialog(self); dlg.setWindowTitle("Manual word swap"); dlg.resize(460, 380)
        v = QtWidgets.QVBoxLayout(dlg)
        v.addWidget(QtWidgets.QLabel("Whenever the left side is heard, the right side is written instead.\n"
                                     "Whole words or phrases, any capitalisation."))
        tab = QtWidgets.QTableWidget(0, 2); tab.setHorizontalHeaderLabels(["Heard as", "Replace with"])
        tab.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        for a, b in self.cfg.get("word_fixes") or []:
            r = tab.rowCount(); tab.insertRow(r)
            tab.setItem(r, 0, QtWidgets.QTableWidgetItem(a)); tab.setItem(r, 1, QtWidgets.QTableWidgetItem(b))
        v.addWidget(tab)
        h = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add"); rem = QtWidgets.QPushButton("Remove selected")
        def do_add():
            r = tab.rowCount(); tab.insertRow(r); tab.setItem(r, 0, QtWidgets.QTableWidgetItem(""))
            tab.setItem(r, 1, QtWidgets.QTableWidgetItem("")); tab.editItem(tab.item(r, 0))
        add.clicked.connect(do_add)
        rem.clicked.connect(lambda: [tab.removeRow(r) for r in sorted({i.row() for i in tab.selectedIndexes()}, reverse=True)])
        h.addWidget(add); h.addWidget(rem); h.addStretch(1)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Save | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept); bb.rejected.connect(dlg.reject); h.addWidget(bb)
        v.addLayout(h)
        if dlg.exec() != QtWidgets.QDialog.Accepted: return
        fixes = []
        for r in range(tab.rowCount()):
            a = (tab.item(r, 0).text() if tab.item(r, 0) else "").strip()
            b = (tab.item(r, 1).text() if tab.item(r, 1) else "").strip()
            if a and b: fixes.append([a, b])
        self.set_group_value("word_fixes", fixes); self.refresh_word_counts()
        for i in range(self.tlist.count()):                 # fix what's already in the transcript too
            it = self.tlist.item(i); d = it.data(Qt.UserRole)
            t = engine.apply_fixes(d["text"], fixes)
            if t != d["text"]: d["text"] = t; it.setData(Qt.UserRole, d)
        for l in self.overlay.lines: l["final"] = engine.apply_fixes(l["final"], fixes)
        self.overlay._docs.clear(); self.overlay.update()
        self.repaint_transcript()

    def open_transcripts(self):
        d = self.group_folder()
        if os.name == "nt": os.startfile(d)
        else: subprocess.Popen(["xdg-open", d])

    def closeEvent(self, ev):
        try:
            if self.worker:
                self.worker.stop_flag = True
                if self.worker.src: self.worker.src.close()
            if getattr(self, "mic_worker", None):
                self.mic_worker.stop_flag = True
                if self.mic_worker.src: self.mic_worker.src.close()
            if self._auto_t.isActive(): self.save_auto_words()
            self.bank.save(force=True)
            self.save_session()
            self._write_state()
        finally:
            QtWidgets.QApplication.quit()


# =============================================================================================== transcript
class Transcript:
    def __init__(self, bank, cfg):
        os.makedirs(TRANSCRIPTS, exist_ok=True)
        self.bank, self.cfg = bank, cfg
        self.f = None
        self.set_group(cfg.get("group", "Group 1"))

    def set_group(self, name):
        if self.f: self.f.close(); self.f = None
        safe = re.sub(r'[<>:"/\\|?*]', "_", name).strip()
        self.path = os.path.join(TRANSCRIPTS, datetime.datetime.now().strftime(f"{safe}_%Y-%m-%d_%H-%M.txt"))

    def write(self, e):
        if e["kind"] != "final" or not e["parts"]: return
        if self.f is None: self.f = open(self.path, "a", encoding="utf-8")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        for v, txt in e["parts"]:
            name = self.cfg.get("my_name", "Me") if v == "me" else (self.bank.get(v)[0] if v is not None else "?")
            self.f.write(f"[{ts}] {name}: {txt}\n")
        self.f.flush()


# =============================================================================================== main
def migrate_words(bank):
    """one time per group: the right-hand words of manual word swaps used to be sent to Whisper as words to expect.
    That is now only done for the words you list under "Words to expect", so keep working the same by adding them there."""
    if bank.extra.get("words_v2"): return
    terms = wordswap.parse_terms(bank.extra.get("vocabulary", ""))
    have = {t.lower() for t in terms}
    for f in bank.extra.get("word_fixes") or []:
        if len(f) > 1 and f[1].strip() and f[1].strip().lower() not in have:
            terms.append(f[1].strip()); have.add(f[1].strip().lower())
    bank.extra["vocabulary"] = ", ".join(terms); bank.extra["words_v2"] = True
    bank.save(force=True)


def migrate_position(cfg):
    """older versions placed the subtitles with left/centre/right + 'distance from side' + 'height from bottom';
    turn that into the new X / Y pixel offsets (0, 0 = centred near the bottom) so nothing moves"""
    old = [k for k in ("align", "overlay_bottom_px", "overlay_side_px") if k in cfg]
    if not old: return
    screens = QtGui.QGuiApplication.screens()
    g = screens[min(max(int(cfg.get("screen", 0)), 0), len(screens) - 1)].geometry()
    w = int(g.width() * cfg.get("overlay_width_pct", 75) / 100)
    room = (g.width() - w) // 2 - int(cfg.get("overlay_side_px", 40))
    al = cfg.get("align", "center")
    cfg["overlay_x_px"] = -room if al == "left" else room if al == "right" else 0
    cfg["overlay_y_px"] = BOTTOM_MARGIN - int(cfg.get("overlay_bottom_px", 90))
    for k in old: cfg.pop(k, None)
    log(f"[config] moved the old subtitle position to X {cfg['overlay_x_px']} px, Y {cfg['overlay_y_px']} px")


def main():
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("D&D Subtitles")
    ico = os.path.join(HERE, "app.ico")
    if os.path.exists(ico): app.setWindowIcon(QtGui.QIcon(ico))
    cfg = engine.load_config(); migrate_position(cfg); engine.save_config(cfg)      # writes any new settings into config.json
    if "--snapshot" in sys.argv:              # demo voices, nothing saved
        import tempfile, numpy as np
        bank = engine.VoiceBank(path=os.path.join(tempfile.gettempdir(), "dndsubs_snapshot_voices.json"), model=cfg["voice_model"])
        bank.voices.clear()
        for n in ["Dana", "Omer", "Yael", "Itai", "Noa", "Gil"]:
            bank.rename(bank.add(np.random.default_rng(len(n) + ord(n[0])).standard_normal(192).astype(np.float32)), n)
    else:
        old = os.path.join(HERE, "voices.json")
        if os.path.exists(old) and not engine.list_groups():      # first run with groups: old voices become Group 1
            os.replace(old, engine.group_path(cfg.get("group", "Group 1")))
        if not engine.list_groups():
            cfg["group"] = cfg.get("group") or "Group 1"
        elif cfg.get("group") not in engine.list_groups():
            cfg["group"] = engine.list_groups()[0]
        engine.save_config(cfg)
        bank = engine.VoiceBank(path=engine.group_path(cfg["group"]), model=cfg["voice_model"])
        if "vocabulary" not in bank.extra and "word_fixes" not in bank.extra:     # lists used to be shared
            bank.extra = {"vocabulary": cfg.get("vocabulary", ""), "word_fixes": cfg.get("word_fixes", [])}
            bank.save(force=True)
        migrate_words(bank)
        cfg["vocabulary"] = bank.extra.get("vocabulary", ""); cfg["word_fixes"] = bank.extra.get("word_fixes", [])
        cfg["auto_words"] = bank.extra.get("auto_words", "")
    bridge = Bridge()
    bank.on_change = lambda k: bridge.voices_changed.emit(k)
    overlay = Overlay(cfg, bank); overlay.show()
    panel = Panel(cfg, bank, overlay, bridge); panel.show()
    panel.open_session()

    def on_event(e):
        bridge.event.emit(e)
    bridge.event.connect(overlay.on_event)
    bridge.event.connect(panel.add_transcript)

    def load():
        try:
            asr = engine.Transcriber(cfg)
            emb = engine.Embedder(cfg["voice_model"])
            panel.pipeline = engine.Pipeline(cfg, bank, asr, emb, on_event)
            panel.worker = engine.Worker(panel.pipeline)
            mp = engine.MicPipeline(cfg, bank, asr, emb, on_event); mp.call = panel.pipeline
            panel.mic_worker = engine.Worker(mp)
            bridge.ready.emit()
        except Exception as e:
            traceback.print_exc()
            bridge.status.emit(f"Startup failed: {e}  (see the CLI window for details)")
    threading.Thread(target=load, daemon=True).start()
    if "--snapshot" in sys.argv:              # self-check: show test lines, save a screenshot of them, quit
        panel.snapshot_mode = True
        def grab():
            g = overlay.geometry(); sc = overlay.screen()
            sc.grabWindow(0, g.x() - sc.geometry().x(), g.y() - sc.geometry().y(), g.width(), g.height()) \
                .save(os.path.join(HERE, "snapshot_overlay.png"))
            panel.grab().save(os.path.join(HERE, "snapshot_panel.png"))
            log(f"[snapshot] saved; status: {panel.status.text()}")
            panel.close()
        def wait_ready():
            if panel.pipeline is None: return QtCore.QTimer.singleShot(500, wait_ready)
            overlay.demo(); QtCore.QTimer.singleShot(1500, grab)
        QtCore.QTimer.singleShot(500, wait_ready)
    code = app.exec()
    os._exit(code)          # make sure nothing (audio, Whisper threads) keeps the app alive after the window closes


if __name__ == "__main__":
    main()
