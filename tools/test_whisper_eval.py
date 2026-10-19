"""Checks for the caption parsing and comparing in whisper_eval (no Whisper needed).   python tools/test_whisper_eval.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whisper_eval as we

VTT = """WEBVTT
Kind: captions

00:00:01.000 --> 00:00:04.000 align:start position:0%
<c.colorE5E5E5>The bee holder</c> attacks [laughs]

00:00:04.500 --> 00:00:07.000
MATT: Roll a d20, then add two.

1
00:00:07.500 --> 00:00:09.000
Roll a d20, then add two.
"""
SRT = """1
00:00:01,000 --> 00:00:04,000
The beholder attacks

2
01:00:04,500 --> 01:00:07,000
- Twenty-one damage!
"""
c = we.parse_captions(VTT)
assert [x[2] for x in c] == ["The bee holder attacks", "Roll a d20, then add two."], c       # tags, [laughs], speaker, repeat
assert c[0][:2] == (1.0, 4.0)
s = we.parse_captions(SRT); assert s[1][0] == 3604.5 and s[1][2] == "Twenty-one damage!", s
assert we.toks("Twenty-one damage, 20 dice") == ["21", "damage", "20", "dice"], we.toks("Twenty-one damage, 20 dice")
assert we.toks("Don’t stop") == ["don't", "stop"]

cues = [(1, 4, "The Beholder attacks the Tiefling"), (5, 8, "Cast Fireball, then roll a d20"), (9, 12, "Make a dexterity saving throw")]
words = [(1.0, 1.3, " The"), (1.3, 1.6, " bee"), (1.6, 2.0, " holder"), (2.0, 2.5, " attacks"), (2.6, 2.8, " the"), (2.8, 3.4, " tifling"),
         (5.0, 5.3, " Cast"), (5.3, 5.5, " fire"), (5.5, 5.8, " ball,"), (6.0, 6.2, " then"), (6.2, 6.4, " roll"), (6.4, 6.5, " a"),
         (6.5, 6.7, " d"), (6.7, 7.0, " 20"), (9.0, 9.5, " Make"), (9.5, 9.6, " a"), (9.6, 10.2, " dexterity"), (10.2, 10.6, " saving"),
         (10.6, 11.0, " throw")]
terms = ["Beholder", "Tiefling", "Fireball", "d20", "saving throw"]
r = we.align(cues, words, terms)
assert r["conf"][("beholder", "bee holder")] == 1 and r["conf"][("tiefling", "tifling")] == 1, r["conf"]
assert r["conf"][("fireball", "fire ball")] == 1 and r["conf"][("d20", "d 20")] == 1, r["conf"]
assert r["term_ok"]["saving throw"] == 1 and r["term_ok"]["Beholder"] == 0 and r["term_n"]["Fireball"] == 1
assert r["term_heard"]["Beholder"]["bee holder"] == 1, r["term_heard"]
assert 0 < r["errors"] / r["n_ref"] < 0.5

# word swaps applied afterwards fix what they should
import wordswap
post = lambda t: wordswap.apply_auto(t, "Beholder, Tiefling, Fireball")
r2 = we.align(cues, words, terms, post)
assert r2["term_ok"]["Beholder"] == 1 and r2["term_ok"]["Tiefling"] == 1 and r2["term_ok"]["Fireball"] == 1, r2["term_ok"]
assert r2["errors"] < r["errors"]
rep = we.build_report("test", "info", r)
assert "| Tiefling |" in rep and "word error rate" in rep
print("all checks passed")
