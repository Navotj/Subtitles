"""Checks for the automatic word swap.   python tools/test_wordswap.py"""
import os, sys, time, random, string
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wordswap import apply_auto, AutoSwap, parse_terms

LIST = "Beholder, Eldritch Blast, Fireball, Neverwinter, Strahd, Tiamat, Mind Flayer, Magic Missile, Tarrasque, Moradin"
cases = [
    ("the bee holder attacks", "the Beholder attacks"),                    # split word
    ("I cast fire ball", "I cast Fireball"),                               # joined word
    ("he used elder rich blast on it", "he used Eldritch Blast on it"),    # sounds alike, several words
    ("two magic missle hits", "two Magic Missile hits"),                   # typo
    ("the beholders gaze", "the Beholders gaze"),                          # plural keeps its s
    ("the Beholder's gaze", "the Beholder's gaze"),                        # already right
    ("beholder's gaze", "Beholder's gaze"),                                # capital only
    ("we found some straw", "we found some straw"),                        # needs a manual swap
    ("the mind flare attacks", "the Mind Flayer attacks"),
    ("never winter is cold", "Neverwinter is cold"),
    ("there is a moron here", "there is a moron here"),                    # similar letters, different word
    ("their water is near", "their water is near"),                        # ordinary words untouched
    ("Fireball and fireball", "Fireball and Fireball"),
]
bad = 0
for src, want in cases:
    got = apply_auto(src, LIST)
    if got != want:
        bad += 1; print(f"FAIL  {src!r}\n   got  {got!r}\n   want {want!r}")
print(f"{len(cases) - bad}/{len(cases)} cases ok")

random.seed(1)
words = ["".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(5, 11))) for _ in range(2000)]
sw = AutoSwap(words)
line = "the old dragon breathes fire across the bridge and everyone makes a dexterity saving throw against it now"
t = time.time()
for _ in range(200): sw.apply(line)
print(f"2000 terms: {(time.time() - t) / 200 * 1000:.1f} ms per line of {len(line.split())} words")
sys.exit(1 if bad else 0)
