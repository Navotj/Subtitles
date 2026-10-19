# Whisper models compared (2026-10-19)

Run on an RTX 5080 (GPU rows) and the same PC's processor (CPU rows) with `tools/whisper_eval.py`.
Test material: the first 20 minutes of three public actual-play videos with human-made captions (8,762 caption words).
CPU rows: first video, 10 minutes only, so their error rate is not comparable with the GPU rows; they are there for speed.
Speed is the time Whisper itself needed (the 80% GPU cap's pauses are not counted).

| model | where | word error | campaign names right | one finished sentence (median / slowest tenth) | live preview (median / slowest tenth) | x real time |
|---|---|---|---|---|---|---|
| large-v3-turbo | GPU | 19.9% | 180/320 (56%) | 132 / 195 ms | 82 / 89 ms | 34x |
| large-v3 | GPU | 20.0% | 182/320 (56%) | 280 / 520 ms | 147 / 176 ms | 16x |
| distil-large-v3 | GPU | 20.9% | 146/320 (46%) | 118-166 / 170-205 ms | 78 / 85 ms | 39-46x |
| distil-medium.en | GPU | 23.3% | 121/320 (38%) | 80-125 / 127-163 ms | 48 / 52 ms | 55-61x |
| small.en | GPU | 23.3% | 142/320 (44%) | 112 / 237 ms | 43 / 55 ms | 37x |
| small | GPU | 24.9% | 130/320 (40%) | 119 / 245 ms | 45 / 61 ms | 30x |
| distil-small.en | GPU | 25.8% | 95/320 (30%) | 64-117 / 124-162 ms | 27 / 32 ms | 63-66x |
| distil-small.en | CPU | 23.5% (10 min) | 23/61 (38%) | 645 / 712 ms | 595 / 622 ms | 7.1x |
| small.en | CPU | 21.2% (10 min) | 26/61 (43%) | 776 / 948 ms | 656 / 684 ms | 6x |
| small | CPU | 21.4% (10 min) | 32/61 (52%) | 806 / 962 ms | 655 / 674 ms | 5.5x |

What it says
- large-v3 is no more accurate than turbo here (19.9% vs 20.0%) and about twice as slow. Turbo is the better default.
- The small models cost about 3 to 5 points of word error and about 12 to 16 points on the names.
- Everything runs far faster than real time on the GPU. On this processor the small models need about 0.8 s per sentence and 0.65 s for a live preview, still 5 to 6 times faster than real time. A weaker processor was not tested.
- English-only small is a little better than multilingual small on English.

Distil models (tested, not adopted)
- Smaller, faster English-only versions of Whisper, tried as alternatives for computers without an NVIDIA card. Word error is weighted over the three videos; the distil ranges are across those videos.
- distil-large-v3 matches turbo on word error (20.9% vs 19.9%) but is worse on names and not faster, and it needs a GPU. distil-medium.en equals small.en on word error, is worse on names, and is bigger.
- On the processor, distil-small.en is about 17% faster than small.en but 2 points less accurate. Not worth offering. distil-medium.en on the processor was not finished at the time of writing.

Caveat: these three videos are D&D table talk with many invented names. The same models on the same videos gave higher error rates in the first run (25% overall for turbo): that run used a 14 s maximum utterance length from the PC's config.json, this one the 10 s default. The two settings were not tested in isolation, but 10 s looks better.
