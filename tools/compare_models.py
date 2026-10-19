"""Table of the model runs (eval/work/cmp_<tag>_<video>.metrics.json made by whisper_eval.py): accuracy and speed per model.
    python tools/compare_models.py"""
import os, sys, json, glob, collections
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = collections.OrderedDict()
for f in sorted(glob.glob(os.path.join(os.environ.get("SUBTITLES_EVAL_DIR") or os.path.join(ROOT, "eval"), "work", "cmp_*.metrics.json"))):
    d = json.load(open(f, encoding="utf-8")); tag = d["name"].split("_")[1]
    rows.setdefault(tag, []).append(d)
out = ["# Whisper models compared", "",
       "Accuracy: word error rate against human-made captions (lower is better), and how many uses of the campaign names came out right.",
       "Speed: time Whisper needs for one finished sentence, and for the live preview text (first 2.5 s). Speech is much faster "
       "than real time when the 'x real time' number is above 1.", "",
       "| model | where | word error | names right | sentence (median / slowest tenth) | live preview (median / slowest tenth) | x real time | load |",
       "|---|---|---|---|---|---|---|---|"]
for tag, ds in rows.items():
    n = sum(d["n_ref"] for d in ds); wer = sum(d["wer"] * d["n_ref"] for d in ds) / max(n, 1)
    tn, to = sum(d["terms_n"] for d in ds), sum(d["terms_ok"] for d in ds)
    i = [d["info"] for d in ds]
    med = lambda k: sorted(x[k] for x in i)[len(i) // 2]
    out.append(f"| {ds[0]['model']} | {ds[0]['device']} | {wer:.1f}% ({len(ds)} video{'s' if len(ds) > 1 else ''}) | "
               f"{to}/{tn} ({100 * to // max(tn, 1)}%) | {med('final_ms_med')} / {med('final_ms_p90')} ms | "
               f"{med('partial_ms_med')} / {med('partial_ms_p90')} ms | {med('final_x_realtime')}x | {med('load_s')} s |")
path = os.path.join(os.environ.get("SUBTITLES_EVAL_DIR") or os.path.join(ROOT, "eval"), "reports", "model_comparison.md")
os.makedirs(os.path.dirname(path), exist_ok=True)
open(path, "w", encoding="utf-8").write("\n".join(out) + "\n"); print("\n".join(out)); print("\nsaved", path)
