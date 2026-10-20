"""Find German videos with HUMAN-MADE German captions on YouTube, download audio + captions, and print the list.

    tools\\eval_video_de.bat            (searches, downloads up to 3 videos of 20-60 min into eval\\media_de)

Auto-generated captions are never used (they contain the same mistakes we want to measure).
"""
import os, sys, glob, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eval", "media_de"); os.makedirs(OUT, exist_ok=True)
QUERIES = ["Podcast Gespräch deutsch Untertitel", "Dokumentation deutsch Untertitel ganze Folge", "Talkshow deutsch ganze Sendung Untertitel",
           "Rollenspiel Runde deutsch Pen and Paper Untertitel", "Interview lang deutsch Untertitel", "Vortrag deutsch Untertitel"]


def main(want=3):
    import yt_dlp
    found = []
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "extract_flat": True}) as y:
        for q in QUERIES:
            try: res = y.extract_info(f"ytsearch25:{q}", download=False)["entries"]
            except Exception as e: print("search failed:", q, e); continue
            for e in res:
                if e and e.get("id") and e["id"] not in [f[0] for f in found] and 900 <= (e.get("duration") or 0) <= 4500:
                    found.append((e["id"], e.get("title"), e.get("duration")))
    print(f"{len(found)} candidates; checking for human-made German captions ...")
    good = []
    with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as y:
        for vid, title, dur in found:
            if len(good) >= want: break
            try: info = y.extract_info("https://www.youtube.com/watch?v=" + vid, download=False)
            except Exception: continue
            subs = info.get("subtitles") or {}
            if any(k == "de" or k.startswith("de-") or k.startswith("de.") for k in subs):
                print("  ok:", vid, title, dur); good.append((vid, title))
    ok = []
    for vid, title in good:
        opts = {"format": "bestaudio/best", "outtmpl": os.path.join(OUT, "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True,
                "writesubtitles": True, "writeautomaticsub": False, "subtitleslangs": ["de", "de-*"], "subtitlesformat": "vtt/srt/best"}
        for fmt in ("bestaudio/best", "worstaudio/worst", "best"):         # a stream that answers 403 is skipped for the next one
            opts["format"] = fmt
            try:
                with yt_dlp.YoutubeDL(opts) as y: y.download(["https://www.youtube.com/watch?v=" + vid])
                ok.append((vid, title)); break
            except Exception as e: print("download failed:", vid, fmt, str(e)[:120])
    good = ok
    json.dump(good, open(os.path.join(OUT, "list.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("downloaded:", good)


if __name__ == "__main__":
    main()
