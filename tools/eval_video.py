"""Download a video's audio and its HUMAN-MADE English captions (never auto-generated ones), then run whisper_eval on it.

    tools\\eval_video.bat "<video url>" ["<another url>" ...] [whisper_eval options, e.g. --minutes 20 --expect words.txt]

Videos without human-made English captions are skipped (auto-generated captions contain the same kind of mistakes we are
trying to find, so they are no use as an answer key). Downloads go to eval\\media for local testing only.
If a download fails, download the audio and the captions yourself and use whisper_eval.py directly.
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whisper_eval as we


def fetch(url):
    import yt_dlp
    out = os.path.join(we.EVAL, "media"); os.makedirs(out, exist_ok=True)
    opts = {"format": "bestaudio/best", "outtmpl": os.path.join(out, "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True,
            "writesubtitles": True, "writeautomaticsub": False, "subtitleslangs": ["en.*"], "subtitlesformat": "vtt/srt/best"}
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=False)
        if not any(k.startswith("en") for k in (info.get("subtitles") or {})):
            return None, None, info.get("title")
        y.download([url])
    vid = info["id"]
    subs = sorted(p for p in glob.glob(os.path.join(out, vid + ".en*")) if p.endswith((".vtt", ".srt")))
    media = [p for p in glob.glob(os.path.join(out, vid + ".*")) if not p.endswith((".vtt", ".srt", ".json", ".part"))]
    return (media[0] if media else None), (subs[0] if subs else None), info.get("title")


def main():
    urls = [a for a in sys.argv[1:] if a.startswith("http")]
    rest = [a for a in sys.argv[1:] if not a.startswith("http")]
    if not urls: raise SystemExit(__doc__)
    for u in urls:
        print(f"\n=== {u}")
        try:
            media, subs, title = fetch(u)
        except Exception as e:
            print(f"download failed: {e}"); continue
        if not subs:
            print(f"skipped: '{title}' has no human-made English captions. Try another video."); continue
        if not media:
            print("skipped: the audio did not download"); continue
        print(f"'{title}': captions {os.path.basename(subs)}")
        we.main([media, subs, "--name", os.path.splitext(os.path.basename(media))[0]] + rest)


if __name__ == "__main__":
    main()
