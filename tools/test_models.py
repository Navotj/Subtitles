"""Offline checks of models.py: recommendations, language rules, speaker download, change + delayed cleanup."""
import os, sys, tempfile, threading, http.server, functools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import models

tmp = tempfile.mkdtemp(); models.MODELS = os.path.join(tmp, "models"); os.makedirs(models.MODELS)
ok = 0
def check(name, cond):
    global ok
    assert cond, name; ok += 1

check("5080 -> turbo", models.recommended_speech(("RTX 5080", 16303), ["en"]) == "large-v3-turbo")
check("small card, english", models.recommended_speech(("GTX 1050", 4096), ["en"]) == "small.en")
check("no card, german", models.recommended_speech(None, ["en", "de"]) == "small")
check("english-only model rejects german", not models.supports(models.by_id(models.SPEECH, "small.en"), ["en", "de"]))
check("multilingual model accepts", models.supports(models.by_id(models.SPEECH, "small"), ["de"]))

srv = os.path.join(tmp, "srv"); os.makedirs(srv)
open(os.path.join(srv, "b.onnx"), "wb").write(b"y" * 200000)
httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(http.server.SimpleHTTPRequestHandler, directory=srv))
httpd.RequestHandlerClass.log_message = lambda *a: None
threading.Thread(target=httpd.serve_forever, daemon=True).start()
models.SHERPA = f"http://127.0.0.1:{httpd.server_address[1]}/"
seen = []; models.download_speaker("b.onnx", seen.append)
check("speaker model downloaded", models.speaker_installed("b.onnx") and seen[-1] == 1.0)
check("no .part left", not os.path.exists(os.path.join(models.MODELS, "b.onnx.part")))

open(os.path.join(models.MODELS, "a.onnx"), "wb").write(b"x")
d = os.path.join(models.MODELS, "models--Systran--faster-whisper-small", "snapshots", "h"); os.makedirs(d)
open(os.path.join(d, "model.bin"), "wb").write(b"x")
os.makedirs(os.path.join(models.MODELS, "models--Systran--faster-whisper-small.en", "snapshots", "h"))
check("small installed, small.en (empty) not", models.speech_installed("small") and not models.speech_installed("small.en"))
stale = models.download_changes(dict(whisper_model="small", voice_model="a.onnx"), dict(whisper_model="small", voice_model="b.onnx"))
check("only the changed model is stale", stale == [("speaker", "a.onnx")])
cfg = dict(whisper_model="small", voice_model="b.onnx", models_to_delete=[["speaker", "a.onnx"], ["speech", "small"]])
check("cleanup keeps the model in use", models.cleanup(cfg) == ["a.onnx"] and models.speech_installed("small") and not models.speaker_installed("a.onnx"))
print(f"{ok} checks ok")
