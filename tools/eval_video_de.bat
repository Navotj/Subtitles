@echo off
cd /d "%~dp0.."
".venv\Scripts\python.exe" -c "import yt_dlp" 2>nul || "%USERPROFILE%\.local\bin\uv.exe" pip install --python ".venv\Scripts\python.exe" yt-dlp
".venv\Scripts\python.exe" -u tools\eval_video_de.py > eval\work\de_fetch.log 2>&1
echo done > eval\work\de_fetch_done.txt
