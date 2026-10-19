@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Run install.bat first.
  exit /b 1
)
".venv\Scripts\python.exe" -c "import yt_dlp" 2>nul
if errorlevel 1 (
  echo Installing yt-dlp...
  "%USERPROFILE%\.local\bin\uv.exe" pip install --python ".venv\Scripts\python.exe" yt-dlp
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" -u tools\eval_video.py %*
