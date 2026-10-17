@echo off
cd /d "%~dp0"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
.venv\Scripts\python.exe -u selftest.py %*
