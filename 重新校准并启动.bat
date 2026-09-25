@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Run setup_windows.ps1 first.
  pause
  exit /b 1
)
set "XIANGQI_RECALIBRATE=1"
".venv\Scripts\python.exe" xiangqi_bot.py
pause
