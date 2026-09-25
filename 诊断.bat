@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Run setup_windows.ps1 first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" windows_diagnostics.py
pause
