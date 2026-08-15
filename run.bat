@echo off
REM AI-ChessMate v1.0 -- digital board + Stockfish hints.
REM   run.bat                 start on http://127.0.0.1:8090 and open a browser
REM   run.bat --port 9000     any chess_ai.server flag is passed straight through
REM   run.bat --no-browser
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

where python >nul 2>nul
if errorlevel 1 (
  echo [x] python is not on PATH. Install Python 3.9+ and try again.
  exit /b 1
)

python -c "import chess" >nul 2>nul
if errorlevel 1 (
  echo [*] installing python-chess ...
  python -m pip install --quiet -r requirements.txt || exit /b 1
)

python tools\get_stockfish.py --check >nul 2>nul
if errorlevel 1 (
  echo [*] no Stockfish found, downloading it once ...
  python tools\get_stockfish.py || exit /b 1
)

python -m chess_ai.server %*
