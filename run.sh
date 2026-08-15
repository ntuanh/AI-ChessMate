#!/usr/bin/env sh
# AI-ChessMate v1.0 -- digital board + Stockfish hints.
#   ./run.sh                 start on http://127.0.0.1:8090
#   ./run.sh --port 9000     any digital_board.server flag is passed straight through
set -e
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python
command -v "$PY" >/dev/null 2>&1 || { echo "[x] python not found"; exit 1; }

"$PY" -c "import chess" >/dev/null 2>&1 || {
  echo "[*] installing python-chess ..."
  "$PY" -m pip install --quiet -r requirements.txt
}

"$PY" tools/get_stockfish.py --check >/dev/null 2>&1 || {
  echo "[*] no Stockfish found, downloading it once ..."
  "$PY" tools/get_stockfish.py
}

exec "$PY" -m digital_board.server "$@"
