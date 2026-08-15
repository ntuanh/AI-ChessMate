# AI-ChessMate

A digital chessboard in your browser, with Stockfish telling you the next move whenever you ask.

**v1.0** is the first milestone: the board and the engine, nothing else. No camera, no vision, no
network calls once the engine is installed — it runs entirely offline on localhost.

```
┌──────────────────────────┐   White to move
│ 8  ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜       │
│ 7  ♟ ♟ ♟ ♟ ♟ ♟ ♟ ♟       │   Hint:  Nf3   (g1f3)
│ …                        │   eval +0.38 · 548 ms
│ 1  ♖ ♘ ♗ ♕ ♔ ♗ ♘ ♖       │   line: Nf3 d6 d4 Nf6
└──────────────────────────┘   also: Nc3 (-0.05), c3 (-0.11)
   a  b  c  d  e  f  g  h
```

## Quick start

```bash
git clone https://github.com/ntuanh/AI-ChessMate.git
cd AI-ChessMate

run.bat            # Windows
./run.sh           # Linux / macOS
```

The launcher installs `python-chess` if it is missing, downloads Stockfish on first run, then
opens <http://127.0.0.1:8090>. Nothing else to configure.

Doing it by hand is three commands:

```bash
pip install -r requirements.txt
python tools/get_stockfish.py       # ~40 MB, once
python -m digital_board.server           # --port 8090 --movetime 300 --no-browser
```

Requires **Python 3.9+**. The only dependency is [python-chess](https://python-chess.readthedocs.io/);
the server, the JSON API and the board page are standard library and plain JavaScript.

## Using the board

| | |
|---|---|
| **Move** | Click a piece, then its destination. Legal targets are dotted; the last move stays highlighted. |
| **Hint** (`H`) | Stockfish's best move, drawn as a green arrow, with the evaluation, the line it expects, and the two runners-up with what each one costs. |
| **Engine plays** | Let it make the move for the side to move. |
| **Engine answers my moves** | Tick it to play a game against Stockfish instead of just consulting it. |
| **Undo** (`U`) | Take back a half-move — or a full move when you are playing the engine, so it does not get a free tempo. |
| **Flip** (`F`) | Put Black at the bottom. |
| **Think time** | 100 ms to 3 s per hint. 300 ms is already far stronger than a club player. |
| **Strength** | Stockfish's own Skill Level, for an opponent you can actually beat. |

Promotion is automatic to a queen: the board sends `e7e8` and the server fills in the piece.

## HTTP API

The page is just one client. Everything it does is available over JSON, so a physical board, a
voice command or a script can drive the same session.

| Method | Route | Body | Returns |
|---|---|---|---|
| `GET` | `/api/state` | — | FEN, turn, status, history, legal moves, last move |
| `GET` | `/api/health` | — | version, engine path, whether it has started |
| `POST` | `/api/move` | `{"move": "e2e4"}` — UCI or SAN | the new state |
| `POST` | `/api/hint` | `{"movetime": 300, "multipv": 3}` | best move, alternatives, eval, PV |
| `POST` | `/api/engine-move` | `{"movetime": 300, "skill": 8}` | the move it played, plus the new state |
| `POST` | `/api/undo` | `{"full": true}` for both halves | the new state |
| `POST` | `/api/new` | `{"fen": "..."}` optional | the new state |

An illegal move is a `400` with a reason, not a silent no-op. A missing engine is a `503`.

```bash
curl -s -X POST -d '{"move":"e2e4"}' http://127.0.0.1:8090/api/move
curl -s -X POST -d '{"movetime":500}' http://127.0.0.1:8090/api/hint
```

Or in Python:

```python
from digital_board import Engine, Game

game = Game()
game.push("e4"); game.push("c5")

with Engine() as engine:
    hint = engine.hint(game.board, multipv=3, movetime_ms=300)
    print(hint.best.san, hint.best.cp, hint.best.pv[:4])   # Nf3 36 ('2.', 'Nf3', 'd6', '3.')
```

## Layout

```
digital_board/
  engine.py      Stockfish discovery + one persistent process, analysis cached by FEN
  game.py        game state, move legality, undo, the JSON the page renders from
  server.py      http.server routes; the engine is started lazily and shared under a lock
  web/index.html the board — one self-contained file, no CDN, no build step
tools/
  get_stockfish.py   downloads the right official build, or copies one you have
tests/
  test_digital_board.py   63 checks; exit 0 when they pass
engine/          the binary lands here; git-ignored
```

`chess_ai/` is the other half of this repository — the camera pipeline that reads a *physical*
board on the AIBOX. The two never import each other; see "Relationship to `chess_ai/`" below.

## Stockfish

Not committed — it is ~80 MB, platform-specific, and GPL-3 with its own source obligations.
`tools/get_stockfish.py` reads the official releases API and picks the right asset for your
platform, so it keeps working when the release tag moves.

`find_binary()` looks in this order, most explicit first:

1. the `--stockfish` flag
2. `$STOCKFISH_PATH` (or `$STOCKFISH`)
3. `engine/` in this repository
4. `stockfish` on `$PATH`
5. `chess_ai.config.STOCKFISH_PATH` — so on the AIBOX both halves use the same binary
6. `/usr/games/stockfish`, `/usr/local/bin/stockfish`, `/opt/homebrew/bin/stockfish`

Already have one? Skip the download:

```bash
python tools/get_stockfish.py --from /path/to/stockfish
python tools/get_stockfish.py --check      # print what would be used
```

## Notes worth knowing

- **The engine is one process, not one per request.** Spawning Stockfish costs more than the
  search does at 300 ms, and analysis is cached by `(FEN, multipv, movetime)` because a session
  revisits the same position constantly — hinting twice, undoing, reloading the page. A repeated
  hint comes back in ~0 ms and says `cached`.
- **Legality is decided on the server.** The page sends two squares; `Game.parse` accepts or
  rejects. A board that quietly ignores a move looks broken, so a rejection carries a reason.
- **Evaluations are always shown from White's point of view.** Stockfish reports from the side to
  move, and a number that flips sign every half-move is unreadable. `describe_score` is the one
  place that conversion happens.
- **Stockfish is not thread-safe** — it is a UCI pipe. The handler is threaded so a slow search
  cannot block the page from loading, but every analysis goes through a single lock.
- **A missing engine does not break the board.** The server starts, the page loads, you can play
  both sides; only Hint reports that no binary was found and how to get one.

## Tests

```bash
python tests/test_digital_board.py
```

63 checks over game state (castling moves the rook, promotion defaults to a queen, checkmate and
stalemate are recognised, illegal moves raise), score formatting, engine discovery, real engine
answers (mate in one is found, a hanging queen is taken) and every HTTP route. The engine tests
**skip** rather than fail when no binary is installed — a red suite on a fresh clone teaches
people to ignore the suite.

## Relationship to `chess_ai/`

The repository has two halves that do **not** import each other:

| | `chess_ai/` | `digital_board/` |
|---|---|---|
| Input | Logitech C270 filming a real board | mouse clicks |
| Where it runs | AIBOX 8550 (aarch64) | any machine with Python |
| Needs | OpenCV, numpy, PieceNet on the Hexagon NPU | `python-chess` only |
| Stockfish | `ChessEngine`, path from `config.STOCKFISH_PATH` | `Engine`, path from `find_binary()` |
| UI | `tools/coach_server.py` on `:8090` | `digital_board/server.py` on `:8090` |

Two Stockfish wrappers is deliberate, not an oversight. `chess_ai.engine.ChessEngine` hardcodes
`/usr/games/stockfish` and has no cache and no `loss` field; it cannot start on Windows, which is
exactly where the digital board has to run before any hardware exists. The one place they meet is
`find_binary()`, which consults `chess_ai.config.STOCKFISH_PATH` so that **on the AIBOX both halves
use the same binary** — inside a `try/except`, because a laptop running the board has no `cv2`.

Run them one at a time: both default to port `8090`.

## Where it could go next

- **v1.1** — PGN save/load, FEN setup in the UI, per-move accuracy review.
- **Feed the board from the camera.** `chess_ai/tracker3.py` already produces a tracked
  `chess.Board`; pushing that into `Game` would make this page a live mirror of the physical board
  instead of a separate game. `POST /api/new {"fen": ...}` is already the seam for it.
- **Explain the hint**, not just name it — `chess_ai/commentary.py` exists for exactly this.

## Licence

Stockfish is **GPL-3.0**, downloaded separately and never bundled here — if you redistribute a
build with the binary included, the GPL applies to that distribution. The repository itself has no
`LICENSE` file yet; that is the owner's call to make.
