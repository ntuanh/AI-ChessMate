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
python -m chess_ai.server           # --port 8090 --movetime 300 --no-browser
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
from chess_ai import Engine, Game

game = Game()
game.push("e4"); game.push("c5")

with Engine() as engine:
    hint = engine.hint(game.board, multipv=3, movetime_ms=300)
    print(hint.best.san, hint.best.cp, hint.best.pv[:4])   # Nf3 36 ('2.', 'Nf3', 'd6', '3.')
```

## Layout

```
chess_ai/
  engine.py      Stockfish discovery + one persistent process, analysis cached by FEN
  game.py        game state, move legality, undo, the JSON the page renders from
  server.py      http.server routes; the engine is started lazily and shared under a lock
  web/index.html the board — one self-contained file, no CDN, no build step
tools/
  get_stockfish.py   downloads the right official build, or copies one you have
tests/
  selftest.py    63 checks; exit 0 when they pass
engine/          the binary lands here; git-ignored
```

## Stockfish

Not committed — it is ~80 MB, platform-specific, and GPL-3 with its own source obligations.
`tools/get_stockfish.py` reads the official releases API and picks the right asset for your
platform, so it keeps working when the release tag moves.

`find_binary()` looks in this order, most explicit first:

1. the `--stockfish` flag
2. `$STOCKFISH_PATH` (or `$STOCKFISH`)
3. `engine/` in this repository
4. `stockfish` on `$PATH`
5. `/usr/games/stockfish`, `/usr/local/bin/stockfish`, `/opt/homebrew/bin/stockfish`

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
python tests/selftest.py
```

63 checks over game state (castling moves the rook, promotion defaults to a queen, checkmate and
stalemate are recognised, illegal moves raise), score formatting, engine discovery, real engine
answers (mate in one is found, a hanging queen is taken) and every HTTP route. The engine tests
**skip** rather than fail when no binary is installed — a red suite on a fresh clone teaches
people to ignore the suite.

## Roadmap

v1.0 is the board and the engine. What the project is aimed at:

- **v1.1** — save and load PGN, position setup from a FEN in the UI, per-move accuracy review.
- **v2.0** — *board vision*: read a real board through a webcam (quad detection, rectification, an
  occupancy CNN and a piece classifier) so the digital board mirrors the physical one.
- **v2.1** — spoken requests, and an explanation of *why* the hinted move is good rather than just
  the move.

## Licence

This project is MIT. Stockfish is **GPL-3.0** and is downloaded separately, never bundled here —
if you redistribute a build with the binary included, the GPL applies to that distribution.
