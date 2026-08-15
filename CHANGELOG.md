# Changelog

All notable changes to AI-ChessMate. Versions follow [semver](https://semver.org/).

## [1.0.0] — 2026-08-15

First packaged release: **a digital board, and a Stockfish hint for the next move.**

### Added

- **Digital board** (`chess_ai/web/index.html`) — a self-contained page, no CDN and no build
  step. Click-to-move with legal-target dots, last-move and check highlighting, board flip,
  move list, and an eval bar. Keyboard: `H` hint, `U` undo, `F` flip.
- **Hints** — `POST /api/hint` returns Stockfish's best move drawn as an arrow on the board,
  with the evaluation, the expected line, and the runners-up priced in centipawns.
- **Play against the engine** — `POST /api/engine-move`, optionally at a reduced Skill Level, and
  an "engine answers my moves" mode. Undo takes back both halves in that mode.
- **`chess_ai.engine`** — one persistent Stockfish process behind `hint()`, `evaluate()` and
  `play()`, with analysis cached by `(FEN, multipv, movetime)`.
- **`find_binary()`** — engine discovery across `--stockfish`, `$STOCKFISH_PATH`, `engine/`,
  `$PATH` and the usual system locations, in that order.
- **`chess_ai.game`** — game state, server-side legality, undo, auto-queen promotion, and the
  JSON shape the page renders from.
- **`tools/get_stockfish.py`** — downloads the right official build for the platform through the
  GitHub releases API (so a moved tag does not break it), or copies one you already have with
  `--from`. `--check` reports what would be used.
- **`run.bat` / `run.sh`** — install the dependency if missing, fetch Stockfish on first run,
  start the server, open the browser.
- **`tests/selftest.py`** — 63 checks across game state, score formatting, engine discovery, real
  engine answers and every HTTP route. Engine tests skip, rather than fail, with no binary
  installed.

### Notes

- Stockfish is **not** committed: ~80 MB, platform-specific, GPL-3. It is fetched into
  `engine/`, which is git-ignored.
- Evaluations are always displayed from White's point of view; the engine reports from the side
  to move and a sign that flips every half-move is unreadable.
- The server is threaded but analysis is serialised under one lock — Stockfish is a UCI pipe and
  is not thread-safe.

[1.0.0]: https://github.com/ntuanh/AI-ChessMate/releases/tag/v1.0.0
