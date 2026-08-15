"""The v1.0 app: a digital board in the browser, a Stockfish hint on request.

Standard library only on the server side (``http.server`` + ``json``) and a single
self-contained page on the client side.  No framework, no CDN, no build step --
the whole thing runs offline, which is the point on a board with no internet.

The engine is started *lazily and once*.  Stockfish speaks UCI over a pipe and is
not thread-safe, so every analysis goes through one lock; the handler is threaded
only so a slow search cannot block the page from loading.

    python -m chess_ai.server --port 8090
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple

import chess

from . import __version__
from .engine import Engine, EngineUnavailable, clamp_pawns, describe_score, find_binary
from .game import Game, MoveRejected

WEB_DIR = Path(__file__).resolve().parent / "web"

#: Default search time per hint.  300 ms is already far stronger than a human
#: club player; the cost of going higher is felt immediately in the UI.
DEFAULT_MOVETIME_MS = 300


class Session:
    """Everything the handlers share: one game, one engine, one lock."""

    def __init__(
        self,
        binary: Optional[str] = None,
        movetime_ms: int = DEFAULT_MOVETIME_MS,
        threads: int = 2,
    ) -> None:
        self.game = Game()
        self.movetime_ms = movetime_ms
        self.threads = threads
        self.binary = binary
        self.lock = threading.Lock()
        self._engine: Optional[Engine] = None
        self._engine_error: Optional[str] = None

    def engine(self) -> Engine:
        """The shared engine, started on first use.

        A failed start is remembered so a missing binary does not cost a process
        spawn attempt on every click.
        """
        if self._engine is not None:
            return self._engine
        if self._engine_error is not None:
            raise EngineUnavailable(self._engine_error)
        try:
            self._engine = Engine(
                binary=self.binary, threads=self.threads, hash_mb=128
            )
        except EngineUnavailable as exc:
            self._engine_error = str(exc)
            raise
        return self._engine

    def engine_ready(self) -> Tuple[bool, str]:
        if self._engine is not None:
            return True, self._engine.name
        path = find_binary(self.binary)
        if path is None:
            return False, "not found"
        return False, path

    def close(self) -> None:
        if self._engine is not None:
            self._engine.close()
            self._engine = None


def _white_pov(board: chess.Board) -> bool:
    """Engine scores are from the side to move; the UI is always White-relative."""
    return board.turn == chess.WHITE


class Handler(BaseHTTPRequestHandler):
    server_version = f"AI-ChessMate/{__version__}"
    session: Session  # injected by serve()

    # -- plumbing ---------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:  # noqa: A003 - base class name
        # One line per request would drown the hint timings that matter.
        if not self.path.startswith("/api/") or self.command != "GET":
            sys.stderr.write(f"  {self.command} {self.path}\n")

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass  # the tab was closed mid-response

    def _json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

    def _error(self, message: str, code: int = 400) -> None:
        self._json({"ok": False, "error": message}, code)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            return {}

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._file("index.html", "text/html; charset=utf-8")
        if path == "/api/state":
            return self._json({"ok": True, **self.session.game.as_dict()})
        if path == "/api/health":
            started, detail = self.session.engine_ready()
            return self._json(
                {
                    "ok": True,
                    "version": __version__,
                    "engine_started": started,
                    "engine": detail,
                    "movetime_ms": self.session.movetime_ms,
                }
            )
        return self._error("not found", 404)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        body = self._body()
        session = self.session

        if path == "/api/move":
            try:
                session.game.push(str(body.get("move", "")))
            except MoveRejected as exc:
                return self._error(str(exc))
            return self._json({"ok": True, **session.game.as_dict()})

        if path == "/api/new":
            try:
                session.game.reset(body.get("fen") or None)
            except MoveRejected as exc:
                return self._error(str(exc))
            return self._json({"ok": True, **session.game.as_dict()})

        if path == "/api/undo":
            # Undo twice when asked, so taking back your move also takes back the
            # engine's reply -- otherwise you hand it a free move.
            count = 2 if body.get("full") else 1
            for _ in range(count):
                if session.game.undo() is None:
                    break
            return self._json({"ok": True, **session.game.as_dict()})

        if path == "/api/hint":
            return self._hint(body)

        if path == "/api/engine-move":
            return self._engine_move(body)

        return self._error("not found", 404)

    # -- engine-backed routes ---------------------------------------------

    def _hint(self, body: dict) -> None:
        session = self.session
        board = session.game.board
        if board.is_game_over(claim_draw=True):
            return self._error("the game is over")

        movetime = int(body.get("movetime") or session.movetime_ms)
        multipv = max(1, min(5, int(body.get("multipv") or 3)))
        try:
            with session.lock:
                hint = session.engine().hint(
                    board, multipv=multipv, movetime_ms=movetime
                )
        except EngineUnavailable as exc:
            return self._error(str(exc), 503)
        except Exception as exc:  # noqa: BLE001 - a dead engine must not 500 blankly
            return self._error(f"engine failed: {exc}", 500)

        if hint is None:
            return self._error("no legal moves")

        white_pov = _white_pov(board)
        return self._json(
            {
                "ok": True,
                **hint.as_dict(),
                "eval": describe_score(hint.best.cp, hint.best.mate, white_pov),
                "eval_pawns": clamp_pawns(hint.best.cp if white_pov else -hint.best.cp),
            }
        )

    def _engine_move(self, body: dict) -> None:
        session = self.session
        board = session.game.board
        if board.is_game_over(claim_draw=True):
            return self._error("the game is over")

        movetime = int(body.get("movetime") or session.movetime_ms)
        skill = body.get("skill")
        try:
            with session.lock:
                move = session.engine().play(
                    board,
                    movetime_ms=movetime,
                    skill=int(skill) if skill is not None else None,
                )
        except EngineUnavailable as exc:
            return self._error(str(exc), 503)
        except Exception as exc:  # noqa: BLE001
            return self._error(f"engine failed: {exc}", 500)

        if move is None:
            return self._error("no legal moves")
        san = board.san(move)
        board.push(move)
        return self._json({"ok": True, "played": san, **session.game.as_dict()})

    # -- static -----------------------------------------------------------

    def _file(self, name: str, ctype: str) -> None:
        target = WEB_DIR / name
        if not target.is_file():
            return self._error(f"missing asset {name}", 404)
        self._send(200, target.read_bytes(), ctype)


def serve(
    host: str = "127.0.0.1",
    port: int = 8090,
    binary: Optional[str] = None,
    movetime_ms: int = DEFAULT_MOVETIME_MS,
    threads: int = 2,
    open_browser: bool = True,
) -> None:
    session = Session(binary=binary, movetime_ms=movetime_ms, threads=threads)
    handler = type("BoundHandler", (Handler,), {"session": session})
    httpd = ThreadingHTTPServer((host, port), handler)

    url = f"http://{host}:{port}"
    found = find_binary(binary)
    print(f"AI-ChessMate {__version__} -- digital board + Stockfish hints")
    print(f"  board   : {url}")
    print(f"  engine  : {found or 'NOT FOUND -- run python tools/get_stockfish.py'}")
    print(f"  movetime: {movetime_ms} ms per hint")
    print("  Ctrl-C to stop")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
        session.close()


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="chess_ai.server",
        description="Digital chessboard with Stockfish hints (AI-ChessMate v1.0)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument(
        "--stockfish", default=None, help="path to the Stockfish binary"
    )
    parser.add_argument(
        "--movetime",
        type=int,
        default=DEFAULT_MOVETIME_MS,
        help="milliseconds of search per hint (default: 300)",
    )
    parser.add_argument("--threads", type=int, default=2, help="engine threads")
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab"
    )
    args = parser.parse_args(argv)

    serve(
        host=args.host,
        port=args.port,
        binary=args.stockfish,
        movetime_ms=args.movetime,
        threads=args.threads,
        open_browser=not args.no_browser,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
