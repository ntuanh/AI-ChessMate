"""Selftest for AI-ChessMate v1.0.  Exit 0 when everything passes.

    python tests/selftest.py

The engine tests are skipped, not failed, when no Stockfish binary is present:
a fresh clone has none until ``tools/get_stockfish.py`` runs, and a red suite on
first checkout teaches people to ignore the suite.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import chess  # noqa: E402

from chess_ai import __version__  # noqa: E402
from chess_ai.engine import (  # noqa: E402
    Engine,
    EngineUnavailable,
    MATE_SCORE,
    clamp_pawns,
    describe_score,
    find_binary,
    sequence_san,
)
from chess_ai.game import Game, MoveRejected  # noqa: E402

PASSED = 0
FAILED = 0
SKIPPED = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  ok   {label}")
    else:
        FAILED += 1
        print(f"  FAIL {label}" + (f" -- {detail}" if detail else ""))


def skip(label: str, why: str) -> None:
    global SKIPPED
    SKIPPED += 1
    print(f"  skip {label} -- {why}")


def section(name: str) -> None:
    print(f"\n{name}")


# -- game ----------------------------------------------------------------

def test_game() -> None:
    section("game state")
    g = Game()
    check("starts from the standard array", g.board.fen() == chess.STARTING_FEN)
    check("white to move", g.as_dict()["turn"] == "white")

    g.push("e2e4")
    check("accepts UCI", g.history_san() == ["e4"])
    g.push("c5")
    check("accepts SAN", g.history_san() == ["e4", "c5"])
    check("turn alternates", g.as_dict()["turn"] == "white")

    g.undo()
    check("undo takes back one half-move", g.history_san() == ["e4"])

    try:
        g.push("e2e4")
        check("illegal move rejected", False, "no exception")
    except MoveRejected:
        check("illegal move rejected", True)

    try:
        g.push("zz99")
        check("garbage rejected", False, "no exception")
    except MoveRejected:
        check("garbage rejected", True)

    # After the undo it is Black to move, so every key must be a black piece.
    state = g.as_dict()
    check("legal map is keyed by from-square", "e7" in state["legal"], str(state["legal"])[:80])
    check(
        "legal map only lists the side to move",
        all(
            g.board.color_at(chess.parse_square(sq)) == chess.BLACK
            for sq in state["legal"]
        ),
    )
    check("legal targets are reachable", state["legal"]["e7"] == ["e5", "e6"], str(state["legal"]["e7"]))
    check("last move reported", state["last_move"]["uci"] == "e2e4")
    check("undo flag set", state["can_undo"] is True)

    # Promotion: the UI sends four characters and the server defaults to a queen.
    p = Game()
    p.reset("8/P7/8/8/8/8/8/K6k w - - 0 1")
    move = p.push("a7a8")
    check("promotion defaults to a queen", move.promotion == chess.QUEEN)

    # Castling has to move the rook too, or every later move is read wrong.
    c = Game()
    for mv in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "O-O"]:
        c.push(mv)
    check("castling moves the rook", str(c.board.piece_at(chess.F1)) == "R")
    check("castling moves the king", str(c.board.piece_at(chess.G1)) == "K")

    m = Game()
    for mv in ["f3", "e5", "g4", "Qh4"]:
        m.push(mv)
    check("checkmate detected", m.board.is_checkmate())
    check("status names the winner", "Black wins" in m.status, m.status)
    check("game reported over", m.over)

    s = Game()
    s.reset("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    check("stalemate detected", "stalemate" in s.status, s.status)

    r = Game()
    r.push("e4")
    r.reset()
    check("reset clears history", r.history_san() == [])
    r.reset("8/8/8/4k3/8/8/4P3/4K3 w - - 0 1")
    check("reset accepts a FEN", r.board.piece_at(chess.E2) == chess.Piece.from_symbol("P"))
    r.reset()
    check("reset with no FEN returns to the standard array", r.board.fen() == chess.STARTING_FEN)

    try:
        r.reset("not a fen")
        check("bad FEN rejected", False, "no exception")
    except MoveRejected:
        check("bad FEN rejected", True)


# -- scoring helpers ------------------------------------------------------

def test_scoring() -> None:
    section("score formatting")
    check("white advantage keeps its sign", describe_score(120, None, True) == "+1.20")
    check("black to move flips the sign", describe_score(120, None, False) == "-1.20")
    check("mate for white", describe_score(MATE_SCORE, 3, True) == "#3")
    check("mate seen from black", describe_score(MATE_SCORE, 3, False) == "#-3")
    check("no score", describe_score(None, None, True) == "--")
    check("eval bar is bounded", clamp_pawns(MATE_SCORE) == 10.0)
    check("eval bar is bounded below", clamp_pawns(-MATE_SCORE) == -10.0)

    board = chess.Board()
    moves = [chess.Move.from_uci("e2e4"), chess.Move.from_uci("e7e5")]
    check("sequence_san converts a line", sequence_san(board, moves) == ["e4", "e5"])
    check("sequence_san does not mutate", board.fen() == chess.STARTING_FEN)


# -- engine ---------------------------------------------------------------

def test_engine() -> None:
    section("engine")
    binary = find_binary()
    if binary is None:
        skip("stockfish analysis", "no binary; run python tools/get_stockfish.py")
        return
    print(f"  ..   using {binary}")

    with Engine(binary=binary, threads=1) as eng:
        board = chess.Board()
        hint = eng.hint(board, multipv=3, movetime_ms=200)
        check("hint returned", hint is not None)
        check("best move is legal", hint.best.move in board.legal_moves)
        check("best move has zero loss", hint.best.loss == 0)
        check("alternatives ranked worse", all(c.loss >= 0 for c in hint.alternatives))
        check("opening eval is near level", abs(hint.best.cp) < 200, f"cp={hint.best.cp}")
        check("pv is populated", len(hint.best.pv) > 0)

        cached = eng.hint(board, multipv=3, movetime_ms=200)
        check("second identical hint is cached", cached.cached is True)

        # Mate in one: the only correct answer is Qxf7#.
        mate = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
        hint = eng.hint(mate, multipv=1, movetime_ms=400)
        check("finds mate in one", hint.best.san == "Qxf7#", hint.best.san)
        check("mate distance reported", hint.best.mate == 1, str(hint.best.mate))

        # A free queen must be taken.
        hanging = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6q1/5P2/PPPPP1PP/RNBQKBNR w KQkq - 0 1")
        hint = eng.hint(hanging, multipv=1, movetime_ms=400)
        check("takes the hanging queen", hint.best.san.startswith("fxg4"), hint.best.san)

        over = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
        check("no hint when there are no moves", eng.hint(over) is None)
        check("stalemate evaluates level", eng.evaluate(over) == 0)

        played = eng.play(chess.Board(), movetime_ms=200)
        check("play returns a legal move", played in chess.Board().legal_moves)


def test_engine_missing() -> None:
    section("engine discovery")
    check("find_binary rejects a bad explicit path", find_binary("/no/such/stockfish") is None)
    try:
        Engine(binary="/no/such/stockfish")
        check("missing binary raises EngineUnavailable", False, "no exception")
    except EngineUnavailable as exc:
        check("missing binary raises EngineUnavailable", True)
        check("error names the fix", "get_stockfish" in str(exc), str(exc))


# -- server ---------------------------------------------------------------

def test_server() -> None:
    section("http api")
    from http.server import ThreadingHTTPServer

    from chess_ai.server import Handler, Session

    session = Session()
    handler = type("BoundHandler", (Handler,), {"session": session})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"

    def call(path: str, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            base + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    try:
        code, body = call("/api/state")
        check("GET /api/state", code == 200 and body["fen"] == chess.STARTING_FEN)

        code, body = call("/api/health")
        check("GET /api/health", code == 200 and body["version"] == __version__)

        code, body = call("/api/move", {"move": "e2e4"})
        check("POST /api/move", code == 200 and body["history"] == ["e4"])

        code, body = call("/api/move", {"move": "e2e4"})
        check("illegal move gets a 400", code == 400 and not body["ok"], str(body))

        code, body = call("/api/undo", {})
        check("POST /api/undo", code == 200 and body["history"] == [])

        code, body = call("/api/new", {"fen": "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1"})
        check("POST /api/new with a FEN", code == 200 and body["fen"].startswith("8/8"))

        code, body = call("/api/new", {})
        check("POST /api/new resets to the start", code == 200 and body["fen"] == chess.STARTING_FEN)

        code, body = call("/api/new", {"fen": "garbage"})
        check("bad FEN gets a 400", code == 400, str(body))

        code, _ = call("/api/nope", {})
        check("unknown route 404s", code == 404)

        req = urllib.request.Request(base + "/")
        with urllib.request.urlopen(req, timeout=10) as resp:
            page = resp.read().decode("utf-8")
        check("board page is served", "AI-ChessMate" in page and "/api/hint" in page)

        if find_binary() is None:
            skip("POST /api/hint", "no Stockfish binary")
        else:
            call("/api/new", {})
            code, body = call("/api/hint", {"movetime": 200})
            check("POST /api/hint", code == 200 and "best" in body, str(body)[:120])
            check("hint carries an eval", isinstance(body.get("eval"), str))
            check("hint names squares for the arrow", "from" in body["best"])

            code, body = call("/api/engine-move", {"movetime": 200})
            check("POST /api/engine-move", code == 200 and body["history"], str(body)[:120])
    finally:
        httpd.shutdown()
        httpd.server_close()
        session.close()


def main() -> int:
    print(f"AI-ChessMate {__version__} selftest")
    test_game()
    test_scoring()
    test_engine_missing()
    test_engine()
    test_server()
    print(f"\n{PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
