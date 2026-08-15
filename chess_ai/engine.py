"""Persistent Stockfish wrapper, plus the search for a binary to run.

One long-lived engine process, not a fresh one per query: on this project's
hardware spawning Stockfish costs more than the search itself.  Analysis is
cached by (FEN, multipv, movetime) because a session revisits the same position
constantly -- asking for a hint twice, undoing and replaying, refreshing the page.

Nothing here is Windows- or Linux-specific.  ``find_binary`` looks in the places
a binary actually turns up, in order of how explicit the intent was, and every
caller gets the same answer.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import chess.engine

#: Score assigned to a forced mate when converting to centipawns.
MATE_SCORE = 100_000

#: Repository root -- ``engine/`` next to this package is the vendored location.
ROOT = Path(__file__).resolve().parent.parent

#: Where a system package manager tends to put it.
SYSTEM_PATHS = (
    "/usr/games/stockfish",
    "/usr/local/bin/stockfish",
    "/opt/homebrew/bin/stockfish",
)


class EngineUnavailable(RuntimeError):
    """Stockfish is not installed, or would not start."""


def find_binary(explicit: Optional[str] = None) -> Optional[str]:
    """Locate a Stockfish executable, or return None.

    Order, most explicit first:

    1. ``explicit`` -- passed on the command line.
    2. ``$STOCKFISH_PATH`` (or ``$STOCKFISH``).
    3. ``engine/`` in the repository, where ``tools/get_stockfish.py`` puts it.
    4. ``stockfish`` on ``$PATH``.
    5. The usual system install locations.
    """
    if explicit:
        return explicit if _runnable(explicit) else None

    for var in ("STOCKFISH_PATH", "STOCKFISH"):
        value = os.environ.get(var)
        if value and _runnable(value):
            return value

    vendored = _search_dir(ROOT / "engine")
    if vendored:
        return vendored

    for name in ("stockfish", "stockfish.exe"):
        found = shutil.which(name)
        if found:
            return found

    for path in SYSTEM_PATHS:
        if _runnable(path):
            return path

    return None


def _search_dir(directory: Path) -> Optional[str]:
    """Newest ``stockfish*`` executable directly inside ``directory``."""
    if not directory.is_dir():
        return None
    matches = [
        p
        for p in directory.iterdir()
        if p.is_file()
        and p.name.lower().startswith("stockfish")
        and p.suffix.lower() in ("", ".exe")
    ]
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0])


def _runnable(path: str) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    # Windows has no execute bit; the extension is what decides there.
    if sys.platform == "win32":
        return True
    return os.access(path, os.X_OK)


@dataclass(frozen=True)
class Candidate:
    """One engine-approved move, with its cost in centipawns."""

    move: chess.Move
    san: str
    #: Evaluation after the move, from the moving side's point of view.
    cp: int
    #: How much worse than the engine's top choice, in centipawns.  0 for the best.
    loss: int
    #: Principal variation in SAN, for explaining the suggestion.
    pv: Tuple[str, ...]
    #: Mate distance if this line forces mate, else None.  Positive = we mate.
    mate: Optional[int] = None

    @property
    def is_best(self) -> bool:
        return self.loss == 0

    def as_dict(self) -> dict:
        return {
            "uci": self.move.uci(),
            "san": self.san,
            "from": chess.square_name(self.move.from_square),
            "to": chess.square_name(self.move.to_square),
            "cp": self.cp,
            "loss": self.loss,
            "mate": self.mate,
            "pv": list(self.pv),
        }


@dataclass(frozen=True)
class Hint:
    """What the UI needs to draw an arrow and say why."""

    best: Candidate
    alternatives: Tuple[Candidate, ...]
    #: Milliseconds actually spent searching.
    elapsed_ms: int
    #: True when the answer came from the cache rather than a fresh search.
    cached: bool

    def as_dict(self) -> dict:
        return {
            "best": self.best.as_dict(),
            "alternatives": [c.as_dict() for c in self.alternatives],
            "elapsed_ms": self.elapsed_ms,
            "cached": self.cached,
        }


class Engine:
    """A persistent Stockfish process behind a small analysis API."""

    def __init__(
        self,
        binary: Optional[str] = None,
        threads: int = 2,
        hash_mb: int = 128,
        cpus: Optional[str] = None,
    ) -> None:
        resolved = find_binary(binary)
        if resolved is None:
            raise EngineUnavailable(
                "Stockfish not found. Run `python tools/get_stockfish.py`, or set "
                "STOCKFISH_PATH to an existing binary."
            )
        self.binary = resolved

        command: List[str] = [resolved]
        # Pinning is a Linux nicety (the QCS8550 board has three slow cores and
        # three fast ones); everywhere else taskset is simply absent.
        if cpus and shutil.which("taskset"):
            command = ["taskset", "-c", cpus] + command

        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(command)
        except Exception as exc:  # noqa: BLE001 - surfaced with context below
            raise EngineUnavailable(f"could not start {' '.join(command)}: {exc}") from exc

        try:
            self._engine.configure({"Threads": threads, "Hash": hash_mb})
        except Exception:  # noqa: BLE001 - a build may not expose both options
            pass

        self.name = self._engine.id.get("name", "Stockfish")
        self._cache: Dict[Tuple[str, int, int], List[Candidate]] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    # -- analysis ---------------------------------------------------------

    def candidates(
        self,
        board: chess.Board,
        multipv: int = 3,
        movetime_ms: int = 300,
    ) -> List[Candidate]:
        """Return up to ``multipv`` legal moves ranked best-first.

        ``loss`` is relative to the top line, so a caller that wants a *different*
        move (a style scorer, a teaching mode) can see what each one costs.
        """
        legal = board.legal_moves.count()
        if legal == 0:
            return []
        multipv = max(1, min(multipv, legal))

        key = (board.fen(), multipv, movetime_ms)
        cached = self._cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1

        infos = self._engine.analyse(
            board,
            chess.engine.Limit(time=movetime_ms / 1000.0),
            multipv=multipv,
        )
        if isinstance(infos, dict):  # multipv=1 returns a bare InfoDict
            infos = [infos]

        raw: List[Tuple[chess.Move, int, Optional[int], Tuple[str, ...]]] = []
        for info in infos:
            pv = info.get("pv") or []
            if not pv:
                continue
            score = info["score"].pov(board.turn)
            cp = score.score(mate_score=MATE_SCORE)
            raw.append((pv[0], cp, score.mate(), tuple(board.variation_san(pv).split())))

        if not raw:
            return []

        best_cp = max(cp for _, cp, _, _ in raw)
        out = [
            Candidate(
                move=move,
                san=board.san(move),
                cp=cp,
                loss=best_cp - cp,
                pv=pv,
                mate=mate,
            )
            for move, cp, mate, pv in raw
        ]
        out.sort(key=lambda c: c.loss)
        self._cache[key] = out
        return out

    def hint(
        self,
        board: chess.Board,
        multipv: int = 3,
        movetime_ms: int = 300,
    ) -> Optional[Hint]:
        """The next move to play, with runners-up.  None when the game is over."""
        import time

        before = self.cache_misses
        start = time.perf_counter()
        cands = self.candidates(board, multipv=multipv, movetime_ms=movetime_ms)
        elapsed = int((time.perf_counter() - start) * 1000)
        if not cands:
            return None
        return Hint(
            best=cands[0],
            alternatives=tuple(cands[1:]),
            elapsed_ms=elapsed,
            cached=self.cache_misses == before,
        )

    def evaluate(self, board: chess.Board, movetime_ms: int = 300) -> int:
        """Centipawn evaluation from the side to move's point of view."""
        cands = self.candidates(board, multipv=1, movetime_ms=movetime_ms)
        if not cands:
            # No legal moves: checkmate is lost, stalemate is level.
            return -MATE_SCORE if board.is_checkmate() else 0
        return cands[0].cp

    def play(
        self,
        board: chess.Board,
        movetime_ms: int = 300,
        skill: Optional[int] = None,
    ) -> Optional[chess.Move]:
        """Let the engine pick a move to actually play.

        ``skill`` maps to Stockfish's own ``Skill Level`` (0-20) so v1.0 can offer
        a weaker opponent without a second search path.
        """
        if skill is not None:
            try:
                self._engine.configure({"Skill Level": max(0, min(20, skill))})
            except Exception:  # noqa: BLE001 - option missing on some builds
                pass
        if not board.legal_moves:
            return None
        result = self._engine.play(board, chess.engine.Limit(time=movetime_ms / 1000.0))
        return result.move

    # -- lifecycle --------------------------------------------------------

    def close(self) -> None:
        try:
            self._engine.quit()
        except Exception:  # noqa: BLE001 - closing must never raise
            pass

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def describe_score(cp: Optional[int], mate: Optional[int], white_pov: bool) -> str:
    """Human-readable evaluation, always from White's point of view.

    The engine reports from the side to move; a UI that flips sign every half-move
    is unreadable, so everything user-facing goes through here.
    """
    if mate is not None:
        return f"#{mate}" if white_pov else f"#{-mate}"
    if cp is None:
        return "--"
    value = cp if white_pov else -cp
    return f"{value / 100.0:+.2f}"


def clamp_pawns(cp: int) -> float:
    """Centipawns to a bounded pawn value, for an eval bar that never runs off."""
    return max(-10.0, min(10.0, cp / 100.0))


def sequence_san(board: chess.Board, moves: Sequence[chess.Move]) -> List[str]:
    """SAN for a sequence of moves from ``board``, without mutating it."""
    scratch = board.copy(stack=False)
    out: List[str] = []
    for move in moves:
        if move not in scratch.legal_moves:
            break
        out.append(scratch.san(move))
        scratch.push(move)
    return out
