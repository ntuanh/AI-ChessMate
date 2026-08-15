"""AI-ChessMate -- a digital chessboard with a Stockfish hint on demand.

Version 1.0 is deliberately small: a board you can play on in the browser and an
engine that will tell you the next move when asked.  Board *vision* (reading a
physical board through a camera) is the next milestone, not this one.

Public surface:

    from chess_ai import Game, Engine, find_binary
"""

from __future__ import annotations

__version__ = "1.0.0"

from .engine import (
    Candidate,
    Engine,
    EngineUnavailable,
    Hint,
    find_binary,
)
from .game import Game, MoveRejected

__all__ = [
    "__version__",
    "Candidate",
    "Engine",
    "EngineUnavailable",
    "Game",
    "Hint",
    "MoveRejected",
    "find_binary",
]
