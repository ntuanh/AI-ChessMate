"""Game state: the board, the move list, and what the browser needs to draw it.

The rules live in python-chess; this module owns only the session -- history for
undo, a starting position to reset to, and a JSON shape stable enough that the
web page can be written against it.

Legality is decided *here*, never in the browser.  The page sends a from-square
and a to-square; anything illegal is rejected with a reason rather than silently
ignored, because a board that quietly refuses a move looks broken.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chess


class MoveRejected(ValueError):
    """The requested move is not legal in the current position."""


@dataclass
class Game:
    """One game in progress."""

    start_fen: str = chess.STARTING_FEN
    board: chess.Board = field(init=False)

    def __post_init__(self) -> None:
        self.board = chess.Board(self.start_fen)

    # -- mutation ---------------------------------------------------------

    def reset(self, fen: Optional[str] = None) -> None:
        """Start over, from ``fen`` if given and the standard array otherwise.

        No argument means the *standard* start, not a replay of whatever position
        was last loaded -- otherwise loading a FEN once would leave the New game
        button permanently stuck on it.
        """
        if fen:
            try:
                chess.Board(fen)
            except ValueError as exc:
                raise MoveRejected(f"invalid FEN: {exc}") from exc
            self.start_fen = fen
        else:
            self.start_fen = chess.STARTING_FEN
        self.board = chess.Board(self.start_fen)

    def push(self, text: str) -> chess.Move:
        """Apply a move given in UCI (``e2e4``) or SAN (``Nf3``).

        Promotion is defaulted to a queen when the UCI string omits it, which is
        what a click-to-move UI sends before it has asked the user.
        """
        move = self.parse(text)
        self.board.push(move)
        return move

    def parse(self, text: str) -> chess.Move:
        """Text to a legal move, or raise ``MoveRejected``."""
        text = (text or "").strip()
        if not text:
            raise MoveRejected("empty move")

        move: Optional[chess.Move] = None
        try:
            move = chess.Move.from_uci(text)
        except ValueError:
            try:
                move = self.board.parse_san(text)
            except ValueError as exc:
                raise MoveRejected(f"cannot read {text!r}: {exc}") from exc

        if move not in self.board.legal_moves:
            # A pawn reaching the last rank must promote; the UI sends e7e8 and
            # lets the server default it rather than guessing in JavaScript.
            promoted = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if promoted in self.board.legal_moves:
                return promoted
            raise MoveRejected(f"{text} is not legal in this position")
        return move

    def undo(self) -> Optional[chess.Move]:
        """Take back the last half-move.  None when there is nothing to take back."""
        if not self.board.move_stack:
            return None
        return self.board.pop()

    # -- reporting --------------------------------------------------------

    @property
    def status(self) -> str:
        """One short phrase describing where the game stands."""
        b = self.board
        if b.is_checkmate():
            return "Checkmate -- " + ("Black" if b.turn == chess.WHITE else "White") + " wins"
        if b.is_stalemate():
            return "Draw -- stalemate"
        if b.is_insufficient_material():
            return "Draw -- insufficient material"
        if b.is_fifty_moves():
            return "Draw -- fifty-move rule"
        if b.is_repetition(3):
            return "Draw -- threefold repetition"
        side = "White" if b.turn == chess.WHITE else "Black"
        return f"{side} to move -- check" if b.is_check() else f"{side} to move"

    @property
    def over(self) -> bool:
        return self.board.is_game_over(claim_draw=True)

    def history_san(self) -> List[str]:
        """The move list in SAN, replayed from the start position."""
        scratch = chess.Board(self.start_fen)
        out: List[str] = []
        for move in self.board.move_stack:
            out.append(scratch.san(move))
            scratch.push(move)
        return out

    def legal_targets(self) -> Dict[str, List[str]]:
        """``{"e2": ["e3", "e4"], ...}`` -- what the page highlights on click."""
        targets: Dict[str, List[str]] = {}
        for move in self.board.legal_moves:
            targets.setdefault(chess.square_name(move.from_square), []).append(
                chess.square_name(move.to_square)
            )
        for squares in targets.values():
            squares.sort()
        return targets

    def as_dict(self) -> dict:
        """The full state the browser renders from."""
        last = self.board.move_stack[-1] if self.board.move_stack else None
        checked_square = None
        if self.board.is_check():
            king = self.board.king(self.board.turn)
            if king is not None:
                checked_square = chess.square_name(king)
        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "status": self.status,
            "over": self.over,
            "check": self.board.is_check(),
            "check_square": checked_square,
            "fullmove": self.board.fullmove_number,
            "history": self.history_san(),
            "can_undo": bool(self.board.move_stack),
            "legal": self.legal_targets(),
            "last_move": (
                {
                    "uci": last.uci(),
                    "from": chess.square_name(last.from_square),
                    "to": chess.square_name(last.to_square),
                }
                if last
                else None
            ),
        }
