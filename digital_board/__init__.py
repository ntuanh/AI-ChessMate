"""digital_board — bàn cờ số trên trình duyệt, Stockfish gợi ý nước tiếp theo.

Đây là **v1.0**, chạy độc lập với `chess_ai/`:

  chess_ai/       đường chạy chính — camera soi bàn thật, PieceNet trên NPU, AIBOX
  digital_board/  bàn cờ số — không camera, không model, chạy trên máy nào cũng được

Hai bên KHÔNG import nhau. Lý do tách hẳn: `chess_ai.engine.ChessEngine` đóng đinh
`config.STOCKFISH_PATH = "/usr/games/stockfish"` (đường dẫn của AIBOX) và không có
cache lẫn dò binary, nên không khởi động được trên Windows/macOS. `digital_board`
cần chạy được trước khi có phần cứng — xem `engine.py`.

    from digital_board import Game, Engine, find_binary
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
