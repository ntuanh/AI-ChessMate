"""Bọc Stockfish (qua python-chess) — vừa để AI ĐI nước, vừa để PHÂN TÍCH thế trận."""
from __future__ import annotations
import chess
import chess.engine
from dataclasses import dataclass
from typing import Optional

from . import config


@dataclass
class CandidateMove:
    """Một phương án nước đi ứng viên."""
    move: chess.Move
    san: str
    score_cp: Optional[int]
    mate_in: Optional[int]
    pv: list


@dataclass
class Analysis:
    """Kết quả phân tích một thế cờ."""
    score_cp: Optional[int]      # điểm theo centipawn, dương = TRẮNG lợi. None nếu là chiếu hết.
    mate_in: Optional[int]       # số nước tới chiếu hết (dương = trắng hết đen), None nếu không.
    best_move: Optional[chess.Move]
    pv: list                     # biến chính (list[chess.Move])
    depth: int
    candidates: list[CandidateMove] = None

    def pov_cp(self, color: bool) -> Optional[int]:
        """Điểm theo góc nhìn của `color` (chess.WHITE/BLACK). Dương = color đang lợi."""
        if self.score_cp is None:
            return None
        return self.score_cp if color == chess.WHITE else -self.score_cp


class ChessEngine:
    def __init__(self, settings: config.Settings):
        self.settings = settings
        self.engine = chess.engine.SimpleEngine.popen_uci(config.STOCKFISH_PATH)
        self._configure()

    def _configure(self):
        opts = {}
        s = self.settings
        # Skill level (0..20)
        try:
            opts["Skill Level"] = int(max(0, min(20, s.skill_level)))
        except Exception:
            pass
        # Giới hạn Elo nếu bật
        if s.limit_elo:
            opts["UCI_LimitStrength"] = True
            opts["UCI_Elo"] = int(max(1350, min(2850, s.elo)))
        try:
            self.engine.configure(opts)
        except Exception:
            pass

    def best_move(self, board: chess.Board) -> Optional[chess.Move]:
        """AI chọn nước đi (theo độ khó đã cấu hình)."""
        if board.is_game_over():
            return None
        limit = chess.engine.Limit(time=self.settings.think_time)
        result = self.engine.play(board, limit)
        return result.move

    def analyse(self, board: chess.Board, multipv: int = 3) -> Analysis:
        """Phân tích khách quan (full sức) với Multi-PV để lấy nhiều phương án."""
        results = self.engine.analyse(
            board, chess.engine.Limit(time=self.settings.analysis_time), multipv=multipv
        )
        if isinstance(results, dict):
            results = [results]

        candidates = []
        best_move = None
        best_pv = []
        depth = 0
        top_cp = None
        top_mate = None

        for idx, info in enumerate(results):
            pov = info["score"]
            white_score = pov.white()
            cp = white_score.score()
            mate = white_score.mate()
            pv = list(info.get("pv", []))
            if pv:
                mv = pv[0]
                try:
                    san_str = board.san(mv)
                except Exception:
                    san_str = mv.uci()
                candidates.append(CandidateMove(move=mv, san=san_str, score_cp=cp, mate_in=mate, pv=pv))
                if idx == 0:
                    best_move = mv
                    best_pv = pv
                    top_cp = cp
                    top_mate = mate
                    depth = int(info.get("depth", 0))

        return Analysis(
            score_cp=top_cp,
            mate_in=top_mate,
            best_move=best_move,
            pv=best_pv,
            depth=depth,
            candidates=candidates or [],
        )

    def close(self):
        try:
            self.engine.quit()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
