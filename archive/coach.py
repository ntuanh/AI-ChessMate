"""BỘ ĐIỀU PHỐI — gắn kết engine + phân tích + bình luận + trợ lý + giọng nói
thành một "huấn luyện viên kiêm đối thủ" hoàn chỉnh."""
from __future__ import annotations
import chess

from chess_ai import config, analysis
from chess_ai.engine import ChessEngine
from chess_ai.commentary import Commentator
from chess_ai.assistant import Assistant
from chess_ai.speaker import Speaker


class Coach:
    def __init__(self, settings: config.Settings):
        self.settings = settings
        self.engine = ChessEngine(settings)
        self.commentator = Commentator(settings)
        self.assistant = Assistant(settings, self.engine,
                                   use_llm=getattr(settings, "use_llm", True))
        self.speaker = Speaker(settings)
        self.board = chess.Board()

    # ---------- tiện ích ----------
    def human_color(self) -> bool:
        return chess.WHITE if self.settings.human_color == "white" else chess.BLACK

    def ai_color(self) -> bool:
        return not self.human_color()

    def is_ai_turn(self) -> bool:
        return self.board.turn == self.ai_color()

    def game_over(self) -> bool:
        return self.board.is_game_over()

    def result_text(self) -> str:
        if not self.board.is_game_over():
            return ""
        if self.board.is_checkmate():
            winner = analysis.COLOR_VI[not self.board.turn]
            return f"Chiếu hết! {winner} thắng."
        if self.board.is_stalemate():
            return "Hòa cờ (pat)."
        if self.board.is_insufficient_material():
            return "Hòa do không đủ lực."
        if self.board.can_claim_threefold_repetition():
            return "Hòa do lặp nước 3 lần."
        return "Ván cờ kết thúc (hòa)."

    def say(self, text: str, wav: str | None = None):
        self.speaker.say(text, save_wav=wav)

    # ---------- nước đi ----------
    def parse_move(self, text: str):
        """Nhận nước từ người: SAN ('Nf3','e4') hoặc UCI ('g1f3')."""
        text = text.strip()
        for parser in (self.board.parse_san, self.board.parse_uci):
            try:
                mv = parser(text)
                if mv in self.board.legal_moves:
                    return mv
            except Exception:
                continue
        return None

    def human_move(self, text: str):
        """Áp dụng nước của người + bình luận. Trả về (ok, message)."""
        mv = self.parse_move(text)
        if mv is None:
            return False, f"Nước '{text}' không hợp lệ. Thử lại (vd e4, Nf3)."
        before = self.engine.analyse(self.board)
        board_before = self.board.copy()
        self.board.push(mv)
        after = self.engine.analyse(self.board)
        msg = self.commentator.comment_move(board_before, mv, before, after, mover_is_ai=False)
        return True, msg

    def ai_move(self):
        """AI tính + đi nước của mình, kèm thông báo và bình luận. Trả về (move, message)."""
        if self.game_over():
            return None, self.result_text()
        board_before = self.board.copy()
        before = self.engine.analyse(self.board)
        mv = self.engine.best_move(self.board)
        if mv is None:
            return None, self.result_text()
        announce = self.commentator.announce_ai_move(board_before, mv)
        self.board.push(mv)
        after = self.engine.analyse(self.board)
        comment = self.commentator.comment_move(board_before, mv, before, after, mover_is_ai=True)
        return mv, announce + " " + comment

    # ---------- phân tích & đối thoại ----------
    def analyse_current(self, verbosity: str | None = None) -> str:
        rep = analysis.build_report(self.board, self.engine)
        lines = analysis.narrate(rep, verbosity or self.settings.verbosity,
                                 for_color=self.human_color())
        return " ".join(lines)

    def ask(self, question: str) -> str:
        return self.assistant.answer(self.board, question)

    def close(self):
        self.engine.close()
