"""Bộ BÌNH LUẬN CHUYÊN GIA — biến số liệu Stockfish thành lời nhận định tiếng Việt
như một huấn luyện viên cờ vua: đánh giá thế trận, chất lượng nước đi, ý tưởng, cảnh báo."""
from __future__ import annotations
import chess
from typing import Optional

from .engine import Analysis

import random

# Tên quân tiếng Việt
PIECE_VI = {
    chess.PAWN: "Tốt", chess.KNIGHT: "Mã", chess.BISHOP: "Tượng",
    chess.ROOK: "Xe", chess.QUEEN: "Hậu", chess.KING: "Vua",
}
COLOR_VI = {chess.WHITE: "Trắng", chess.BLACK: "Đen"}

_EVAL_GREAT = [
    "Cục diện đang hoàn toàn nghiêng về phía bạn, hãy mở rộng kiểm soát.",
    "Bạn đang có ưu thế rất lớn trên bàn cờ, duy trì sức ép.",
    "Thế trận cực kỳ thuận lợi, hãy tận dụng cơ hội gia tăng khoảng cách.",
]

_EVAL_GOOD = [
    "Thế cờ của bạn đang nhỉnh hơn, quân cờ đứng ở các vị trí khá đẹp.",
    "Vị thế tương đối thuận lợi cho bạn, hãy tiếp tục tích lũy ưu thế.",
    "Bạn đang có vị trí quân khá tốt trên bàn cờ.",
]

_EVAL_EVEN = [
    "Thế cờ hiện tại đang giằng co rất cân bằng.",
    "Cục diện hai bên đang khá ngang ngửa.",
    "Trận đấu đang ở thế cân bằng, cần đi quân cẩn thận.",
]

_EVAL_BAD = [
    "Thế cờ đang có chút bất lợi, chú ý phòng thủ và củng cố vị trí.",
    "Hãy cẩn trọng đường cờ trên bàn, đưa Vua về vị trí an toàn.",
    "Bàn cờ đang gặp chút áp lực, cẩn thận từng nước đi.",
]

_OPTION_INTROS_3 = [
    "Tôi gợi ý 3 lựa chọn cho bạn: Nước ưu tiên nhất là {c0}, thứ hai là {c1}, hoặc thử {c2}.",
    "Bạn có thể cân nhắc các phương án: Nước mạnh nhất {c0}, ngoài ra có {c1} hoặc {c2}.",
    "Một số phương án hay lúc này: Đầu tiên là {c0}, kế tiếp là {c1}, hoặc lựa chọn {c2}.",
    "Ý tưởng chiến thuật lúc này: Gợi ý {c0}, ngoài ra có thể chọn {c1} hoặc {c2}.",
]

_OPTION_INTROS_2 = [
    "Gợi ý cho bạn 2 nước đi tốt: Phương án chính là {c0}, hoặc bạn có thể chọn {c1}.",
    "Bạn có 2 lựa chọn sáng giá: Nước tối ưu là {c0}, phương án thay thế là {c1}.",
    "Ý tưởng tiếp theo: Nước đi mạnh nhất là {c0}, ngoài ra {c1} cũng rất tốt.",
]

_OPTION_INTROS_1 = [
    "Nước đi tối ưu nhất lúc này là {c0}.",
    "Gợi ý nước đi hàng đầu cho bạn là {c0}.",
    "Lựa chọn mạnh nhất lúc này là {c0}.",
]


def generate_short_move(board: chess.Board, analysis: Analysis) -> str:
    """Quy tắc 1: Trạng thái bình thường — CHỈ hiển thị/đọc duy nhất nước đi tốt nhất."""
    candidates = getattr(analysis, "candidates", []) or []
    if candidates and hasattr(candidates[0], "san"):
        return f"Nước đi tốt nhất: {candidates[0].san}"
    if analysis.best_move is not None:
        try:
            san = board.san(analysis.best_move)
            return f"Nước đi tốt nhất: {san}"
        except Exception:
            return f"Nước đi tốt nhất: {analysis.best_move.uci()}"
    return "Nước đi tốt nhất: —"


def generate_detailed_analysis(board: chess.Board, analysis: Analysis) -> str:
    """Quy tắc 2: Trạng thái phân tích chi tiết khi có yêu cầu.
    Cung cấp báo cáo gồm: Đánh giá thế trận + Điểm mạnh/Điểm yếu + Kế hoạch/Gợi ý chiến thuật."""
    parts = []

    # 1. ĐÁNH GIÁ THẾ TRẬN (Ưu thế thuộc về ai)
    cp = analysis.score_cp
    mate = analysis.mate_in
    to_move = board.turn

    parts.append("📊 BÁO CÁO PHÂN TÍCH CHI TIẾT:")
    if mate is not None:
        if (mate > 0 and to_move == chess.WHITE) or (mate < 0 and to_move == chess.BLACK):
            parts.append("• Đánh giá: Có đòn chiếu hết cận kề, ưu thế tuyệt đối thuộc về bạn.")
        else:
            parts.append("• Đánh giá: Đang gặp đòn chiếu hết nguy hiểm từ đối phương, ưu thế thuộc về đối thủ.")
    elif cp is not None:
        turn_cp = cp if to_move == chess.WHITE else -cp
        if turn_cp > 250:
            parts.append("• Đánh giá: Ưu thế lớn hoàn toàn thuộc về bạn, kiểm soát tốt thế trận.")
        elif turn_cp > 80:
            parts.append("• Đánh giá: Ưu thế nhẹ thuộc về bạn, vị trí các quân khá thuận lợi.")
        elif turn_cp > -80:
            parts.append("• Đánh giá: Thế cờ hiện tại rất cân bằng cho cả hai bên.")
        elif turn_cp > -250:
            parts.append("• Đánh giá: Ưu thế thuộc về đối phương, bạn cần cẩn trọng.")
        else:
            parts.append("• Đánh giá: Bất lợi lớn nghiêng về phía bạn, cần tập trung phòng thủ.")

    # 2. ĐIỂM MẠNH & ĐIỂM YẾU
    strengths = []
    weaknesses = []

    hang_user = _find_hanging(board, to_move)
    if hang_user:
        weaknesses.append(f"Cẩn thận: {hang_user}.")

    hang_opp = _find_hanging(board, not to_move)
    if hang_opp:
        strengths.append(f"Cơ hội tấn công: Đối phương đang có {hang_opp}.")

    n = board.fullmove_number
    if n <= 10:
        if board.has_castling_rights(to_move):
            strengths.append("Vẫn còn quyền nhập thành để bảo vệ Vua.")
        else:
            weaknesses.append("Chưa nhập thành, cần chú ý an toàn Vua.")

    if not strengths:
        strengths.append("Cấu trúc quân đứng vị trí tương đối ổn định.")
    if not weaknesses:
        weaknesses.append("Chưa phát hiện điểm yếu chí mạng nào.")

    parts.append("• Điểm mạnh: " + " ".join(strengths))
    parts.append("• Điểm yếu: " + " ".join(weaknesses))

    # 3. KẾ HOẠCH & GỢI Ý CHIẾN THUẬT
    candidates = getattr(analysis, "candidates", []) or []
    cand_sans = [c.san for c in candidates if hasattr(c, "san")]

    if cand_sans:
        opts_str = ", ".join(cand_sans[:3])
        parts.append(f"💡 Kế hoạch & Gợi ý chiến thuật: Gợi ý các phương án nước đi hàng đầu: {opts_str}.")
    elif analysis.best_move is not None:
        try:
            san = board.san(analysis.best_move)
            parts.append(f"💡 Kế hoạch chiến thuật: Nên triển khai nước đi {san}.")
        except Exception:
            pass

    return "\n".join(parts)


def _mate_to_cp(mate_in: Optional[int]) -> Optional[int]:
    if mate_in is None:
        return None
    # mate dương (trắng thắng) -> điểm lớn dương; càng gần càng lớn
    sign = 1 if mate_in > 0 else -1
    return sign * (100000 - abs(mate_in) * 100)


def _score_white(a: Analysis) -> Optional[int]:
    """Điểm theo góc TRẮNG, gộp cả mate. None nếu không có gì (hiếm)."""
    if a.mate_in is not None:
        return _mate_to_cp(a.mate_in)
    return a.score_cp


def _score_pov(a: Analysis, color: bool) -> Optional[int]:
    w = _score_white(a)
    if w is None:
        return None
    return w if color == chess.WHITE else -w


def describe_move(board_before: chess.Board, move: chess.Move) -> str:
    """Mô tả nước đi bằng tiếng Việt, vd 'Mã đi f3', 'Tượng ăn c4', 'nhập thành cánh vua'."""
    if board_before.is_castling(move):
        if chess.square_file(move.to_square) > 4:
            return "nhập thành cánh vua (O-O)"
        return "nhập thành cánh hậu (O-O-O)"
    piece = board_before.piece_at(move.from_square)
    name = PIECE_VI.get(piece.piece_type, "Quân") if piece else "Quân"
    to_sq = chess.square_name(move.to_square)
    verb = "ăn" if board_before.is_capture(move) else "đi"
    txt = f"{name} {verb} {to_sq}"
    if move.promotion:
        txt += f", phong {PIECE_VI[move.promotion]}"
    tmp = board_before.copy()
    tmp.push(move)
    if tmp.is_checkmate():
        txt += " — chiếu hết!"
    elif tmp.is_check():
        txt += " — chiếu!"
    return txt


def eval_phrase(a: Analysis) -> str:
    """Câu đánh giá thế trận hiện tại (ai lợi, mức nào)."""
    if a.mate_in is not None:
        m = a.mate_in
        who = COLOR_VI[chess.WHITE] if m > 0 else COLOR_VI[chess.BLACK]
        return f"{who} chiếu hết sau {abs(m)} nước"
    cp = a.score_cp
    if cp is None:
        return "thế cờ không rõ ràng"
    white = cp >= 0
    who = COLOR_VI[chess.WHITE] if white else COLOR_VI[chess.BLACK]
    v = abs(cp)
    pawns = v / 100.0
    if v < 30:
        return "thế cờ cân bằng"
    if v < 90:
        level = "hơi lợi hơn"
    elif v < 200:
        level = "lợi rõ rệt"
    elif v < 500:
        level = "ưu thế lớn"
    else:
        level = "gần như thắng"
    return f"{who} {level} (khoảng +{pawns:.1f})"


def move_quality(before: Analysis, after: Analysis, mover: bool,
                 played_best: bool) -> Optional[str]:
    """Nhận xét chất lượng nước vừa đi. So sánh eval trước (giả định đi tốt nhất) với eval thực sau khi đi."""
    b = _score_pov(before, mover)
    a = _score_pov(after, mover)
    if b is None or a is None:
        return None
    drop = b - a  # >0 = tệ hơn so với nước tốt nhất
    if played_best or drop <= 15:
        return "Nước đi chính xác!"
    if drop <= 50:
        return "Nước đi tốt."
    if drop <= 120:
        return "Hơi thiếu chính xác (?!)."
    if drop <= 280:
        return "Đây là một sai lầm (?), làm hỏng thế cờ."
    return "Một nước đi rất tệ (??), bỏ lỡ ưu thế lớn!"


def _find_hanging(board: chess.Board, color: bool) -> Optional[str]:
    """Tìm quân của `color` đang bị treo (đối phương ăn được mà lời quân). Xấp xỉ bằng SEE."""
    opp = not color
    best = None
    best_gain = 0
    for mv in board.legal_moves:
        # nước của bên tới lượt (opp) ăn quân color
        if board.turn != opp:
            break
        if board.is_capture(mv):
            try:
                if board.is_en_passant(mv):
                    continue
                gain = board.piece_at(mv.to_square)
                if gain and gain.color == color:
                    # dùng SEE của python-chess
                    see = board.see(mv) if hasattr(board, "see") else None
                    val = see if see is not None else _pval(gain.piece_type)
                    if val > best_gain:
                        best_gain = val
                        best = mv
            except Exception:
                continue
    if best is not None and best_gain >= 100:
        sq = chess.square_name(best.to_square)
        pc = board.piece_at(best.to_square)
        nm = PIECE_VI.get(pc.piece_type, "quân") if pc else "quân"
        return f"{nm} ở {sq} đang bị treo (mất khoảng {best_gain/100:.0f} điểm)"
    return None


_PVAL = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
         chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0}


def _pval(pt) -> int:
    return _PVAL.get(pt, 0)


def strategic_tips(board: chess.Board, analysis: Analysis, to_move: bool,
                   verbosity: str) -> list[str]:
    """Gợi ý kế hoạch cho bên `to_move` (thường là con người sắp đi)."""
    tips = []
    n = board.fullmove_number
    # cảnh báo quân bị treo của bên sắp đi
    hang = _find_hanging(board, to_move)
    if hang:
        tips.append("Cẩn thận: " + hang + ".")
    # gợi ý nước theo engine
    if analysis.best_move is not None:
        try:
            san = board.san(analysis.best_move)
            tips.append(f"Ý tưởng mạnh: {san}.")
        except Exception:
            pass
    if verbosity == "short":
        return tips[:1]
    # theo giai đoạn
    piece_count = len(board.piece_map())
    if n <= 10:
        if board.has_castling_rights(to_move):
            tips.append("Giai đoạn khai cuộc: nên phát triển quân nhẹ và nhập thành sớm để an toàn Vua.")
        else:
            tips.append("Khai cuộc: tiếp tục phát triển quân, giành kiểm soát trung tâm.")
    elif piece_count <= 12:
        tips.append("Tàn cuộc: hãy chủ động dùng Vua và đẩy Tốt thông.")
    else:
        tips.append("Trung cuộc: chú ý an toàn Vua và tăng tính chủ động cho các quân.")
    return tips


class Commentator:
    """Sinh lời bình cho mỗi nước đi."""

    _GENERIC_PREFIXES = ("Giai đoạn", "Khai cuộc", "Trung cuộc", "Tàn cuộc")

    def __init__(self, settings):
        self.settings = settings
        self._last_generic = None    # tránh lặp lại lời khuyên giai đoạn y hệt

    def comment_move(self, board_before: chess.Board, move: chess.Move,
                     before: Analysis, after: Analysis,
                     mover_is_ai: bool) -> str:
        mover = board_before.turn  # bên vừa đi
        played_best = (before.best_move == move)
        parts = []

        who = "Tôi" if mover_is_ai else "Bạn"
        parts.append(f"{who} vừa {describe_move(board_before, move)}.")

        # chất lượng nước (chỉ bình cho nước của con người hoặc khi verbose)
        q = move_quality(before, after, mover, played_best)
        if q and (not mover_is_ai or self.settings.verbosity == "detailed"):
            parts.append(q)

        # đánh giá thế trận sau nước đi
        parts.append("Thế trận: " + eval_phrase(after) + ".")

        # tình huống đặc biệt
        board_after = board_before.copy()
        board_after.push(move)
        if board_after.is_checkmate():
            winner = COLOR_VI[mover]
            parts.append(f"Chiếu hết! {winner} thắng ván này.")
            return " ".join(parts)
        if board_after.is_stalemate():
            parts.append("Hết nước đi — hòa do pat.")
            return " ".join(parts)
        if board_after.is_insufficient_material():
            parts.append("Hòa do không đủ lực chiếu hết.")
            return " ".join(parts)

        # gợi ý cho bên sắp đi (đối thủ của người vừa đi)
        to_move = board_after.turn
        if self.settings.verbosity != "short" or not mover_is_ai:
            for t in strategic_tips(board_after, after, to_move, self.settings.verbosity):
                # bỏ lời khuyên giai đoạn nếu trùng y hệt lần trước (đỡ nhàm)
                if t.startswith(self._GENERIC_PREFIXES):
                    if t == self._last_generic:
                        continue
                    self._last_generic = t
                parts.append(t)

        return " ".join(parts)

    def announce_ai_move(self, board_before: chess.Board, move: chess.Move) -> str:
        """Câu đọc to nước đi của AI để người chơi đặt quân trên bàn thật."""
        try:
            san = board_before.san(move)
        except Exception:
            san = move.uci()
        return f"Tôi đi: {describe_move(board_before, move)}. Ký hiệu {san}."
