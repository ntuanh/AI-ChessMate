"""PHÂN TÍCH THẾ TRẬN CHUYÊN SÂU — trái tim của "chuyên gia cờ vua".
Không chỉ đưa điểm số, mà giải thích VÌ SAO: vật chất, an toàn Vua, cấu trúc Tốt,
tính chủ động, kiểm soát trung tâm, cột mở cho Xe, quân bị treo, đòn chiến thuật...
rồi diễn đạt thành lời khuyên tiếng Việt như một huấn luyện viên."""
from __future__ import annotations
import chess
from dataclasses import dataclass, field
from typing import Optional

from .engine import Analysis, ChessEngine

PIECE_VI = {
    chess.PAWN: "Tốt", chess.KNIGHT: "Mã", chess.BISHOP: "Tượng",
    chess.ROOK: "Xe", chess.QUEEN: "Hậu", chess.KING: "Vua",
}
COLOR_VI = {chess.WHITE: "Trắng", chess.BLACK: "Đen"}
PVAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
CENTER = [chess.D4, chess.E4, chess.D5, chess.E5]
BIG_CENTER = CENTER + [chess.C3, chess.D3, chess.E3, chess.F3,
                       chess.C4, chess.F4, chess.C5, chess.F5,
                       chess.C6, chess.D6, chess.E6, chess.F6]


@dataclass
class SideReport:
    material: int = 0
    bishop_pair: bool = False
    doubled: int = 0
    isolated: int = 0
    passed: list = field(default_factory=list)      # danh sách ô Tốt thông
    open_rooks: int = 0                              # số Xe trên cột mở/nửa mở
    king_shield: int = 0                             # số Tốt che Vua
    king_open_files: int = 0                         # số cột mở cạnh Vua
    center_control: int = 0                          # số ô trung tâm kiểm soát
    developed: int = 0                               # quân nhẹ đã ra khỏi hàng gốc
    mobility: int = 0                                # số nước hợp lệ (độ chủ động)


@dataclass
class PositionReport:
    fen: str
    turn: bool
    phase: str
    white: SideReport
    black: SideReport
    eval: Analysis
    hanging: dict = field(default_factory=dict)     # color -> mô tả quân treo lời nhất
    best_line_san: list = field(default_factory=list)
    checks_available: bool = False


def _material(board, color):
    return sum(PVAL[p.piece_type] for p in board.piece_map().values()
               if p.color == color and p.piece_type != chess.KING)


def _pawn_files(board, color):
    files = [0] * 8
    for sq in board.pieces(chess.PAWN, color):
        files[chess.square_file(sq)] += 1
    return files


def _doubled_isolated(board, color):
    files = _pawn_files(board, color)
    doubled = sum(c - 1 for c in files if c > 1)
    isolated = 0
    for f in range(8):
        if files[f] == 0:
            continue
        left = files[f - 1] if f > 0 else 0
        right = files[f + 1] if f < 7 else 0
        if left == 0 and right == 0:
            isolated += files[f]
    return doubled, isolated


def _passed_pawns(board, color):
    passed = []
    opp = not color
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        blocked = False
        for of in (f - 1, f, f + 1):
            if of < 0 or of > 7:
                continue
            for osq in board.pieces(chess.PAWN, opp):
                if chess.square_file(osq) != of:
                    continue
                orank = chess.square_rank(osq)
                if color == chess.WHITE and orank > r:
                    blocked = True
                elif color == chess.BLACK and orank < r:
                    blocked = True
        if not blocked:
            passed.append(sq)
    return passed


def _bishop_pair(board, color):
    return len(board.pieces(chess.BISHOP, color)) >= 2


def _king_safety(board, color):
    ksq = board.king(color)
    if ksq is None:
        return 0, 0
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    shield = 0
    for df in (-1, 0, 1):
        f = kf + df
        if f < 0 or f > 7:
            continue
        # Tốt ngay trước Vua (theo hướng tiến của màu)
        for dr in (1, 2):
            r = kr + dr if color == chess.WHITE else kr - dr
            if 0 <= r <= 7:
                sq = chess.square(f, r)
                p = board.piece_at(sq)
                if p and p.piece_type == chess.PAWN and p.color == color:
                    shield += 1
                    break
    # cột mở/nửa mở cạnh Vua (nguy hiểm)
    open_files = 0
    my_pawn_files = _pawn_files(board, color)
    for df in (-1, 0, 1):
        f = kf + df
        if 0 <= f <= 7 and my_pawn_files[f] == 0:
            open_files += 1
    return shield, open_files


def _center_control(board, color):
    return sum(1 for sq in CENTER if board.is_attacked_by(color, sq))


def _developed(board, color):
    back = 0 if color == chess.WHITE else 7
    cnt = 0
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            if chess.square_rank(sq) != back:
                cnt += 1
    return cnt


def _open_rooks(board, color):
    my = _pawn_files(board, color)
    opp = _pawn_files(board, not color)
    cnt = 0
    for sq in board.pieces(chess.ROOK, color):
        f = chess.square_file(sq)
        if my[f] == 0:  # cột mở hoặc nửa mở
            cnt += 1
    return cnt


def _mobility(board, color):
    if board.turn == color:
        return board.legal_moves.count()
    b = board.copy()
    b.turn = color
    try:
        return b.legal_moves.count()
    except Exception:
        return 0


def _hanging(board, color) -> Optional[str]:
    """Quân của `color` bị đối phương ăn lời nhất (theo SEE)."""
    opp = not color
    b = board.copy()
    if b.turn != opp:
        b.turn = opp
    best, best_gain = None, 0
    try:
        moves = list(b.legal_moves)
    except Exception:
        return None
    for mv in moves:
        if not b.is_capture(mv) or b.is_en_passant(mv):
            continue
        tgt = b.piece_at(mv.to_square)
        if not tgt or tgt.color != color:
            continue
        try:
            see = b.see(mv)
        except Exception:
            see = PVAL[tgt.piece_type] * 100
        if see > best_gain:
            best_gain, best = see, mv
    if best is not None and best_gain >= 100:
        pc = board.piece_at(best.to_square)
        nm = PIECE_VI.get(pc.piece_type, "quân") if pc else "quân"
        return f"{nm} ở {chess.square_name(best.to_square)} (thiệt ~{best_gain/100:.0f})"
    return None


def _phase(board) -> str:
    n = len(board.piece_map())
    if board.fullmove_number <= 10 and n >= 26:
        return "khai cuộc"
    if n <= 12:
        return "tàn cuộc"
    return "trung cuộc"


def side_report(board, color) -> SideReport:
    doubled, isolated = _doubled_isolated(board, color)
    shield, open_f = _king_safety(board, color)
    return SideReport(
        material=_material(board, color),
        bishop_pair=_bishop_pair(board, color),
        doubled=doubled, isolated=isolated,
        passed=_passed_pawns(board, color),
        open_rooks=_open_rooks(board, color),
        king_shield=shield, king_open_files=open_f,
        center_control=_center_control(board, color),
        developed=_developed(board, color),
        mobility=_mobility(board, color),
    )


def build_report(board: chess.Board, engine: ChessEngine) -> PositionReport:
    ana = engine.analyse(board)
    # dịch biến chính sang SAN
    line = []
    tmp = board.copy()
    for mv in ana.pv[:6]:
        try:
            line.append(tmp.san(mv)); tmp.push(mv)
        except Exception:
            break
    checks = any(board.gives_check(m) for m in board.legal_moves)
    return PositionReport(
        fen=board.fen(), turn=board.turn, phase=_phase(board),
        white=side_report(board, chess.WHITE),
        black=side_report(board, chess.BLACK),
        eval=ana,
        hanging={chess.WHITE: _hanging(board, chess.WHITE),
                 chess.BLACK: _hanging(board, chess.BLACK)},
        best_line_san=line, checks_available=checks,
    )


# ---------- DIỄN ĐẠT THÀNH LỜI CHUYÊN GIA ----------

def _eval_sentence(ana: Analysis) -> str:
    if ana.mate_in is not None:
        who = COLOR_VI[chess.WHITE] if ana.mate_in > 0 else COLOR_VI[chess.BLACK]
        return f"{who} có đòn chiếu hết sau {abs(ana.mate_in)} nước."
    cp = ana.score_cp or 0
    who = COLOR_VI[chess.WHITE] if cp >= 0 else COLOR_VI[chess.BLACK]
    v = abs(cp)
    if v < 30:
        return "Thế cờ cân bằng, chưa bên nào chiếm ưu thế rõ."
    lvl = ("hơi nhỉnh hơn" if v < 90 else "lợi thế rõ" if v < 200
           else "ưu thế lớn" if v < 500 else "gần như thắng")
    return f"{who} {lvl}, khoảng +{v/100:.1f} điểm."


def narrate(report: PositionReport, verbosity: str = "normal",
            for_color: Optional[bool] = None) -> list[str]:
    """Sinh danh sách câu phân tích/lời khuyên. `for_color` = màu người chơi để hướng lời khuyên."""
    out = [_eval_sentence(report.eval)]
    w, b = report.white, report.black

    # 1) VẬT CHẤT
    diff = w.material - b.material
    if abs(diff) >= 1:
        lead = COLOR_VI[chess.WHITE] if diff > 0 else COLOR_VI[chess.BLACK]
        out.append(f"Vật chất: {lead} hơn {abs(diff)} điểm quân.")

    # 2) QUÂN BỊ TREO (chiến thuật quan trọng nhất)
    for color in (chess.WHITE, chess.BLACK):
        h = report.hanging.get(color)
        if h:
            out.append(f"Cảnh báo: {COLOR_VI[color]} đang treo {h}.")

    # 3) AN TOÀN VUA
    for color, sr in ((chess.WHITE, w), (chess.BLACK, b)):
        if sr.king_open_files >= 2 or (sr.king_shield == 0 and report.phase != "tàn cuộc"):
            out.append(f"Vua {COLOR_VI[color]} hơi lộ (ít Tốt che / có cột mở) — cần lưu ý phòng thủ.")

    if verbosity != "short":
        # 4) TRUNG TÂM & PHÁT TRIỂN (khai cuộc)
        if report.phase == "khai cuộc":
            if w.center_control != b.center_control:
                lead = COLOR_VI[chess.WHITE] if w.center_control > b.center_control else COLOR_VI[chess.BLACK]
                out.append(f"{lead} kiểm soát trung tâm tốt hơn.")
            if abs(w.developed - b.developed) >= 2:
                lead = COLOR_VI[chess.WHITE] if w.developed > b.developed else COLOR_VI[chess.BLACK]
                out.append(f"{lead} phát triển quân nhanh hơn — lợi thế nhịp độ.")

        # 5) CẤU TRÚC TỐT
        for color, sr in ((chess.WHITE, w), (chess.BLACK, b)):
            probs = []
            if sr.doubled:
                probs.append(f"{sr.doubled} Tốt chồng")
            if sr.isolated:
                probs.append(f"{sr.isolated} Tốt cô lập")
            if probs:
                out.append(f"Cấu trúc Tốt {COLOR_VI[color]} có điểm yếu: {', '.join(probs)}.")
            if sr.passed:
                sqs = ", ".join(chess.square_name(s) for s in sr.passed)
                out.append(f"{COLOR_VI[color]} có Tốt thông ở {sqs} — quân bài mạnh ở tàn cuộc.")

        # 6) XE TRÊN CỘT MỞ, CẶP TƯỢNG
        for color, sr in ((chess.WHITE, w), (chess.BLACK, b)):
            if sr.open_rooks:
                out.append(f"Xe {COLOR_VI[color]} đã chiếm cột mở — áp lực tốt.")
        if w.bishop_pair != b.bishop_pair:
            lead = COLOR_VI[chess.WHITE] if w.bishop_pair else COLOR_VI[chess.BLACK]
            out.append(f"{lead} giữ cặp Tượng — lợi thế dài hơi.")

    # 7) Ý TƯỞNG THEO ENGINE (biến chính)
    if report.best_line_san:
        line = " ".join(report.best_line_san[:4])
        out.append(f"Phương án gợi ý: {line}.")

    # 8) LỜI KHUYÊN THEO GIAI ĐOẠN cho người chơi
    if for_color is not None and verbosity == "detailed":
        sr = w if for_color == chess.WHITE else b
        if report.phase == "khai cuộc" and report.turn == for_color:
            out.append("Lời khuyên: ưu tiên phát triển quân nhẹ, chiếm trung tâm và nhập thành sớm.")
        elif report.phase == "tàn cuộc":
            out.append("Lời khuyên: kích hoạt Vua và đẩy Tốt thông để tạo Hậu.")

    if verbosity == "short":
        return out[:2]
    return out
