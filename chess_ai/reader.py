"""Đọc THẾ CỜ TUYỆT ĐỐI từ frame camera: rectify → PieceNet → FEN, lọc bằng luật cờ.

Khác hẳn đường occupancy cũ. Occupancy chỉ biết ô có quân hay trống, nên phải suy
nước đi từ mặt nạ chênh lệch — và cách đó KHÔNG phân biệt được nước ăn quân: ô đích
vốn đã có quân nên mặt nạ trước/sau giống hệt nhau. Ví dụ tốt e4 ăn d5 hay ăn f5 cho
cùng một mặt nạ, chọn sai là lệch cả ván mà không có cách nào tự phát hiện.

Ở đây mỗi frame cho một thế cờ đầy đủ (biết quân gì, màu gì, từng ô), nên:
  * không tích luỹ sai — sai một frame thì frame sau đọc lại là đúng;
  * ăn quân, nhập thành, phong cấp (kể cả phong Mã/Xe/Tượng) đều đọc được;
  * mỗi ô có confidence riêng → tay che bàn thì biết là đang không tin được,
    thay vì đoán bừa.

Luật cờ vẫn được dùng, nhưng làm LỚP LỌC chứ không phải nguồn thông tin: chỉ nhận
thế mới khi nó cách thế hiện tại đúng một (hoặc hai) nước hợp lệ, và phải lặp lại
đủ số phiếu để loại nhiễu thoáng qua.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from . import piece_net, rectify

CONF_MIN = 0.60          # dưới mức này coi như ô không đọc được
MAX_UNSURE = 2           # quá số ô không đọc được thì bỏ cả frame (tay đang che)
VOTES = 3                # số frame liên tiếp phải cho cùng kết quả (~0.4s ở 7fps)
RESYNC_AFTER = 12         # số frame ổn định mà không giải thích được → thử đồng bộ lại


def placement(board, flipped=False):
    """Thế cờ dưới dạng 8×8 ký hiệu, theo đúng thứ tự ô của ảnh đã nắn.

    flipped=False: a8 ở góc trên-trái (Trắng ở dưới ảnh).
    flipped=True : h1 ở góc trên-trái (Đen ở dưới ảnh).
    """
    out = []
    for r in range(8):
        row = []
        for f in range(8):
            rank = r if flipped else 7 - r
            file = 7 - f if flipped else f
            pc = board.piece_at(chess.square(file, rank))
            row.append(pc.symbol() if pc else ".")
        out.append(row)
    return out


def _same(a, b):
    return all(a[r][f] == b[r][f] for r in range(8) for f in range(8))


def matches(board, grid, flipped=False):
    """Thế cờ `board` có khớp đúng lưới ký hiệu `grid` đọc từ ảnh hay không."""
    return _same(placement(board, flipped), grid)


def pick_by_identity(board, cands, grid, flipped=False):
    """Chọn 1 phương án trong `cands` bằng DANH TÍNH QUÂN mà PieceNet đọc được.

    Đây là chỗ duy nhất occupancy thật sự bó tay: nhiều nước khác nhau cho cùng
    một mặt nạ ô-có-quân (mọi nước ăn quân đều vậy — ô đích vốn đã có quân), nên
    xét theo mặt nạ thì chúng bằng điểm nhau. So thêm quân gì ở ô nào là tách được.

    Trả None nếu vẫn không phân giải được (0 hoặc >1 phương án khớp) — thà chờ
    thêm bằng chứng còn hơn đoán.
    """
    keep = []
    for seq in cands:
        b = board.copy()
        for mv in seq:
            b.push(mv)
        if matches(b, grid, flipped):
            keep.append(seq)
    return keep[0] if len(keep) == 1 else None


def board_from_grid(grid, flipped=False):
    """Dựng chess.Board TRỰC TIẾP từ lưới ký hiệu đọc được.

    Đây là thứ occupancy không làm được và là lý do PieceNet đáng giá: không cần
    đoán "chắc đang ở thế khai cuộc", cứ đọc bàn đang có là ra thế cờ — vào giữa
    ván cũng canh bàn được.

    Hai thông tin không nằm trong ảnh phải suy ra:
      * quyền nhập thành: chỉ cho khi Vua và Xe còn đúng ô gốc;
      * bên đi: ảnh tĩnh không nói được, nên thử Trắng trước, nếu thế cờ vô lý
        (bên KHÔNG đi lại đang bị chiếu) thì đổi sang Đen.
    Trả None nếu thế đọc được không hợp lệ (thiếu Vua, thừa quân...).
    """
    ref = chess.Board(None)                       # bàn trống
    for r in range(8):
        for f in range(8):
            s = grid[r][f]
            if s == ".":
                continue
            rank = r if flipped else 7 - r         # phép nghịch của placement()
            file = 7 - f if flipped else f
            try:
                ref.set_piece_at(chess.square(file, rank), chess.Piece.from_symbol(s))
            except ValueError:
                return None
    fen_pl = ref.board_fen()

    def has(sym, sq):
        pc = ref.piece_at(sq)
        return pc is not None and pc.symbol() == sym

    rights = ""
    if has("K", chess.E1):
        rights += "K" if has("R", chess.H1) else ""
        rights += "Q" if has("R", chess.A1) else ""
    if has("k", chess.E8):
        rights += "k" if has("r", chess.H8) else ""
        rights += "q" if has("r", chess.A8) else ""

    for turn in ("w", "b"):
        try:
            b = chess.Board(f"{fen_pl} {turn} {rights or '-'} - 0 1")
        except ValueError:
            continue
        if b.is_valid():
            return b
    return None


def _pl_diff(board, grid, flipped):
    p = placement(board, flipped)
    return sum(1 for r in range(8) for f in range(8) if p[r][f] != grid[r][f])


def locate_from_start(grid, flipped=False, start=None, max_plies=3):
    """Thế cờ ĐẦY ĐỦ khớp `grid`, dò từ thế khai cuộc. None nếu không tìm thấy.

    Vì sao cần: ảnh tĩnh không nói được AI ĐANG ĐI. board_from_grid() phải đoán và
    nó thử Trắng trước — nhưng thế sau 1.e4 với "Trắng đi" vẫn là thế hợp lệ, nên
    nó trả về sai lượt. Hậu quả không phải chậm mà là ĐỨNG IM: explain() và
    explain_occ() chỉ liệt kê nước của bên đang đi, nên nước của bên kia không bao
    giờ khớp được. Đúng cảnh bạn cầm Đen mà Trắng đã đi trước.

    Dò từ thế khai cuộc thì lượt đi, quyền nhập thành VÀ ô bắt tốt qua đường đều là
    dữ kiện chắc chắn chứ không phải suy đoán — cả ba thứ board_from_grid đều phải
    đoán.

    Cắt nhánh: một nước đổi nhiều nhất 4 ô (nhập thành đổi 4, thường 2), nên thế ở
    độ sâu d còn lệch quá 4*(max_plies-d) ô thì không thể về đích — bỏ luôn. Nhờ đó
    không phải nở hết cây nước đi.
    """
    root = start.copy() if start is not None else chess.Board()
    if _pl_diff(root, grid, flipped) == 0:
        return root

    level = [root]
    for depth in range(1, max_plies + 1):
        budget = 4 * (max_plies - depth)
        nxt, hit = [], []
        for cur in level:
            for mv in cur.legal_moves:
                nb = cur.copy()
                nb.push(mv)
                d = _pl_diff(nb, grid, flipped)
                if d == 0:
                    hit.append(nb)
                elif d <= budget:
                    nxt.append(nb)
        if hit:
            # Nhiều đường đi tới cùng một thế là chuyện thường (chỉ đổi thứ tự
            # nước), nên chỉ nhận khi chúng NHẤT TRÍ về những gì ảnh hưởng tới thế
            # cờ; không nhất trí thì trả None để bên gọi rơi về đường đoán, hơn là
            # chốt một lượt đi có thể sai.
            #
            # Ô bắt tốt qua đường chỉ tính khi thật sự bắt được: python-chess đặt
            # ep_square cho MỌI nước tốt đi hai bước, nên 1.e4 c5 2.Nf3 và
            # 1.Nf3 c5 2.e4 cho cùng thế nhưng khác ep_square — khác biệt không có
            # hậu quả gì. Xét ep thô là tự loại chính mình ở mốc 3 nước.
            def sig(b):
                return (b.turn, b.castling_rights,
                        b.ep_square if b.has_legal_en_passant() else None)
            if len({sig(b) for b in hit}) != 1:
                return None
            # Cùng thế, cùng lượt, khác thứ tự nước: biên bản có thể ghi thứ tự
            # khác thực tế, nhưng THẾ CỜ — thứ duy nhất dùng để bám nước về sau —
            # là đúng.
            return hit[0]
        level = nxt
        if not level:
            break
    return None


def explain(board, target, flipped, max_plies=2):
    """Tìm chuỗi ≤ max_plies nước hợp lệ dẫn từ `board` tới đúng thế `target`.

    Trả (danh sách nước, ) hoặc None. Nhập nhằng — nhiều chuỗi khác nhau cùng khớp
    — cũng trả None: thà chờ thêm bằng chứng còn hơn đoán rồi lệch cả ván. Với thế
    cờ đầy đủ thì nhập nhằng gần như không xảy ra, khác hẳn occupancy.
    """
    if _same(placement(board, flipped), target):
        return []
    found = []
    for mv1 in board.legal_moves:
        board.push(mv1)
        if _same(placement(board, flipped), target):
            found.append([mv1])
        elif max_plies >= 2:
            for mv2 in board.legal_moves:
                board.push(mv2)
                if _same(placement(board, flipped), target):
                    found.append([mv1, mv2])
                board.pop()
        board.pop()
        if len(found) > 1:
            return None
    return found[0] if len(found) == 1 else None


def occ_of(board, flipped=False):
    """Mặt nạ 8×8 ô-có-quân của một thế cờ, theo thứ tự ô của ảnh đã nắn."""
    g = placement(board, flipped)
    return np.array([[g[r][f] != "." for f in range(8)] for r in range(8)], bool)


def explain_occ(board, occ, flipped=False, max_plies=2):
    """Các nước hợp lệ khớp NHẤT với mặt nạ occupancy đọc được.

    Trả (danh sách phương án, residual) hoặc None. Danh sách CÓ THỂ nhiều phần tử:
    mọi nước ăn quân đều cho cùng một mặt nạ (ô đích vốn đã có quân), nên xét theo
    occupancy thì chúng bằng điểm nhau và không được phép tự chọn một cái. Bên gọi
    dùng pick_by_identity() để phân giải, hoặc chờ thêm frame.
    """
    base = int((occ_of(board, flipped) != occ).sum())
    if base == 0:
        return None

    def collect(depth):
        best_d, out = base, []
        level = [([], board.copy())]
        for _ in range(depth):
            nxt = []
            for seq, cur in level:
                for mv in cur.legal_moves:
                    nb = cur.copy()
                    nb.push(mv)
                    nxt.append((seq + [mv], nb))
            level = nxt
        for seq, cur in level:
            d = int((occ_of(cur, flipped) != occ).sum())
            if d < best_d:
                best_d, out = d, [seq]
            elif d == best_d and out:
                out.append(seq)
        return out, best_d

    for depth in range(1, max_plies + 1):
        if depth > 1 and base < 2:
            break
        cands, d = collect(depth)
        if cands and d <= 1:
            return cands, d
    return None


@dataclass
class Reading:
    """Kết quả đọc một frame."""
    ok: bool = False                 # rectify tin được và đủ ô rõ ràng
    moves: list = field(default_factory=list)   # nước vừa được CHỐT (đã đủ phiếu)
    grid: list | None = None         # 8×8 ký hiệu vừa đọc
    unsure: int = 0                  # số ô confidence thấp
    mode: str = ""                   # trạng thái bám bàn
    score: float = 0.0
    desync: bool = False             # đọc rõ nhưng không giải thích được bằng luật


class BoardReader:
    """Trạng thái đọc bàn qua nhiều frame."""

    def __init__(self, net=None, board=None, votes=VOTES):
        self.net = net or piece_net.load()
        if self.net is None:
            raise RuntimeError("thiếu piece_net.onnx hoặc onnxruntime")
        self.board = board.copy() if board is not None else chess.Board()
        self.flipped = False
        self.tracker = None
        self.votes = votes
        self._pending, self._count, self._stuck = None, 0, 0

    # ---------- canh bàn lần đầu ----------
    def calibrate(self, frame_bgr):
        """Dò bàn + xác định hướng. Trả True khi sẵn sàng đọc."""
        res = rectify.rectify(frame_bgr)
        if not res.ok:
            return False
        # rectify chỉ bảo đảm a8 sáng → còn nhập nhằng 180°; màu quân do CNN đọc
        # mới quyết được bên nào ở dưới.
        wb = self.net.white_is_bottom(res.board)
        if wb is None:
            return False
        quad = res.quad if wb else rectify.rot_quad(res.quad, 2)
        self.flipped = not wb
        self.tracker = rectify.Tracker(quad, res.score)
        return True

    # ---------- mỗi frame ----------
    def step(self, frame_bgr):
        if self.tracker is None:
            return Reading(mode="CHƯA CANH BÀN")
        quad, score, mode = self.tracker.update(frame_bgr)
        if mode == "MẤT BÀN":
            self._pending, self._count = None, 0
            return Reading(mode=mode, score=score)

        board_img = rectify.rectify(frame_bgr, quad).board
        grid, conf = self.net.predict(board_img)
        unsure = int((np.asarray(conf) < CONF_MIN).sum())
        out = Reading(ok=True, grid=grid, unsure=unsure, mode=mode, score=score)
        if unsure > MAX_UNSURE:            # tay che / mờ: không kết luận gì
            self._pending, self._count = None, 0
            return out

        seq = explain(self.board, grid, self.flipped)
        if seq is None:                    # đọc rõ mà luật không giải thích được
            self._stuck += 1
            out.desync = self._stuck >= RESYNC_AFTER
            return out
        self._stuck = 0
        if not seq:                        # trùng thế hiện tại
            self._pending, self._count = None, 0
            return out

        key = tuple(m.uci() for m in seq)
        if key == self._pending:
            self._count += 1
        else:
            self._pending, self._count = key, 1
        if self._count >= self.votes:
            for mv in seq:
                self.board.push(mv)
            out.moves = seq
            self._pending, self._count = None, 0
        return out

    def fen(self):
        return self.board.fen()
