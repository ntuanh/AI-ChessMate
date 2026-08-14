"""BÁM NƯỚC ĐI KIỂU ĐỌC-TRỰC-TIẾP — thay đường liệt kê tổ hợp + bỏ phiếu.

Chỉ nhắm bàn Lichess 2D.

Vòng đời một frame:

    S = đọc 64 ô  →  {trống, trắng, đen}
    Δ = S khác THẾ CỜ ĐANG GIỮ ở những ô nào     (neo vào thế cờ, KHÔNG neo vào frame trước)
    Δ ổn định  →  tra hình dạng Δ ra (from, to)  →  lọc bằng luật cờ  →  chốt

Vì sao neo vào thế cờ: thế cờ chỉ đổi khi chốt được một nước HỢP LỆ, nên nó là
mốc không bao giờ trôi. Đọc nhầm một frame chỉ gây trễ, không gây lệch — khác
hẳn kiểu so frame-với-frame, sai một lần là sai vĩnh viễn.

Vì sao dùng luật cờ để LỌC chứ không để SINH: sinh thì phải nở cây nước đi (64ms,
và chết cứng khi lỡ quá 2 nước); lọc thì chỉ là một phép tra trong `legal_moves`.

Phân vai hai nguồn đọc, theo số đo trên 1000 khung Lichess thật:
  - occupancy pixel  : quyết "có quân hay không"  (0 ô bỏ sót)
  - PieceNet 3 lớp   : quyết "màu gì"             (99,96%)
Độ sáng pixel KHÔNG mang thông tin màu (đo được 50%, đúng bằng tung đồng xu) —
quân Lichess màu nào cũng gồm cả thân sáng lẫn viền tối. Đừng thử lại.
"""
from __future__ import annotations

import chess
import numpy as np

TRONG, TRANG, DEN, KHONG_CHAC = 0, 1, 2, 3

CONF_MAU = 0.75      # dưới mức này thì không dám kết luận màu
ON_DINH = 3          # số frame liên tiếp phải đọc giống nhau


def rf_of(sq, flipped=False):
    """Ô cờ → toạ độ ảnh (hàng, cột)."""
    rank, file = chess.square_rank(sq), chess.square_file(sq)
    return (rank if flipped else 7 - rank), ((7 - file) if flipped else file)


def state_of(board, flipped=False):
    """Thế cờ → mảng 8×8 {trống, trắng, đen} theo TOẠ ĐỘ ẢNH.

    Duyệt piece_map() (chỉ các ô CÓ quân) thay vì gọi piece_at 64 lần — hàm này
    nằm trong vòng nóng, gọi lại cho từng phương án nước đi.
    """
    s = np.zeros((8, 8), np.int8)
    for sq, pc in board.piece_map().items():
        r, f = rf_of(sq, flipped)
        s[r, f] = TRANG if pc.color == chess.WHITE else DEN
    return s


def square_of(r, f, flipped=False):
    """Toạ độ ảnh → ô cờ."""
    rank = r if flipped else (7 - r)
    file = (7 - f) if flipped else f
    return chess.square(file, rank)


def color_logp(pnet_p3, occ):
    """→ 8×8×2 log-xác suất {trắng, đen} cho ô CÓ QUÂN, đã chuẩn hoá lại.

    Bỏ nhánh "trống" của model và chia lại cho hai màu: occupancy đã chốt ô này
    có quân rồi (đo được 0 ô bỏ sót), nên xác suất model dành cho "trống" là kiến
    thức lạc đề, giữ lại chỉ làm loãng.
    """
    p = np.asarray(pnet_p3, np.float64)
    w = p[:, :, TRANG] + 1e-9
    b = p[:, :, DEN] + 1e-9
    t = w + b
    return np.stack([np.log(w / t), np.log(b / t)], axis=2)


def read_state(occ, pnet_state, pnet_conf):
    """Gộp hai nguồn thành 8×8 {trống, trắng, đen, KHÔNG CHẮC}.

    Mỗi nguồn chỉ được hỏi đúng câu nó giỏi: occupancy quyết có/không, PieceNet
    quyết màu. Ô có quân mà PieceNet không chắc màu → KHÔNG CHẮC, không đoán.
    """
    s = np.zeros((8, 8), np.int8)
    for r in range(8):
        for f in range(8):
            if not occ[r][f]:
                s[r, f] = TRONG
            elif pnet_state is None:
                s[r, f] = KHONG_CHAC
            elif pnet_state[r][f] == TRONG or pnet_conf[r][f] < CONF_MAU:
                s[r, f] = KHONG_CHAC          # pixel bảo có, model bảo trống/mờ
            else:
                s[r, f] = int(pnet_state[r][f])
    return s


class Stabilizer:
    """Bỏ phiếu TRÊN KẾT QUẢ ĐỌC TỪNG Ô, không phải trên giả thuyết nước đi.

    Nhờ vậy tay che góc bàn chỉ làm mất ổn định mấy ô bị che; đường cũ thì mất
    luôn cả giả thuyết và trừ phiếu về 0.
    """

    def __init__(self, k=ON_DINH):
        self.k = k
        self.buf = []

    def push(self, s):
        self.buf.append(np.asarray(s))
        if len(self.buf) > self.k:
            self.buf.pop(0)

    def stable(self):
        """8×8 bool: ô đã đọc giống nhau đủ k lần và không lần nào KHÔNG CHẮC."""
        if len(self.buf) < self.k:
            return np.zeros((8, 8), bool)
        a = self.buf[0]
        ok = np.ones((8, 8), bool)
        for b in self.buf[1:]:
            ok &= (a == b)
        for b in self.buf:
            ok &= (b != KHONG_CHAC)
        return ok

    def reset(self):
        self.buf.clear()


def _touched(board, mv, flipped):
    """Các ô ẢNH mà nước này làm đổi trạng thái, kèm giá trị MỚI.

    Chỉ 2–4 ô, nên chấm điểm một phương án là O(4) chứ không phải dựng lại cả
    bàn 8×8 rồi so 64 ô như trước.
    """
    mau = TRANG if board.turn == chess.WHITE else DEN
    out = {rf_of(mv.from_square, flipped): TRONG,
           rf_of(mv.to_square, flipped): mau}
    if board.is_castling(mv):
        back = chess.square_rank(mv.from_square)
        if chess.square_file(mv.to_square) > chess.square_file(mv.from_square):
            r_tu, r_den = chess.square(7, back), chess.square(5, back)
        else:
            r_tu, r_den = chess.square(0, back), chess.square(3, back)
        out[rf_of(r_tu, flipped)] = TRONG
        out[rf_of(r_den, flipped)] = mau
    elif board.is_en_passant(mv):
        cap = chess.square(chess.square_file(mv.to_square),
                           chess.square_rank(mv.from_square))
        out[rf_of(cap, flipped)] = TRONG
    return out


def _shapes(board, S, stable, flipped):
    """Δ giữa thế cờ và ảnh → các cặp (from_sq, to_sq) đáng xét.

    Chỉ nhìn ô ĐÃ ỔN ĐỊNH. Bốn hình dạng, tra bảng chứ không nở cây:
      từ  : ô có quân bên đi → thành trống
      đến : ô trống → có quân bên đi   (nước thường)
            hoặc ô quân đối phương → quân bên đi   (ĂN QUÂN — occupancy mù chỗ này)
    """
    cur = state_of(board, flipped)
    mau_di = TRANG if board.turn == chess.WHITE else DEN
    mau_dich = DEN if board.turn == chess.WHITE else TRANG

    tu, den = [], []
    for r in range(8):
        for f in range(8):
            if not stable[r][f] or S[r][f] == cur[r][f]:
                continue
            if cur[r][f] == mau_di and S[r][f] == TRONG:
                tu.append(square_of(r, f, flipped))
            elif S[r][f] == mau_di and cur[r][f] in (TRONG, mau_dich):
                den.append(square_of(r, f, flipped))
    return tu, den


def _cost(hyp, r, f, S, logp):
    """Giá phải trả khi giả thuyết bảo ô (r,f) mang nhãn `hyp`.

    Hai nguồn có độ tin RẤT khác nhau nên phải phạt khác nhau:
      - lệch về CÓ/KHÔNG  → occupancy nói, mà nó không bỏ sót ô nào → phạt CỨNG.
      - lệch về MÀU       → model nói, và nó sai 1-3 ô ở 7/10 ca trượt → phạt
        đúng bằng −log xác suất model gán cho màu đó. Ô model lưỡng lự
        (0,55/0,45) gần như miễn phí; ô model chắc chắn thì rất đắt.
    Nhờ vậy một nước đi hợp lệ không còn bị giết bởi một ô model đọc sai mà chính
    nó cũng không tự tin.
    """
    co_quan = S[r][f] != TRONG
    if (hyp == TRONG) != (not co_quan):
        return 4.0                     # mâu thuẫn occupancy: chặn thẳng
    if hyp == TRONG:
        return 0.0
    if logp is None:
        return 0.0 if S[r][f] == hyp else 1.0
    return -float(logp[r][f][0 if hyp == TRANG else 1])


def detect(board, S, stable, flipped=False, last_move=None, logp=None):
    """→ (danh sách nước hợp lệ khớp ảnh, số ô Δ chưa giải thích được).

    Trả đúng một nước thì bên gọi chốt. Trả nhiều nước là nhập nhằng, trả rỗng là
    ảnh mâu thuẫn luật cờ — cả hai trường hợp đều KHÔNG được đoán.

    `last_move`: nước vừa chốt, dùng cho RULE CHỐNG DỘI NGƯỢC bên dưới.
    """
    # RULE 1 — SỐ QUÂN KHÔNG BAO GIỜ TĂNG. Không nước cờ nào sinh thêm quân
    # (phong cấp cũng chỉ đổi loại). Ảnh đọc ra nhiều quân hơn thế cờ đang giữ
    # nghĩa là occupancy đang báo thừa vì loá/tô sáng — frame rác, bỏ hẳn thay vì
    # cố ghép cho ra một nước. Đây là bộ lọc rẻ nhất mà chặn được nhiều nước ma nhất.
    # Đếm cả ô KHÔNG CHẮC: occupancy đã khẳng định ô đó CÓ quân, chỉ là chưa rõ
    # màu. Bỏ nó ra khỏi phép đếm là tự tay đếm hụt rồi rule dưới báo mất quân —
    # đo được là kéo tụt từ 96,4% xuống 88,0%.
    n_anh = int(np.sum(S != TRONG))
    n_co = len(board.piece_map())
    # Nới 2 ô: occupancy BÁO THỪA trung bình ~0,5 ô mỗi khung (đo trên 300 khung:
    # 169 ô thừa, 0 ô sót). Siết đúng bằng 0 thì rule vứt luôn phần lớn frame
    # đúng — đo được là tụt từ 96,4% xuống 86,9%.
    if n_anh > n_co + 2:
        return [], 0
    # RULE 2 — MỘT NƯỚC ĂN NHIỀU NHẤT MỘT QUÂN. Ảnh thiếu từ 2 quân trở lên so
    # với thế cờ thì không nước đơn nào giải thích nổi; đó là bàn tay che hoặc
    # occupancy sập, không phải nước đi. Bỏ frame, đừng ghép bừa.
    if n_co - n_anh > 1:
        return [], 0

    tu, den = _shapes(board, S, stable, flipped)
    if not tu and not den:
        return [], 0

    hop = []
    for mv in board.legal_moves:
        if mv.from_square not in tu:
            continue
        if mv.to_square not in den:
            # nhập thành: `to` của python-chess là ô VUA về, xe đi kèm không nằm
            # trong `den` nên phải xét riêng, nếu không mọi nước O-O đều trượt.
            if not board.is_castling(mv):
                continue
        if mv.promotion not in (None, chess.QUEEN):
            continue                      # phong cấp: mặc định Hậu, ảnh không phân biệt được
        # RULE 3 — CHỐNG DỘI NGƯỢC. Nước đảo ngược đúng nước vừa chốt (đi B→A ngay
        # sau A→B) gần như luôn là đọc nhầm khung giữa lúc quân đang trượt: bàn
        # trông như quân đã về chỗ cũ. Người chơi có đi lùi thật thì cũng phải qua
        # ít nhất một nước của đối phương, nên chặn ở đây không mất nước hợp lệ nào.
        if (last_move is not None and mv.from_square == last_move.to_square
                and mv.to_square == last_move.from_square):
            continue
        hop.append(mv)

    # Chấm điểm bằng SAI LỆCH CÒN LẠI, không đòi khớp tuyệt đối. Đòi tuyệt đối
    # nghe chặt chẽ nhưng đo được là thua hẳn: occupancy dư trung bình ~0,5 ô mỗi
    # khung, nên một ô nhiễu ở góc bàn cũng đủ giết nước đi đúng (67,9% so với
    # 91,6%). Giữ mọi phương án cùng đạt điểm thấp nhất, nhập nhằng thì trả nhiều
    # và bên gọi chờ thêm frame chứ không đoán.
    cur = state_of(board, flipped)

    # CHẤM HAI TẦNG.
    #  tầng 1 (quyết định) — ĐẾM Ô LỆCH, thang cứng. Đo được 96,4%.
    #  tầng 2 (phá hoà)    — tổng −log xác suất màu, chỉ dùng khi tầng 1 hoà.
    # Từng thử cho xác suất quyết định luôn: tụt xuống 86,9%. Model quá tự tin nên
    # một ô đọc sai màu tốn −log p tới hàng chục, đắt hơn cả "không đi nước nào",
    # thành ra nước đúng bị loại. Xác suất chỉ đáng tin ở vai trò so sánh tương
    # đối giữa các phương án đã ngang điểm cứng.
    def cung(hyp, r, f):
        return int(hyp != S[r][f]) if S[r][f] != KHONG_CHAC else 0

    def mem(hyp, r, f):
        if logp is None or hyp == TRONG or S[r][f] == TRONG:
            return 0.0
        return -float(logp[r][f][0 if hyp == TRANG else 1])

    base = sum(cung(cur[r][f], r, f)
               for r in range(8) for f in range(8) if stable[r][f])
    tot, best, best_m = [], base, None
    for mv in hop:
        # Chỉ cộng/trừ ở 2–4 ô nước này chạm tới, không dựng lại cả bàn.
        d, m = base, 0.0
        for (r, f), moi in _touched(board, mv, flipped).items():
            if not stable[r][f]:
                continue
            d += cung(moi, r, f) - cung(cur[r][f], r, f)
            m += mem(moi, r, f)
        if d < best or (d == best and tot and m < best_m - 1e-9):
            best, best_m, tot = d, m, [mv]
        elif d == best and tot and abs(m - best_m) <= 1e-9:
            tot.append(mv)
    return tot, best
