"""Render một thế cờ thành ẢNH BÀN CỜ 2D — mô phỏng bàn cờ hiển thị trên MÀN HÌNH
điện thoại (kịch bản thực tế: camera soi vào app cờ). Dùng để TỰ TEST thị giác."""
from __future__ import annotations
import numpy as np
import chess

try:
    import cv2
except Exception:
    cv2 = None

# màu (BGR) kiểu bàn cờ gỗ nhạt
LIGHT = (181, 217, 240)
DARK = (99, 136, 181)
WHITE_PIECE = (245, 245, 245)
BLACK_PIECE = (35, 35, 35)
SYMBOL = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
          chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"}


def render_position(board: chess.Board, size: int = 800) -> np.ndarray:
    """Trả về ảnh BGR (size x size) của thế cờ. Hàng trên = rank 8 (nhìn từ phía Trắng)."""
    step = size // 8
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(8):           # r = hàng tính từ trên
        for f in range(8):       # f = cột tính từ trái
            y0, x0 = r * step, f * step
            light = (f + r) % 2 == 0
            img[y0:y0 + step, x0:x0 + step] = LIGHT if light else DARK
            sq = chess.square(f, 7 - r)
            pc = board.piece_at(sq)
            if pc is None:
                continue
            cx, cy = x0 + step // 2, y0 + step // 2
            pcolor = WHITE_PIECE if pc.color == chess.WHITE else BLACK_PIECE
            tcolor = BLACK_PIECE if pc.color == chess.WHITE else WHITE_PIECE
            cv2.circle(img, (cx, cy), int(step * 0.36), pcolor, -1)
            cv2.circle(img, (cx, cy), int(step * 0.36), tcolor, 2)
            sym = SYMBOL[pc.piece_type]
            fs = step / 60.0
            (tw, th), _ = cv2.getTextSize(sym, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            cv2.putText(img, sym, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, tcolor, 2, cv2.LINE_AA)
    return img


def render_grid(grid, size: int = 480) -> np.ndarray:
    """Vẽ bàn cờ từ ma trận ký hiệu 8x8 (grid[0]=rank8) — dùng hiển thị 'AI nhận diện'."""
    step = size // 8
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for r in range(8):
        for f in range(8):
            y0, x0 = r * step, f * step
            light = (f + r) % 2 == 0
            img[y0:y0 + step, x0:x0 + step] = LIGHT if light else DARK
            sym = grid[r][f]
            if sym == ".":
                continue
            white = sym.isupper()
            cx, cy = x0 + step // 2, y0 + step // 2
            pcolor = WHITE_PIECE if white else BLACK_PIECE
            tcolor = BLACK_PIECE if white else WHITE_PIECE
            cv2.circle(img, (cx, cy), int(step * 0.36), pcolor, -1)
            cv2.circle(img, (cx, cy), int(step * 0.36), tcolor, 2)
            s = sym.upper()
            fs = step / 62.0
            (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
            cv2.putText(img, s, (cx - tw // 2, cy + th // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, tcolor, 2, cv2.LINE_AA)
    return img


def embed_in_phone(board_img: np.ndarray, frame_wh=(720, 1280),
                   jitter: float = 0.0) -> np.ndarray:
    """Đặt ảnh bàn cờ vào giữa một 'màn hình điện thoại' (có lề trên/dưới như app thật),
    tùy chọn thêm phối cảnh nhẹ (jitter) để giả lập camera soi lệch."""
    W, H = frame_wh
    frame = np.full((H, W, 3), 25, dtype=np.uint8)          # nền tối như app
    bs = int(min(W, H * 0.62))                               # cạnh bàn trên màn
    board = cv2.resize(board_img, (bs, bs))
    x0 = (W - bs) // 2
    y0 = int(H * 0.22)
    frame[y0:y0 + bs, x0:x0 + bs] = board
    # thanh UI giả phía trên/dưới
    cv2.rectangle(frame, (0, 0), (W, int(H * 0.08)), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, H - int(H * 0.08)), (W, H), (40, 40, 40), -1)
    if jitter <= 0:
        return frame
    # phối cảnh nhẹ
    src = np.float32([[x0, y0], [x0 + bs, y0], [x0 + bs, y0 + bs], [x0, y0 + bs]])
    d = bs * jitter
    dst = src + np.float32([[d, d], [-d, d * 0.5], [-d * 0.5, -d], [d, -d * 0.5]])
    M = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [W, 0], [W, H], [0, H]]),
        np.float32([[0, 0], [W, 0], [W, H], [0, H]]))
    Mp = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, Mp, (W, H), borderValue=(25, 25, 25))
