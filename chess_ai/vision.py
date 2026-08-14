"""THỊ GIÁC — nhìn bàn cờ thật qua camera C270 và suy ra NƯỚC ĐI.

CHIẾN LƯỢC (quan trọng): KHÔNG cần nhận diện TỪNG LOẠI quân. Vì ta theo dõi ván cờ
từ đầu bằng python-chess, chỉ cần biết Ô NÀO CÓ QUÂN / Ô NÀO TRỐNG (occupancy).
So sánh occupancy trước và sau khi đối thủ đi, rồi khớp với danh sách nước hợp lệ:
  - ô 'from' chuyển từ CÓ -> TRỐNG
  - ô 'to'   chuyển sang CÓ (hoặc vẫn CÓ nếu là nước ăn quân)
Cách này cực bền, chạy được cả với bộ quân bất kỳ.

CẦN HIỆU CHỈNH VỚI BÀN THẬT (làm cùng bạn buổi sáng):
  1. Chọn 4 góc bàn cờ (calibrate) -> lưu vào file.
  2. Chụp 'bàn trống lúc đầu ván' làm tham chiếu (đúng ra là thế cờ đầu đã có quân,
     nên ta dùng ngưỡng foreground theo từng ô).
  3. Tinh chỉnh ngưỡng occupancy theo ánh sáng/màu quân.
"""
from __future__ import annotations
import json
import os
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:  # cho phép import module khi máy không có cv2
    cv2 = None

import chess
from . import config

WARP = 800  # kích thước ảnh bàn cờ sau khi nắn phẳng (px)
CALIB_FILE = os.path.join(os.path.dirname(__file__), "..", "captures", "board_calib.json")


# ---------------- Camera ----------------
def open_camera(index: int | None = None):
    idx = config.CAM_INDEX if index is None else index
    cap = cv2.VideoCapture(idx)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def grab(cap, warmup=5):
    for _ in range(warmup):
        cap.read()
    ok, frame = cap.read()
    return frame if ok else None


# ---------------- Hiệu chỉnh 4 góc ----------------
def save_calibration(corners):
    """corners: 4 điểm (x,y) theo thứ tự [trên-trái, trên-phải, dưới-phải, dưới-trái]."""
    os.makedirs(os.path.dirname(CALIB_FILE), exist_ok=True)
    with open(CALIB_FILE, "w") as f:
        json.dump({"corners": [[float(x), float(y)] for x, y in corners]}, f)


def load_calibration():
    if not os.path.exists(CALIB_FILE):
        return None
    with open(CALIB_FILE) as f:
        return np.array(json.load(f)["corners"], dtype=np.float32)


def auto_detect_corners(frame) -> Optional[np.ndarray]:
    """Thử tự dò 4 góc bàn cờ = tứ giác lớn nhất. Nếu không chắc, hãy calibrate tay."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.2 * frame.shape[0] * frame.shape[1]:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) == 4 and area > best_area:
            best, best_area = approx.reshape(4, 2), area
    return _order_corners(best) if best is not None else None


def _order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).reshape(-1)
    return np.array([
        pts[np.argmin(s)],   # trên-trái
        pts[np.argmin(d)],   # trên-phải
        pts[np.argmax(s)],   # dưới-phải
        pts[np.argmax(d)],   # dưới-trái
    ], dtype=np.float32)


def warp_board(frame, corners):
    dst = np.array([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), dst)
    return cv2.warpPerspective(frame, M, (WARP, WARP))


# ---------------- Occupancy ----------------
def square_cells(warped):
    """Chia ảnh bàn phẳng thành 8x8 ô, trả về ma trận [rank][file] ảnh con."""
    step = WARP // 8
    cells = []
    for r in range(8):
        row = []
        for f in range(8):
            y, x = r * step, f * step
            row.append(warped[y:y + step, x:x + step])
        cells.append(row)
    return cells


def occupancy_grid(warped, thresh=25.0):
    """Trả về ma trận bool 8x8: ô có quân hay không. Lấy VÙNG GIỮA mỗi ô (tránh viền)
    rồi đo độ biến thiên cạnh (Laplacian): ô có quân nhiều cạnh hơn ô trống.
    Ngưỡng `thresh` cần tinh chỉnh theo ánh sáng/app thật."""
    cells = square_cells(warped)
    grid = np.zeros((8, 8), dtype=bool)
    for r in range(8):
        for f in range(8):
            cell = cells[r][f]
            h, w = cell.shape[:2]
            m = int(min(h, w) * 0.2)                      # bỏ 20% viền mỗi phía
            inner = cell[m:h - m, m:w - m]
            if inner.size == 0:
                continue
            g = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
            edges = cv2.Laplacian(g, cv2.CV_64F).var()
            grid[r][f] = edges > thresh
    return grid


def _sq_index(rank_from_top, file_left) -> int:
    """Đổi (hàng tính từ trên, cột tính từ trái của ảnh) sang ô chess.
    Giả định camera nhìn từ phía Trắng: hàng trên ảnh = rank 8. Cần chỉnh theo hướng đặt thật."""
    rank = 7 - rank_from_top
    file = file_left
    return chess.square(file, rank)


def _expected_change(board: chess.Board, mv: chess.Move):
    """Tập ô KỲ VỌNG chuyển TRỐNG (emptied) và CHUYỂN CÓ QUÂN (filled) khi đi `mv`,
    xét theo occupancy (không quan tâm loại quân)."""
    emptied = {mv.from_square}
    filled = set()
    to = mv.to_square
    if board.is_castling(mv):
        rank = chess.square_rank(mv.from_square)
        if chess.square_file(to) > 4:            # nhập thành cánh vua
            rook_from, rook_to = chess.square(7, rank), chess.square(5, rank)
        else:                                    # cánh hậu
            rook_from, rook_to = chess.square(0, rank), chess.square(3, rank)
        emptied.add(rook_from)
        filled.update({to, rook_to})
    elif board.is_en_passant(mv):
        filled.add(to)
        emptied.add(chess.square(chess.square_file(to), chess.square_rank(mv.from_square)))
    elif board.is_capture(mv):
        pass                                     # ô 'to' vẫn có quân -> occupancy không đổi
    else:
        filled.add(to)
    return emptied, filled


def diff_to_move(board: chess.Board, prev: np.ndarray, curr: np.ndarray) -> Optional[chess.Move]:
    """Suy ra nước đi hợp lệ khớp NHẤT với thay đổi occupancy giữa 2 khung hình."""
    obs_empty, obs_fill = set(), set()
    for r in range(8):
        for f in range(8):
            if prev[r][f] and not curr[r][f]:
                obs_empty.add(_sq_index(r, f))
            elif not prev[r][f] and curr[r][f]:
                obs_fill.add(_sq_index(r, f))

    exact, subset = [], []
    for mv in board.legal_moves:
        e, fl = _expected_change(board, mv)
        if e == obs_empty and fl == obs_fill:
            exact.append(mv)                     # khớp hoàn hảo
        elif e <= obs_empty and fl <= obs_fill:
            subset.append(mv)                    # khớp một phần (dự phòng khi nhiễu)

    def _pick(cands):
        # Phong cấp: occupancy KHÔNG phân biệt được phong con gì (e8=Q/R/B/N cùng
        # dấu hiệu) -> mặc định chọn HẬU.
        for mv in cands:
            if mv.promotion in (None, chess.QUEEN):
                return mv
        return cands[0] if cands else None

    return _pick(exact) or _pick(subset)


# ---------------- Vòng chụp một nước ----------------
class BoardWatcher:
    """Theo dõi bàn cờ: giữ occupancy hiện tại, mỗi lần gọi read_move() để lấy nước mới."""

    def __init__(self, cap, corners):
        self.cap = cap
        self.corners = corners
        self.prev_grid = None

    def snapshot(self):
        frame = grab(self.cap)
        if frame is None:
            return None
        warped = warp_board(frame, self.corners)
        return occupancy_grid(warped)

    def sync(self):
        self.prev_grid = self.snapshot()

    def read_move(self, board: chess.Board) -> Optional[chess.Move]:
        curr = self.snapshot()
        if curr is None or self.prev_grid is None:
            self.prev_grid = curr
            return None
        mv = diff_to_move(board, self.prev_grid, curr)
        if mv is not None:
            self.prev_grid = curr
        return mv
