"""NHẬN DIỆN QUÂN CỜ bằng template-matching TỰ HIỆU CHỈNH.
Ý tưởng: bàn Lichess sạch & cố định. Ở thế ĐẦU VÁN ta BIẾT CHẮC ô nào quân gì →
lấy mỗi ô làm MẪU cho loại quân đó (phân theo ô sáng/tối). Frame sau: so khớp mỗi ô
với các mẫu (cùng màu ô) bằng tương quan chuẩn hoá → ra loại quân → dựng FEN.

Ưu điểm: nhận đúng TỪNG LOẠI quân (không chỉ occupancy), bền với quân số hoá."""
from __future__ import annotations
import cv2
import numpy as np
import chess

CELL = 48  # kích thước chuẩn hoá mỗi ô để so khớp


class Tracker:
    """BÁM BÀN ĐỘNG: từ 1 khung tham chiếu (đã canh chuẩn), mỗi frame tính bàn đã
    trôi đi đâu bằng khớp đặc trưng ORB -> homography -> 4 góc tự đuổi theo.
    Thích ứng khi camera/màn hình rung, trôi."""

    def __init__(self, ref_gray, ref_corners):
        self.orb = cv2.ORB_create(1500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(ref_gray, None)
        self.ref_corners = np.array(ref_corners, dtype=np.float32).reshape(-1, 1, 2)

    def track(self, cur_gray, fallback):
        if self.ref_des is None:
            return fallback
        kp, des = self.orb.detectAndCompute(cur_gray, None)
        if des is None or len(kp) < 12:
            return fallback
        try:
            matches = self.bf.match(self.ref_des, des)
        except Exception:
            return fallback
        if len(matches) < 15:
            return fallback
        matches = sorted(matches, key=lambda m: m.distance)[:150]
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return fallback
        cur = cv2.perspectiveTransform(self.ref_corners, H)
        c = cur.reshape(4, 2).astype(np.float32)
        # kiểm tra hợp lý: 4 góc không được méo quá / ra ngoài khung
        h, w = cur_gray.shape[:2]
        if (c[:, 0].min() < -40 or c[:, 0].max() > w + 40 or
                c[:, 1].min() < -40 or c[:, 1].max() > h + 40):
            return fallback
        return c


def preprocess(bgr):
    """Làm NÉT (unsharp mask) + tăng tương phản → chống mờ, tăng chi tiết quân cờ."""
    blur = cv2.GaussianBlur(bgr, (0, 0), 3)
    sharp = cv2.addWeighted(bgr, 1.7, blur, -0.7, 0)
    # tăng tương phản kênh sáng (CLAHE)
    lab = cv2.cvtColor(sharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def cell_gray(warped, r, f):
    W = warped.shape[0]
    step = W // 8
    c = warped[r * step:(r + 1) * step, f * step:(f + 1) * step]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (CELL, CELL))


def is_light(r, f):
    return (r + f) % 2 == 0


def build_templates(warped_start, start_board=None):
    """Từ ảnh warp thế ĐẦU VÁN → danh sách mẫu (symbol, is_light, cell)."""
    b = start_board or chess.Board()
    tmpl = []
    for r in range(8):
        for f in range(8):
            sq = chess.square(f, 7 - r)     # r=0 (trên) = rank 8
            pc = b.piece_at(sq)
            sym = pc.symbol() if pc else "."
            tmpl.append((sym, is_light(r, f), cell_gray(warped_start, r, f)))
    return tmpl


def classify_grid(warped, tmpl):
    """Trả về ma trận 8x8 ký hiệu quân ('.'=trống, 'P'/'n'...).
    Kết hợp template-matching + KIỂM TRA ĐỘ PHẲNG: ô quá phẳng (giống nền, không quân)
    thì ép về trống → chống nhận nhầm ô trống thành quân."""
    grid = [["." for _ in range(8)] for _ in range(8)]
    for r in range(8):
        for f in range(8):
            cell = cell_gray(warped, r, f)
            light = is_light(r, f)
            # độ phẳng vùng giữa ô (ô trống rất phẳng; ô có quân có nét/đường viền)
            h, w = cell.shape
            m = int(min(h, w) * 0.22)
            var = float(cell[m:h - m, m:w - m].std())
            best, best_s = ".", -2.0
            for sym, tl, tc in tmpl:
                if tl != light:
                    continue
                s = float(cv2.matchTemplate(cell, tc, cv2.TM_CCOEFF_NORMED)[0, 0])
                if s > best_s:
                    best_s, best = s, sym
            # ô quá phẳng -> chắc chắn trống (loại quân "ma")
            if best != "." and var < 13:
                best = "."
            grid[r][f] = best
    return grid


def grid_to_fen(grid, turn="w"):
    """Dựng FEN (chỉ phần vị trí + lượt) từ ma trận 8x8 (grid[0]=rank8)."""
    rows = []
    for r in range(8):
        row = ""
        empty = 0
        for f in range(8):
            s = grid[r][f]
            if s == ".":
                empty += 1
            else:
                if empty:
                    row += str(empty); empty = 0
                row += s
        if empty:
            row += str(empty)
        rows.append(row)
    placement = "/".join(rows)
    return f"{placement} {turn} - - 0 1"


def grid_is_valid(grid):
    """Kiểm tra thô: đúng 1 Vua mỗi bên, mỗi bên ≤ 16 quân, ≤ 8 Tốt. Lọc read rác."""
    flat = [c for row in grid for c in row if c != "."]
    if flat.count("K") != 1 or flat.count("k") != 1:
        return False
    white = [c for c in flat if c.isupper()]
    black = [c for c in flat if c.islower()]
    if len(white) > 16 or len(black) > 16:
        return False
    if flat.count("P") > 8 or flat.count("p") > 8:
        return False
    return True


def read_board(warped, tmpl, turn="w"):
    """Ảnh warp → (chess.Board|None, fen, grid). Board=None nếu read không hợp lệ."""
    grid = classify_grid(warped, tmpl)
    fen = grid_to_fen(grid, turn)
    if not grid_is_valid(grid):
        return None, fen, grid
    try:
        b = chess.Board(fen)
        return b, fen, grid
    except Exception:
        return None, fen, grid
