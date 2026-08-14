"""DÒ + BÁM LƯỚI BÀN CỜ với camera DI ĐỘNG — tận dụng tri thức CỜ VUA làm chỗ dựa:

  1. Bàn cờ là lưới 8x8 caro sáng/tối xen kẽ  → mọi kết quả dò đều TỰ CHẤM ĐIỂM
     (caro_score); điểm thấp = lưới lệch = VỨT, thà im lặng còn hơn báo sai.
  2. Thế KHAI CUỘC đã biết (32 ô nào có quân) → tự hiệu chỉnh ngưỡng occupancy
     cho đúng điều kiện sáng/loá của phiên hiện tại, không cần chỉnh tay.
  3. Quân Trắng sáng / Đen tối                → tự xác định hướng bàn (Trắng ở dưới?).
  4. Trong ván, game state cho biết occupancy KỲ VỌNG → chọn hướng khi dò lại.

Pipeline camera di động:  bám (ORB, rẻ, mỗi frame) → mất dấu thì dò lại (đắt,
thi thoảng) → luôn luôn tự chấm điểm trước khi tin.
"""
from __future__ import annotations
import json
from typing import Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

# Kích thước warp: nhỏ để CHẤM ĐIỂM/dò (nhanh), lớn để đọc occupancy (nét)
SCORE_WARP = 320
CARO_OK = 0.2          # điểm caro tối thiểu để tin một lưới (t-statistic sáng/tối)
SHARP_MIN = 10.0       # độ nét tối thiểu của frame (var Laplacian toàn ảnh)


# ---------------- tiện ích hình học ----------------
def order_corners(pts):
    """Sắp 4 điểm thành [trên-trái, trên-phải, dưới-phải, dưới-trái]."""
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(1)
    d = (pts[:, 1] - pts[:, 0])
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def rot_corners(corners, k):
    """Xoay Ý NGHĨA của 4 góc đi k*90° (ảnh warp ra sẽ xoay theo, không cần rotate ảnh)."""
    c = list(corners)
    k = k % 4
    # xoay 90° CW: tl mới = bl cũ
    for _ in range(k):
        c = [c[3], c[0], c[1], c[2]]
    return np.array(c, dtype=np.float32)


def warp(img, corners, size):
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), dst)
    return cv2.warpPerspective(img, M, (size, size))


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ---------------- chữ ký caro 8x8 ----------------
def cell_bg(warped_gray):
    """Ước lượng độ sáng NỀN từng ô (8x8): lấy 4 mảnh nhỏ ở 4 GÓC ô (quân đứng giữa
    ô nên góc ô thường là nền), dùng trung vị → bền cả khi ô có quân.
    Vector hoá bằng ảnh tích phân (gọi rất nhiều lần trong refine nên cần nhanh)."""
    g = warped_gray
    H, W = g.shape[:2]
    step = W / 8.0
    half = max(2, int(step * 0.07))
    ii = cv2.integral(g.astype(np.float32))          # (H+1, W+1)
    # 16 toạ độ tâm mảnh theo mỗi trục: mỗi ô 2 vị trí (0.16, 0.84)
    cs = (np.add.outer(np.arange(8), (0.16, 0.84)).ravel() * step)
    lo = np.clip((cs - half).astype(int), 0, None)
    hi = np.clip((cs + half).astype(int), None, H)
    y0, y1 = lo[:, None], hi[:, None]
    x0, x1 = lo[None, :], hi[None, :]
    area = (y1 - y0) * (x1 - x0) + 1e-6
    sums = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    means = sums / area                               # (16,16) mảnh [2r+i, 2f+j]
    quads = means.reshape(8, 2, 8, 2).transpose(0, 2, 1, 3).reshape(8, 8, 4)
    return np.median(quads, axis=2).astype(np.float32)


def caro_score(warped_gray):
    """Điểm 'đúng là bàn cờ': tách biệt giữa nhóm ô sáng và ô tối (kiểu t-statistic).
    Khử VIGNETTE/đổ sáng lệch trước bằng high-pass (trừ trung bình lân cận) — màn tối
    hay sáng không đều vẫn chấm đúng, vì caro là tín hiệu TẦN SỐ CAO ô-kề-ô.
    DẤU: dương = ô (0,0) thuộc nhóm sáng (parity chuẩn lichess: a8 sáng);
         âm = lưới đang lệch 90° (xoay góc 1 nấc là về chuẩn)."""
    bg = cell_bg(warped_gray)
    hp = bg - cv2.blur(bg, (3, 3))
    mask = (np.add.outer(np.arange(8), np.arange(8)) % 2) == 0
    light, dark = hp[mask], hp[~mask]
    spread = (light.std() + dark.std()) / 2.0 + 1e-6
    sc = float((light.mean() - dark.mean()) / spread)
    return max(-99.0, min(99.0, sc)), bg           # chặn trần (ảnh render sạch -> vô cực)


CARO = (np.add.outer(np.arange(8), np.arange(8)) % 2) == 0   # ô sáng của bàn cờ


def board_contrast(bg):
    """Chênh lệch sáng/tối của nền bàn — dùng chuẩn hoá đặc trưng occupancy
    để miễn nhiễm với auto-exposure của camera."""
    return abs(float(bg[CARO].mean() - bg[~CARO].mean())) + 1e-6


# ---------------- dò bàn từ đầu (chạy khi mất dấu) ----------------
def _quad_candidates(gray, area_min_frac=0.03):
    """Sinh các tứ giác ứng viên từ NHIỀU cách nhị phân hoá: Otsu, các ngưỡng theo
    bách phân vị (bàn không phải lúc nào cũng là vùng sáng nhất — phòng tối, màn phụ),
    và cạnh Canny. Trả về list 4 điểm đã sắp thứ tự."""
    h, w = gray.shape[:2]
    area_min = area_min_frac * h * w
    cands = []

    def _from_binary(bw):
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:6]:
            if cv2.contourArea(c) < area_min:
                break
            for eps in (0.02, 0.04):
                ap = cv2.approxPolyDP(c, eps * cv2.arcLength(c, True), True)
                if len(ap) == 4 and cv2.isContourConvex(ap):
                    cands.append(order_corners(ap.reshape(4, 2)))
                    break
            else:
                # không ra 4 đỉnh -> lấy hình chữ nhật bao nhỏ nhất (refine sửa sau)
                cands.append(order_corners(cv2.boxPoints(cv2.minAreaRect(c))))

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _from_binary(otsu)
    for p in (55, 70, 85):
        _, bw = cv2.threshold(blur, float(np.percentile(blur, p)), 255, cv2.THRESH_BINARY)
        _from_binary(bw)
    # CLAHE cứu cảnh TỐI/tương phản thấp (phòng tắt đèn, màn để độ sáng thấp)
    cl = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(blur)
    _, bw = cv2.threshold(cl, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _from_binary(bw)
    edges = cv2.dilate(cv2.Canny(blur, 40, 120), np.ones((3, 3), np.uint8))
    _from_binary(edges)
    return _dedupe(cands)


def _dedupe(cands, tol=0.05, keep=14):
    """Gộp các tứ giác gần trùng nhau (tâm + diện tích sát nhau)."""
    out, seen = [], []
    for c in cands:
        cx, cy = c[:, 0].mean(), c[:, 1].mean()
        a = cv2.contourArea(c.astype(np.int32).reshape(-1, 1, 2))
        side = max(np.sqrt(a), 1.0)
        if any(abs(cx - kx) < tol * side and abs(cy - ky) < tol * side and
               0.85 < a / max(ka, 1.0) < 1.18 for kx, ky, ka in seen):
            continue
        seen.append((cx, cy, a))
        out.append(c)
        if len(out) >= keep:
            break
    return out


def refine_corners(gray, corners, steps=None):
    """Tinh chỉnh 4 góc bằng leo đồi: xê dịch từng góc để TỐI ĐA |caro_score|.
    Đây là chỗ prior 8x8 phát huy — tín hiệu caro tuần hoàn kéo lưới về khớp ô.
    Bước đi tỉ lệ theo CỠ tứ giác (tứ giác to thì bước to mới thoát nổi lệch lớn)."""
    corners = np.array(corners, dtype=np.float32).copy()
    if steps is None:
        side = float(np.linalg.norm(corners[1] - corners[0]) +
                     np.linalg.norm(corners[2] - corners[3])) / 2.0
        steps = [max(2, int(side * k)) for k in (0.05, 0.02, 0.008)]
    def _score(c):
        sc, _ = caro_score(warp(gray, c, SCORE_WARP))
        return abs(sc)

    def _climb(c, best):
        for s in steps:
            improved = True
            while improved:
                improved = False
                for i in range(4):
                    for dx, dy in ((s, 0), (-s, 0), (0, s), (0, -s)):
                        trial = c.copy()
                        trial[i, 0] += dx
                        trial[i, 1] += dy
                        sc = _score(trial)
                        if sc > best + 1e-3:
                            best, c, improved = sc, trial, True
        return c, best

    # QUÉT PHA trước khi leo: caro gần như bằng 0 cho tới khi lưới ≈ khớp ô (mặt
    # phẳng tuần hoàn), leo đồi từ điểm mù sẽ kẹt. Tịnh tiến cả tứ giác ±0.4 ô
    # theo 2 trục, lấy pha tốt nhất làm điểm xuất phát.
    tl0, tr0, br0, bl0 = corners
    ex0 = ((tr0 - tl0) + (br0 - bl0)) / 16.0
    ey0 = ((bl0 - tl0) + (br0 - tr0)) / 16.0
    start, s0 = corners, _score(corners)
    for ax in (-0.4, -0.2, 0.0, 0.2, 0.4):
        for ay in (-0.4, -0.2, 0.0, 0.2, 0.4):
            if ax == 0.0 and ay == 0.0:
                continue
            v = (corners + ax * ex0 + ay * ey0).astype(np.float32)
            sc = _score(v)
            if sc > s0:
                start, s0 = v, sc
    corners, best = _climb(start, s0)

    # SNAP THEO Ô: caro tuần hoàn chu kỳ 1 ô -> leo đồi có thể khoá nhầm lưới
    # LỆCH NGUYÊN 1 Ô (7 cột khớp, 1 cột nằm ngoài bàn). Thử tịnh tiến / nở / co
    # đúng 1 ô theo vector cạnh của chính tứ giác, cái nào điểm cao hơn thì lấy.
    tl, tr, br, bl = corners
    ex = ((tr - tl) + (br - bl)) / 16.0        # vector 1 ô theo trục ngang
    ey = ((bl - tl) + (br - tr)) / 16.0        # vector 1 ô theo trục dọc
    variants = [
        corners + ex, corners - ex, corners + ey, corners - ey,          # tịnh tiến
        np.array([tl - ex, tr, br, bl - ex]),                            # nở trái
        np.array([tl, tr + ex, br + ex, bl]),                            # nở phải
        np.array([tl - ey, tr - ey, br, bl]),                            # nở trên
        np.array([tl, tr, br + ey, bl + ey]),                            # nở dưới
        np.array([tl + ex, tr, br, bl + ex]),                            # co trái
        np.array([tl, tr - ex, br - ex, bl]),                            # co phải
        np.array([tl + ey, tr + ey, br, bl]),                            # co trên
        np.array([tl, tr, br - ey, bl - ey]),                            # co dưới
    ]
    for v in variants:
        v = v.astype(np.float32)
        sc = _score(v)
        if sc > best + 0.3:                    # phải hơn RÕ RỆT mới nhảy
            c2, s2 = _climb(v, sc)
            if s2 > best:
                corners, best = c2, s2
    return corners, best


def find_board(frame_bgr):
    """Dò bàn cờ trong frame bất kỳ. Trả về (corners đã sửa parity, |score|) hoặc None.
    LƯU Ý: parity chỉ khử được lệch 90°; lệch 180° (Trắng trên/dưới) phải phân xử
    bằng màu quân (sync đầu ván) hoặc game state (orient_to_board)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cands = _quad_candidates(gray)

    # ĐỆ QUY 1 CẤP: bàn cờ hay nằm BÊN TRONG một tứ giác to hơn (màn hình laptop).
    # Với 2 ứng viên to nhất, quét lại ngay bên trong ROI của nó.
    by_area = sorted(cands, key=lambda c: -cv2.contourArea(
        c.astype(np.int32).reshape(-1, 1, 2)))
    h, w = gray.shape[:2]
    for c in by_area[:2]:
        x0, y0 = max(0, int(c[:, 0].min())), max(0, int(c[:, 1].min()))
        x1, y1 = min(w, int(c[:, 0].max())), min(h, int(c[:, 1].max()))
        if x1 - x0 < 80 or y1 - y0 < 80:
            continue
        for sub in _quad_candidates(gray[y0:y1, x0:x1], area_min_frac=0.08):
            cands.append(sub + np.array([x0, y0], dtype=np.float32))

    scored = []
    for cand in _dedupe(cands, keep=20):
        sc, _ = caro_score(warp(gray, cand, 160))
        scored.append((abs(sc), cand))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    best_c, best_s = None, 0.0
    for _, cand in scored[:3]:                      # refine 3 ứng viên tốt nhất
        c, s = refine_corners(gray, cand)
        if s > best_s:
            best_c, best_s = c, s
        if best_s >= CARO_OK * 2:                   # đã rất chắc -> khỏi thử thêm
            break
    if best_c is None or best_s < CARO_OK:
        return None
    sc, _ = caro_score(warp(gray, best_c, SCORE_WARP))
    if sc < 0:                                       # parity lệch 90° -> xoay 1 nấc
        best_c = rot_corners(best_c, 1)
    return best_c, best_s


# ---------------- occupancy tự hiệu chỉnh ----------------
START_MASK = np.zeros((8, 8), dtype=bool)
START_MASK[0:2, :] = True    # rank 8,7 (hàng trên ảnh)
START_MASK[6:8, :] = True    # rank 2,1


GRID_SPAN = 8          # biên độ căn lại lưới, tính bằng pixel trên ảnh warp 512


def _cell_bg_box(ii, box):
    """cell_bg nhưng cho một hộp bàn tuỳ ý, dùng lại ảnh tích phân đã dựng sẵn."""
    x0, y0, x1, y1 = box
    sx, sy = (x1 - x0) / 8.0, (y1 - y0) / 8.0
    half = max(2, int(min(sx, sy) * 0.07))
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    cx = np.add.outer(np.arange(8), (0.16, 0.84)).ravel() * sx + x0
    cy = np.add.outer(np.arange(8), (0.16, 0.84)).ravel() * sy + y0
    lox = np.clip((cx - half).astype(int), 0, W)
    hix = np.clip((cx + half).astype(int), 0, W)
    loy = np.clip((cy - half).astype(int), 0, H)
    hiy = np.clip((cy + half).astype(int), 0, H)
    area = (hiy[:, None] - loy[:, None]) * (hix[None, :] - lox[None, :]) + 1e-6
    s = (ii[hiy[:, None], hix[None, :]] - ii[loy[:, None], hix[None, :]]
         - ii[hiy[:, None], lox[None, :]] + ii[loy[:, None], lox[None, :]])
    quads = (s / area).reshape(8, 2, 8, 2).transpose(0, 2, 1, 3).reshape(8, 8, 4)
    return np.median(quads, axis=2).astype(np.float32)


_LAST_BOX = {}         # shape ảnh -> hộp bàn của khung trước (xem `grid_box`)


def grid_box(gray, span=GRID_SPAN, init=None, ii=None):
    """Căn lại lưới 8×8 NGAY TRÊN ẢNH WARP → hộp bàn (x0,y0,x1,y1).

    Vì sao cần, dù `rectify` đã nắn: warp chỉ cần quad đúng ~vài pixel là đạt
    ngưỡng caro, nhưng lõi ô chỉ cách mép ô đúng 15px (m = 0.24×64). Đo trên
    1000 khung thật, mép trái warp còn dính ~14px nền ngoài bàn — biên an toàn
    của cột a gần như bằng 0, nên hễ camera rung là dải nền trắng lọt vào lõi và
    ô trống bị đọc thành CÓ QUÂN. Đó là nguồn gốc 121/291 ô báo thừa dồn vào một
    cột duy nhất.

    Leo đồi trên TƯƠNG PHẢN CARO của nền ô (sáng/tối xen kẽ), bốn cạnh ĐỘC LẬP:
    lệch thường chỉ ở một cạnh, nên co đều cả bốn là tự kéo lệch cả lưới — đã đo
    và nó tụt xuống 73,5%.
    """
    H, W = gray.shape[:2]
    # Biên độ tính THEO TỈ LỆ ảnh: dataset thật lẫn cả warp 512 lẫn 800, và một
    # hằng số pixel cứng sẽ siết quá chặt ở cỡ này và quá lỏng ở cỡ kia.
    span = max(4.0, span * W / 512.0)
    if ii is None:
        ii = cv2.integral(gray.astype(np.float32))
    goc = [0.0, 0.0, float(W), float(H)]

    def sc(b):
        bg = _cell_bg_box(ii, b)
        return abs(float(bg[CARO].mean() - bg[~CARO].mean()))

    # Khởi từ lưới của khung TRƯỚC, đúng cách `rectify.Tracker` bám quad: giữa hai
    # khung liền nhau lưới gần như không dịch, nên chỉ cần bước tinh. Leo lại từ
    # đầu mỗi khung tốn ~12 ms — nhiều hơn cả `rectify` — mà kết quả y hệt.
    box = list(init) if init is not None else [0.0, 0.0, float(W), float(H)]
    buoc = (2, 1) if init is not None else (8, 4, 2, 1)
    best = sc(box)
    for step in buoc:
        moved = True
        while moved:
            moved = False
            for e in range(4):
                for d in (-step, step):
                    nb = list(box)
                    nb[e] += d
                    if abs(nb[e] - goc[e]) > span:
                        continue
                    if not (0 <= nb[0] < nb[2] <= W and 0 <= nb[1] < nb[3] <= H):
                        continue
                    s = sc(nb)
                    if s > best + 1e-6:
                        box, best, moved = nb, s, True
    return box


def _features(warped_bgr):
    """(8,8,2) đặc trưng mỗi ô, đã CHUẨN HOÁ theo độ tương phản bàn (chống auto-exposure):
       f0 = độ gồ ghề lõi ô (std)   f1 = lệch màu lõi ô so với nền ước lượng của ô.

    Lưới được CĂN LẠI trên chính ảnh warp trước khi cắt ô (xem `grid_box`). Đo
    trên 1000 khung thật: khung khớp 100% 86,9% → 90,6%, ô báo thừa 291 → 185,
    và khung lệch ≥2 ô — đúng cái đẩy hệ thống vào đường nở cây 282 ms — 72 → 48.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ii = cv2.integral(gray)                       # dựng MỘT lần, dùng cho cả hai
    box = grid_box(gray, init=_LAST_BOX.get(gray.shape), ii=ii)
    _LAST_BOX[gray.shape] = box
    x0, y0, x1, y1 = box
    sx, sy = (x1 - x0) / 8.0, (y1 - y0) / 8.0
    bg = _cell_bg_box(ii, box)
    contrast = abs(float(bg[CARO].mean() - bg[~CARO].mean())) + 1e-6
    mx, my = int(sx * 0.24), int(sy * 0.24)
    feats = np.zeros((8, 8, 2), dtype=np.float32)
    for r in range(8):
        for f in range(8):
            inner = gray[int(y0 + r * sy) + my:int(y0 + (r + 1) * sy) - my,
                         int(x0 + f * sx) + mx:int(x0 + (f + 1) * sx) - mx]
            if inner.size == 0:
                continue
            feats[r, f, 0] = inner.std() / contrast
            feats[r, f, 1] = np.abs(inner - bg[r, f]).mean() / contrast
    return feats


class OccupancyModel:
    """Phân loại ô CÓ QUÂN/TRỐNG với ngưỡng TỰ HỌC từ thế khai cuộc."""

    def __init__(self, t=(1e9, 1e9)):
        self.t = list(t)

    @staticmethod
    def fit(warped_bgr, occ_mask=START_MASK):
        feats = _features(warped_bgr)
        model = OccupancyModel()
        margins = []
        for k in range(feats.shape[2]):
            occ, emp = feats[occ_mask, k], feats[~occ_mask, k]
            if len(occ) == 0 or len(emp) == 0:
                margins.append(-1.0)
                continue
            # Dùng percentile 95/5 để loại bỏ 1-2 ô nhiễu do vệt sáng/bóng râm trên màn hình
            lo = float(np.percentile(emp, 95))
            hi = float(np.percentile(occ, 5))
            margin = (hi - lo) / (float(occ.mean()) + 1e-6)
            margins.append(margin)
            model.t[k] = (lo + hi) / 2.0 if margin > 0 else 1e9
        return model, margins

    def predict(self, warped_bgr):
        feats = _features(warped_bgr)
        grid = np.zeros((8, 8), dtype=bool)
        for k, t in enumerate(self.t):
            grid |= feats[:, :, k] > t
        return grid

    def usable(self):
        return any(t < 1e8 for t in self.t)

    def to_dict(self):
        return {"t": self.t}

    @staticmethod
    def from_dict(d):
        return OccupancyModel(tuple(d["t"]))


# ---------------- hướng bàn ----------------
def detect_human_color(warped_bgr):
    """Xác định màu quân ở PHÍA DƯỚI ảnh camera (hàng 6,7):
    Nếu hàng 6,7 tối hơn hàng 0,1 -> Người chơi cầm ĐEN (Black at bottom).
    Nếu hàng 6,7 sáng hơn hàng 0,1 -> Người chơi cầm TRẮNG (White at bottom)."""
    import chess
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    W = gray.shape[0]
    step = W // 8
    m = int(step * 0.24)

    def piece_darkness(rows):
        dark_vals = []
        for r in rows:
            for f in range(8):
                inner = gray[r * step + m:(r + 1) * step - m,
                             f * step + m:(f + 1) * step - m].ravel()
                k = max(1, inner.size // 5)
                dark_vals.append(float(np.mean(np.partition(inner, k)[:k])))
        dark_vals.sort()
        return float(np.mean(dark_vals[:8]))

    bot = piece_darkness([6, 7])
    top = piece_darkness([0, 1])
    # Hàng dưới tối hơn hẳn (bot < top - 15) -> Bạn cầm ĐEN ở dưới
    if bot < top - 15:
        return chess.BLACK
    return chess.WHITE


def white_is_bottom(warped_bgr, occ_mask=START_MASK):
    """Thế khai cuộc: so độ TỐI của quân 2 hàng trên vs 2 hàng dưới
    (lấy trung vị 25% pixel tối nhất lõi ô — quân Đen tối hơn hẳn quân Trắng)."""
    import chess
    return detect_human_color(warped_bgr) == chess.WHITE


def color_match(warped_bgr, board):
    """Điểm khớp MÀU QUÂN với thế cờ `board` ở đúng hướng warp hiện tại:
    quân Đen phải TỐI hơn quân Trắng. Trả về (độ tối Đen - độ tối Trắng) —
    dương = hướng đúng. Dùng phân xử lệch 180° với THẾ BẤT KỲ (không chỉ khai cuộc)."""
    import chess
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    W = gray.shape[0]
    step = W // 8
    m = int(step * 0.24)
    dw, db = [], []
    for r in range(8):
        for f in range(8):
            pc = board.piece_at(chess.square(f, 7 - r))
            if pc is None:
                continue
            inner = gray[r * step + m:(r + 1) * step - m,
                         f * step + m:(f + 1) * step - m].ravel()
            k = max(1, inner.size // 4)
            dark = float(np.median(np.partition(inner, k)[:k]))
            (dw if pc.color == chess.WHITE else db).append(dark)
    if not dw or not db:
        return 0.0
    return float(np.median(dw) - np.median(db))         # Trắng sáng hơn -> dương


def expected_mask(board, flipped=False):
    """Occupancy KỲ VỌNG từ game state (hàng ảnh 0 = rank 8 khi flipped=False; hàng ảnh 7 = rank 8 khi flipped=True)."""
    import chess
    mask = np.zeros((8, 8), dtype=bool)
    for r in range(8):
        for f in range(8):
            rank = r if flipped else (7 - r)
            file = (7 - f) if flipped else f
            if board.piece_at(chess.square(file, rank)) is not None:
                mask[r, f] = True
    return mask


def orient_to_board(frame_bgr, corners, model, expect, warp_size=800):
    """Sau khi DÒ LẠI giữa ván: parity đã khử 90°, còn 2 khả năng (0°/180°).
    Chọn hướng có occupancy khớp game state nhất; khớp kém cả 2 -> None (chưa tin)."""
    best = None
    for k in (0, 2):
        c = rot_corners(corners, k)
        grid = model.predict(warp(frame_bgr, c, warp_size))
        agree = int((grid == expect).sum())
        if best is None or agree > best[0]:
            best = (agree, c)
    return best[1] if best and best[0] >= 60 else None      # >=60/64 ô khớp mới tin


# ---------------- bám theo thời gian thực ----------------
class GridTracker:
    """Bám 4 góc bàn qua từng frame bằng ORB + homography (camera rung/trôi vẫn đuổi kịp).
    Mỗi lần bám xong PHẢI tự kiểm bằng caro_score ở phía gọi."""

    def __init__(self, ref_gray, ref_corners):
        self.orb = cv2.ORB_create(1200)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(ref_gray, None)
        self.ref_corners = np.array(ref_corners, dtype=np.float32).reshape(-1, 1, 2)

    def track(self, cur_gray):
        if self.ref_des is None or len(self.ref_kp) < 12:
            return None
        kp, des = self.orb.detectAndCompute(cur_gray, None)
        if des is None or len(kp) < 12:
            return None
        matches = self.bf.match(self.ref_des, des)
        if len(matches) < 15:
            return None
        matches = sorted(matches, key=lambda m: m.distance)[:150]
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return None
        c = cv2.perspectiveTransform(self.ref_corners, H).reshape(4, 2)
        h, w = cur_gray.shape[:2]
        if (c[:, 0].min() < -60 or c[:, 0].max() > w + 60 or
                c[:, 1].min() < -60 or c[:, 1].max() > h + 60):
            return None
        return c.astype(np.float32)

    def rebase(self, gray, corners):
        """Cập nhật frame tham chiếu (gọi định kỳ khi đang bám tốt, chống trôi luỹ kế)."""
        self.ref_kp, self.ref_des = self.orb.detectAndCompute(gray, None)
        self.ref_corners = np.array(corners, dtype=np.float32).reshape(-1, 1, 2)


# ---------------- lưu / nạp hiệu chỉnh ----------------
def save_calib(path, corners, model, score):
    with open(path, "w") as f:
        json.dump({"corners": np.asarray(corners, dtype=float).tolist(),
                   "occupancy": model.to_dict(), "caro": float(score)}, f)


def load_calib(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return (np.array(d["corners"], dtype=np.float32),
                OccupancyModel.from_dict(d["occupancy"]), float(d.get("caro", 0)))
    except Exception:
        return None
