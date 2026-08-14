"""THỊ GIÁC BÀN CỜ v2 — tổng hợp từ 2 agent nghiên cứu + thiết kế.
Pipeline OpenCV thuần (không cần deep model) cho ảnh bàn Lichess NGHIÊNG trên màn hình:

  1) localize_board(frame): dò 4 góc bàn = Hough lines -> 2 họ song song -> giao điểm,
     và XÁC MINH TÍNH TUẦN HOÀN (ô median xen kẽ sáng/tối 8x8) để loại vùng rác.
  2) warp -> occupancy_grid: đa tín hiệu (cạnh + lệch màu Lab so nền + phương sai),
     chuẩn hoá z-score 64 ô, chịu được 'trắng trên ô sáng'.
  3) verify_periodic: 'oracle' chấm điểm một homography có phải bàn cờ thật không.

Occupancy trả bool 8x8; suy nước đi do coach làm (đối chiếu luật cờ, python-chess)."""
from __future__ import annotations
import numpy as np
import cv2

WARP = 800
STEP = WARP // 8


# ---------------- Oracle: xác minh tính tuần hoàn ----------------
def cell_medians(warped_gray):
    """Màu nền mỗi ô = TRUNG VỊ của VIỀN ô (quân ở giữa nên viền luôn là màu nền)."""
    med = np.zeros((8, 8), np.float32)
    for r in range(8):
        for f in range(8):
            blk = warped_gray[r * STEP:(r + 1) * STEP, f * STEP:(f + 1) * STEP]
            h, w = blk.shape
            b = max(2, int(min(h, w) * 0.16))
            ring = np.concatenate([blk[:b, :].ravel(), blk[-b:, :].ravel(),
                                   blk[:, :b].ravel(), blk[:, -b:].ravel()])
            med[r, f] = np.median(ring)
    return med


def periodic_score(warped_gray):
    """Điểm 'giống bàn cờ' = tỉ số Fisher: 2 nhóm ô sáng/tối phải tách bạch & đồng nhất.
    Bàn thật -> điểm cao (>2). Vùng rác -> thấp."""
    med = cell_medians(warped_gray)
    checker = (np.indices((8, 8)).sum(0) % 2)
    best = 0.0
    for pat in (checker, 1 - checker):
        A = med[pat == 1]; B = med[pat == 0]
        within = A.std() + B.std() + 1e-6
        between = abs(A.mean() - B.mean())
        best = max(best, between / within)
    return float(best)


# ---------------- Dò góc bàn ----------------
def _order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(1); d = np.diff(pts, axis=1).reshape(-1)
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _line_intersect(l1, l2):
    (x1, y1, x2, y2) = l1; (x3, y3, x4, y4) = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
    return (px, py)


def _cluster_lines_by_angle(lines):
    """Chia đoạn thẳng thành 2 họ theo góc (2 đỉnh histogram góc)."""
    segs = []
    for l in lines[:, 0]:
        x1, y1, x2, y2 = l
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        segs.append((ang, l))
    angs = np.array([a for a, _ in segs])
    hist, edges = np.histogram(angs, bins=36, range=(0, 180))
    order = np.argsort(hist)[::-1]
    peaks = []
    for idx in order:
        c = (edges[idx] + edges[idx + 1]) / 2
        if all(min(abs(c - p), 180 - abs(c - p)) > 20 for p in peaks):
            peaks.append(c)
        if len(peaks) == 2:
            break
    if len(peaks) < 2:
        return None
    fam = {0: [], 1: []}
    for a, l in segs:
        d0 = min(abs(a - peaks[0]), 180 - abs(a - peaks[0]))
        d1 = min(abs(a - peaks[1]), 180 - abs(a - peaks[1]))
        if min(d0, d1) > 12:
            continue
        fam[0 if d0 < d1 else 1].append(l)
    if len(fam[0]) < 2 or len(fam[1]) < 2:
        return None
    return fam[0], fam[1]


def _outer_lines(family, w, h):
    """Lấy 2 đường ngoài cùng của 1 họ song song (theo khoảng cách tới tâm ảnh)."""
    cx, cy = w / 2, h / 2
    scored = []
    for l in family:
        x1, y1, x2, y2 = l
        # khoảng cách có dấu từ tâm tới đường
        dx, dy = x2 - x1, y2 - y1
        nn = np.hypot(dx, dy) + 1e-6
        dist = ((x1 - cx) * dy - (y1 - cy) * dx) / nn
        scored.append((dist, l))
    scored.sort(key=lambda t: t[0])
    return scored[0][1], scored[-1][1]


def localize_hough(frame):
    """Trả về 4 góc (hoặc None) bằng Hough lines + giao điểm 2 họ ngoài cùng."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 50, 50)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), 1)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60,
                            minLineLength=int(w * 0.12), maxLineGap=25)
    if lines is None or len(lines) < 8:
        return None
    fams = _cluster_lines_by_angle(lines)
    if fams is None:
        return None
    fa, fb = fams
    a0, a1 = _outer_lines(fa, w, h)
    b0, b1 = _outer_lines(fb, w, h)
    corners = []
    for la in (a0, a1):
        for lb in (b0, b1):
            p = _line_intersect(la, lb)
            if p is None:
                return None
            corners.append(p)
    return _order_corners(corners)


def localize_contour(frame):
    """Fallback: tứ giác lồi lớn nhất."""
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8), 1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:8]:
        if cv2.contourArea(c) < 0.05 * w * h:
            break
        q = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(q) == 4 and cv2.isContourConvex(q):
            yield _order_corners(q.reshape(4, 2))


def warp(frame, corners):
    dst = np.array([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.array(corners, dtype=np.float32), dst)
    return cv2.warpPerspective(frame, M, (WARP, WARP))


def localize_board(frame, min_score=1.5):
    """Dò bàn: thử Hough + các tứ giác contour, chọn cái có điểm tuần hoàn cao nhất."""
    cands = []
    hc = localize_hough(frame)
    if hc is not None:
        cands.append(hc)
    for cc in localize_contour(frame):
        cands.append(cc)
    best, best_s = None, min_score
    for c in cands:
        try:
            wg = cv2.cvtColor(warp(frame, c), cv2.COLOR_BGR2GRAY)
            s = periodic_score(wg)
        except Exception:
            continue
        if s > best_s:
            best_s, best = s, c
    return best, best_s


# ---------------- Occupancy đa tín hiệu ----------------
def occupancy_grid(warped_bgr, empty_light=None, empty_dark=None):
    """Trả bool 8x8: ô có quân không. Đa tín hiệu + chuẩn hoá z-score 64 ô."""
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2LAB)
    edge_c = np.zeros((8, 8)); grad_c = np.zeros((8, 8)); var_c = np.zeros((8, 8))
    for r in range(8):
        for f in range(8):
            m = int(STEP * 0.14)
            y0, x0 = r * STEP + m, f * STEP + m
            y1, x1 = (r + 1) * STEP - m, (f + 1) * STEP - m
            g = gray[y0:y1, x0:x1]
            e = cv2.Canny(g, 40, 120)
            edge_c[r, f] = e.mean()
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
            grad_c[r, f] = np.mean(np.hypot(gx, gy))
            var_c[r, f] = g.std()

    def z(a):
        return (a - a.mean()) / (a.std() + 1e-6)
    score = z(edge_c) + z(grad_c) + 0.5 * z(var_c)
    return score > 0.0
