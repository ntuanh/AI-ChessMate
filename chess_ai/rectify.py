"""Nắn ảnh camera thành BÀN CỜ CHUẨN (top-down 512×512) — bước tiền xử lý duy nhất
trước khi phân loại quân.

Vì sao tách thành module riêng: pipeline cũ dò được một tứ giác rồi TIN LUÔN, nên
một quad sai (viền laptop, khung browser) vẫn bị warp và mọi bước sau sai theo —
xem captures/coach_warp.png. Ở đây quad chỉ là ĐIỂM KHỞI ĐẦU:

    dò nhiều ứng viên  →  tinh chỉnh 4 góc để TỐI ĐA HOÁ điểm caro
                       →  TỪ CHỐI nếu điểm cuối vẫn thấp

Điểm caro chỉ cao khi lưới 8×8 trùng đúng biên ô thật, nên việc tối đa hoá nó tự
sửa cả lệch phối cảnh lẫn lệch pha nửa ô — hai lỗi mà bước "dò quad rồi warp"
không hề phát hiện được.

Sau khi nắn, ảnh còn được CHUẨN HOÁ MÀU về đúng tông dữ liệu train (gen_cells):
màu ô sáng/ô tối được đưa về mốc cố định bằng phép affine từng kênh, nên frame
tối om lệch màu xanh (captures/r720.jpg) và frame sáng chói ra cùng một tông.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SIZE = 512                  # cạnh ảnh bàn cờ đầu ra (8 ô × 64px — khớp PieceNet)
CELL = SIZE // 8
SCORE_SIZE = 256            # cỡ warp dùng khi chấm điểm (nhỏ cho nhanh)
ACCEPT = 5.0                # điểm caro tối thiểu để tin một quad (đo trên bộ test
                            # captures/: frame nắn đúng 11.1–17.8, quad sai ≤ 0.9)
MIN_INSIDE = 0.995          # bàn phải nằm gần trọn trong khung ảnh
EARLY_ACCEPT = 10.0         # đường contour đạt mức này thì khỏi quét toàn khung
                            # (gấp đôi ACCEPT — chỉ bỏ quét khi đã rất chắc)
PARITY = (np.indices((8, 8)).sum(0) % 2 == 0)   # True ở ô (0,0) — a8 sáng

# Tông bàn đích (BGR) — lichess brown, trùng theme đầu trong gen_cells.BOARD_THEMES
CANON_LIGHT = np.array([181.0, 217.0, 240.0], dtype=np.float32)
CANON_DARK = np.array([99.0, 136.0, 181.0], dtype=np.float32)


@dataclass
class Rectified:
    """Kết quả nắn một frame."""
    board: np.ndarray       # (SIZE,SIZE,3) BGR đã nắn + chuẩn hoá màu
    quad: np.ndarray        # (4,2) float32 góc bàn trong ảnh GỐC, thứ tự TL,TR,BR,BL
    score: float            # điểm caro cuối cùng
    inside: float           # tỉ lệ diện tích bàn NẰM TRONG khung ảnh (0..1)
    ok: bool                # score >= ACCEPT và bàn nằm trọn trong khung

    def cells(self):
        """Cắt thành (64,CELL,CELL,3) theo thứ tự a8..h8, a7..h7, ... a1..h1."""
        b = self.board
        return np.stack([b[r * CELL:(r + 1) * CELL, f * CELL:(f + 1) * CELL]
                         for r in range(8) for f in range(8)])


# ---------------- hình học cơ bản ----------------
def order_quad(pts):
    """Sắp 4 điểm theo TL, TR, BR, BL (thuận chiều kim đồng hồ từ góc trên-trái)."""
    p = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]                       # theo góc: bắt đầu từ ~-180°
    k = int(np.argmin(p.sum(axis=1)))            # góc gần gốc toạ độ nhất = TL
    return np.roll(p, -k, axis=0)


def rot_quad(quad, k):
    """Xoay nhãn góc đi k nấc 90° (đổi hướng bàn, KHÔNG đổi vùng ảnh)."""
    return np.roll(np.asarray(quad, dtype=np.float32), -k, axis=0)


def _warp_to(img, quad, size, flags=cv2.INTER_LINEAR):
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [size, 0], [size, size], [0, size]], dtype=np.float32)
    m = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, m, (size, size), flags=flags)


def warp(img, quad, size):
    """Nắn vùng tứ giác thành ảnh vuông size×size, CÓ khử moiré.

    Render ở 2× rồi INTER_AREA về size: INTER_AREA là bộ lọc thông thấp thật sự
    nên xoá được moiré của ảnh soi màn hình, còn warp trực tiếp (INTER_LINEAR)
    lấy mẫu thưa và GIỮ LẠI moiré.

    Dùng cho ẢNH ĐẦU RA (đưa vào PieceNet) — nơi chất lượng ảnh quyết định độ
    chính xác. Bước tìm kiếm thì dùng warp_fast trên ảnh đã blur sẵn.
    """
    return cv2.resize(_warp_to(img, quad, size * 2), (size, size),
                      interpolation=cv2.INTER_AREA)


def warp_fast(img, quad, size):
    """Nắn 1 lần, KHÔNG oversample — cho bước chấm điểm (gọi hàng trăm lần/frame).

    Bỏ oversample ở đây không mất khả năng chống moiré, vì `img` truyền vào đã
    được blur một lần trước cả vòng tìm kiếm (xem `prep_gray`): lọc thông thấp
    làm 1 lần cho cả frame thay vì lặp lại ở từng lần chấm. Đo trên AIBOX: rẻ
    hơn ~4× số pixel phải warp.
    """
    return _warp_to(img, quad, size)


def prep_gray(frame_bgr, work_width=640):
    """Ảnh xám đã thu nhỏ + blur nhẹ, dùng cho toàn bộ bước dò/chấm điểm.

    Thu nhỏ bằng INTER_AREA và blur σ=1.0 chính là phần khử moiré, làm MỘT LẦN
    mỗi frame. Trả (gray, tỉ lệ thu nhỏ).
    """
    h, w = frame_bgr.shape[:2]
    sc = work_width / float(w)
    small = cv2.resize(frame_bgr, (work_width, int(round(h * sc))),
                       interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (0, 0), 1.0), sc


# ---------------- điểm caro ----------------
def cell_bg(board_gray):
    """(8,8) độ sáng NỀN từng ô: trung vị 4 mảnh ở 4 GÓC ô.

    Quân cờ đứng giữa ô nên góc ô gần như luôn là nền — nhờ vậy hàm này đọc được
    màu bàn kể cả khi bàn đầy quân. Vector hoá bằng ảnh tích phân vì bước tinh
    chỉnh gọi nó hàng trăm lần mỗi frame.
    """
    g = board_gray.astype(np.float32)
    h, w = g.shape[:2]
    step = w / 8.0
    half = max(2, int(step * 0.07))
    ii = cv2.integral(g)
    cs = (np.add.outer(np.arange(8), (0.16, 0.84)).ravel() * step)
    lo = np.clip((cs - half).astype(int), 0, None)
    hi = np.clip((cs + half).astype(int), None, h - 1)
    y0, y1 = lo[:, None], hi[:, None]
    x0, x1 = lo[None, :], hi[None, :]
    area = (y1 - y0) * (x1 - x0) + 1e-6
    s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
    quads = (s / area).reshape(8, 2, 8, 2).transpose(0, 2, 1, 3).reshape(8, 8, 4)
    return np.median(quads, axis=2).astype(np.float32)


def caro_score(bg):
    """Độ tách biệt giữa nhóm ô sáng và ô tối, tính theo TRUNG VỊ/MAD.

    Khử đổ sáng không đều trước bằng high-pass (trừ trung bình 3×3 lân cận): caro
    là tín hiệu ô-kề-ô, còn vignette/ánh đèn là tín hiệu biến thiên chậm.

    Dùng trung vị + MAD thay vì trung bình + phương sai vì đây là thống kê BỀN:
    bàn tay che vài ô (captures/chk1.jpg) hay vệt loá chỉ làm lệch một nhóm nhỏ
    mẫu, trung bình/phương sai bị kéo theo còn trung vị/MAD thì không.

    DẤU: dương = ô (0,0) thuộc nhóm sáng (đúng chuẩn lichess: a8 sáng);
         âm = lưới lệch 1 ô hoặc bàn đang xoay 90° → xoay quad 1 nấc là về chuẩn.
    """
    hp = bg - cv2.blur(bg, (3, 3))
    a, b = hp[PARITY], hp[~PARITY]
    d = np.median(a) - np.median(b)
    mad = np.median(np.abs(np.concatenate([a - np.median(a), b - np.median(b)])))
    return float(d / (1.4826 * mad + 1e-6))


def score_quad(gray, quad, size=SCORE_SIZE):
    """Điểm |caro| của một quad — dùng trị tuyệt đối để bước tinh chỉnh không bị
    kẹt ở nghiệm lệch 1 ô (parity ngược nhưng lưới vẫn trùng biên ô)."""
    try:
        b = warp_fast(gray, quad, size)
    except cv2.error:
        return -1e9
    return abs(caro_score(cell_bg(b)))


# ---------------- sinh ứng viên quad ----------------
def _dedupe(quads, tol=0.04):
    """Bỏ các quad gần trùng nhau (so tâm + cạnh, tương đối theo kích cỡ quad)."""
    keep = []
    for q in quads:
        c, s = q.mean(axis=0), np.linalg.norm(q[2] - q[0])
        if s < 1e-3:
            continue
        if any(np.linalg.norm(c - k.mean(axis=0)) < tol * s
               and abs(s - np.linalg.norm(k[2] - k[0])) < tol * s for k in keep):
            continue
        keep.append(q)
    return keep


def _quad_ok(q, h, w, min_area=0.03):
    """Quad có còn là ẢNH PHỐI CẢNH CỦA MỘT HÌNH VUÔNG hay không.

    Ràng buộc này là bắt buộc cho cả bước tinh chỉnh, không chỉ bước lọc ứng
    viên: điểm caro một mình có thể bị "lừa" bởi quad suy biến thành mảnh mỏng —
    nó vắt qua vài ô rồi khớp parity một cách tình cờ và ăn điểm cao giả
    (captures/chk1.jpg từng cho dương tính giả đúng kiểu này).
    """
    q = np.asarray(q, dtype=np.float32)
    if not np.isfinite(q).all():
        return False
    a = cv2.contourArea(q)
    if a < min_area * h * w or a > 0.99 * h * w:
        return False
    e = np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)])
    if e.min() < 1e-3 or e.max() / e.min() > 2.5:
        return False
    def cross2(u, v):                            # numpy 2 bỏ cross cho vector 2D
        return float(u[0] * v[1] - u[1] * v[0])

    cr = [cross2(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4])
          for i in range(4)]
    if not (all(c > 0 for c in cr) or all(c < 0 for c in cr)):
        return False                             # không lồi hoặc bị xoắn
    return a > 0.5 * float(e.mean()) ** 2        # không quá dẹt


def _from_contours(blur, h, w):
    """Quad từ contour trên nhiều bản đồ cạnh khác nhau — bắt được biên ngoài bàn
    trong cả frame sáng lẫn frame tối."""
    out = []
    k5 = np.ones((5, 5), np.uint8)
    maps = [cv2.morphologyEx(cv2.Canny(blur, 30, 90), cv2.MORPH_CLOSE, k5),
            cv2.morphologyEx(cv2.Canny(blur, 60, 180), cv2.MORPH_CLOSE, k5),
            cv2.morphologyEx(cv2.Canny(blur, 100, 250), cv2.MORPH_CLOSE, k5)]
    for m in maps:
        cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:12]:
            p = cv2.arcLength(c, True)
            for eps in (0.02, 0.05):
                ap = cv2.approxPolyDP(c, eps * p, True)
                if len(ap) == 4 and cv2.isContourConvex(ap):
                    q = order_quad(ap)
                    if _quad_ok(q, h, w):
                        out.append(q)
                    break
    return out


def _from_checker_energy(gray, h, w):
    """Quad từ VÙNG CÓ NĂNG LƯỢNG CARO cao.

    Bàn cờ là vùng duy nhất trong khung có tín hiệu tần số cao trải đều thành
    mảng lớn. Chấm bằng |ảnh - trung bình lân cận| rồi làm mượt: chữ trên browser
    cũng nhiễu cao nhưng thành vệt mảnh, còn bàn cờ thành một khối vuông đặc.
    Trả về minAreaRect — chưa đúng phối cảnh nhưng là mồi rất tốt cho tinh chỉnh.
    """
    g = gray.astype(np.float32)
    hp = np.abs(g - cv2.blur(g, (9, 9)))
    e = cv2.blur(hp, (41, 41))
    out = []
    for pct in (75, 85):
        m = (e > np.percentile(e, pct)).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:3]:
            q = order_quad(cv2.boxPoints(cv2.minAreaRect(c)))
            if _quad_ok(q, h, w):
                out.append(q)
    return out


_HANN = {}


def _hann(size):
    """Cửa sổ Hanning 2D, cache lại — board_likeness gọi hàng nghìn lần mỗi frame."""
    if size not in _HANN:
        w = np.hanning(size).astype(np.float32)
        _HANN[size] = np.outer(w, w)
    return _HANN[size]


def board_likeness(gray, quad, size=64):
    """Điểm "vùng này giống mảng caro 8×8", KHÔNG phụ thuộc pha lưới.

    Đây là thước đo dùng để XẾP HẠNG MỒI, và nó khác hẳn score_quad. Điểm caro chỉ
    cao khi lưới đã trùng đúng biên ô: một mồi lệch 20px tụt về ~0.9, không phân
    biệt nổi với vùng vô nghĩa — nên xếp hạng mồi bằng điểm caro là vô ích (đo
    được trên captures/chk1.jpg: quad thật chỉ 0.86 thô nhưng lên 13.28 sau khi
    tinh chỉnh).

    Ở đây đo trực tiếp thành phần TẦN SỐ đặc trưng của bàn cờ: mảng 8×8 đổi màu
    mỗi ô → chu kỳ 2 ô → đúng 4 chu kỳ mỗi chiều trên cả bàn, tức bin (±4, 4) của
    FFT 2D. Biên độ bin không đổi khi dịch pha, nên mồi lệch vẫn được xếp hạng
    đúng; chuẩn hoá theo tổng năng lượng để không thiên vị vùng tương phản cao.
    """
    try:
        b = warp_fast(gray, quad, size).astype(np.float32)
    except cv2.error:
        return -1e9
    b -= b.mean()
    b = b * _hann(size)
    f = np.abs(np.fft.rfft2(b))
    peak = float(f[4, 4] + f[size - 4, 4])
    return peak / (float(np.sqrt((b * b).sum())) + 1e-6)


def _grid_seeds(h, w):
    """Rải mồi hình vuông có xoay khắp khung: nhiều tâm × nhiều cỡ × vài góc nghiêng.

    Không cần thấy biên bàn nên vẫn chạy được khi contour bị hở (bàn bị tay che
    góc) hoặc khi trong khung có hình chữ nhật khác hấp dẫn hơn bàn thật (màn
    hình thứ hai, khung browser). Trả về TOÀN BỘ mồi — việc sàng lọc để dành cho
    board_likeness.
    """
    m = min(h, w)
    base = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
    seeds = []
    for frac in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        corners = base * (frac * m / 2.0)
        for cy in np.linspace(0.25, 0.75, 4) * h:
            for cx in np.linspace(0.2, 0.8, 5) * w:
                for deg in (-12.0, -6.0, 0.0, 6.0, 12.0):
                    t = np.deg2rad(deg)
                    ct, st = np.cos(t), np.sin(t)
                    rot = corners @ np.array([[ct, st], [-st, ct]], dtype=np.float32)
                    seeds.append(rot + np.array([cx, cy], dtype=np.float32))
    return seeds


def candidates(gray):
    """Danh sách quad ứng viên hình học, đã lọc thô và bỏ trùng."""
    h, w = gray.shape[:2]
    # `gray` vào đây đã qua prep_gray (thu nhỏ INTER_AREA + blur) nên không blur lại
    return _dedupe(_from_checker_energy(gray, h, w) + _from_contours(gray, h, w))


# ---------------- tinh chỉnh 4 góc ----------------
def _hill_climb(gray, quad, score, steps, rounds=6):
    """Leo đồi từng toạ độ: thử dịch mỗi góc theo x/y với bước giảm dần, giữ lại
    thay đổi nào làm `score` tăng. Quad suy biến bị loại ngay trong vòng lặp."""
    h, w = gray.shape[:2]
    best = np.asarray(quad, dtype=np.float32).copy()
    bs = score(best)
    diag = np.linalg.norm(best[2] - best[0])
    for st in steps:
        d = st * diag
        for _ in range(rounds):                  # trần vòng lặp: tránh kẹt lâu
            moved = False
            for i in range(4):
                for ax in (0, 1):
                    for sgn in (1.0, -1.0):
                        cand = best.copy()
                        cand[i, ax] += sgn * d
                        if not _quad_ok(cand, h, w):
                            continue
                        s = score(cand)
                        if s > bs + 1e-4:
                            bs, best, moved = s, cand, True
            if not moved:
                break
    return best, bs


def _phase_sweep(gray, quad, size):
    """Quét pha lưới: dịch cả quad theo hai trục CỦA BÀN với các phân số ô khác
    nhau, giữ lại pha cho điểm caro cao nhất.

    Bắt buộc phải có bước này giữa hai tầng leo đồi. Khi lưới lệch nửa ô, mọi ô
    lấy mẫu đều vắt qua hai ô thật nên caro ≈ 0 ĐỀU KHẮP: leo đồi không có hướng
    nào để đi và đứng luôn tại chỗ. Quét pha là cách vượt vùng phẳng đó — chỉ 2
    tham số nên vét cạn được, rất rẻ.
    """
    q = np.asarray(quad, dtype=np.float32)
    u = ((q[1] - q[0]) + (q[2] - q[3])) / 2.0    # trục ngang của bàn (8 ô)
    v = ((q[3] - q[0]) + (q[2] - q[1])) / 2.0    # trục dọc của bàn (8 ô)
    best, bs = q, -1e9
    for a in np.arange(-0.5, 0.5, 0.125):
        for b in np.arange(-0.5, 0.5, 0.125):
            cand = q + (a * u + b * v) / 8.0
            s = score_quad(gray, cand, size)
            if s > bs:
                bs, best = s, cand
    return best, bs


def lock_on(gray, quad):
    """TẦNG 1 tách riêng: bám khối bàn bằng board_likeness.

    Tách ra vì đây là tầng QUYẾT ĐỊNH mồi nào đáng theo, và nó rẻ nhất (mục tiêu
    trơn, warp chỉ 64px). Nhờ vậy detect() sàng 12 mồi bằng tầng này rồi mới trả
    giá cho tầng 2–3 ở 3 mồi dẫn đầu — đo trên AIBOX: full refine tốn ~180-400ms
    mỗi mồi, nên bỏ 9 lần chạy vô ích là phần tiết kiệm lớn nhất.
    """
    return _hill_climb(gray, quad, lambda x: board_likeness(gray, x),
                       (0.06, 0.03, 0.015, 0.008))


def refine(gray, quad, size=SCORE_SIZE):
    """Tinh chỉnh 4 góc theo 3 tầng, mỗi tầng gỡ một loại lệch khác nhau:

    1. leo đồi trên board_likeness — mục tiêu TRƠN và không phụ thuộc pha, nên
       kéo được mồi lệch nhiều về đúng cỡ/vị trí/độ nghiêng của khối bàn;
    2. quét pha để lưới trùng biên ô (vượt vùng caro phẳng bằng 0);
    3. leo đồi trên caro để khớp nốt phối cảnh còn dư.

    Thứ tự này quan trọng: chạy caro ngay từ mồi thô thì đứng im tại chỗ, đo được
    trên captures/chk1.jpg (mồi tốt vẫn chỉ lên 2.6 trong khi quad thật đạt 13.3).
    """
    q, _ = lock_on(gray, quad)
    q, _ = _phase_sweep(gray, q, size)
    return _hill_climb(gray, q, lambda x: score_quad(gray, x, size),
                       (0.03, 0.015, 0.008, 0.004))


# ---------------- chuẩn hoá màu ----------------
def normalize_colors(board_bgr):
    """Đưa màu ô sáng/ô tối về mốc cố định (CANON_LIGHT/CANON_DARK) bằng affine
    từng kênh.

    Mỗi kênh có 2 mốc đo (trung vị nền nhóm ô sáng và nhóm ô tối) và 2 ẩn
    (gain, offset) → giải đúng nghiệm. Nhờ vậy một phép biến đổi xử luôn cả sai
    phơi sáng lẫn lệch cân bằng trắng, và mọi frame ra cùng một tông với dữ liệu
    train — đúng ý "sửa ảnh cho giống data".
    """
    out = np.empty_like(board_bgr, dtype=np.float32)
    for c in range(3):
        bg = cell_bg(board_bgr[:, :, c])
        lo, hi = float(np.median(bg[~PARITY])), float(np.median(bg[PARITY]))
        if hi < lo:                              # phòng khi parity bị đảo
            lo, hi = hi, lo
        span = max(hi - lo, 1.0)
        gain = float(CANON_LIGHT[c] - CANON_DARK[c]) / span
        gain = float(np.clip(gain, 0.25, 4.0))   # chặn khuếch đại nhiễu ở frame tối
        out[:, :, c] = (board_bgr[:, :, c].astype(np.float32) - lo) * gain + CANON_DARK[c]
    return np.clip(out, 0, 255).astype(np.uint8)


def frame_coverage(frame_bgr, quad):
    """Tỉ lệ diện tích bàn NẰM TRONG khung ảnh.

    Cần chỉ số này tách riêng khỏi điểm caro vì hai lỗi rất khác nhau: quad có thể
    nắn ĐÚNG hình học mà bàn vẫn thiếu dữ liệu, khi camera chỉ soi được một phần
    bàn và phần còn lại bị suy ra ngoài mép khung (captures/coach_raw.png — hàng 8
    ra thành dải đen nhưng điểm caro vẫn 12.6). Điểm caro không thấy được lỗi đó,
    nên nếu chỉ dựa vào nó thì pipeline sẽ đọc quân trên vùng không có thật.
    """
    h, w = frame_bgr.shape[:2]
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    rect = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    inter, _ = cv2.intersectConvexConvex(q, rect)
    return float(inter) / (float(cv2.contourArea(q)) + 1e-6)


# ---------------- API chính ----------------
def detect(frame_bgr, work_width=640):
    """Dò + tinh chỉnh quad bàn cờ. Trả (quad ảnh gốc, score) hoặc (None, score).

    Dò ở ảnh thu nhỏ (mặc định rộng 640) cho nhanh và bớt nhiễu, rồi đưa toạ độ
    về thang ảnh gốc. Cuối cùng chỉnh parity: nếu ô (0,0) đang tối thì xoay quad
    1 nấc để về chuẩn lichess (a8 sáng).
    """
    gray, sc = prep_gray(frame_bgr, work_width)
    gh, gw = gray.shape[:2]

    # Sàng 3 tầng, mỗi tầng đắt hơn tầng trước ~10 lần nên phải lọc dần:
    #   1. gộp mồi hình học + mồi rải khắp khung, xếp hạng bằng board_likeness
    #   2. tinh chỉnh 12 mồi dẫn đầu ở cỡ 128 (nhanh)
    #   3. tinh chỉnh lại 3 mồi dẫn đầu ở cỡ đầy đủ
    def best_of(seeds, top=3):
        """Sàng bằng tầng 1 (rẻ) rồi chỉ trả giá full refine cho `top` mồi dẫn đầu."""
        seeds = [q for q in seeds if _quad_ok(q, gh, gw)]
        if not seeds:
            return None, -1e9
        seeds.sort(key=lambda q: -board_likeness(gray, q))
        locked = sorted((lock_on(gray, q) for q in seeds[:12]), key=lambda x: -x[1])
        out = [refine(gray, q, SCORE_SIZE) for q, _ in locked[:top]]
        out = [(q, s) for q, s in out if _quad_ok(q, gh, gw)]
        return max(out, key=lambda x: x[1]) if out else (None, -1e9)

    # Thử đường hình học trước: nó chỉ có vài ứng viên nên rẻ. Đạt EARLY_ACCEPT
    # (gấp đôi ngưỡng tin) thì DỪNG, khỏi quét 600 mồi toàn khung — đo trên AIBOX
    # riêng bước quét đó đã tốn ~350ms xếp hạng + ~150ms mỗi mồi sàng.
    # Chỉ bỏ quét khi quad đã tự chứng minh rất chắc, nên không nới lỏng gì.
    best_q, best_s = best_of(candidates(gray))
    if best_q is None or best_s < EARLY_ACCEPT:
        gq, gs = best_of(_grid_seeds(gh, gw))
        if gq is not None and gs > best_s:
            best_q, best_s = gq, gs
    if best_q is None:
        return None, 0.0

    if caro_score(cell_bg(warp_fast(gray, best_q, SCORE_SIZE))) < 0:
        best_q = rot_quad(best_q, 1)
    quad = best_q / sc
    return quad.astype(np.float32), float(best_s)


def verify(frame_bgr, quad, work_width=640):
    """Kiểm lại một quad ĐÃ BIẾT (lần canh bàn trước) mà không quét toàn khung.

    Dùng ĐÚNG bộ tinh chỉnh 3 tầng của detect(), rồi đòi ĐÚNG ngưỡng ACCEPT và
    đúng ràng buộc hình học _quad_ok. Không có chỗ nào được nới: quad cũ phải tự
    chứng minh lại trên frame hiện tại, camera xê dịch hay bàn đổi chỗ là nó tụt
    điểm và bị loại, khi đó bên gọi quay về detect() đầy đủ. Chỗ tiết kiệm duy nhất
    là không rải 600 mồi khắp khung — riêng bước đó đã tốn ~350ms xếp hạng cộng
    ~150ms mỗi mồi sàng trên AIBOX.

    Trả (quad ảnh gốc, score) hoặc (None, score).
    """
    gray, sc = prep_gray(frame_bgr, work_width)
    gh, gw = gray.shape[:2]
    q = np.asarray(quad, dtype=np.float32).reshape(4, 2) * sc
    if not _quad_ok(q, gh, gw):
        return None, 0.0
    q, s = refine(gray, q, SCORE_SIZE)
    if not _quad_ok(q, gh, gw) or s < ACCEPT:
        return None, float(s)
    if caro_score(cell_bg(warp_fast(gray, q, SCORE_SIZE))) < 0:
        q = rot_quad(q, 1)
    return (q / sc).astype(np.float32), float(s)


class Tracker:
    """Bám quad bàn cờ giữa các lần dò đầy đủ.

    detect() mất ~0.9s vì phải rải mồi khắp khung, quá chậm cho vòng lặp video.
    Nhưng giữa hai frame liền nhau bàn gần như không dịch, nên chỉ cần tinh chỉnh
    CỤC BỘ quanh quad trước: quét pha hẹp + leo đồi bước nhỏ ở cỡ 128, tốn khoảng
    10-30ms. Khi điểm tụt dưới ngưỡng liên tiếp `patience` frame (tay che, camera
    bị xoay, mất bàn) thì tự dò lại toàn khung.
    """

    def __init__(self, quad, score=0.0, patience=6, work_width=640):
        self.quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        self.score = float(score)
        self.patience = patience
        self.work_width = work_width
        self.misses = 0

    def _small(self, frame_bgr):
        return prep_gray(frame_bgr, self.work_width)

    def update(self, frame_bgr):
        """Trả (quad, score, mode) với mode ∈ {"BÁM", "DÒ LẠI", "MẤT BÀN"}."""
        gray, sc = self._small(frame_bgr)
        # Ngân sách nhỏ: giữa hai frame liền nhau bàn gần như không dịch, nên chỉ
        # cần bước tinh. Nếu vẫn không đạt ngưỡng thì bậc thang phía dưới (quét
        # pha → dò lại toàn khung) mới lo, nên siết ở đây KHÔNG hạ độ chính xác.
        q, s = _hill_climb(gray, self.quad * sc,
                           lambda x: score_quad(gray, x, 128),
                           (0.008, 0.004), rounds=3)
        if s < ACCEPT:                           # thử gỡ lệch pha trước khi bỏ
            q2, s2 = _phase_sweep(gray, q, 128)
            if s2 > s:
                q, s = _hill_climb(gray, q2, lambda x: score_quad(gray, x, 128),
                                   (0.012, 0.006, 0.003), rounds=4)
        if s >= ACCEPT:
            self.quad, self.score, self.misses = q / sc, s, 0
            return self.quad, s, "BÁM"

        self.misses += 1
        if self.misses < self.patience:          # nhiễu thoáng qua: giữ quad cũ
            return self.quad, s, "BÁM"

        nq, ns = detect(frame_bgr)
        if nq is None or ns < ACCEPT:
            return self.quad, ns, "MẤT BÀN"
        self.quad, self.score, self.misses = nq, ns, 0
        return self.quad, ns, "DÒ LẠI"


def rectify(frame_bgr, quad=None, normalize=True):
    """Nắn frame thành bàn cờ chuẩn. Truyền `quad` sẵn để bỏ qua bước dò.

    Lưu ý hướng bàn: hàm này chỉ bảo đảm parity đúng (a8 sáng), tức còn lại hai
    khả năng lệch nhau 180° (bên nào ở dưới). Việc chọn giữa hai khả năng đó cần
    màu quân nên thuộc bước nhận diện quân, không thuộc bước nắn ảnh.
    """
    score = 0.0
    if quad is None:
        quad, score = detect(frame_bgr)
        if quad is None:
            return Rectified(np.zeros((SIZE, SIZE, 3), np.uint8),
                             np.zeros((4, 2), np.float32), 0.0, 0.0, False)
    else:
        quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
        g, sc = prep_gray(frame_bgr)
        score = score_quad(g, quad * sc, SCORE_SIZE)

    board = warp(frame_bgr, quad, SIZE)
    if normalize:
        board = normalize_colors(board)
    inside = frame_coverage(frame_bgr, quad)
    return Rectified(board, quad, score, inside,
                     score >= ACCEPT and inside >= MIN_INSIDE)
