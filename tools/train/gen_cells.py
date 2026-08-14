"""SINH DỮ LIỆU Ô CỜ TỔNG HỢP cho PieceNet (13 lớp: trống + 6 loại quân × 2 màu).

Bối cảnh thật: camera soi MÀN HÌNH điện thoại chạy app cờ 2D (lichess/chess.com/...).
→ Ta có thể sinh vô hạn ảnh ô cờ giống hệt điều kiện thật:
   bộ quân đa dạng (12 bộ lichess) × màu bàn đa dạng × nhiễu camera-soi-màn-hình
   (mờ, moiré, loá, lệch lưới, JPEG, phơi sáng, highlight nước đi, chấm gợi ý).

Mỗi mẫu được dựng trên canvas 3×3 ô (ô đích ở giữa) để mô phỏng đúng thực tế:
crop lệch lưới sẽ dính mép quân hàng xóm — model phải học cách bỏ qua chúng.

Ảnh trả về: 64×64×3 uint8 **BGR** (khớp OpenCV ở pipeline suy luận).
"""
from __future__ import annotations
import io
import os
import random

import numpy as np
import cv2

CLASSES = ".PNBRQKpnbrqk"          # 0=trống; hoa=Trắng, thường=Đen
CELL = 64                           # cạnh ô đầu ra
ASSET_PX = 128                      # cỡ render mỗi quân từ SVG

# 12 bộ quân của lichess (GPL) — tải 1 lần rồi cache PNG
PIECE_SETS = ["cburnett", "merida", "alpha", "staunty", "fresca", "gioco",
              "tatiana", "kosal", "maestro", "pirouetti", "california", "cardinal"]
_LILA_RAW = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece"

# (sáng, tối) BGR — gom các theme phổ biến của lichess/chess.com + biến thể
BOARD_THEMES = [
    ((181, 217, 240), (99, 136, 181)),     # lichess brown
    ((230, 227, 222), (173, 162, 140)),    # lichess blue/xám
    ((210, 238, 238), (86, 150, 118)),     # chess.com green
    ((221, 255, 255), (102, 166, 134)),    # lichess green
    ((235, 235, 235), (130, 130, 130)),    # xám trung tính
    ((201, 189, 232), (110, 84, 160)),     # tím
    ((176, 210, 232), (96, 142, 178)),     # nâu gỗ nhạt (BGR đảo của tan)
    ((205, 224, 235), (120, 158, 186)),    # kem/xanh nhạt
]


# ---------------- tải & cache bộ quân ----------------
def _asset_dir():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
    os.makedirs(d, exist_ok=True)
    return d


def _fetch_svg(pset, code):
    import urllib.request
    url = f"{_LILA_RAW}/{pset}/{code}.svg"
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read()


def prepare_assets(verbose=True):
    """Tải 12 bộ × 12 quân về assets/<set>_<code>.png (RGBA). Chạy 1 lần, có cache."""
    import cairosvg
    d = _asset_dir()
    for pset in PIECE_SETS:
        for color in "wb":
            for pt in "PNBRQK":
                code = color + pt
                out = os.path.join(d, f"{pset}_{code}.png")
                if os.path.exists(out):
                    continue
                try:
                    svg = _fetch_svg(pset, code)
                    cairosvg.svg2png(bytestring=svg, write_to=out,
                                     output_width=ASSET_PX, output_height=ASSET_PX)
                    if verbose:
                        print("ok", pset, code)
                except Exception as e:
                    print("FAIL", pset, code, e)


_CACHE = {}


def load_assets():
    """{(set, 'wP'): RGBA float32 (ASSET_PX,ASSET_PX,4)} — chỉ giữ bộ tải đủ 12 quân."""
    if _CACHE:
        return _CACHE
    d = _asset_dir()
    for pset in PIECE_SETS:
        imgs = {}
        for color in "wb":
            for pt in "PNBRQK":
                p = os.path.join(d, f"{pset}_{color+pt}.png")
                im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if im is None or im.shape[2] != 4:
                    imgs = None
                    break
            else:
                continue
            break
        if imgs is None:
            continue
        for color in "wb":
            for pt in "PNBRQK":
                im = cv2.imread(os.path.join(d, f"{pset}_{color+pt}.png"),
                                cv2.IMREAD_UNCHANGED).astype(np.float32)
                _CACHE[(pset, color + pt)] = im
    if not _CACHE:
        raise RuntimeError("Chưa có asset quân cờ — chạy prepare_assets() trước.")
    return _CACHE


def _sets_available():
    return sorted({k[0] for k in load_assets()})


# ---------------- dựng ô ----------------
def _sym_code(sym):
    """'P'→'wP', 'p'→'bP'."""
    return ("w" if sym.isupper() else "b") + sym.upper()


def _paste_piece(canvas, piece_rgba, cx, cy, size):
    """Dán quân (RGBA) vào canvas BGR, tâm (cx,cy), cạnh `size` px, alpha-blend."""
    im = cv2.resize(piece_rgba, (size, size), interpolation=cv2.INTER_AREA)
    a = im[:, :, 3:4] / 255.0
    x0, y0 = int(cx - size / 2), int(cy - size / 2)
    H, W = canvas.shape[:2]
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(W, x0 + size - sx0), min(H, y0 + size - sy0)
    if x1 <= x0 or y1 <= y0:
        return
    roi = canvas[y0:y1, x0:x1].astype(np.float32)
    src = im[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0), :3]
    al = a[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    canvas[y0:y1, x0:x1] = (roi * (1 - al) + src * al).astype(np.uint8)


def _jitter_color(bgr, rng, amt=18):
    return tuple(int(np.clip(c + rng.uniform(-amt, amt), 0, 255)) for c in bgr)


def synth_cell(rng: random.Random, sym: str | None = None):
    """Sinh 1 mẫu. Trả về (img 64×64×3 uint8 BGR, nhãn int trong CLASSES)."""
    assets = load_assets()
    sets = _sets_available()
    nprng = np.random.RandomState(rng.randrange(2**31))

    if sym is None:
        # cân bằng lớp: 25% trống, còn lại đều cho 12 loại quân
        sym = "." if rng.random() < 0.25 else rng.choice(CLASSES[1:])
    label = CLASSES.index(sym)

    pset = rng.choice(sets)
    light, dark = rng.choice(BOARD_THEMES)
    light, dark = _jitter_color(light, rng), _jitter_color(dark, rng)
    center_light = rng.random() < 0.5

    # canvas 3×3 ô
    CS = CELL
    canvas = np.zeros((3 * CS, 3 * CS, 3), np.uint8)
    for r in range(3):
        for f in range(3):
            is_l = center_light ^ ((r + f) % 2 == 1)
            canvas[r * CS:(r + 1) * CS, f * CS:(f + 1) * CS] = light if is_l else dark

    # highlight nước đi (ô vàng/xanh) — app hay tô 2 ô
    if rng.random() < 0.35:
        tint = np.array(rng.choice([(80, 200, 205), (60, 190, 155), (0, 170, 230)]), np.float32)
        for _ in range(rng.randint(1, 2)):
            r, f = rng.randrange(3), rng.randrange(3)
            roi = canvas[r * CS:(r + 1) * CS, f * CS:(f + 1) * CS].astype(np.float32)
            k = rng.uniform(0.25, 0.55)
            canvas[r * CS:(r + 1) * CS, f * CS:(f + 1) * CS] = \
                (roi * (1 - k) + tint[None, None] * k).astype(np.uint8)

    # quân hàng xóm (mép chúng sẽ lọt vào crop khi lưới lệch)
    for r in range(3):
        for f in range(3):
            if r == 1 and f == 1:
                continue
            if rng.random() < 0.45:
                s = rng.choice(CLASSES[1:])
                sz = int(CS * rng.uniform(0.78, 0.98))
                _paste_piece(canvas, assets[(pset, _sym_code(s))],
                             f * CS + CS // 2 + rng.randint(-3, 3),
                             r * CS + CS // 2 + rng.randint(-3, 3), sz)

    # ô đích
    if sym != ".":
        sz = int(CS * rng.uniform(0.78, 1.0))
        _paste_piece(canvas, assets[(pset, _sym_code(sym))],
                     CS + CS // 2 + rng.randint(-4, 4),
                     CS + CS // 2 + rng.randint(-4, 4), sz)
    elif rng.random() < 0.25:
        # chấm gợi ý nước đi trên ô trống (app vẽ khi đang chọn quân)
        col = rng.choice([(90, 90, 90), (60, 140, 60), (120, 120, 120)])
        cv2.circle(canvas, (CS + CS // 2, CS + CS // 2), max(4, CS // rng.choice([6, 7, 8])),
                   col, -1, cv2.LINE_AA)

    # ---- hình học: xoay/phối cảnh nhẹ + crop lệch lưới ----
    if rng.random() < 0.7:
        ang = rng.uniform(-4, 4)
        M = cv2.getRotationMatrix2D((1.5 * CS, 1.5 * CS), ang, rng.uniform(0.97, 1.03))
        canvas = cv2.warpAffine(canvas, M, (3 * CS, 3 * CS), borderMode=cv2.BORDER_REFLECT)
    if rng.random() < 0.5:
        d = CS * 0.06
        src = np.float32([[0, 0], [3 * CS, 0], [3 * CS, 3 * CS], [0, 3 * CS]])
        dst = src + nprng.uniform(-d, d, (4, 2)).astype(np.float32)
        canvas = cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst),
                                     (3 * CS, 3 * CS), borderMode=cv2.BORDER_REFLECT)
    jx, jy = rng.randint(-int(CS * 0.12), int(CS * 0.12)), rng.randint(-int(CS * 0.12), int(CS * 0.12))
    sc = rng.uniform(0.92, 1.10)
    half = int(CS * sc / 2)
    cx, cy = 1.5 * CS + jx, 1.5 * CS + jy
    x0, y0 = int(cx - half), int(cy - half)
    img = canvas[max(0, y0):y0 + 2 * half, max(0, x0):x0 + 2 * half]
    img = cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_AREA)

    # ---- quang học camera-soi-màn-hình ----
    f = img.astype(np.float32)
    # cân bằng trắng lệch
    f *= nprng.uniform(0.82, 1.18, (1, 1, 3))
    # gamma / sáng / tương phản
    f = np.clip(f, 0, 255) / 255.0
    f = f ** rng.uniform(0.65, 1.5)
    f = np.clip((f - 0.5) * rng.uniform(0.7, 1.25) + 0.5 + rng.uniform(-0.12, 0.12), 0, 1) * 255
    # loá: gradient sáng cục bộ
    if rng.random() < 0.35:
        gx, gy = nprng.uniform(0, CELL, 2)
        yy, xx = np.mgrid[0:CELL, 0:CELL]
        glare = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * rng.uniform(15, 45) ** 2)))
        f += glare[:, :, None] * rng.uniform(20, 90)
    # moiré: vân sin tần số cao (chụp màn hình)
    if rng.random() < 0.4:
        yy, xx = np.mgrid[0:CELL, 0:CELL]
        th = rng.uniform(0, np.pi)
        wave = np.sin((xx * np.cos(th) + yy * np.sin(th)) * rng.uniform(0.35, 1.4)
                      + rng.uniform(0, 6.28))
        f += wave[:, :, None] * rng.uniform(2, 9)
    # mờ: gaussian hoặc motion
    f = np.clip(f, 0, 255).astype(np.uint8)
    r = rng.random()
    if r < 0.45:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 1.8))
    elif r < 0.6:
        k = rng.choice([3, 5, 7])
        kern = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            kern[k // 2, :] = 1.0 / k
        else:
            kern[:, k // 2] = 1.0 / k
        f = cv2.filter2D(f, -1, kern)
    # phân giải thấp (camera xa)
    if rng.random() < 0.45:
        s = rng.uniform(0.35, 0.8)
        f = cv2.resize(cv2.resize(f, (int(CELL * s), int(CELL * s))), (CELL, CELL))
    # nhiễu cảm biến
    if rng.random() < 0.6:
        f = np.clip(f.astype(np.float32) +
                    nprng.normal(0, rng.uniform(1, 7), f.shape), 0, 255).astype(np.uint8)
    # JPEG
    if rng.random() < 0.5:
        q = rng.randint(30, 92)
        f = cv2.imdecode(cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, q])[1],
                         cv2.IMREAD_COLOR)
    return f, label


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "assets":
        prepare_assets()
        print("Bộ đủ quân:", _sets_available())
    else:
        # xem thử 1 lưới mẫu 13×10
        rng = random.Random(0)
        rows = []
        for c in range(13):
            row = [synth_cell(rng, CLASSES[c])[0] for _ in range(10)]
            rows.append(np.hstack(row))
        cv2.imwrite("preview.png", np.vstack(rows))
        print("Đã ghi preview.png (mỗi hàng 1 lớp: . P N B R Q K p n b r q k)")
