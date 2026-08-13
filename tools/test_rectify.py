"""Kiểm tra rectify.py trên ảnh camera THẬT và xuất ảnh debug để soi bằng mắt.

    python3 tools/test_rectify.py captures/*.jpg --out /tmp/rect

Với mỗi ảnh vào, xuất một tấm debug ghép 3 phần:
    [ảnh gốc + quad đã dò]  [bàn đã nắn + lưới 8×8]  [bàn đã chuẩn hoá màu]
và in một dòng: tên ảnh, điểm caro, thời gian dò, kết luận đạt/không.

Cuối cùng xuất contact_sheet.png gộp tất cả bàn đã nắn để so sánh nhanh.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_ai import rectify as R      # noqa: E402


def panel(img, h):
    """Đưa ảnh về cùng chiều cao h, giữ tỉ lệ."""
    s = h / img.shape[0]
    return cv2.resize(img, (int(round(img.shape[1] * s)), h))


def draw_grid(board):
    """Vẽ lưới 8×8 + đánh dấu ô a8 để kiểm tra parity bằng mắt."""
    out = board.copy()
    for i in range(9):
        p = i * R.CELL
        cv2.line(out, (p, 0), (p, R.SIZE), (0, 0, 255), 1)
        cv2.line(out, (0, p), (R.SIZE, p), (0, 0, 255), 1)
    cv2.putText(out, "a8", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--out", default="/tmp/rect")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    boards, names, nok = [], [], 0
    for p in args.images:
        frame = cv2.imread(p)
        if frame is None:
            print(f"{os.path.basename(p):18s} KHÔNG ĐỌC ĐƯỢC")
            continue
        # ảnh 800×800 trong captures/ là warp cũ (đầu ra), không phải frame camera
        t0 = time.time()
        quad, score = R.detect(frame)
        dt = time.time() - t0
        if quad is None:
            print(f"{os.path.basename(p):18s} score=  n/a  {dt*1000:6.0f}ms  KHÔNG THẤY BÀN")
            continue
        res = R.rectify(frame, quad)
        raw = R.warp(frame, quad, R.SIZE)

        vis = frame.copy()
        cv2.polylines(vis, [quad.astype(np.int32)], True, (60, 220, 60), 3)
        for i, (x, y) in enumerate(quad.astype(int)):
            cv2.circle(vis, (x, y), 6, (0, 140, 255), -1)
            cv2.putText(vis, "TL TR BR BL".split()[i], (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        h = R.SIZE
        sheet = np.hstack([panel(vis, h), draw_grid(raw), res.board])
        cv2.imwrite(os.path.join(args.out, os.path.basename(p) + ".debug.png"), sheet)

        if res.ok:
            flag = "ĐẠT"
        elif score >= R.ACCEPT:
            flag = "BÀN BỊ CẮT"      # nắn đúng nhưng thiếu dữ liệu ngoài khung
        else:
            flag = "THẤP"
        nok += int(res.ok)
        print(f"{os.path.basename(p):18s} score={score:6.2f}  "
              f"trong_khung={res.inside*100:5.1f}%  {dt*1000:6.0f}ms  {flag}")
        boards.append(res.board)
        names.append(os.path.basename(p))

    if boards:
        cols = min(4, len(boards))
        rows = (len(boards) + cols - 1) // cols
        thumb = 256
        sheet = np.full((rows * thumb, cols * thumb, 3), 30, np.uint8)
        for i, (b, n) in enumerate(zip(boards, names)):
            r, c = divmod(i, cols)
            t = cv2.resize(b, (thumb, thumb))
            cv2.putText(t, n[:18], (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 0, 255), 1)
            sheet[r * thumb:(r + 1) * thumb, c * thumb:(c + 1) * thumb] = t
        cv2.imwrite(os.path.join(args.out, "contact_sheet.png"), sheet)

    print(f"\n{nok}/{len(boards)} frame đạt ngưỡng ACCEPT={R.ACCEPT}")


if __name__ == "__main__":
    main()
