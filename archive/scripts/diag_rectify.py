"""Chẩn đoán một frame khó: xem điểm caro của từng nguồn mồi, và so với QUAD THẬT
do người chỉ định.

    python3 tools/diag_rectify.py captures/chk1.jpg \
        --quad 124,12,537,25,500,370,86,355

Mục đích là tách hai loại lỗi thường bị lẫn:
  * quad thật ĐIỂM THẤP  → lỗi ở cách chấm điểm (che khuất, loá, mờ)
  * quad thật ĐIỂM CAO nhưng dò không ra → lỗi ở bước sinh mồi / tìm kiếm
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_ai import rectify as R      # noqa: E402


def stats(q, h, w):
    q = np.asarray(q, np.float32)
    a = cv2.contourArea(q)
    e = np.array([np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)])
    return (f"dt={a/(h*w)*100:5.1f}%  canh={e.max()/e.min():4.2f}  "
            f"det={a/max(e.mean()**2,1):4.2f}  ok={R._quad_ok(q, h, w)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--quad", help="x1,y1,x2,y2,x3,y3,x4,y4 (thang ảnh gốc)")
    ap.add_argument("--out", default="/tmp/rect")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frame = cv2.imread(args.image)
    h0, w0 = frame.shape[:2]
    sc = 640.0 / w0
    small = cv2.resize(frame, (640, int(round(h0 * sc))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gh, gw = gray.shape[:2]
    print(f"{args.image}  {w0}x{h0} -> {gw}x{gh}\n")

    for label, seeds in (("HÌNH HỌC", R.candidates(gray)),
                         ("RẢI KHẮP KHUNG", R._grid_seeds(gh, gw))):
        seeds = [q for q in seeds if R._quad_ok(q, gh, gw)]
        seeds.sort(key=lambda q: -R.board_likeness(gray, q))
        print(f"--- {label}: {len(seeds)} mồi hợp lệ, xét 8 mồi dẫn đầu ---")
        rows = [(R.board_likeness(gray, q), R.score_quad(gray, q)) + R.refine(gray, q)
                for q in seeds[:8]]
        for lk, s0, rq, rs in sorted(rows, key=lambda x: -x[3]):
            print(f"  giống_bàn={lk:5.3f}  caro_thô={s0:6.2f} -> tinh chỉnh={rs:6.2f}"
                  f"   {stats(rq, gh, gw)}")
        print()

    if args.quad:
        v = [float(x) for x in args.quad.split(",")]
        q0 = R.order_quad(np.array(v, np.float32).reshape(4, 2) * sc)
        s0 = R.score_quad(gray, q0)
        rq, rs = R.refine(gray, q0)
        print(f"--- QUAD THẬT (người chỉ định) ---")
        print(f"  nguyên trạng ={s0:6.2f}   {stats(q0, gh, gw)}")
        print(f"  sau tinh chỉnh={rs:6.2f}   {stats(rq, gh, gw)}")
        base = os.path.basename(args.image)
        for tag, q in (("truth", q0), ("truth_refined", rq)):
            b = R.normalize_colors(R.warp(small, q, R.SIZE))
            for i in range(9):
                p = i * R.CELL
                cv2.line(b, (p, 0), (p, R.SIZE), (0, 0, 255), 1)
                cv2.line(b, (0, p), (R.SIZE, p), (0, 0, 255), 1)
            cv2.imwrite(os.path.join(args.out, f"{base}.{tag}.png"), b)
        print(f"  đã ghi {base}.truth.png / {base}.truth_refined.png")


if __name__ == "__main__":
    main()
