"""Đọc thế cờ từ một ảnh camera TĨNH — kiểm tra cả chuỗi rectify → occupancy.

    python3 tools/read_frame.py captures/live_raw.jpg
    python3 tools/read_frame.py captures/*.jpg --bench

Phép kiểm khách quan: frame nào đang ở thế khai cuộc thì phải ra ĐÚNG FEN chuẩn
(mismatch = 0). Không cần nhìn ảnh, chỉ đọc số.

Cách chấm giống build_reference trong coach_server: thử thế đầu ván và cả 20 nước
đi đầu tiên, chọn thế cho occupancy tách bạch nhất.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import chess
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_ai import gridfind, piece_net, reader, rectify      # noqa: E402


def fit_best(board_img):
    """Thử 21 thế mở màn, trả (thế tốt nhất, model, margins, mismatch)."""
    sb = chess.Board()
    cands = [sb.copy()]
    for mv in sb.legal_moves:
        b = sb.copy()
        b.push(mv)
        cands.append(b)

    best = (None, None, None, 10 ** 9, -1e9)
    for flipped in (False, True):
        for b in cands:
            exp = gridfind.expected_mask(b, flipped=flipped)
            m, margins = gridfind.OccupancyModel.fit(board_img, exp)
            if not m.usable():
                continue
            occ = m.predict(board_img)
            d = int((exp != occ).sum())
            tot = sum(margins)
            if (d, -tot) < (best[3], -best[4]):
                best = (b, m, margins, d, tot)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--bench", action="store_true",
                    help="đo thêm tốc độ Tracker.update (vòng lặp video)")
    ap.add_argument("--model", default=None, help="đường dẫn piece_net.onnx")
    args = ap.parse_args()

    net = piece_net.load(args.model)
    print("PieceNet:", net.path if net else "KHÔNG CÓ (chỉ chạy đường occupancy)")

    for p in args.images:
        frame = cv2.imread(p)
        name = os.path.basename(p)
        if frame is None:
            print(f"{name:18s} KHÔNG ĐỌC ĐƯỢC")
            continue

        t0 = time.time()
        res = rectify.rectify(frame)
        t_det = (time.time() - t0) * 1000
        if not res.ok:
            print(f"{name:18s} score={res.score:5.2f} trong_khung={res.inside*100:5.1f}%"
                  f"  BỎ QUA (chưa tin được)")
            continue

        print(f"{name:18s} score={res.score:5.2f} trong_khung={res.inside*100:5.1f}%"
              f" nắn={t_det:5.0f}ms")

        if net is not None:
            t0 = time.time()
            grid, conf = net.predict(res.board)
            t_net = (time.time() - t0) * 1000
            wb = net.white_is_bottom(res.board)
            unsure = int((np.asarray(conf) < reader.CONF_MIN).sum())
            b_read = reader.board_from_grid(grid, flipped=not wb) if wb is not None else None
            info = (f"{'':18s}   PieceNet {t_net:4.0f}ms  trắng_ở_dưới={wb}  "
                    f"ô_mờ={unsure}  conf_min={np.min(conf):.2f}  ")
            if b_read is None:
                print(info + "thế cờ đọc được KHÔNG HỢP LỆ")
            else:
                std = (b_read.board_fen() == chess.Board().board_fen())
                print(info + ("= THẾ KHAI CUỘC CHUẨN" if std else b_read.board_fen()))

        b, m, margins, d, _ = fit_best(res.board)
        if b is None:
            print(f"{'':18s}   occupancy KHÔNG TÁCH ĐƯỢC ô quân/ô trống")
        else:
            mg = "/".join(f"{x:.2f}" for x in margins)
            tag = "KHỚP 100%" if d == 0 else f"lệch {d} ô"
            print(f"{'':18s}   occupancy margin={mg}  {tag}")

        if args.bench:
            tr = rectify.Tracker(res.quad, res.score)
            tr.update(frame)                       # bỏ lần đầu (khởi động cache)
            t0 = time.time()
            for _ in range(10):
                _, sc, mode = tr.update(frame)
            print(f"{'':18s} Tracker.update = {(time.time()-t0)*100:5.1f}ms/frame"
                  f"  ({mode}, score={sc:.2f})")


if __name__ == "__main__":
    main()
