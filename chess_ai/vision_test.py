"""TỰ TEST THỊ GIÁC (không cần bàn thật) — kịch bản: camera soi MÀN HÌNH điện thoại.
Render thế cờ 2D, chạy pipeline nhận nước đi, so với nước thật. Báo độ chính xác.

Chạy:  python3 -m chess_ai.vision_test
"""
from __future__ import annotations
import argparse
import os
import random

import chess

from . import render, vision

try:
    import cv2
except Exception:
    cv2 = None

CAP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "captures"))


def _grid_from_board_direct(board):
    return vision.occupancy_grid(render.render_position(board))


def _grid_from_phone(board, jitter):
    img = render.render_position(board)
    frame = render.embed_in_phone(img, jitter=jitter)
    corners = vision.auto_detect_corners(frame)
    if corners is None:
        return None
    warped = vision.warp_board(frame, corners)
    return vision.occupancy_grid(warped)


def run(n=40, seed=3, mode="direct", jitter=0.0, save_samples=True):
    random.seed(seed)
    board = chess.Board()
    getgrid = (lambda b: _grid_from_phone(b, jitter)) if mode == "phone" else _grid_from_board_direct
    prev = getgrid(board)
    ok = tot = 0
    fails = []
    saved = False
    for i in range(n):
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = random.choice(legal)
        san = board.san(mv)
        bb = board.copy()
        board.push(mv)
        grid = getgrid(board)
        if grid is None:
            fails.append((san, mv.uci(), "KHONG_DO_DUOC_BAN"))
            prev = None
            continue
        det = vision.diff_to_move(bb, prev, grid) if prev is not None else None
        tot += 1
        if det == mv:
            ok += 1
        else:
            fails.append((san, mv.uci(), det.uci() if det else None))
        prev = grid

        # lưu vài ảnh mẫu để xem
        if save_samples and not saved and cv2 is not None and i == 4:
            os.makedirs(CAP_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(CAP_DIR, "render_board.png"), render.render_position(board))
            cv2.imwrite(os.path.join(CAP_DIR, "render_phone.png"),
                        render.embed_in_phone(render.render_position(board), jitter=jitter))
            saved = True

    acc = (100.0 * ok / tot) if tot else 0.0
    print(f"[{mode}] jitter={jitter}: nhận đúng {ok}/{tot} nước ({acc:.1f}%)")
    if fails:
        print("  Sai/không nhận:")
        for san, uci, det in fails[:8]:
            print(f"    thật={san}({uci})  ->  máy nhận={det}")
    return ok, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    print("=== TỰ TEST THỊ GIÁC (mô phỏng camera soi màn hình điện thoại) ===")
    run(args.n, args.seed, mode="direct")
    run(args.n, args.seed, mode="phone", jitter=0.0)
    run(args.n, args.seed, mode="phone", jitter=0.03)
    print(f"\nẢnh mẫu đã lưu ở: {CAP_DIR}/render_board.png , render_phone.png")


if __name__ == "__main__":
    main()
