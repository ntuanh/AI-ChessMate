"""TEST DÒ LƯỚI + OCCUPANCY cho camera DI ĐỘNG.

  python3 -m chess_ai.gridfind_test real    # chạy trên ảnh THẬT trong captures/
  python3 -m chess_ai.gridfind_test synth   # mô phỏng 40 nước, camera RUNG mỗi frame
"""
from __future__ import annotations
import argparse
import glob
import os
import random

import numpy as np
import chess

from chess_ai import gridfind, render, vision

try:
    import cv2
except Exception:
    cv2 = None

CAP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "captures"))
OUT_DIR = os.path.join(CAP_DIR, "gridtest")

REAL_FRAMES = ["cur.jpg", "r2.jpg", "live_now.jpg", "live_raw.jpg", "r720.jpg",
               "read_now.png", "coach_raw.png", "watch_raw.png"]


def _occupancy_unsupervised(warped):
    """Không có nhãn (ảnh giữa ván) -> tách 2 cụm trên đặc trưng lệch-nền."""
    feats = gridfind._features(warped)[:, :, 1].ravel()
    lo, hi = feats.min(), feats.max()
    t = (lo + hi) / 2.0
    for _ in range(10):
        a, b = feats[feats <= t], feats[feats > t]
        if not len(a) or not len(b):
            break
        t = (a.mean() + b.mean()) / 2.0
    return (feats > t).reshape(8, 8)


def run_real():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"{'ảnh':<16}{'caro':>7}  {'nét':>6}  {'ô quân':>7}  kết quả")
    for name in REAL_FRAMES:
        path = os.path.join(CAP_DIR, name)
        if not os.path.exists(path):
            continue
        frame = cv2.imread(path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = gridfind.sharpness(gray)
        found = gridfind.find_board(frame)
        if found is None:
            print(f"{name:<16}{'—':>7}  {sharp:>6.0f}  {'—':>7}  KHÔNG DÒ ĐƯỢC")
            continue
        corners, score = found
        warped = gridfind.warp(frame, corners, vision.WARP)
        grid = _occupancy_unsupervised(warped)
        occ = int(grid.sum())

        step = vision.WARP // 8
        vis = warped.copy()
        for i in range(9):
            cv2.line(vis, (i * step, 0), (i * step, vision.WARP), (0, 0, 255), 2)
            cv2.line(vis, (0, i * step), (vision.WARP, i * step), (0, 0, 255), 2)
        for r in range(8):
            for f in range(8):
                if grid[r][f]:
                    cv2.circle(vis, (f * step + step // 2, r * step + step // 2),
                               8, (255, 0, 0), -1)
        out = os.path.join(OUT_DIR, "grid_" + os.path.splitext(name)[0] + ".png")
        cv2.imwrite(out, vis)
        print(f"{name:<16}{score:>7.1f}  {sharp:>6.0f}  {occ:>7}  -> {os.path.relpath(out, CAP_DIR)}")


def run_synth(n=40, seed=3, jmax=0.05):
    """Camera 'rung': MỖI frame một phối cảnh jitter khác nhau -> ép đi đường dò-lại
    + orient_to_board liên tục (không có tracker/continuity nào cứu)."""
    random.seed(seed)
    board = chess.Board()

    def frame_of(b):
        j = random.uniform(0.0, jmax)
        return render.embed_in_phone(render.render_position(b), jitter=j)

    # --- hiệu chỉnh từ thế khai cuộc ---
    f0 = frame_of(board)
    found = gridfind.find_board(f0)
    assert found is not None, "khong do duoc ban o the khai cuoc"
    corners, score = found
    warped = gridfind.warp(f0, corners, vision.WARP)
    model, margins = gridfind.OccupancyModel.fit(warped)
    assert model.usable(), f"khong tach duoc quan/trong: margins={margins}"
    if not gridfind.white_is_bottom(warped):
        corners = gridfind.rot_corners(corners, 2)
        warped = gridfind.warp(f0, corners, vision.WARP)
        model, _ = gridfind.OccupancyModel.fit(warped)
    g0 = model.predict(warped)
    sync = int((g0 == gridfind.START_MASK).sum())
    print(f"[synth] hiệu chỉnh: caro={score:.1f}, margins={['%.2f' % m for m in margins]}, "
          f"khớp khai cuộc {sync}/64")
    assert sync >= 62, "occupancy khai cuoc lech"

    # --- 40 nước, camera rung ---
    prev = g0
    ok = tot = lost = 0
    fails = []
    for _ in range(n):
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = random.choice(legal)
        bb = board.copy()
        board.push(mv)

        fr = frame_of(board)
        found = gridfind.find_board(fr)
        grid = None
        if found is not None:
            c = gridfind.orient_to_board(fr, found[0], model, gridfind.expected_mask(board))
            if c is not None:
                grid = model.predict(gridfind.warp(fr, c, vision.WARP))
        tot += 1
        if grid is None:
            lost += 1
            fails.append((bb.san(mv), "MAT_BAN"))
            prev = None
            continue
        det = vision.diff_to_move(bb, prev, grid) if prev is not None else None
        if det == mv:
            ok += 1
        else:
            fails.append((bb.san(mv), det.uci() if det else None))
        prev = grid

    print(f"[synth] camera rung jitter≤{jmax}: đúng {ok}/{tot} nước, mất bàn {lost} lần")
    for san, det in fails[:8]:
        print(f"    thật={san}  máy={det}")
    return ok, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["real", "synth", "all"], nargs="?", default="all")
    args = ap.parse_args()
    if args.mode in ("real", "all"):
        run_real()
    if args.mode in ("synth", "all"):
        run_synth()


if __name__ == "__main__":
    main()
