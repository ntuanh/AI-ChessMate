"""ĐÁNH GIÁ piece_net.onnx trên dữ liệu THẬT do coach_server dump (png + fen)
và đo tốc độ suy luận 1 bàn (64 ô).

    python3 eval_real.py --model piece_net.onnx --data /data/kit/captures/dataset
    python3 eval_real.py --model piece_net.onnx --bench        # chỉ đo tốc độ

Chạy được cả trên laptop lẫn AIBOX (chỉ cần onnxruntime + cv2 + numpy + chess).
"""
from __future__ import annotations
import argparse
import glob
import os
import time

import numpy as np
import cv2
import onnxruntime as ort

CLASSES = ".PNBRQKpnbrqk"
CELL = 64


def cells_of(warped):
    W = warped.shape[0]
    step = W / 8.0
    out = np.empty((64, 3, CELL, CELL), np.float32)
    for r in range(8):
        for f in range(8):
            c = cv2.resize(warped[int(r * step):int((r + 1) * step),
                                  int(f * step):int((f + 1) * step)],
                           (CELL, CELL), interpolation=cv2.INTER_AREA)
            out[r * 8 + f] = c.transpose(2, 0, 1)
    return out


def truth_grid(fen_path):
    import chess
    txt = open(fen_path).read()
    lines = txt.split()
    board = chess.Board(" ".join(lines[:6]) if len(lines) >= 6 else lines[0] + " w - - 0 1")
    flipped = "flipped=1" in txt
    g = []
    for r in range(8):
        for f in range(8):
            rank = r if flipped else (7 - r)
            file = (7 - f) if flipped else f
            pc = board.piece_at(chess.square(file, rank))
            g.append(pc.symbol() if pc else ".")
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="piece_net.onnx")
    ap.add_argument("--data", default=None)
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    if args.bench or not args.data:
        x = np.random.randint(0, 255, (64, 3, CELL, CELL)).astype(np.float32)
        for _ in range(3):
            sess.run(None, {"cells": x})            # làm nóng
        t0 = time.time()
        n = 30
        for _ in range(n):
            sess.run(None, {"cells": x})
        dt = (time.time() - t0) / n * 1000
        print(f"suy luận 64 ô: {dt:.1f} ms/bàn ({1000/dt:.1f} bàn/s)")
        if not args.data:
            return

    pairs = sorted(glob.glob(os.path.join(args.data, "*.fen")))
    if not pairs:
        print("Không có dữ liệu ở", args.data)
        return
    tot = np.zeros(13, int)
    cor = np.zeros(13, int)
    n_cell = n_frame = frame_ok = 0
    occ_err = col_err = 0
    for fp in pairs:
        img = cv2.imread(fp[:-4] + ".png")
        if img is None:
            continue
        truth = truth_grid(fp)
        lg = sess.run(None, {"cells": cells_of(img)})[0]
        pred = [CLASSES[i] for i in lg.argmax(1)]
        n_frame += 1
        if pred == truth:
            frame_ok += 1
        for p, t in zip(pred, truth):
            c = CLASSES.index(t)
            tot[c] += 1
            cor[c] += int(p == t)
            n_cell += 1
            if (p == ".") != (t == "."):
                occ_err += 1
            elif p != "." and p.isupper() != t.isupper():
                col_err += 1
    acc = cor.sum() / max(1, tot.sum())
    print(f"{n_frame} frame, {n_cell} ô | ô đúng {acc*100:.2f}% | "
          f"frame đúng cả 64 ô: {frame_ok}/{n_frame}")
    print(f"lỗi occupancy: {occ_err} | lỗi màu (đúng chỗ, sai màu): {col_err}")
    for c in range(13):
        if tot[c]:
            print(f"  {CLASSES[c]!r}: {cor[c]}/{tot[c]} = {100*cor[c]/tot[c]:.2f}%")


if __name__ == "__main__":
    main()
