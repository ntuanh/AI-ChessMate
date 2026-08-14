#!/usr/bin/env python3
"""THU DATA THẬT để train CNN occupancy đúng domain của bạn.
Chạy trên LAPTOP trong lúc bạn chơi (server dashboard phiên kia đang bám bàn tốt).
Mỗi vài giây: lấy /raw (khung camera) + góc bàn + nước đã đi từ http://localhost:8090,
tự warp + gán nhãn 64 ô (từ game state) → lưu ảnh ô + nhãn vào captures/occ_data/.

Chỉ thu khi bám TỐT (caro cao) để nhãn đáng tin. Sau đó train lại trên GPU.
"""
import argparse, io, json, os, time, urllib.request
import numpy as np
import cv2
import chess

WARP = 800
STEP = 100


def get(url, binary=False):
    with urllib.request.urlopen(url, timeout=4) as r:
        return r.read() if binary else json.loads(r.read())


def build_board_from_sans(sans):
    b = chess.Board()
    for s in sans:
        try:
            b.push_san(s)
        except Exception:
            break
    return b


def occ_mask(board, flipped=False):
    g = np.zeros((8, 8), np.int64)
    for sq in chess.SQUARES:
        if board.piece_at(sq) is None:
            continue
        r = chess.square_rank(sq); f = chess.square_file(sq)
        row = r if flipped else (7 - r)
        col = (7 - f) if flipped else f
        g[row][col] = 1
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8090")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "captures", "occ_data"))
    ap.add_argument("--min-caro", type=float, default=6.0)
    ap.add_argument("--interval", type=float, default=1.2)
    ap.add_argument("--max", type=int, default=8000, help="dừng khi đủ số ô")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)

    saved = 0
    idx = len(os.listdir(out))
    last_sig = None
    print(f"Bắt đầu thu vào {out}. Chơi trên Lichess đi, tôi thu khi bám tốt (caro>={args.min_caro}).", flush=True)
    while saved < args.max:
        time.sleep(args.interval)
        try:
            st = get(args.url + "/status")
        except Exception:
            continue
        caro = st.get("caro", 0) or 0
        corners = st.get("corners")
        if caro < args.min_caro or not corners or not st.get("ready"):
            continue
        moves = st.get("moves", [])
        sig = (len(moves), round(caro, 1))
        if sig == last_sig:            # chưa đổi gì -> đỡ trùng
            continue
        last_sig = sig
        try:
            raw = get(args.url + "/raw", binary=True)
            frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            corners = np.array(corners, dtype=np.float32).reshape(4, 2)
            M = cv2.getPerspectiveTransform(
                corners, np.array([[0, 0], [WARP, 0], [WARP, WARP], [0, WARP]], np.float32))
            w = cv2.warpPerspective(frame, M, (WARP, WARP))
            board = build_board_from_sans(moves)
            # hướng: đoán theo st.get flip? mặc định trắng dưới
            mask = occ_mask(board, flipped=False)
        except Exception as e:
            continue
        for r in range(8):
            for f in range(8):
                m = 14
                cell = w[r * STEP + m:(r + 1) * STEP - m, f * STEP + m:(f + 1) * STEP - m]
                if cell.size == 0:
                    continue
                lbl = int(mask[r][f])
                fn = os.path.join(out, f"{idx:06d}_{lbl}.png")
                cv2.imwrite(fn, cv2.resize(cell, (32, 32)))
                idx += 1; saved += 1
        print(f"  +64 ô (nước {len(moves)}, caro {caro:.1f}) — tổng {saved}", flush=True)
    print(f"✅ Xong: {saved} ô thật tại {out}", flush=True)


if __name__ == "__main__":
    main()
