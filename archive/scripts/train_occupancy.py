#!/usr/bin/env python3
"""Train CNN NHỊ PHÂN OCCUPANCY (ô có quân / ô trống) — chạy trên máy GPU (hoặc CPU i9).
Tự sinh data: render bàn cờ ngẫu nhiên + AUGMENT mạnh (tối, loá, mờ, nhiễu, lệch, color
jitter) mô phỏng camera soi màn hình nghiêng. Học đặc trưng 'ô phẳng=trống / ô có hình=quân'
— tín hiệu bất biến domain nên chuyển sang Lichess thật tốt. Xuất ONNX cho AIBOX.

Chạy:  python train_occupancy.py --n 4000 --epochs 8 --out occ_cnn.onnx
"""
import argparse, random, math
import numpy as np
import cv2
import chess
import torch
import torch.nn as nn

CELL = 32  # kích thước ô đưa vào CNN

LIGHT = (181, 217, 240); DARK = (99, 136, 181)      # màu bàn (BGR) kiểu Lichess
# nhiều theme để CNN không học tủ 1 màu
THEMES = [((181, 217, 240), (99, 136, 181)),        # nâu gỗ
          ((168, 184, 118), (98, 122, 60)),         # xanh green
          ((210, 210, 210), (130, 130, 130)),       # xám
          ((200, 230, 240), (120, 160, 190))]       # be sáng


def rand_piece(img, cx, cy, r):
    """Vẽ 'một quân' hình dạng ngẫu nhiên (đại diện cho 'ô có nội dung')."""
    white = random.random() < 0.5
    body = (random.randint(225, 255),) * 3 if white else (random.randint(20, 55),) * 3
    outline = (30, 30, 30) if white else (210, 210, 210)
    shape = random.random()
    if shape < 0.6:                       # tròn (đầu quân)
        cv2.circle(img, (cx, cy), r, body, -1)
        cv2.circle(img, (cx, cy), r, outline, 2)
    elif shape < 0.85:                    # tam giác (tốt/mã cách điệu)
        pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
        cv2.fillPoly(img, [pts], body); cv2.polylines(img, [pts], True, outline, 2)
    else:                                 # chữ nhật (xe)
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), body, -1)
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), outline, 2)


def render_board(board, size=512, theme=None):
    light, dark = theme or random.choice(THEMES)
    step = size // 8
    img = np.zeros((size, size, 3), np.uint8)
    for r in range(8):
        for f in range(8):
            y0, x0 = r * step, f * step
            img[y0:y0 + step, x0:x0 + step] = light if (r + f) % 2 == 0 else dark
            sq = chess.square(f, 7 - r)
            if board.piece_at(sq) is not None:
                rr = int(step * random.uniform(0.30, 0.40))
                jx = random.randint(-3, 3); jy = random.randint(-3, 3)
                rand_piece(img, x0 + step // 2 + jx, y0 + step // 2 + jy, rr)
    return img, step


def augment(cell):
    """AUGMENT mạnh mô phỏng camera soi màn hình: mờ, tối, loá, nhiễu, lệch, jitter."""
    h, w = cell.shape[:2]
    # lệch nhỏ (misalignment sau warp)
    M = np.float32([[1, 0, random.randint(-4, 4)], [0, 1, random.randint(-4, 4)]])
    cell = cv2.warpAffine(cell, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    # nghiêng nhẹ (perspective còn dư)
    if random.random() < 0.5:
        d = random.uniform(0, 0.06) * w
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = src + np.float32([[d, d], [-d, d], [-d, -d], [d, -d]]) * random.choice([-1, 1])
        cell = cv2.warpPerspective(cell, cv2.getPerspectiveTransform(src, dst), (w, h),
                                   borderMode=cv2.BORDER_REPLICATE)
    # mờ
    if random.random() < 0.7:
        k = random.choice([3, 3, 5]); cell = cv2.GaussianBlur(cell, (k, k), 0)
    # sáng/tương phản (tối, bệt màu như ảnh camera)
    alpha = random.uniform(0.45, 1.25); beta = random.uniform(-40, 30)
    cell = cv2.convertScaleAbs(cell, alpha=alpha, beta=beta)
    # color shift
    cell = cv2.cvtColor(cell, cv2.COLOR_BGR2HSV).astype(np.int16)
    cell[..., 0] = (cell[..., 0] + random.randint(-8, 8)) % 180
    cell[..., 1] = np.clip(cell[..., 1] * random.uniform(0.6, 1.2), 0, 255)
    cell = cv2.cvtColor(np.clip(cell, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    # loá (blob sáng)
    if random.random() < 0.25:
        gx, gy = random.randint(0, w), random.randint(0, h)
        ov = cell.copy(); cv2.circle(ov, (gx, gy), random.randint(4, 12), (255, 255, 255), -1)
        cell = cv2.addWeighted(cell, 0.7, ov, 0.3, 0)
    # nhiễu
    if random.random() < 0.6:
        cell = np.clip(cell.astype(np.int16) +
                       np.random.randint(-14, 14, cell.shape), 0, 255).astype(np.uint8)
    return cell


def gen_dataset(n_boards):
    X, Y = [], []
    for _ in range(n_boards):
        b = chess.Board()
        for _ in range(random.randint(0, 40)):        # đi ngẫu nhiên ra thế giữa ván
            mv = list(b.legal_moves)
            if not mv or b.is_game_over():
                break
            b.push(random.choice(mv))
        img, step = render_board(b)
        for r in range(8):
            for f in range(8):
                m = int(step * random.uniform(0.10, 0.18))
                y0, x0 = r * step + m, f * step + m
                cell = img[y0:y0 + step - 2 * m, x0:x0 + step - 2 * m]
                cell = cv2.resize(cell, (CELL, CELL))
                cell = augment(cell)
                g = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                X.append(g)
                sq = chess.square(f, 7 - r)
                Y.append(1 if b.piece_at(sq) is not None else 0)
    X = np.array(X, np.float32)[:, None, :, :]         # (N,1,32,32)
    return X, np.array(Y, np.int64)


class OccNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.BatchNorm2d(12), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(24, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="số bàn cờ sinh (mỗi bàn 64 ô)")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="occ_cnn.onnx")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị: {dev}. Sinh {args.n} bàn (~{args.n*64} ô)...", flush=True)
    X, Y = gen_dataset(args.n)
    Xv, Yv = gen_dataset(max(200, args.n // 10))       # tập kiểm tra
    print(f"Train: {len(X)} ô ({Y.mean()*100:.0f}% có quân). Val: {len(Xv)} ô.", flush=True)

    Xt = torch.tensor(X); Yt = torch.tensor(Y)
    Xvt = torch.tensor(Xv).to(dev); Yvt = torch.tensor(Yv).to(dev)
    net = OccNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    lossf = nn.CrossEntropyLoss()
    bs = 512
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            xb = Xt[idx].to(dev); yb = Yt[idx].to(dev)
            opt.zero_grad(); out = net(xb); l = lossf(out, yb); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = (net(Xvt).argmax(1) == Yvt).float().mean().item()
        print(f"  epoch {ep+1}/{args.epochs}  val_acc={acc*100:.2f}%", flush=True)

    net.eval()
    dummy = torch.randn(1, 1, CELL, CELL).to(dev)
    torch.onnx.export(net, dummy, args.out, input_names=["cell"], output_names=["logits"],
                      dynamic_axes={"cell": {0: "batch"}, "logits": {0: "batch"}}, opset_version=12)
    print(f"✅ Đã xuất ONNX: {args.out}  (val_acc cuối {acc*100:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
