"""TRAIN PieceNet — CNN nhỏ 13 lớp nhận diện quân cờ từng ô (64×64 BGR).

Chạy trên máy GPU (WSL):
    python train_piece_net.py            # train + export piece_net.onnx
    python train_piece_net.py --steps 200 --val 2000   # chạy thử nhanh

Dữ liệu sinh VÔ HẠN on-the-fly từ gen_cells.synth_cell (worker DataLoader).
Chuẩn hoá nằm TRONG model: input float32 0..255 → (x-127.5)/63.75
→ phía suy luận chỉ cần đưa ảnh uint8→float, không nhớ hệ số; và thuận lợi
lượng tử hoá QNN sau này (dải input cố định).
"""
from __future__ import annotations
import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

import gen_cells
from gen_cells import CLASSES, synth_cell


def load_real_cells(real_dir):
    """Nạp dataset THẬT do coach_server dump: cặp <ts>.png (warp) + <ts>.fen
    (dòng 1: FEN; dòng 2 tuỳ chọn: 'flipped=1' nếu Trắng ở TRÊN ảnh warp).
    → (cells float32 (N,3,64,64), labels (N,)). Mỗi frame đẻ 64 ô có nhãn."""
    import glob
    import cv2
    import chess
    xs, ys = [], []
    for fen_path in sorted(glob.glob(os.path.join(real_dir, "*.fen"))):
        png = fen_path[:-4] + ".png"
        img = cv2.imread(png)
        if img is None:
            continue
        lines = open(fen_path).read().split()
        board = chess.Board(" ".join(lines[:6]) if len(lines) >= 6 else lines[0] + " w - - 0 1")
        flipped = "flipped=1" in open(fen_path).read()
        W = img.shape[0]
        step = W / 8.0
        for r in range(8):
            for f in range(8):
                rank = r if flipped else (7 - r)
                file = (7 - f) if flipped else f
                pc = board.piece_at(chess.square(file, rank))
                sym = pc.symbol() if pc else "."
                c = cv2.resize(img[int(r * step):int((r + 1) * step),
                                   int(f * step):int((f + 1) * step)], (64, 64),
                               interpolation=cv2.INTER_AREA)
                xs.append(c.transpose(2, 0, 1).astype(np.float32))
                ys.append(CLASSES.index(sym))
    if not xs:
        return None, None
    return np.stack(xs), np.array(ys)


class SynthCells(IterableDataset):
    """Sinh mẫu vô hạn; nếu có dữ liệu thật thì trộn vào với xác suất real_p
    (ô thật được augment nhẹ: sáng/mờ/nhiễu — không phá nhãn)."""

    def __init__(self, seed=0, real=None, real_p=0.3):
        self.seed = seed
        self.real = real          # (xs, ys) hoặc None
        self.real_p = real_p

    def _augment_real(self, img_chw, rng):
        import cv2
        img = img_chw.transpose(1, 2, 0).astype(np.uint8).copy()
        if rng.random() < 0.5:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
        f = np.clip(img.astype(np.float32) * rng.uniform(0.8, 1.2)
                    + rng.uniform(-15, 15), 0, 255)
        if rng.random() < 0.5:
            f = np.clip(f + np.random.default_rng(rng.randrange(2**31))
                        .normal(0, rng.uniform(1, 5), f.shape), 0, 255)
        return f.transpose(2, 0, 1).astype(np.float32)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        rng = random.Random(self.seed * 1009 + wid)
        while True:
            if self.real is not None and rng.random() < self.real_p:
                i = rng.randrange(len(self.real[1]))
                yield torch.from_numpy(self._augment_real(self.real[0][i], rng)), \
                    int(self.real[1][i])
                continue
            img, y = synth_cell(rng)
            # BGR HWC uint8 -> CHW float32 (giữ 0..255, model tự chuẩn hoá)
            yield torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32)), y


def conv_block(ci, co, stride=1):
    return nn.Sequential(nn.Conv2d(ci, co, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class PieceNet(nn.Module):
    """~430k tham số — đủ nhỏ cho NPU/CPU, đủ lớn cho 13 lớp ảnh 64×64."""

    def __init__(self, n=13):
        super().__init__()
        self.body = nn.Sequential(
            conv_block(3, 32), conv_block(32, 32), nn.MaxPool2d(2),    # 32
            conv_block(32, 64), conv_block(64, 64), nn.MaxPool2d(2),   # 16
            conv_block(64, 128), conv_block(128, 128), nn.MaxPool2d(2),# 8
            conv_block(128, 128), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n)

    def forward(self, x):
        x = (x - 127.5) / 63.75
        return self.head(self.body(x).flatten(1))


def make_val(n, seed=999):
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        img, y = synth_cell(rng)
        xs.append(img.transpose(2, 0, 1).astype(np.float32))
        ys.append(y)
    return torch.from_numpy(np.stack(xs)), torch.tensor(ys)


@torch.no_grad()
def evaluate(model, xs, ys, dev, bs=1024):
    model.eval()
    correct = np.zeros(13, int)
    total = np.zeros(13, int)
    for i in range(0, len(xs), bs):
        p = model(xs[i:i + bs].to(dev)).argmax(1).cpu()
        y = ys[i:i + bs]
        for c in range(13):
            m = y == c
            total[c] += int(m.sum())
            correct[c] += int((p[m] == c).sum())
    acc = correct.sum() / max(1, total.sum())
    return acc, correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val", type=int, default=13000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default="piece_net")
    ap.add_argument("--real-dir", default=None,
                    help="thư mục dump (png+fen) từ coach_server để trộn dữ liệu thật")
    ap.add_argument("--real-p", type=float, default=0.3)
    ap.add_argument("--init", default=None, help="checkpoint .pt để fine-tune tiếp")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    gen_cells.load_assets()
    real = None
    if args.real_dir:
        rx, ry = load_real_cells(args.real_dir)
        if rx is not None:
            real = (rx, ry)
            print(f"dữ liệu thật: {len(ry)} ô từ {args.real_dir}")

    model = PieceNet().to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev, weights_only=True))
        print("fine-tune từ", args.init)
    print("params:", sum(p.numel() for p in model.parameters()))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)

    dl = DataLoader(SynthCells(real=real, real_p=args.real_p), batch_size=args.batch,
                    num_workers=args.workers, prefetch_factor=4, persistent_workers=True)
    print("sinh tập val...")
    vx, vy = make_val(args.val)

    model.train()
    t0 = time.time()
    it = iter(dl)
    best = 0.0
    for step in range(1, args.steps + 1):
        x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        loss = lossf(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            acc, cor, tot = evaluate(model, vx, vy, dev)
            model.train()
            print(f"step {step}/{args.steps} loss={loss.item():.4f} "
                  f"val_acc={acc*100:.2f}% ({time.time()-t0:.0f}s)")
            if acc > best:
                best = acc
                torch.save(model.state_dict(), args.out + ".pt")

    # nạp checkpoint tốt nhất, in chi tiết từng lớp
    model.load_state_dict(torch.load(args.out + ".pt", weights_only=True))
    acc, cor, tot = evaluate(model, vx, vy, dev)
    print(f"BEST val_acc={acc*100:.2f}%")
    for c in range(13):
        print(f"  {CLASSES[c]!r}: {cor[c]}/{tot[c]} = {100*cor[c]/max(1,tot[c]):.2f}%")

    # export ONNX (batch động — suy luận 64 ô 1 lần)
    model.cpu().eval()
    dummy = torch.zeros(1, 3, 64, 64)
    torch.onnx.export(model, dummy, args.out + ".onnx",
                      input_names=["cells"], output_names=["logits"],
                      dynamic_axes={"cells": {0: "n"}, "logits": {0: "n"}},
                      opset_version=17, dynamo=False)
    print("Đã export", args.out + ".onnx")


if __name__ == "__main__":
    main()
