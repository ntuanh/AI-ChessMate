"""PieceNet — nhận diện TỪNG QUÂN mỗi ô bằng CNN 13 lớp (ONNX, chạy onnxruntime).

Nâng cấp so với OccupancyModel (chỉ biết ô có quân hay không):
  - biết CHÍNH XÁC quân gì, MÀU gì ở từng ô  → dựng FEN, bắt phong cấp,
    xác định hướng bàn bằng màu quân tin cậy hơn heuristic độ tối.
  - trả kèm CONFIDENCE từng ô → phía gọi tự quyết ngưỡng tin.

Model: models/piece_net.onnx (train ở tools/train/, xem README trong đó).
Input:  (N,3,64,64) float32 BGR 0..255 (chuẩn hoá nằm TRONG model).
Output: (N,13) logits theo thứ tự lớp ".PNBRQKpnbrqk".

Module này KHÔNG bắt buộc: thiếu onnxruntime/model → load() trả None,
pipeline giữ nguyên OccupancyModel cũ.
"""
from __future__ import annotations
import os

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

CLASSES = ".PNBRQKpnbrqk"
CELL = 64

# thứ tự tìm model: cạnh repo (dev laptop) rồi chỗ deploy trên AIBOX
_MODEL_PATHS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "piece_net.onnx"),
    "/data/kit/models/piece_net.onnx",
]


class PieceNet:
    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.path = path

    # ---------- tách ô ----------
    @staticmethod
    def cells(warped_bgr):
        """Ảnh bàn đã warp (vuông) → (64,3,64,64) float32, thứ tự hàng-trước (r*8+f)."""
        W = warped_bgr.shape[0]
        step = W / 8.0
        out = np.empty((64, 3, CELL, CELL), np.float32)
        for r in range(8):
            for f in range(8):
                y0, y1 = int(r * step), int((r + 1) * step)
                x0, x1 = int(f * step), int((f + 1) * step)
                c = cv2.resize(warped_bgr[y0:y1, x0:x1], (CELL, CELL),
                               interpolation=cv2.INTER_AREA)
                out[r * 8 + f] = c.transpose(2, 0, 1)
        return out

    # ---------- suy luận ----------
    def logits(self, warped_bgr):
        return self.sess.run(None, {"cells": self.cells(warped_bgr)})[0]

    def predict(self, warped_bgr):
        """→ (grid 8×8 ký hiệu '.'/'P'/…​, conf 8×8 float xác suất lớp thắng)."""
        lg = self.logits(warped_bgr)
        e = np.exp(lg - lg.max(axis=1, keepdims=True))
        prob = e / e.sum(axis=1, keepdims=True)
        idx = prob.argmax(axis=1)
        conf = prob[np.arange(64), idx].reshape(8, 8)
        grid = [[CLASSES[idx[r * 8 + f]] for f in range(8)] for r in range(8)]
        return grid, conf

    def occupancy(self, warped_bgr, conf_min=0.0):
        """8×8 bool CÓ QUÂN — thay thế trực tiếp OccupancyModel.predict."""
        grid, conf = self.predict(warped_bgr)
        return np.array([[grid[r][f] != "." and conf[r][f] >= conf_min
                          for f in range(8)] for r in range(8)], bool)

    def white_is_bottom(self, warped_bgr):
        """Hướng bàn theo MÀU QUÂN do CNN đọc: đếm quân Trắng nửa dưới vs nửa trên.
        Trả về True/False, hoặc None nếu chưa đủ tin (ít quân đọc được)."""
        grid, conf = self.predict(warped_bgr)
        top = bot = 0.0
        for r in range(8):
            for f in range(8):
                s = grid[r][f]
                if s == "." or conf[r][f] < 0.5:
                    continue
                w = 1.0 if s.isupper() else -1.0
                if r < 4:
                    top += w
                else:
                    bot += w
        if abs(bot - top) < 4:      # chênh lệch quá mỏng -> không kết luận
            return None
        return bot > top


def load(path=None):
    """PieceNet | None (thiếu cv2/onnxruntime/model thì im lặng trả None)."""
    if cv2 is None:
        return None
    cands = [path] if path else _MODEL_PATHS
    for p in cands:
        if p and os.path.exists(p):
            try:
                return PieceNet(p)
            except Exception:
                return None
    return None
