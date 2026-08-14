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
import socket
import struct
import threading

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
    os.path.join(os.environ.get("KIT_ROOT", "/home/khacthu/kit"), "models", "piece_net.onnx"),
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

    # nhóm chỉ số lớp theo màu, tính sẵn
    _W = [CLASSES.index(c) for c in "PNBRQK"]
    _B = [CLASSES.index(c) for c in "pnbrqk"]

    def prob3(self, warped_bgr):
        """→ 8×8×3 xác suất {trống, trắng, đen}. Bên gọi chấm điểm bằng xác suất
        thay vì nhãn cứng thì một ô model lưỡng lự không giết được nước đi đúng."""
        lg = self.logits(warped_bgr)
        e = np.exp(lg - lg.max(axis=1, keepdims=True))
        pr = e / e.sum(axis=1, keepdims=True)
        return np.stack([pr[:, 0], pr[:, self._W].sum(1),
                         pr[:, self._B].sum(1)], 1).reshape(8, 8, 3)

    def read3(self, warped_bgr):
        """→ (state 8×8 ∈ {0 trống, 1 trắng, 2 đen}, conf 8×8).

        CỘNG DỒN xác suất theo màu, KHÔNG phải argmax 13 lớp rồi map: đọc nhầm
        Tốt thành Tượng vẫn ra đúng màu, mà đó đúng là loại lỗi model hay mắc
        trên frame thật. Ba lớp là đủ để bám nước đi — occupancy mù trước nước ăn
        quân (ô đích có quân cả trước lẫn sau), còn màu thì thấy rõ ô đích đổi
        chủ; quân gì thì không cần biết.
        """
        p3 = self.prob3(warped_bgr)
        idx = p3.argmax(2)
        return idx, p3.max(2)

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


class NpuPieceNet(PieceNet):
    """Cùng giao diện PieceNet nhưng suy luận trên NPU Hexagon qua sidecar.

    Sidecar `tools/npu_server.py` chạy python3.12 (API QAIRT bắt buộc), nên
    không nạp thẳng vào tiến trình này được. Đo trên AIBOX: 495ms → ~12ms.
    Mọi trục trặc socket đều rơi về `fallback` (ONNX CPU) chứ không ném lỗi
    lên vòng nhận diện.
    """

    def __init__(self, sock_path, fallback=None):
        self.path = sock_path
        self.fallback = fallback
        self._lock = threading.Lock()
        self.sock = None
        self._connect()

    def _connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect(self.path)
        self.sock = s

    def _rpc(self, payload):
        self.sock.sendall(struct.pack("<I", len(payload)) + payload)
        hdr = self.sock.recv(4)
        if len(hdr) != 4:
            raise ConnectionError("sidecar NPU dong ket noi")
        n = struct.unpack("<I", hdr)[0]
        if n == 0:
            raise RuntimeError("sidecar NPU bao loi suy luan")
        buf = bytearray()
        while len(buf) < n:
            b = self.sock.recv(n - len(buf))
            if not b:
                raise ConnectionError("sidecar NPU dut giua chung")
            buf += b
        return np.frombuffer(bytes(buf), np.float32).reshape(64, 13)

    def logits(self, warped_bgr):
        # cells() trả float32 nhưng giá trị là pixel 0..255 nguyên → uint8 không
        # mất mát, mà đường truyền nhẹ đi 4 lần (3MB → 786KB).
        cells = self.cells(warped_bgr).astype(np.uint8)
        with self._lock:
            try:
                return self._rpc(cells.tobytes())
            except Exception:
                try:                    # thử nối lại đúng 1 lần (sidecar restart)
                    self._connect()
                    return self._rpc(cells.tobytes())
                except Exception:
                    if self.fallback is None:
                        raise
                    self.sock = None
                    return self.fallback.logits(warped_bgr)


def load(path=None):
    """PieceNet | None (thiếu cv2/onnxruntime/model thì im lặng trả None).

    Ưu tiên NPU: nếu sidecar đang nghe ở KIT_NPU_SOCK thì dùng nó, giữ bản
    ONNX CPU làm lưới đỡ.
    """
    if cv2 is None:
        return None
    cpu = None
    cands = [path] if path else _MODEL_PATHS
    for p in cands:
        if p and os.path.exists(p):
            try:
                cpu = PieceNet(p)
            except Exception:
                cpu = None
            break
    sock = os.environ.get("KIT_NPU_SOCK", "/tmp/kit_npu.sock")
    if os.path.exists(sock):
        try:
            return NpuPieceNet(sock, fallback=cpu)
        except Exception:
            pass
    return cpu
