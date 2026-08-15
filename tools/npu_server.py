#!/usr/bin/env python3
"""Sidecar chạy PieceNet trên NPU Hexagon (QNN HTP) — nghe unix socket.

Vì sao tách tiến trình: API python của QAIRT **bắt buộc python 3.12**, còn
coach_server chạy python 3.10 (nơi có cv2/chess/onnxruntime). Sidecar này chạy
bằng <KIT_ROOT>/qnnenv/bin/python (3.12) và chỉ cần numpy.

Giao thức (unix stream socket, /tmp/kit_npu.sock):
    gửi:  4 byte little-endian độ dài  +  64*3*64*64 byte uint8 (cells)
    nhận: 4 byte little-endian độ dài  +  64*13 float32 (logits)
    độ dài = 0  → lỗi phía server, client tự lùi về CPU.

Graph nạp 1 lần (~1.2 s) rồi giữ; mỗi lần suy luận ~4.7 ms.
"""
from __future__ import annotations
import os
import socket
import struct
import sys
import threading

import numpy as np

SOCK = os.environ.get("KIT_NPU_SOCK", "/tmp/kit_npu.sock")
KIT_ROOT = os.environ.get("KIT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DLC = os.environ.get("KIT_NPU_DLC") or os.path.join(KIT_ROOT, "qnn_build", "piece_net_q.dlc")
N_IN = 64 * 3 * 64 * 64
N_OUT = 64 * 13


def build_inferencer():
    import qti.aisw.core.model_level_api as mlapi
    from qti.aisw.tools.core.modules.api.backend.htp_backend import HtpBackend
    from qti.aisw.tools.core.modules.api.definitions.common import ModelConfig
    be = HtpBackend(target=mlapi.OELinuxTarget())
    inf = mlapi.Inferencer(be, ModelConfig(path=DLC), executor=mlapi.NativeExecutor())
    inf.setup()
    return inf


def recvall(c, n):
    buf = bytearray()
    while len(buf) < n:
        b = c.recv(n - len(buf))
        if not b:
            return None
        buf += b
    return buf


def serve(inf, lock, c):
    with c:
        while True:
            hdr = recvall(c, 4)
            if hdr is None:
                return
            n = struct.unpack("<I", hdr)[0]
            body = recvall(c, n)
            if body is None or n != N_IN:
                return
            x = np.frombuffer(body, np.uint8).astype(np.float32).reshape(64, 3, 64, 64)
            try:
                with lock:                       # graph QNN không an toàn đa luồng
                    out = inf.run({"cells": x})
                lg = np.asarray(list(out[0].values())[0], np.float32).reshape(N_OUT)
                c.sendall(struct.pack("<I", lg.nbytes) + lg.tobytes())
            except Exception as e:               # lỗi NPU: báo rỗng, client lùi CPU
                print(f"[npu] loi suy luan: {e!r}", flush=True)
                c.sendall(struct.pack("<I", 0))


def main():
    # Xoá socket cũ NGAY, trước khi nạp graph (~20s): nếu để nguyên thì suốt thời
    # gian nạp, `test -S` của bên ngoài vẫn thấy socket chết và tưởng đã sẵn sàng,
    # nên coach_server nối vào rồi thất bại rồi lùi về CPU.
    if os.path.exists(SOCK):
        os.unlink(SOCK)
    inf = build_inferencer()
    print("[npu] graph san sang", flush=True)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(SOCK)
    os.chmod(SOCK, 0o666)
    s.listen(8)
    lock = threading.Lock()
    print(f"[npu] nghe tai {SOCK}", flush=True)
    while True:
        c, _ = s.accept()
        threading.Thread(target=serve, args=(inf, lock, c), daemon=True).start()


if __name__ == "__main__":
    sys.exit(main())
