#!/usr/bin/env python3
"""Live view HIỆU CHỈNH: vừa xem camera vừa dò bàn cờ theo thời gian thực.
- Viền XANH = đã dò được bàn (căn tốt). Viền ĐỎ chữ = chưa dò được (lùi ra/để cả bàn trong khung).
- Khi dò ổn định vài khung -> TỰ LƯU hiệu chỉnh vào captures/board_calib.json.
Mở http://127.0.0.1:8090 trên laptop.
"""
import os, sys, time
import numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.environ.get("KIT_ROOT", "/home/khacthu/kit"))
from chess_ai import vision
import cv2

cap = vision.open_camera(2)
_stable = {"n": 0, "corners": None, "saved": False}


def process(frame):
    raw = frame.copy()
    corners = vision.auto_detect_corners(raw)
    b = float(np.mean(frame))
    if corners is not None:
        cv2.polylines(frame, [corners.astype(int)], True, (0, 255, 0), 3)
        try:
            warped = vision.warp_board(raw, corners)
            occ = int(vision.occupancy_grid(warped).sum())
        except Exception:
            occ = -1
        cv2.putText(frame, f"DA DO DUOC BAN - o co quan: {occ} (dau van ~32)",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # ổn định -> lưu
        prev = _stable["corners"]
        if prev is not None and np.mean(np.abs(prev - corners)) < 12:
            _stable["n"] += 1
        else:
            _stable["n"] = 0
        _stable["corners"] = corners
        if _stable["n"] >= 6 and not _stable["saved"]:
            vision.save_calibration(corners)
            with open(os.path.join(os.path.dirname(vision.CALIB_FILE), "board_rotate.txt"), "w") as f:
                f.write("0")
            _stable["saved"] = True
        if _stable["saved"]:
            cv2.putText(frame, "DA LUU HIEU CHINH! San sang choi.",
                        (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        msg = "CHUA DO DUOC - lui camera ra, de CA ban trong khung, bot nghieng/loa"
        color = (0, 0, 255) if b > 20 else (0, 140, 255)
        cv2.putText(frame, msg, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"SANG:{b:.0f}", (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    ok, f = cap.read()
                    if not ok: continue
                    f = process(f)
                    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ok: continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers()
            self.wfile.write(b"<body style='background:#111;text-align:center'><h3 style='color:#eee;font-family:sans-serif'>Hieu chinh ban co - canh toi khi vien XANH</h3><img src='/stream' style='max-width:100%'></body>")


if __name__ == "__main__":
    print("calib stream on :8090", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8090), H).serve_forever()
