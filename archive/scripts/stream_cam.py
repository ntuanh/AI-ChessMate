#!/usr/bin/env python3
"""Live MJPEG stream cua camera AIBOX de xem tren trinh duyet laptop (qua adb forward).
Mo http://127.0.0.1:8090 tren laptop. Co overlay do sang + luoi canh giua de aim camera."""
import cv2, numpy as np
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CAM_INDEX = 2  # C270 = /dev/video2
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>AIBOX Camera Live</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0}
img{max-width:100%;height:auto;border:2px solid #444;margin-top:8px}
h3{margin:8px}</style></head>
<body><h3>Camera AIBOX (C270) - Live</h3>
<img src="/stream"><p>Do sang hien tren goc trai. Canh sao cho ban co nam giua khung.</p>
</body></html>"""

def draw_overlay(f):
    b = float(np.mean(f))
    h, w = f.shape[:2]
    # luoi canh giua
    cv2.line(f, (w // 2, 0), (w // 2, h), (0, 120, 0), 1)
    cv2.line(f, (0, h // 2), (w, h // 2), (0, 120, 0), 1)
    cv2.rectangle(f, (w // 4, h // 8), (w * 3 // 4, h * 7 // 8), (0, 180, 0), 1)
    color = (0, 0, 255) if b < 20 else (0, 255, 0)
    txt = f"SANG:{b:5.1f}/255  {'-> QUA TOI, them den!' if b < 20 else 'OK'}"
    cv2.putText(f, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return f

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    r, f = cap.read()
                    if not r:
                        continue
                    f = draw_overlay(f)
                    ok, jpg = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if not ok:
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(PAGE)

if __name__ == "__main__":
    print("stream on :8090", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8090), H).serve_forever()
