#!/usr/bin/env python3
"""Proxy HTTP/HTTPS tối giản, không cần sudo, để chia internet cho AIBOX qua USB.
Laptop chạy proxy này (127.0.0.1:8899). `adb reverse tcp:8899 tcp:8899` bắc cầu
sang AIBOX; AIBOX đặt http_proxy/https_proxy = http://127.0.0.1:8899.
Proxy phân giải DNS + nối mạng ở phía laptop (nơi có internet)."""
import socket, threading, select
from urllib.parse import urlparse

LISTEN = ("127.0.0.1", 8899)

def pipe(a, b):
    a.setblocking(False); b.setblocking(False)
    socks = [a, b]
    while True:
        try:
            r, _, x = select.select(socks, [], socks, 900)
        except Exception:
            break
        if x:
            break
        if not r:
            continue  # idle: GIU ket noi (Tailscale control song lau) - khong ngat
        for s in r:
            try:
                data = s.recv(65535)
            except Exception:
                return
            if not data:
                return
            try:
                (b if s is a else a).sendall(data)
            except Exception:
                return

def handle(client):
    remote = None
    try:
        client.settimeout(15)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = client.recv(65535)
            if not chunk:
                client.close(); return
            data += chunk
            if len(data) > 65535:
                break
        line = data.split(b"\r\n", 1)[0].split(b" ")
        method, url = line[0], line[1]
        if method == b"CONNECT":
            host, _, port = url.partition(b":")
            remote = socket.create_connection((host.decode(), int(port or b"443")), timeout=15)
            client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        else:
            u = urlparse(url.decode())
            remote = socket.create_connection((u.hostname, u.port or 80), timeout=15)
            remote.sendall(data)
        client.settimeout(None)
        pipe(client, remote)
    except Exception:
        pass
    finally:
        for s in (client, remote):
            try:
                if s: s.close()
            except Exception:
                pass

def main():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(LISTEN); srv.listen(200)
    print(f"[usb_proxy] listening on {LISTEN[0]}:{LISTEN[1]}", flush=True)
    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()

if __name__ == "__main__":
    main()
