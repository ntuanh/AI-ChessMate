"""Tải ảnh bàn cờ THẬT từ Wikimedia Commons để mở rộng bộ test cho rectify.py.

    python3 tools/fetch_test_images.py --out captures/web --n 12

Chỉ lấy ảnh có giấy phép tự do trên Commons. Đây là bàn cờ VẬT LÝ (quân 3D, chụp
xiên, nền lộn xộn) — khác miền với ca chính của dự án (camera soi màn hình 2D),
nên dùng để đo độ bền của bước dò bàn, KHÔNG dùng làm chuẩn nghiệm thu.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
# Commons yêu cầu User-Agent mô tả rõ công cụ; UA chung chung bị trả 429
# ("does not comply with our robot policy").
UA = ("KIT-chess-vision/0.1 (offline computer-vision test-set builder; "
      "non-commercial research; low request rate)")
DELAY = 1.5              # giây giữa hai request — tôn trọng giới hạn của Commons


def get(url, tries=3):
    """Tải một URL, có tiết lưu và thử lại khi bị 429."""
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as r:
                out = r.read()
            time.sleep(DELAY)
            return out
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(DELAY * (k + 2) * 2)      # lùi dần rồi thử lại
    raise RuntimeError("không tải được")


def ext_of(url):
    """Đuôi file của URL, bỏ query string.

    Bắt buộc phải bỏ query: Commons gắn thêm tham số theo dõi vào cuối thumburl
    (`...9676.jpg?...utm_campaign=imageinfo`), nên xét đuôi trên URL thô thì mọi
    ảnh đều bị loại.
    """
    return urllib.parse.urlparse(url).path.lower().rsplit(".", 1)[-1]


def search(term, limit, width):
    q = {"action": "query", "generator": "search", "gsrnamespace": "6",
         "gsrsearch": term, "gsrlimit": str(limit), "prop": "imageinfo",
         "iiprop": "url", "iiurlwidth": str(width), "format": "json"}
    data = json.loads(get(API + "?" + urllib.parse.urlencode(q)))
    out = []
    for page in data.get("query", {}).get("pages", {}).values():
        for ii in page.get("imageinfo", []):
            u = ii.get("thumburl") or ii.get("url")
            if u and ext_of(u) in ("jpg", "jpeg", "png"):
                out.append((page["title"].replace("File:", ""), u))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="captures/web")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    terms = ["chessboard pieces game photo", "chess tournament board position",
             "chess set wooden board", "chess board top view pieces"]
    seen, saved = set(), 0
    for t in terms:
        if saved >= args.n:
            break
        try:
            hits = search(t, 10, args.width)
        except Exception as e:
            print("tìm lỗi:", t, e)
            continue
        for title, url in hits:
            if saved >= args.n or title in seen:
                continue
            seen.add(title)
            ext = ext_of(url)
            name = "".join(c if c.isalnum() else "_" for c in title)[:40] + "." + ext
            try:
                blob = get(url)
            except Exception as e:
                print("tải lỗi:", title, e)
                continue
            with open(os.path.join(args.out, name), "wb") as f:
                f.write(blob)
            saved += 1
            print(f"{saved:2d}. {name}  ({len(blob)//1024} KB)")
    print(f"\nĐã lưu {saved} ảnh vào {args.out}")


if __name__ == "__main__":
    main()
