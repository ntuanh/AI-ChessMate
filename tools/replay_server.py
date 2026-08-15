#!/usr/bin/env python3
"""REPLAY — chạy vòng nhận diện trên DATASET đã dump, không cần camera/AIBOX.

Mỗi mẫu trong `captures/dataset/` là một cặp <ts>.png (bàn đã nắn 512×512) +
<ts>.fen (thế cờ THẬT lúc chụp, dòng 2 có `flipped=1` nếu Trắng ở trên). Vì nhãn
luôn đúng (coach_server chỉ dump khi khớp 100%), bộ này vừa là đầu vào vừa là
đáp án — nên ở đây bám được hay trượt đều ĐO ĐƯỢC, khác hẳn ngoài thực địa.

LƯU Ý về bản chất dữ liệu: coach_server CHỈ dump khi khớp 100% và tối đa 1 mẫu
mỗi 2 giây, nên hai khung liên tiếp có thể cách nhau NHIỀU nước. Vì vậy "bám
xuyên suốt dataset" là chuyện không thể và không phải phép đo có nghĩa. Hai phép
đo có nghĩa, và cũng chính là hai thứ đang hỏng ngoài thực địa:

  --audit  ĐỘ CHÍNH XÁC OCCUPANCY — mặt nạ đo được so với thế cờ THẬT từng khung.
           Đây chính là nguồn sinh chấm đỏ và là thứ đẩy `need` từ 3 lên 5 phiếu.
  --moves  KHẢ NĂNG GIẢI THÍCH NƯỚC — với mỗi cặp khung cách nhau ≤2 nước,
           explain_occ có tìm ra đúng nước đã đi không.

    python3 tools/replay_server.py                 # web ở http://127.0.0.1:8091
    python3 tools/replay_server.py --audit          # chấm occupancy
    python3 tools/replay_server.py --moves          # chấm khả năng bắt nước

Ở laptop chạy bằng venv: .venv/bin/python tools/replay_server.py
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

KIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT_ROOT)

import numpy as np
import cv2
import chess

from chess_ai import gridfind, reader, render, piece_net

CONF_MIN = reader.CONF_MIN
MAX_UNSURE = reader.MAX_UNSURE


def split_sessions(samples, gap_s=120):
    """Cắt dataset thành các PHIÊN riêng.

    Mốc thời gian trong dataset nhảy hàng chục giờ giữa các lần chạy — nối liền
    chúng lại thì bàn cờ đổi đột ngột và mọi thứ sau đó lệch dồn, chấm điểm thành
    vô nghĩa. Khoảng trống > gap_s giây = ván khác, phải neo lại từ đầu.
    """
    out, cur = [], []
    prev = None
    for s in samples:
        t = int(s[0]) / 1000.0
        if prev is not None and t - prev > gap_s and cur:
            out.append(cur)
            cur = []
        cur.append(s)
        prev = t
    if cur:
        out.append(cur)
    return out


def load_samples(data_dir):
    """[(ts, png_path, board_thật, flipped)] sắp theo thời gian."""
    out = []
    for fen_path in sorted(glob.glob(os.path.join(data_dir, "*.fen"))):
        png = fen_path[:-4] + ".png"
        if not os.path.exists(png):
            continue
        txt = open(fen_path).read()
        try:
            bd = chess.Board(txt.split("\n")[0].strip())
        except Exception:
            continue
        out.append((os.path.basename(fen_path)[:-4], png, bd, "flipped=1" in txt))
    return out


def board_grid(b, flipped=False):
    g = [["."] * 8 for _ in range(8)]
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p:
            r, f = chess.square_rank(sq), chess.square_file(sq)
            g[r if flipped else 7 - r][(7 - f) if flipped else f] = p.symbol()
    return g


class Replay:
    """Vòng nhận diện y như coach_server, nhưng ảnh lấy từ đĩa.

    Khác một điểm DUY NHẤT và cố ý: bỏ qua rectify/Tracker, vì dataset lưu ảnh đã
    nắn sẵn. Nhờ vậy mọi sai lệch quan sát được ở đây thuần tuý là của occupancy
    + explain_occ + bỏ phiếu, không lẫn lỗi định vị bàn.
    """

    def __init__(self, samples, use_pnet=True, gap_s=120):
        self.samples = samples
        # chỉ số khung đầu của mỗi phiên
        self.starts, k = set(), 0
        for sess in split_sessions(samples, gap_s):
            self.starts.add(k)
            k += len(sess)
        self.i = 0
        self.model = None
        self.pnet = piece_net.load() if use_pnet else None
        self.lock = threading.Lock()
        self.playing = True
        self.delay = 0.12
        self.reset()

    def _anchor(self, idx):
        """Neo lại thế cờ + ngưỡng occupancy vào khung `idx` (đầu mỗi phiên)."""
        ts, png, truth, flipped = self.samples[idx]
        self.flipped = flipped
        self.board = truth.copy()
        self.model, self.margins = gridfind.OccupancyModel.fit(
            cv2.imread(png), gridfind.expected_mask(truth, flipped=flipped))
        self.pend_mv, self.pend_cnt = None, 0

    def reset(self):
        ts, png, truth, flipped = self.samples[0]
        self.i = 0
        self._anchor(0)
        self.stat = {"frames": 0, "moves": 0, "khop": 0, "lech": 0,
                     "max_lech": 0, "ms": 0.0, "pnet_calls": 0, "phien": 1}
        self.log = []
        self.view = {}

    # ---------- một bước ----------
    def step(self):
        if self.i >= len(self.samples):
            self.playing = False
            return
        if self.i in self.starts and self.i:      # sang phiên mới: neo lại
            self._anchor(self.i)
            self.stat["phien"] += 1
            self.log.append(f"#{self.i} ── PHIÊN MỚI, neo lại thế cờ ──")
        ts, png, truth, flipped = self.samples[self.i]
        t0 = time.time()
        warped = cv2.imread(png)
        occ = self.model.predict(warped)

        _pn = {}

        def read_pieces():
            if not _pn:
                if self.pnet is None:
                    _pn["v"] = (None, 99)
                else:
                    g, c = self.pnet.predict(warped)
                    self.stat["pnet_calls"] += 1
                    _pn["v"] = (g, int((np.asarray(c) < CONF_MIN).sum()))
            return _pn["v"]

        note = ""
        det = None
        res = reader.explain_occ(self.board, occ, flipped=self.flipped)
        if res is not None:
            cands, bd = res
            if len(cands) == 1:
                det = (cands[0], bd)
            else:
                grid, n_unsure = read_pieces()
                if grid is not None and n_unsure <= MAX_UNSURE:
                    seq = reader.pick_by_identity(self.board, cands, grid, self.flipped)
                    if seq:
                        det, note = (seq, bd), f"{len(cands)} phương án bằng điểm → PieceNet"
            if det is None:
                grid, n_unsure = read_pieces()
                if grid is not None and n_unsure <= MAX_UNSURE:
                    seq = reader.explain(self.board, grid, self.flipped)
                    if seq:
                        det, note = (seq, 0), "occupancy bó tay → PieceNet đọc trọn bàn"

        moved = None
        if det is not None:
            mvs, bd = det
            need = 3 if (bd == 0 and len(mvs) == 1) else 5
            if mvs == self.pend_mv:
                self.pend_cnt += 1
            else:
                self.pend_mv, self.pend_cnt = mvs, 1
            if self.pend_cnt >= need:
                moved = []
                for mv in mvs:
                    try:
                        moved.append(self.board.san(mv))
                    except Exception:
                        moved.append(mv.uci())
                    self.board.push(mv)
                self.pend_mv, self.pend_cnt = None, 0
                self.stat["moves"] += len(moved)
        else:
            self.pend_cnt = max(0, self.pend_cnt - 1)

        # CHẤM ĐIỂM: thế cờ ta bám được so với thế cờ THẬT của khung này
        lech = sum(1 for sq in chess.SQUARES
                   if self.board.piece_at(sq) != truth.piece_at(sq))
        self.stat["frames"] += 1
        self.stat["ms"] += (time.time() - t0) * 1000
        if lech == 0:
            self.stat["khop"] += 1
        else:
            self.stat["lech"] += 1
            self.stat["max_lech"] = max(self.stat["max_lech"], lech)

        exp = gridfind.expected_mask(self.board, flipped=self.flipped)
        dots = [[bool(occ[r][f] != exp[r][f]) for f in range(8)] for r in range(8)]
        if moved:
            self.log.append(f"#{self.i} {ts}: {' '.join(moved)}"
                            + (f"  ({note})" if note else ""))
        elif lech and self.i and lech != getattr(self, "_last_lech", 0):
            self.log.append(f"#{self.i} {ts}: LỆCH {lech} ô so với thế thật")
        self._last_lech = lech

        with self.lock:
            self.view = {
                "i": self.i, "total": len(self.samples), "ts": ts,
                "warp": warped, "dots": dots, "occ_n": int(occ.sum()),
                "lech": lech, "residual": (res[1] if res else 0),
                "pend": f"{self.pend_cnt}", "note": note,
                "fen": self.board.board_fen(), "truth": truth.board_fen(),
                "moves": self.stat["moves"],
            }
        self.i += 1

    def loop(self):
        while True:
            if self.playing and self.i < len(self.samples):
                self.step()
                time.sleep(self.delay)
            else:
                time.sleep(0.05)

    # ---------- ảnh cho web ----------
    def jpegs(self):
        with self.lock:
            v = dict(self.view)
        if not v:
            return None, None, {}
        warp = v.pop("warp")
        img = render.render_grid(board_grid(self.board, flipped=self.flipped))
        step = img.shape[0] // 8
        for r in range(8):
            for f in range(8):
                if v["dots"][r][f]:
                    cv2.circle(img, (f * step + step - 12, r * step + 12), 7, (0, 0, 255), -1)
        v.pop("dots")
        ok1, j1 = cv2.imencode(".jpg", warp, [cv2.IMWRITE_JPEG_QUALITY, 82])
        ok2, j2 = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        return (j1.tobytes() if ok1 else None), (j2.tobytes() if ok2 else None), v


PAGE = """<!doctype html><html lang=vi><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>KIT · Replay dataset</title><style>
body{margin:0;font-family:system-ui,sans-serif;background:#0a1020;color:#e8eef7}
header{padding:14px 20px;background:#0d1830;border-bottom:1px solid #22406b;
  display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:16px;margin:0;letter-spacing:.5px}
.chip{background:#12203c;border:1px solid #2b4a7a;border-radius:8px;
  padding:4px 10px;font-size:12px}
.chip b{color:#7fd1ff}
.bad{background:#3a1622;border-color:#7a2b45;color:#ff8fb0}
.good{background:#123021;border-color:#2b7a4a;color:#7fffb0}
button{background:#1e4f75;color:#e8eef7;border:1px solid #2b6ea0;border-radius:8px;
  padding:6px 14px;font-size:13px;cursor:pointer}
button:hover{background:#2b6ea0}
.wrap{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:18px;max-width:1100px;margin:0 auto}
.card{background:#0d1830;border:1px solid #22406b;border-radius:12px;padding:14px}
.card h3{margin:0 0 10px;font-size:12px;color:#9db2d6;text-transform:uppercase;letter-spacing:1px}
img{width:100%;display:block;border-radius:8px;border:1px solid #2b6ea0}
pre{margin:0;font-size:12px;line-height:1.6;color:#9db2d6;max-height:220px;overflow:auto;
  white-space:pre-wrap}
.full{grid-column:1/3}
</style></head><body>
<header>
  <h1>KIT · REPLAY DATASET</h1>
  <span class=chip>khung <b id=i>-</b>/<b id=tot>-</b></span>
  <span class=chip>ô có quân <b id=n>-</b></span>
  <span class=chip>dư <b id=res>-</b></span>
  <span class=chip>phiếu <b id=pend>-</b></span>
  <span class=chip>nước đã bám <b id=mv>-</b></span>
  <span class=chip id=lechchip>lệch <b id=lech>-</b> ô</span>
  <button onclick=cmd('toggle')>⏯ chạy/dừng</button>
  <button onclick=cmd('step')>⏭ 1 khung</button>
  <button onclick=cmd('reset')>↺ về đầu</button>
  <button onclick=cmd('faster')>⏩ nhanh</button>
  <button onclick=cmd('slower')>⏪ chậm</button>
</header>
<div class=wrap>
  <div class=card><h3>Ảnh bàn đã nắn (từ dataset)</h3><img id=warp></div>
  <div class=card><h3>Bàn AI bám được — chấm đỏ = occupancy khác thế cờ</h3><img id=board></div>
  <div class="card full"><h3>Nhật ký</h3><pre id=log>—</pre></div>
  <div class="card full"><h3>Tổng kết</h3><pre id=stat>—</pre></div>
</div>
<script>
function cmd(c){fetch('/cmd?c='+c)}
async function tick(){
  try{
    const r=await fetch('/state'); const d=await r.json();
    for(const k of ['i','tot','n','res','pend','mv','lech'])
      document.getElementById(k).textContent=d[k];
    document.getElementById('lechchip').className='chip '+(d.lech==0?'good':'bad');
    document.getElementById('warp').src='/warp.jpg?'+d.i;
    document.getElementById('board').src='/board.jpg?'+d.i;
    document.getElementById('log').textContent=d.log||'—';
    document.getElementById('stat').textContent=d.stat||'—';
  }catch(e){}
  setTimeout(tick,150);
}
tick();
</script></body></html>"""


def summary(rp):
    s = rp.stat
    n = max(1, s["frames"])
    return (f"khung đã chạy      : {s['frames']}/{len(rp.samples)}\n"
            f"khớp thế cờ thật   : {s['khop']}  ({100*s['khop']/n:.1f}%)\n"
            f"lệch                : {s['lech']}  (lệch nhiều nhất {s['max_lech']} ô)\n"
            f"số phiên            : {s['phien']}\n"
            f"nước đã bám         : {s['moves']}\n"
            f"gọi PieceNet        : {s['pnet_calls']}\n"
            f"thời gian mỗi khung : {s['ms']/n:.1f} ms\n"
            f"PieceNet backend    : {type(rp.pnet).__name__ if rp.pnet else 'KHÔNG CÓ'}")


def make_handler(rp):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            p = self.path.split("?")[0]
            if p == "/":
                return self._send(200, "text/html; charset=utf-8", PAGE.encode())
            if p == "/cmd":
                c = self.path.split("c=")[-1]
                if c == "toggle":
                    rp.playing = not rp.playing
                elif c == "step":
                    rp.playing = False
                    rp.step()
                elif c == "reset":
                    rp.reset()
                elif c == "faster":
                    rp.delay = max(0.0, rp.delay - 0.04)
                elif c == "slower":
                    rp.delay = min(1.0, rp.delay + 0.04)
                return self._send(200, "text/plain", b"ok")
            j1, j2, v = rp.jpegs()
            if p == "/warp.jpg":
                return self._send(200, "image/jpeg", j1 or b"")
            if p == "/board.jpg":
                return self._send(200, "image/jpeg", j2 or b"")
            if p == "/state":
                d = {"i": v.get("i", 0), "tot": len(rp.samples), "n": v.get("occ_n", 0),
                     "res": v.get("residual", 0), "pend": v.get("pend", "0"),
                     "mv": v.get("moves", 0), "lech": v.get("lech", 0),
                     "log": "\n".join(rp.log[-14:]), "stat": summary(rp)}
                return self._send(200, "application/json", json.dumps(d).encode())
            self._send(404, "text/plain", b"?")
    return H


def audit_occupancy(samples, gap_s=120):
    """Occupancy đọc được có khớp thế cờ THẬT không — đo trên từng khung.

    Ngưỡng được fit ở khung đầu mỗi phiên (đúng như coach_server làm lúc canh
    bàn), rồi áp cho cả phiên. Số ô lệch ở đây CHÍNH LÀ số chấm đỏ sẽ hiện lên.
    """
    tot = perfect = cells_bad = 0
    dist = {}
    thua = thieu = 0
    for sess in split_sessions(samples, gap_s):
        ts, png, truth, flipped = sess[0]
        model, _ = gridfind.OccupancyModel.fit(
            cv2.imread(png), gridfind.expected_mask(truth, flipped=flipped))
        for ts, png, truth, flipped in sess:
            occ = model.predict(cv2.imread(png))
            exp = gridfind.expected_mask(truth, flipped=flipped)
            d = int((occ != exp).sum())
            thua += int((occ & ~exp).sum())      # báo CÓ quân ở ô trống
            thieu += int((~occ & exp).sum())     # bỏ sót quân
            tot += 1
            cells_bad += d
            dist[d] = dist.get(d, 0) + 1
            if d == 0:
                perfect += 1
    n = max(1, tot)
    print(f"\nĐỘ CHÍNH XÁC OCCUPANCY  ({tot} khung, {len(split_sessions(samples, gap_s))} phiên)")
    print(f"  khung khớp hoàn toàn : {perfect}  ({100*perfect/n:.1f}%)")
    print(f"  ô sai trung bình     : {cells_bad/n:.2f}/64")
    print(f"  báo thừa (ô trống→có): {thua}      bỏ sót (có→trống): {thieu}")
    print("  phân bố số ô lệch    :")
    for d in sorted(dist)[:10]:
        print(f"     {d:2d} ô lệch : {dist[d]:5d} khung ({100*dist[d]/n:5.1f}%)"
              + ("   ← need nhảy 3→5 từ đây" if d == 1 else ""))
    return perfect / n


def audit_moves(samples, gap_s=120, max_plies=2):
    """Giữa hai khung liên tiếp, explain_occ có tìm đúng nước đã đi không?"""
    import itertools
    tried = ok = amb = miss = 0
    for sess in split_sessions(samples, gap_s):
        ts, png, truth, flipped = sess[0]
        model, _ = gridfind.OccupancyModel.fit(
            cv2.imread(png), gridfind.expected_mask(truth, flipped=flipped))
        for (t0, p0, b0, f0), (t1, p1, b1, f1) in zip(sess, sess[1:]):
            # chỉ xét cặp thật sự cách nhau <= max_plies nước hợp lệ
            seq_true = None
            for depth in range(1, max_plies + 1):
                lvl = [([], b0.copy())]
                for _ in range(depth):
                    lvl = [(sq + [m], (lambda c, m: (c.push(m), c)[1])(cur.copy(), m))
                           for sq, cur in lvl for m in cur.legal_moves]
                for sq, cur in lvl:
                    if cur.board_fen() == b1.board_fen():
                        seq_true = sq
                        break
                if seq_true:
                    break
            if seq_true is None:
                continue
            tried += 1
            occ = model.predict(cv2.imread(p1))
            res = reader.explain_occ(b0, occ, flipped=f0, max_plies=max_plies)
            if res is None:
                miss += 1
                continue
            cands, _ = res
            uci = [[m.uci() for m in c] for c in cands]
            want = [m.uci() for m in seq_true]
            if want not in uci:
                miss += 1
            elif len(cands) == 1:
                ok += 1
            else:
                amb += 1
    n = max(1, tried)
    print(f"\nKHẢ NĂNG GIẢI THÍCH NƯỚC  ({tried} cặp khung cách nhau ≤{max_plies} nước)")
    print(f"  occupancy tự quyết được : {ok}   ({100*ok/n:.1f}%)")
    print(f"  bằng điểm, cần PieceNet  : {amb}   ({100*amb/n:.1f}%)")
    print(f"  KHÔNG giải thích được    : {miss}   ({100*miss/n:.1f}%)")
    return ok / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(KIT_ROOT, "captures", "dataset"))
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--limit", type=int, default=0, help="chỉ lấy N mẫu đầu")
    ap.add_argument("--score", action="store_true", help="chạy hết rồi in kết quả, không mở web")
    ap.add_argument("--audit", action="store_true", help="chấm độ chính xác occupancy")
    ap.add_argument("--moves", action="store_true", help="chấm khả năng bắt nước")
    ap.add_argument("--no-pnet", action="store_true")
    ap.add_argument("--gap", type=float, default=120,
                    help="khoảng trống (giây) coi là sang phiên/ván khác")
    a = ap.parse_args()

    samples = load_samples(a.data)
    if a.limit:
        samples = samples[:a.limit]
    if not samples:
        print(f"!! không thấy mẫu nào trong {a.data}\n"
              f"   kéo về bằng: adb pull /home/khacthu/kit/captures/dataset captures/")
        return 1
    print(f">> {len(samples)} mẫu từ {a.data}")

    if a.audit or a.moves:
        if a.audit:
            audit_occupancy(samples, a.gap)
        if a.moves:
            audit_moves(samples, a.gap)
        return 0

    rp = Replay(samples, use_pnet=not a.no_pnet, gap_s=a.gap)
    print(f">> PieceNet: {type(rp.pnet).__name__ if rp.pnet else 'KHÔNG CÓ'}")

    if a.score:
        t0 = time.time()
        while rp.i < len(rp.samples):
            rp.step()
        print("\n" + summary(rp))
        print(f"\ntổng thời gian      : {time.time()-t0:.1f}s")
        if rp.log:
            print("\n15 dòng nhật ký cuối:\n  " + "\n  ".join(rp.log[-15:]))
        return 0

    threading.Thread(target=rp.loop, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(rp))
    print(f">> mở http://127.0.0.1:{a.port}")
    srv.serve_forever()


if __name__ == "__main__":
    sys.exit(main())
