"""Kiểm chứng logic reader.py — không cần camera, không cần model.

    python3 tools/test_reader.py

Trọng tâm là phép thử QUYẾT ĐỊNH: cùng một thế, hai nước ăn quân khác nhau cho
mặt nạ occupancy GIỐNG HỆT nhau (nên đường cũ phải tung xúc xắc), nhưng cho thế
cờ đầy đủ KHÁC nhau (nên đường mới đọc đúng). Test này chạy được ở mọi máy có
python-chess.
"""
from __future__ import annotations

import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chess_ai.reader import (board_from_grid, explain, explain_occ, occ_of,   # noqa: E402
                            pick_by_identity, placement)

fails = []


def check(name, cond, detail=""):
    print(f"  [{'OK ' if cond else 'SAI'}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        fails.append(name)


def occ_mask(board, flipped=False):
    g = placement(board, flipped)
    return tuple(tuple(c != "." for c in row) for row in g)


print("1) placement ↔ board_from_grid đi vòng có giữ nguyên thế cờ")
for flipped in (False, True):
    b = chess.Board()
    got = board_from_grid(placement(b, flipped), flipped=flipped)
    check(f"thế khai cuộc, flipped={flipped}",
          got is not None and got.board_fen() == b.board_fen())

b = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3")
for flipped in (False, True):
    got = board_from_grid(placement(b, flipped), flipped=flipped)
    check(f"thế giữa ván, flipped={flipped}",
          got is not None and got.board_fen() == b.board_fen())

print("\n2) explain() nhận đúng một nước thường")
b = chess.Board()
after = b.copy()
after.push(chess.Move.from_uci("e2e4"))
seq = explain(b, placement(after), False)
check("e2e4", seq is not None and len(seq) == 1 and seq[0].uci() == "e2e4",
      f"đọc được: {[m.uci() for m in seq] if seq else None}")

print("\n3) PHÉP THỬ QUYẾT ĐỊNH — hai nước ăn quân khác nhau")
# Tốt Trắng e4; tốt Đen d5 và f5. exd5 và exf5 đều làm e4 trống, d5 và f5 đều vẫn
# có quân → mặt nạ occupancy y hệt nhau.
pos = chess.Board("rnbqkbnr/ppp1p1pp/8/3p1p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
m1, m2 = chess.Move.from_uci("e4d5"), chess.Move.from_uci("e4f5")
a1, a2 = pos.copy(), pos.copy()
a1.push(m1)
a2.push(m2)
check("occupancy của exd5 và exf5 GIỐNG NHAU (nên đường cũ bó tay)",
      occ_mask(a1) == occ_mask(a2))
check("thế cờ đầy đủ của exd5 và exf5 KHÁC NHAU",
      placement(a1) != placement(a2))
s1 = explain(pos, placement(a1), False)
s2 = explain(pos, placement(a2), False)
check("explain đọc đúng exd5", s1 is not None and len(s1) == 1 and s1[0] == m1,
      f"đọc được: {[m.uci() for m in s1] if s1 else None}")
check("explain đọc đúng exf5", s2 is not None and len(s2) == 1 and s2[0] == m2,
      f"đọc được: {[m.uci() for m in s2] if s2 else None}")

print("\n4) Phong cấp — occupancy không thể biết là quân gì")
pr = chess.Board("8/P6k/8/8/8/8/7K/8 w - - 0 1")
for uci, tag in (("a7a8q", "Hậu"), ("a7a8n", "Mã")):
    mv = chess.Move.from_uci(uci)
    aft = pr.copy()
    aft.push(mv)
    seq = explain(pr, placement(aft), False)
    check(f"phong {tag}", seq is not None and len(seq) == 1 and seq[0] == mv,
          f"đọc được: {[m.uci() for m in seq] if seq else None}")

print("\n5) Nhập thành và chuỗi 2 nước nhanh")
cs = chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 6 4")
aft = cs.copy()
aft.push(chess.Move.from_uci("e1g1"))
seq = explain(cs, placement(aft), False)
check("nhập thành gần (O-O)", seq is not None and len(seq) == 1
      and seq[0].uci() == "e1g1", f"đọc được: {[m.uci() for m in seq] if seq else None}")

two = chess.Board()
aft = two.copy()
aft.push(chess.Move.from_uci("e2e4"))
aft.push(chess.Move.from_uci("c7c5"))
seq = explain(two, placement(aft), False)
check("bắt kịp 2 nước trong 1 frame", seq is not None and len(seq) == 2
      and [m.uci() for m in seq] == ["e2e4", "c7c5"],
      f"đọc được: {[m.uci() for m in seq] if seq else None}")

print("\n6) Thế không giải thích được bằng luật thì phải TRẢ None, không đoán bừa")
bad = chess.Board()
g = placement(bad)
g[4][4] = "q"          # nhét một quân Hậu Đen vào giữa bàn: vô lý
check("thế vô lý → None", explain(bad, g, False) is None)

# ---- phần bổ sung: kết hợp occupancy + danh tính quân ----

print("\n7) occupancy bằng điểm ở nước ăn quân, PieceNet phân giải")
pos2 = chess.Board("rnbqkbnr/ppp1p1pp/8/3p1p2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 3")
truth = pos2.copy()
truth.push(chess.Move.from_uci("e4d5"))          # thực tế đi exd5
res = explain_occ(pos2, occ_of(truth))
check("occupancy trả về NHIỀU phương án (không tự chọn)",
      res is not None and len(res[0]) >= 2,
      f"số phương án = {len(res[0]) if res else 0}: "
      f"{[[m.uci() for m in s] for s in res[0]] if res else None}")
if res:
    got = pick_by_identity(pos2, res[0], placement(truth), False)
    check("PieceNet chọn đúng exd5",
          got is not None and len(got) == 1 and got[0].uci() == "e4d5",
          f"chọn: {[m.uci() for m in got] if got else None}")

print("\n8) nước thường thì occupancy tự quyết được, không cần PieceNet")
p3 = chess.Board()
t3 = p3.copy(); t3.push(chess.Move.from_uci("e2e4"))
r3 = explain_occ(p3, occ_of(t3))
check("chỉ 1 phương án cho e2e4",
      r3 is not None and len(r3[0]) == 1 and r3[0][0][0].uci() == "e2e4",
      f"{[[m.uci() for m in s] for s in r3[0]] if r3 else None}")

print("\n" + ("TẤT CẢ ĐẠT" if not fails else f"CÓ {len(fails)} PHÉP THỬ SAI: {fails}"))
sys.exit(1 if fails else 0)
