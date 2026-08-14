"""Kiểm chứng logic reader.py — không cần camera, không cần model.

    python3 tests/test_reader.py

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
from chess_ai.reader import (board_from_grid, explain, explain_occ,   # noqa: E402
                            locate_from_start, occ_of, pick_by_identity, placement)

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

print("\n9) LƯỢT ĐI — bạn cầm Đen, Trắng đi trước (đây là ca từng làm bàn ĐỨNG IM)")
# Ảnh tĩnh không nói được ai đang đi. board_from_grid thử Trắng trước, mà thế sau
# 1.e4 với "Trắng đi" vẫn hợp lệ -> nó trả sai lượt. Sai lượt thì explain() chỉ sinh
# nước Trắng, nên nước Đen của người chơi không bao giờ khớp: bàn không nhận nước nào.
w1 = chess.Board()
w1.push(chess.Move.from_uci("e2e4"))
g_b = placement(w1, flipped=True)                 # cầm Đen => Đen ở dưới ảnh

old = board_from_grid(g_b, flipped=True)
check("board_from_grid ĐOÁN SAI lượt (lý do có locate_from_start)",
      old is not None and old.turn == chess.WHITE,
      f"đoán: {'Trắng' if old and old.turn else 'Đen'}, đúng phải là Đen")

fix = locate_from_start(g_b, flipped=True)
check("locate_from_start cho ĐÚNG lượt Đen",
      fix is not None and fix.turn == chess.BLACK,
      f"đọc: {'Trắng' if fix and fix.turn else 'Đen'}")
check("locate_from_start giữ đúng thế cờ",
      fix is not None and fix.board_fen() == w1.board_fen())
check("có cả ô bắt tốt qua đường (board_from_grid không có)",
      fix is not None and fix.ep_square == w1.ep_square,
      f"ep={chess.square_name(fix.ep_square) if fix and fix.ep_square else None}")
check("biên bản giữ được nước mở màn của đối thủ",
      fix is not None and [m.uci() for m in fix.move_stack] == ["e2e4"],
      f"{[m.uci() for m in fix.move_stack] if fix else None}")

nxt = w1.copy()
nxt.push(chess.Move.from_uci("c7c5"))
check("SAU KHI SỬA: nhận ra nước Đen c7c5",
      (s := explain(fix, placement(nxt, flipped=True), True)) is not None
      and [m.uci() for m in s] == ["c7c5"],
      f"đọc được: {[m.uci() for m in s] if s else 'KHÔNG'}")
check("TRƯỚC KHI SỬA: nước đó bị bỏ (bàn đứng im)",
      explain(old, placement(nxt, flipped=True), True) is None)

print("\n10) locate_from_start ở các mốc khác")
check("chưa ai đi -> lượt Trắng",
      (r := locate_from_start(placement(chess.Board()), flipped=False)) is not None
      and r.turn == chess.WHITE and not r.move_stack)
t2 = chess.Board()
for u in ("e2e4", "c7c5"):
    t2.push(chess.Move.from_uci(u))
check("đi 2 nước -> lượt Trắng, đủ 2 nước trong biên bản",
      (r := locate_from_start(placement(t2), flipped=False)) is not None
      and r.turn == chess.WHITE and len(r.move_stack) == 2)
# 3 nước: tới được bằng 2 thứ tự (1.e4 c5 2.Nf3 và 1.Nf3 c5 2.e4) nên ep_square thô
# khác nhau, dù cả hai đều KHÔNG bắt được qua đường. Chốt chặn phải bỏ qua khác biệt
# vô hậu quả đó, nhưng vẫn phải cho ĐÚNG lượt.
t3b = chess.Board()
for u in ("e2e4", "c7c5", "g1f3"):
    t3b.push(chess.Move.from_uci(u))
check("đi 3 nước (2 thứ tự cùng đích) -> vẫn ra đúng lượt Đen",
      (r := locate_from_start(placement(t3b, flipped=True), flipped=True)) is not None
      and r.turn == chess.BLACK and r.board_fen() == t3b.board_fen(),
      f"{'None' if r is None else ('Trắng' if r.turn else 'Đen')}")
# Nhập nhằng THẬT thì phải nhận là nhập nhằng. Sau 1.e4 a6 2.e5 d5 Trắng bắt được
# qua đường; nhưng cùng ảnh đó cũng ra từ 1.e4 d5 2.e5 a6 — khi ấy KHÔNG bắt được.
# Ảnh không nói được Đen vừa đi nước nào, nên trả None để rơi về đường đoán là đúng;
# đoán bừa một quyền bắt qua đường là mở đường cho nước sai được nhận.
amb = chess.Board()
for u in ("e2e4", "a7a6", "e4e5", "d7d5"):
    amb.push(chess.Move.from_uci(u))
check("nhập nhằng thật (2 thứ tự cho khác quyền bắt qua đường) -> None",
      locate_from_start(placement(amb), flipped=False, max_plies=4) is None)

# Còn khi lịch sử là DUY NHẤT thì quyền bắt qua đường phải được giữ đúng — đây là
# dữ kiện board_from_grid không bao giờ có.
st = chess.Board("rnbqkbnr/1ppppppp/p7/4P3/8/8/PPPP1PPP/RNBQKBNR b - - 0 3")
ep = st.copy()
ep.push(chess.Move.from_uci("d7d5"))
r = locate_from_start(placement(ep), flipped=False, start=st, max_plies=1)
check("lịch sử duy nhất -> giữ đúng quyền bắt tốt qua đường",
      r is not None and r.has_legal_en_passant() and r.ep_square == ep.ep_square,
      f"ep={chess.square_name(r.ep_square) if r and r.ep_square is not None else None}")
mid = chess.Board("8/5k2/8/8/8/3K4/8/8 w - - 0 1")
check("thế giữa ván xa khai cuộc -> None (nhường cho đường đoán)",
      locate_from_start(placement(mid), flipped=False) is None)

print("\n" + ("TẤT CẢ ĐẠT" if not fails else f"CÓ {len(fails)} PHÉP THỬ SAI: {fails}"))
sys.exit(1 if fails else 0)
