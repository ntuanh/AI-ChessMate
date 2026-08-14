"""VÒNG CHẠY TRỰC TIẾP — camera (CÓ THỂ RUNG/DI CHUYỂN) soi màn hình → tự nhận
nước đi → bình luận + đọc to.

  python3 -m chess_ai.watch calib      # màn hình để THẾ KHAI CUỘC, chụp + tự hiệu chỉnh
  python3 -m chess_ai.watch play       # chơi trực tiếp: bám bàn liên tục, tự dò lại khi mất

Khác bản cũ: KHÔNG cần camera cố định. Mỗi frame: bám 4 góc bằng ORB; mất dấu thì
dò lại từ đầu; mọi kết quả đều tự chấm điểm caro trước khi tin (điểm thấp = bỏ frame).
"""
from __future__ import annotations
import argparse
import os
import time

import numpy as np
import chess

from chess_ai import config, vision, gridfind
from chess_ai.coach import Coach

try:
    import cv2
except Exception:
    cv2 = None

CAP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "captures"))
REF_FILE = os.path.join(CAP_DIR, "watch_ref.png")


def _draw_grid(warped, grid=None):
    vis = warped.copy()
    step = warped.shape[0] // 8
    for i in range(9):
        cv2.line(vis, (i * step, 0), (i * step, warped.shape[0]), (0, 200, 0), 1)
        cv2.line(vis, (0, i * step), (warped.shape[0], i * step), (0, 200, 0), 1)
    if grid is not None:
        for r in range(8):
            for f in range(8):
                if grid[r][f]:
                    cv2.circle(vis, (f * step + step // 2, r * step + step // 2),
                               6, (0, 0, 255), -1)
    return vis


# ---------------- hiệu chỉnh đầu ván ----------------
def calibrate(cam_index=None, save=True):
    """Màn hình đang để THẾ KHAI CUỘC. Chụp 1 khung → dò lưới → tự học ngưỡng
    occupancy từ 32 ô có quân đã biết → tự xác định hướng bàn theo màu quân."""
    cap = vision.open_camera(cam_index)
    frame = vision.grab(cap, 10)
    cap.release()
    if frame is None:
        print("Không đọc được camera."); return False
    os.makedirs(CAP_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(CAP_DIR, "watch_raw.png"), frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharp = gridfind.sharpness(gray)
    if sharp < gridfind.SHARP_MIN:
        print(f"⚠️  Ảnh MỜ (độ nét {sharp:.0f} < {gridfind.SHARP_MIN}). Giữ camera yên/"
              f"đúng tầm nét rồi chạy lại. Ảnh: captures/watch_raw.png")

    found = gridfind.find_board(frame)
    if found is None:
        print("❌ Chưa dò được bàn cờ (điểm caro thấp). Để bàn chiếm phần lớn khung, "
              "giảm loá, rồi thử lại. Ảnh thô: captures/watch_raw.png")
        return False
    corners, score = found

    warped = gridfind.warp(frame, corners, vision.WARP)
    model, margins = gridfind.OccupancyModel.fit(warped)
    if not model.usable():
        print(f"❌ Không tách được ô quân/ô trống (margin={['%.2f' % m for m in margins]}). "
              "Kiểm tra: màn hình có đúng THẾ KHAI CUỘC không? loá che mất hàng quân nào không?")
        return False

    # hướng bàn: Trắng phải ở DƯỚI ảnh; sai thì xoay ý nghĩa 4 góc 180°
    if not gridfind.white_is_bottom(warped):
        corners = gridfind.rot_corners(corners, 2)
        warped = gridfind.warp(frame, corners, vision.WARP)
        model, margins = gridfind.OccupancyModel.fit(warped)
        print("↻ Phát hiện Trắng ở trên → đã tự xoay bàn 180°.")

    grid = model.predict(warped)
    occ = int(grid.sum())
    match = int((grid == gridfind.START_MASK).sum())

    annot = frame.copy()
    cv2.polylines(annot, [corners.astype(int)], True, (0, 255, 0), 2)
    for (x, y) in corners:
        cv2.circle(annot, (int(x), int(y)), 8, (0, 0, 255), -1)
    cv2.imwrite(os.path.join(CAP_DIR, "watch_corners.png"), annot)
    cv2.imwrite(os.path.join(CAP_DIR, "watch_warped.png"), _draw_grid(warped, grid))

    print(f"✅ Dò được bàn: caro={score:.1f}, độ nét={sharp:.0f}, "
          f"ô có quân={occ}/32 đúng chuẩn, khớp thế khai cuộc {match}/64 ô.")
    print("   Kiểm tra mắt: captures/watch_corners.png & watch_warped.png (chấm đỏ = ô có quân).")
    if match < 62:
        print("⚠️  Khớp thế khai cuộc chưa đủ tốt (<62/64) — xem watch_warped.png xem lệch chỗ nào.")
    if save:
        gridfind.save_calib(vision.CALIB_FILE, corners, model, score)
        cv2.imwrite(REF_FILE, frame)
        print("   Đã lưu hiệu chỉnh (4 góc + ngưỡng occupancy + frame tham chiếu).")
    return match >= 62


# ---------------- theo dõi khi chơi ----------------
class Watcher:
    """Mỗi lần gọi grid() trả về occupancy 8x8 ĐÃ QUA tự kiểm, hoặc None (frame xấu/
    mất bàn — cứ bỏ qua, KHÔNG bao giờ trả kết quả từ lưới chưa tin được)."""

    REBASE_EVERY = 20      # bám tốt N lần thì làm mới frame tham chiếu (chống trôi)

    def __init__(self, cap, corners, model, calib_score):
        self.cap = cap
        self.corners = corners
        self.model = model
        self.score_min = max(gridfind.CARO_OK, 0.4 * calib_score)
        ref = cv2.imread(REF_FILE, cv2.IMREAD_GRAYSCALE)
        self.tracker = (gridfind.GridTracker(ref, corners) if ref is not None else None)
        self.ok_streak = 0
        self.lost = False

    def _validated(self, frame, corners):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sc, _ = gridfind.caro_score(gridfind.warp(gray, corners, gridfind.SCORE_WARP))
        return sc >= self.score_min

    def grid(self, board):
        frame = vision.grab(self.cap, 2)
        if frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gridfind.sharpness(gray) < gridfind.SHARP_MIN:      # mờ (đang di chuyển)
            return None

        corners = None
        if self.tracker is not None:
            c = self.tracker.track(gray)
            if c is not None and self._validated(frame, c):
                corners = c
        if corners is None:                                     # mất dấu -> dò lại
            found = gridfind.find_board(frame)
            if found is not None:
                c = gridfind.orient_to_board(frame, found[0], self.model,
                                             gridfind.expected_mask(board))
                if c is not None:
                    corners = c
                    if self.tracker is not None:
                        self.tracker.rebase(gray, corners)
                    if self.lost:
                        print("🔄 Đã tìm lại được bàn cờ.")
            if corners is None:
                if not self.lost:
                    print("👀 Mất bàn cờ trong khung hình — đưa camera về là tôi tự bắt lại.")
                self.lost = True
                return None

        self.lost = False
        self.corners = corners
        self.ok_streak += 1
        if self.tracker is not None and self.ok_streak % self.REBASE_EVERY == 0:
            self.tracker.rebase(gray, corners)
        return self.model.predict(gridfind.warp(frame, corners, vision.WARP))

    def stable_grid(self, board, tries=3, delay=0.25):
        """Chỉ chấp nhận khi 2 lần đọc LIÊN TIẾP giống nhau (đỡ nhiễu/animation/tay che)."""
        last = None
        for _ in range(tries):
            g = self.grid(board)
            if g is not None and last is not None and np.array_equal(g, last):
                return g
            last = g
            time.sleep(delay)
        return None


def play(skill=10, cam_index=None, voice="auto", poll=0.6, human_color="white"):
    calib = gridfind.load_calib(vision.CALIB_FILE)
    if calib is None:
        print("Chưa hiệu chỉnh. Để màn hình ở THẾ KHAI CUỘC rồi chạy: "
              "python3 -m chess_ai.watch calib")
        return
    corners, model, calib_score = calib
    s = config.Settings(human_color=human_color, skill_level=skill,
                        tts_engine=voice, verbosity="normal", speak=True)
    coach = Coach(s)
    cap = vision.open_camera(cam_index)
    watcher = Watcher(cap, corners, model, calib_score)

    print("👁️  Đang theo dõi. Camera rung/di chuyển vẫn được — tôi tự bám và tự dò lại.")
    print("   (Ctrl+C để dừng)")
    prev = watcher.stable_grid(coach.board)
    try:
        if coach.is_ai_turn():
            _, msg = coach.ai_move()
            print("[AI] " + msg); coach.say(msg)

        while not coach.game_over():
            time.sleep(poll)
            curr = watcher.stable_grid(coach.board)
            if curr is None or prev is None:
                prev = curr if curr is not None else prev
                continue
            if np.array_equal(curr, prev):
                continue
            mv = vision.diff_to_move(coach.board, prev, curr)
            if mv is None:
                continue                    # đổi nhưng chưa khớp nước hợp lệ (đang kéo quân)
            ok, msg = coach.human_move(coach.board.san(mv))
            if not ok:
                continue
            print("[Bạn] " + msg); coach.say(msg)
            prev = curr
            if coach.game_over():
                break
            _, aimsg = coach.ai_move()
            print("[AI] " + aimsg); coach.say(aimsg)
            time.sleep(1.0)
            prev = watcher.stable_grid(coach.board)

        print("==== " + (coach.result_text() or "Kết thúc") + " ====")
    except KeyboardInterrupt:
        print("\nĐã dừng theo dõi.")
    finally:
        cap.release(); coach.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "play"])
    ap.add_argument("--skill", type=int, default=10)
    ap.add_argument("--color", default="white", choices=["white", "black"])
    args = ap.parse_args()
    if args.cmd == "calib":
        calibrate()
    else:
        play(skill=args.skill, human_color=args.color)


if __name__ == "__main__":
    main()
