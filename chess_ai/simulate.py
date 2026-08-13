"""TỰ TEST BẰNG MÔ PHỎNG — không cần bàn cờ thật.
AI (huấn luyện viên) đánh trọn một ván với 'người chơi ảo' (đôi lúc đi ngẫu nhiên
để tạo sai lầm cho AI bình luận). In toàn bộ bình luận + thử hỏi–đáp + tạo giọng nói."""
from __future__ import annotations
import argparse
import os
import random

import chess

from . import config
from .coach import Coach


def pick_human_move(coach: Coach, blunder_rate: float) -> str:
    """Người chơi ảo: đa số đi nước khá, thỉnh thoảng đi ngẫu nhiên (sai lầm)."""
    legal = list(coach.board.legal_moves)
    if not legal:
        return ""
    if random.random() < blunder_rate:
        mv = random.choice(legal)
    else:
        mv = coach.engine.best_move(coach.board) or random.choice(legal)
    return coach.board.san(mv)


def run(max_plies=60, blunder_rate=0.25, speak=False, voices_demo=True, seed=7):
    random.seed(seed)
    s = config.Settings(
        human_color="white", skill_level=14,
        think_time=0.15, analysis_time=0.15, verbosity="normal", speak=speak,
    )
    coach = Coach(s)
    print("=" * 70)
    print("  MÔ PHỎNG: Người (Trắng, có lúc sai lầm)  vs  AI Huấn luyện viên (Đen)")
    print("=" * 70)

    ply = 0
    questions = ["đánh giá thế cờ", "có gì nguy hiểm không", "nên đi nước nào", "vật chất thế nào"]
    try:
        while not coach.game_over() and ply < max_plies:
            if coach.is_ai_turn():
                mv, msg = coach.ai_move()
                print(f"\n[{coach.board.fullmove_number}. AI] {msg}")
            else:
                san = pick_human_move(coach, blunder_rate)
                ok, msg = coach.human_move(san)
                print(f"\n[{coach.board.fullmove_number}. Người] ({san}) {msg}")
            ply += 1

            # thỉnh thoảng người chơi hỏi trợ lý
            if ply % 8 == 0 and not coach.game_over():
                q = random.choice(questions)
                a = coach.ask(q)
                print(f"   💬 Người hỏi: \"{q}\"  ->  AI: {a}")

        print("\n" + "=" * 70)
        print("  KẾT THÚC: " + (coach.result_text() or "hết số nước mô phỏng"))
        print("=" * 70)

        # phân tích tổng thế cờ cuối
        print("\n--- PHÂN TÍCH THẾ CỜ CUỐI (chi tiết) ---")
        print(coach.analyse_current("detailed"))

        # demo giọng nói: tạo WAV cho một câu bình luận
        if voices_demo:
            out = os.path.join(os.path.dirname(__file__), "..", "captures", "voices")
            out = os.path.abspath(out)
            text = "Đen lợi thế rõ. Bạn nên nhập thành sớm và coi chừng Mã e5."
            made = coach.speaker.demo_all_voices(text, out)
            print(f"\n--- ĐÃ TẠO {len(made)} FILE GIỌNG NÓI tại {out} ---")
            for name, p in made.items():
                sz = os.path.getsize(p) if os.path.exists(p) else 0
                print(f"   {name:12s} -> {os.path.basename(p)} ({sz} bytes)")
    finally:
        coach.close()


def main():
    ap = argparse.ArgumentParser(description="Tự test AI cờ vua bằng mô phỏng")
    ap.add_argument("--plies", type=int, default=60)
    ap.add_argument("--blunder", type=float, default=0.25)
    ap.add_argument("--speak", action="store_true", help="đọc to bằng loa")
    ap.add_argument("--no-voices", action="store_true", help="bỏ tạo file giọng nói")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    run(max_plies=args.plies, blunder_rate=args.blunder, speak=args.speak,
        voices_demo=not args.no_voices, seed=args.seed)


if __name__ == "__main__":
    main()
