"""CHƠI TƯƠNG TÁC với chuyên gia cờ vua qua bàn phím (và giọng nói nếu có loa).
Bạn gõ nước đi (vd 'e4', 'Nf3'), hoặc HỎI ('đánh giá thế cờ', 'nên đi gì'),
hoặc lệnh: 'phân tích', 'giọng nu_chuan', 'thoát'."""
from __future__ import annotations
import argparse

from . import config
from .coach import Coach

BANNER = """
============================================================
   CHUYÊN GIA CỜ VUA AI  —  vừa là đối thủ, vừa là HLV
------------------------------------------------------------
  Gõ nước đi:   e4 , Nf3 , O-O , e7e8q ...
  Hỏi trợ lý:   "đánh giá thế cờ" , "nên đi gì" , "có gì nguy hiểm"
  Lệnh:         phân tích | giọng <tên> | trợ giúp | thoát
============================================================
"""

CMD_QUIT = {"thoat", "thoát", "quit", "exit"}
CMD_ANALYSE = {"phan tich", "phân tích", "analyse", "analyze"}
CMD_HELP = {"tro giup", "trợ giúp", "help", "?"}


def emit(coach: Coach, text: str):
    print(text)
    coach.say(text)


def main():
    ap = argparse.ArgumentParser(description="Chơi với chuyên gia cờ vua AI")
    ap.add_argument("--color", choices=["white", "black"], default="white",
                    help="màu bạn cầm (AI cầm màu còn lại)")
    ap.add_argument("--skill", type=int, default=12, help="độ khó 0..20")
    ap.add_argument("--elo", type=int, default=0, help="giới hạn Elo (>0 để bật)")
    ap.add_argument("--voice", default="nam_chuan", help="giọng đọc (xem config.VOICES)")
    ap.add_argument("--verbosity", choices=["short", "normal", "detailed"], default="normal")
    ap.add_argument("--no-speak", action="store_true", help="không đọc to")
    ap.add_argument("--listen", action="store_true", help="nghe lệnh bằng giọng nói (mic + Vosk)")
    ap.add_argument("--listen-seconds", type=float, default=4.0, help="số giây thu mỗi lượt")
    ap.add_argument("--no-llm", action="store_true",
                    help="tắt LLM, chỉ dùng mẫu câu offline (mặc định: bật LLM nếu gọi được)")
    args = ap.parse_args()

    listener = None
    if args.listen:
        try:
            from .listener import Listener
            listener = Listener()
            print("🎤 Chế độ giọng nói BẬT — nói vào mic khi tới lượt bạn.")
        except Exception as e:
            print(f"(Không bật được giọng nói: {e}) — dùng bàn phím.")

    s = config.Settings(
        human_color=args.color, skill_level=args.skill,
        limit_elo=args.elo > 0, elo=args.elo or 1600,
        voice=args.voice, verbosity=args.verbosity, speak=not args.no_speak,
        use_llm=not args.no_llm,
    )
    coach = Coach(s)
    print(BANNER)
    print(f"Bạn cầm {('Trắng' if args.color=='white' else 'Đen')}. "
          f"AI cầm {('Đen' if args.color=='white' else 'Trắng')}. Độ khó {args.skill}.")
    print(coach.assistant.status_line())

    try:
        # Nếu AI đi trước
        if coach.is_ai_turn():
            _, msg = coach.ai_move()
            emit(coach, "\n[AI] " + msg)

        while not coach.game_over():
            try:
                if listener is not None:
                    print("\n🎤 (đang nghe...)", flush=True)
                    raw = listener.listen_once(args.listen_seconds).strip()
                    print(f"Bạn (nói) > {raw!r}")
                else:
                    raw = input("\nBạn > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not raw:
                continue
            low = raw.lower()

            if low in CMD_QUIT:
                break
            if low in CMD_HELP:
                emit(coach, coach.ask("giúp"))
                continue
            if low in CMD_ANALYSE:
                emit(coach, "[Phân tích] " + coach.analyse_current("detailed"))
                continue
            if low.startswith("giọng ") or low.startswith("giong "):
                name = raw.split(" ", 1)[1].strip()
                if name in config.VOICES:
                    coach.settings.voice = name
                    emit(coach, f"Đã đổi giọng sang {name}.")
                else:
                    print("Giọng không có. Các giọng: " + ", ".join(config.VOICES))
                continue

            # thử coi là nước đi
            mv = coach.parse_move(raw)
            if mv is not None:
                ok, msg = coach.human_move(raw)
                emit(coach, "[Bạn] " + msg)
                if coach.game_over():
                    break
                _, aimsg = coach.ai_move()
                emit(coach, "\n[AI] " + aimsg)
            else:
                # không phải nước đi -> coi là câu hỏi cho trợ lý
                emit(coach, "[AI] " + coach.ask(raw))

        print("\n==== " + (coach.result_text() or "Kết thúc") + " ====")
    finally:
        coach.close()


if __name__ == "__main__":
    main()
