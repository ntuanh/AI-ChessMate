#!/usr/bin/env python3
"""Kiểm chứng đường GIỌNG NÓI: nói → Whisper (NPU) → lệnh → Stockfish → nước đi.

    python3 tools/test_voice.py              # dùng TTS thay người nói (không cần mic)
    python3 tools/test_voice.py --mic 4      # thu 4 giây từ micro thật
    python3 tools/test_voice.py --text "đánh cho tôi nước tiếp theo"   # bỏ qua Whisper

Vì sao có chế độ TTS: nó chạy được **toàn bộ chuỗi** — âm thanh thật, log-mel
thật, NPU thật — mà không cần có người ngồi trước micro, nên hợp cho test tự động
và cho việc kiểm tra sau khi sửa code.

Điều nó KHÔNG thay được: giọng Piper/espeak đều đặn và sạch hơn giọng người rất
nhiều, nên tỉ lệ nghe đúng ở đây là chặn TRÊN chứ không phải số đo thực tế. Đúng
lúc chữ nghe lệch là lúc lớp LLM chứng minh giá trị của nó, nên chạy đủ ba lớp.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time

KIT_ROOT = os.environ.get("KIT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KIT_ROOT)

import chess                                          # noqa: E402

from chess_ai import commentary, config, voice        # noqa: E402
from chess_ai.engine import ChessEngine               # noqa: E402
from chess_ai.speaker import Speaker                  # noqa: E402

PHRASES = [
    "đánh cho tôi nước tiếp theo",
    "thế cờ đang thế nào",
    "mình phải làm gì bây giờ",          # không khớp từ khoá -> LLM
]


def stockfish_move(engine, board):
    t0 = time.time()
    a = engine.analyse(board, multipv=3)
    ms = (time.time() - t0) * 1000
    san = board.san(a.best_move) if a.best_move else "—"
    return san, commentary.generate_short_move(board, a), ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mic", type=float, metavar="GIAY",
                    help="thu từ micro thay vì dùng TTS")
    ap.add_argument("--text", action="append", help="câu có sẵn, bỏ qua Whisper")
    ap.add_argument("--no-llm", action="store_true", help="chỉ dùng lớp từ khoá")
    ap.add_argument("--fen", default=chess.STARTING_FEN)
    args = ap.parse_args()

    board = chess.Board(args.fen)
    S = config.Settings(human_color="white", skill_level=12,
                        think_time=0.3, analysis_time=0.35, speak=False)
    engine = ChessEngine(S)
    # Lần gọi đầu tốn ~1 s (nạp NNUE + bắt tay UCI), các lần sau ~340 ms. Trả
    # trước ở đây để số đo bên dưới là số ổn định chứ không phải số lần đầu.
    engine.analyse(board, multipv=1)

    vc = None
    if not args.text:
        vc = voice.VoiceCommander()
        t0 = time.time()
        if not vc.start():
            print("KHONG khoi dong duoc Whisper:", vc.error)
            return 1
        print(f"Whisper NPU san sang sau {time.time() - t0:.2f}s")

    speaker = None
    if not args.text and args.mic is None:
        speaker = Speaker(S)
        print(f"TTS: {speaker._engine()}")

    items = args.text or PHRASES
    fails = 0
    for want in items:
        print("\n" + "=" * 68)
        if args.text:
            heard, st = want, {}
        elif args.mic is not None:
            print(f">>> NOI DI ({args.mic:.0f}s): \"{want}\"")
            heard, st = vc.listen(args.mic)
        else:
            with tempfile.TemporaryDirectory() as td:
                wav = os.path.join(td, "say.wav")
                if not speaker.synth_to_wav(want, wav):
                    print("TTS hong, bo qua"); fails += 1; continue
                t0 = time.time()
                heard = vc._asr.transcribe_wav(wav)
                st = dict(vc._asr.last_stats)
                st["wall_ms"] = (time.time() - t0) * 1000
        print(f'[1] noi   : "{want}"')
        print(f'    nghe  : "{heard}"'
              + (f'   ({st.get("wall_ms", 0)/1000:.2f}s, '
                 f'logprob {st.get("avg_logprob", 0):.2f})' if st else ""))
        if not heard:
            print(f'    -> BI LOAI: {st.get("reject") or "im lang"}')
            fails += 1
            continue

        cmd = voice.resolve(heard, use_llm=not args.no_llm)
        print(f"[2] lenh  : {cmd.lenh}   [{cmd.source}, {cmd.ms:.0f} ms]")
        if not cmd.understood:
            fails += 1
            continue

        if cmd.lenh == "goi_y":
            san, say, ms = stockfish_move(engine, board)
            print(f"[3] engine: {ms:.0f} ms")
            print(f"    ==> NUOC DI: {san}   · \"{say}\"")
        elif cmd.lenh == "danh_gia":
            t0 = time.time()
            a = engine.analyse(board, multipv=3)
            rep = commentary.generate_detailed_analysis(board, a)
            print(f"[3] engine: {(time.time() - t0) * 1000:.0f} ms")
            print(f"    ==> {rep[:160]}")
        else:
            print(f"[3] hanh dong: {cmd.lenh} (coach_server thi hanh)")

    if vc:
        vc.close()
    engine.close() if hasattr(engine, "close") else None
    print("\n" + "=" * 68)
    print(f"{len(items) - fails}/{len(items)} cau di het chuoi")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
