#!/usr/bin/env python3
"""Kiểm chứng trợ lý đối thoại: tầng LLM + tầng dự phòng mẫu câu.
Chạy trên AIBOX:  cd /data/kit && python3 -m tools.test_assistant_llm
Hoặc:             cd /data/kit && python3 tools/test_assistant_llm.py
"""
import sys, os
sys.path.insert(0, "/data/kit")

import chess
from chess_ai import config, llm
from chess_ai.engine import ChessEngine
from chess_ai.assistant import Assistant

# Ruy Lopez sau 1.e4 e5 2.Nf3 Nc6 3.Bb5
b = chess.Board()
for san in ["e4", "e5", "Nf3", "Nc6", "Bb5"]:
    b.push_san(san)

S = config.Settings(human_color="white", analysis_time=0.4, speak=False)
eng = ChessEngine(S)
asst = Assistant(S, eng)

print("=" * 66)
print("OLLAMA_URL   :", llm.OLLAMA_URL)
print("LLM san sang :", llm.available())
print("Trang thai   :", asst.status_line())
print("=" * 66)

print("\n--- KHOI DU LIEU GUI CHO LLM (day la thu quyet dinh chat luong) ---")
print(asst._build_facts(b))

QUESTIONS = [
    "thế cờ này thế nào, tôi nên làm gì tiếp?",          # câu mở, bộ từ khóa cũ CHỊU THUA
    "khai cuộc này tên là gì và ý đồ của nó ra sao?",     # câu hoàn toàn ngoài INTENTS
    "có gì nguy hiểm không",                             # câu khớp từ khóa cũ
]
for q in QUESTIONS:
    print("\n" + "-" * 66)
    print("HỎI :", q)
    print("ĐÁP :", asst.answer(b, q))

# --- kiem tra tang du phong: tro OLLAMA_URL vao cong chet ---
print("\n" + "=" * 66)
print("NGAT LLM -> phai tu dong quay ve mau cau, KHONG duoc crash")
llm.OLLAMA_URL = "http://127.0.0.1:9"
llm._HEALTH["ok"] = False
llm._HEALTH["checked_at"] = 0.0
asst2 = Assistant(S, eng)
print("LLM san sang :", llm.available())
print("Trang thai   :", asst2.status_line())
for q in ["có gì nguy hiểm không", "nên đi nước nào"]:
    print(f"HỎI : {q}\nĐÁP : {asst2.answer(b, q)}")
print("=" * 66)
print("XONG.")
