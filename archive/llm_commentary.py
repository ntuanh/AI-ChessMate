"""Module kết nối LLM (Local Ollama / Llama.cpp / Qwen) để biến phân tích Stockfish Multi-PV thành lời bình luận HLV siêu mềm mại."""
import os
import json
import urllib.request

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

SYSTEM_PROMPT = """Bạn là một Huấn luyện viên Cờ vua chuyên nghiệp, thân thiện, giảng giải tự nhiên bằng tiếng Việt.
Nhiệm vụ: Dựa vào thông tin phân tích bàn cờ từ Stockfish, hãy đưa ra nhận xét ngắn gọn (1-2 câu), giàu tính chiến thuật.
Quy tắc quan trọng:
1. Tuyệt đối KHÔNG dùng từ 'Đối thủ vừa đi'. Chỉ tập trung phân tích từ góc nhìn của NGƯỜI CHƠI (tập trung vào lợi thế và các nước đi của người chơi).
2. Đọc trực tiếp tên nước đi (ví dụ: 'Nước e4.', 'Nước Nf3.').
3. KHÔNG dùng các con số kỹ thuật thô cứng như '+0.5 Tốt', '50 centipawns', 'eval score'.
4. Đưa ra 2-3 phương án nước đi gợi ý cho người chơi với văn phong tự nhiên, biến hóa linh hoạt."""


def ask_llm_coach(board_fen: str, last_move_text: str, candidate_sans: list[str], timeout: float = 1.5) -> str | None:
    """Gửi yêu cầu tới LLM local (nếu có). Trả về câu bình luận mềm mại hoặc None nếu không kết nối được."""
    moves_str = ", ".join(candidate_sans) if candidate_sans else "không rõ"
    prompt = f"""Thế cờ FEN: {board_fen}
Hành động vừa xảy ra: {last_move_text}
Các phương án gợi ý từ Stockfish: {moves_str}

Hãy đóng vai HLV cờ vua nhận xét ngắn gọn, tự nhiên bằng tiếng Việt."""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": SYSTEM_PROMPT,
        "stream": False,
        "options": {"temperature": 0.7, "max_tokens": 150}
    }
    try:
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            text = data.get("response", "").strip()
            if text:
                return text
    except Exception:
        pass
    return None
