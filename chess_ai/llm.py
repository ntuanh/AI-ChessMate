"""KẾT NỐI LLM (Ollama) — dùng chung cho trợ lý đối thoại và bình luận.

TRIẾT LÝ QUAN TRỌNG:
    Stockfish là BỘ NÃO CỜ — mọi con số, nước đi, đánh giá đều do nó quyết.
    LLM chỉ là BỘ NÃO NGÔN NGỮ — diễn đạt lại cho tự nhiên và trả lời câu hỏi mở.
    Không bao giờ để LLM tự nghĩ ra nước đi: LLM chơi cờ rất kém và bịa rất giỏi.

Cấu hình bằng biến môi trường:
    OLLAMA_URL    mặc định http://127.0.0.1:11434
    OLLAMA_MODEL  mặc định qwen2.5:14b   (khớp theo tên trước dấu ':' nên
                  qwen2.5:14b-instruct, qwen2.5:14b-instruct-q4_K_M... đều nhận)
    LLM_TIMEOUT   giây, mặc định 20
"""
from __future__ import annotations
import os
import json
import time
import urllib.request
import urllib.error

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))

_HEALTH = {"ok": False, "checked_at": 0.0, "model": ""}
_HEALTH_TTL = 30.0          # giây — nhớ kết quả để khỏi dò lại liên tục


def _post(path: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def available(force: bool = False) -> bool:
    """Có LLM dùng được không? Kết quả được nhớ 30s để khỏi dò lại mỗi câu."""
    now = time.time()
    if not force and (now - _HEALTH["checked_at"]) < _HEALTH_TTL:
        return _HEALTH["ok"]
    _HEALTH["checked_at"] = now
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name", "") for m in tags.get("models", [])]
        if not names:
            _HEALTH["ok"] = False
            return False
        # khớp chính xác, hoặc khớp phần tên trước dấu ':'
        want = OLLAMA_MODEL
        hit = next((n for n in names if n == want or n.split(":")[0] == want.split(":")[0]), None)
        _HEALTH["ok"] = hit is not None
        _HEALTH["model"] = hit or ""
    except Exception:
        _HEALTH["ok"] = False
    return _HEALTH["ok"]


def model_name() -> str:
    """Tên model thật sự đang có trên server (có thể khác OLLAMA_MODEL về tag)."""
    return _HEALTH.get("model") or OLLAMA_MODEL


def chat(messages: list[dict], temperature: float = 0.4,
         max_tokens: int = 320, timeout: float | None = None) -> str | None:
    """Gọi /api/chat với lịch sử hội thoại. Trả None nếu không kết nối được."""
    if not available():
        return None
    payload = {
        "model": model_name(),
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
        },
    }
    try:
        data = _post("/api/chat", payload, timeout or LLM_TIMEOUT)
        text = (data.get("message") or {}).get("content", "").strip()
        return text or None
    except Exception:
        _HEALTH["ok"] = False           # đánh dấu hỏng để lần sau dò lại ngay
        _HEALTH["checked_at"] = 0.0
        return None


def warmup() -> bool:
    """Nạp sẵn model vào VRAM để câu hỏi đầu tiên không bị chậm."""
    if not available(force=True):
        return False
    try:
        _post("/api/chat", {"model": model_name(), "messages": [], "stream": False}, 120)
        return True
    except Exception:
        return False
