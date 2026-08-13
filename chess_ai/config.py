"""Cấu hình chung cho AI cờ vua chạy trên AIBOX 8550."""
from dataclasses import dataclass, field

# Đường dẫn công cụ trên AIBOX
STOCKFISH_PATH = "/usr/games/stockfish"
ESPEAK_VOICE = "vi"          # giọng tiếng Việt của espeak-ng
APLAY_DEVICE = None          # None = card mặc định; hoặc "plughw:0,0"

# Piper TTS (giọng neural tự nhiên) — ưu tiên hơn espeak
PIPER_BIN = "/data/piper/piper"
PIPER_LIB = "/data/piper"
PIPER_MODEL = "/data/kit/models_tts/vi.onnx"

# Camera
CAM_INDEX = 2                # C270 = /dev/video2

# Các giọng đọc espeak-ng (tên thân thiện -> chuỗi voice của espeak).
# Biến thể: m1..m7 = nam, f1..f5 = nữ; khác nhau về cao độ/chất giọng.
VOICES = {
    "nam_tram":   "vi+m1",
    "nam_chuan":  "vi+m3",
    "nam_cao":    "vi+m5",
    "nu_chuan":   "vi+f2",
    "nu_cao":     "vi+f4",
    "thi_tham":   "vi+whisper",
    "tieng_anh":  "en+m3",
}


@dataclass
class Settings:
    # Người chơi cầm màu nào: "white" hoặc "black". AI cầm màu còn lại.
    human_color: str = "white"

    # Độ khó của AI (đối thủ):
    #   skill_level: 0..20  (Stockfish Skill Level)
    #   elo: nếu limit_elo=True thì giới hạn sức cờ theo Elo (1350..2850)
    skill_level: int = 12
    limit_elo: bool = False
    elo: int = 1600

    # Thời gian/độ sâu suy nghĩ cho mỗi nước
    think_time: float = 0.6      # giây cho nước ĐI của AI
    analysis_time: float = 0.5   # giây để PHÂN TÍCH bình luận

    # Trợ lý đối thoại
    # True  = ưu tiên LLM (thông minh, hiểu câu hỏi mở) — cần OLLAMA_URL với tới được
    # False = chỉ dùng mẫu câu offline
    # Dù bật, nếu không gọi được LLM thì tự động lùi về mẫu câu, không bao giờ chết.
    use_llm: bool = True

    # Bình luận
    lang: str = "vi"             # ngôn ngữ bình luận
    verbosity: str = "normal"    # "short" | "normal" | "detailed"
    speak: bool = True           # đọc to bằng loa (nếu có)
    tts_engine: str = "auto"     # "auto" (Piper nếu có) | "piper" | "espeak"
    voice: str = "nam_chuan"     # (espeak) tên giọng trong config.VOICES
    speed: int = 150             # (espeak) tốc độ đọc từ/phút
    piper_length: float = 1.0    # (piper) >1 chậm & rõ hơn, <1 nhanh hơn

    def ai_color(self) -> str:
        return "black" if self.human_color == "white" else "white"
