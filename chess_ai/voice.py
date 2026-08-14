"""LỆNH BẰNG GIỌNG NÓI — nói tiếng Việt → Whisper (NPU) → LLM → lệnh cho coach.

    nói "đánh cho tôi nước tiếp theo"
      → Whisper-Small trên Hexagon NPU  (~1 s)   → "đánh cho tôi nước tiếp theo"
      → lớp hiểu lệnh                            → goi_y
      → coach_server gọi Stockfish               → hiện + đọc "Nước đi tốt nhất: e4"

BA LỚP, THỨ TỰ QUAN TRỌNG HƠN BẢN THÂN TỪNG LỚP:

1. **Cổng tin cậy của Whisper** (nằm trong `asr.py`) loại âm thanh không phải
   tiếng nói. Phải đứng đầu. Whisper bịa RẤT TRÔI CHẢY — tiếng ồn phòng trên
   chính board này nhiều lần cho ra "Hãy subscribe cho kênh Ghiền Mì Gõ…" — nên
   nếu không chặn ở đây thì mọi lớp sau đang xử lý một câu tự tin, hợp lý, và
   hoàn toàn bịa. Trên board này `no_speech` VÔ DỤNG (0,72–0,90 cho cả tiếng nói
   sạch lẫn im lặng); `avg_logprob` mới tách được, ngưỡng −1,0.
2. **Bộ khớp từ khoá** — 0–1 ms, phủ hầu hết cách nói thường gặp.
3. **LLM**, chỉ cho những cách diễn đạt lớp 2 bỏ sót, và bị GBNF ràng buộc vào
   đúng tập lệnh — trong đó luôn có `khong_hieu` để nó còn đường từ chối.

VÌ SAO LLM PHẢI ĐỨNG CUỐI VÀ PHẢI BỊ RÀNG BUỘC: đưa thẳng chữ ASR thô cho LLM và
bảo "sửa lại giúp" thì một câu quảng cáo YouTube sẽ được nó biến thành một lệnh cờ
nghe rất hợp lý, tức là rửa nhiễu thành mệnh lệnh mà engine sau đó thi hành thật.

Bù lại, LLM làm được đúng việc mà từ khoá không làm nổi: **đọc xuyên qua chữ ASR
nghe lệch**. Model lượng tử hoá này nghe "tấn công cánh phải" thành "tính công
cảnh phải" — sai 2/4 âm tiết, regex chịu thua, LLM vẫn hiểu.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Optional

from . import llm

# Bộ Whisper (encoder.bin / decoder.bin / whisper_worker / asr.py) được deploy
# vào KIT_ROOT/whisper — xem tools/install_voice.sh. Không trỏ thẳng vào
# /home/tuananh/whisper: thư mục đó là drwxr-x--- nên user khacthu (user chạy
# dịch vụ) không vào được, và một đường dẫn nhà người khác là chỗ dựa mong manh.
KIT_ROOT = os.environ.get("KIT_ROOT") or os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))
WHISPER_DIR = os.environ.get("KIT_WHISPER_DIR", os.path.join(KIT_ROOT, "whisper"))

# Lệnh coach_server thi hành được. Cố tình KHÔNG có lệnh phong cách (tấn công /
# phòng thủ / thí quân): engine ở đây chỉ trả nước mạnh nhất, chưa có bộ chấm
# điểm phong cách, nên nhận những lệnh đó là hứa một việc không làm được.
COMMANDS = ("goi_y", "danh_gia", "van_moi", "dong_bo", "khong_hieu")

# Trả lời chỉ MỘT TỪ. Có grammar thì llama.cpp phải kiểm ràng buộc trên toàn bộ
# 151936 token ở mỗi bước và tụt từ ~6 xuống ~1,7 tok/s, nên độ dài câu trả lời
# chính là độ trễ. Một từ ≈ 2–3 token.
GRAMMAR = "root ::= " + " | ".join(f'"{c}"' for c in COMMANDS) + "\n"

SYSTEM = (
    "Ban chuyen cau noi tieng Viet cua nguoi choi co vua thanh MOT lenh duy nhat.\n"
    "goi_y      : xin nuoc di tiep theo, hoi nen di gi, xin goi y.\n"
    "danh_gia   : hoi the co the nao, ai dang loi, phan tich van co.\n"
    "van_moi    : bat dau van moi, choi lai tu dau.\n"
    "dong_bo    : bao AI nhin lai ban co, canh lai camera, dong bo lai the co.\n"
    "khong_hieu : cau KHONG lien quan den co vua (quang cao, phu de video, tap am).\n"
    "Tra loi dung MOT tu trong danh sach tren, khong giai thich.\n"
    "Chi tra ve khong_hieu khi cau RO RANG khong phai chuyen co vua. Con moi cau "
    "noi ve the co, nuoc di hay cach choi deu phai chon mot lenh phu hop."
)

# Few-shot KHÔNG phải tuỳ chọn: chỉ với chỉ thị "đừng đoán bừa", Qwen3-4B đọc đó
# là giấy phép để từ chối tất cả và loại luôn cả những yêu cầu cờ hợp lệ.
EXAMPLES = [
    ("đánh cho tôi nước tiếp theo", "goi_y"),
    ("cho tôi nước đi mạnh nhất", "goi_y"),
    # Chữ ASR nghe lệch có thật, quan sát được trên chính board này.
    ("đánh cho tôi nướt tiếp theo", "goi_y"),
    ("thế cờ đang thế nào", "danh_gia"),
    ("mình với đối thủ ai đang hơn", "danh_gia"),
    ("chơi lại từ đầu đi", "van_moi"),
    ("nhìn lại bàn cờ giúp tôi", "dong_bo"),
    ("Hãy subscribe cho kênh để không bỏ lỡ video hấp dẫn", "khong_hieu"),
]


@dataclass
class Command:
    lenh: str
    source: str          # "tu_khoa" | "llm" | "loai" | "llm_loi"
    text: str
    ms: float = 0.0

    @property
    def understood(self) -> bool:
        return self.lenh != "khong_hieu"


# ---------- lớp 2: từ khoá ----------

def _norm(text: str) -> str:
    """Bỏ dấu + thường hoá. Whisper trả dấu thanh không nhất quán, nên
    "tấn công" và "tan cong" đều phải khớp."""
    text = text.lower().replace("đ", "d")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


_CMD = [
    (r"\b(van moi|choi lai|lam lai tu dau|bat dau lai|choi van moi)\b", "van_moi"),
    (r"\b(dong bo|canh lai|nhin lai ban|hieu chinh|do lai ban)\b", "dong_bo"),
    (r"\b(danh gia|the co|ai dang loi|ai hon|ai thang|tinh hinh|phan tich)\b", "danh_gia"),
    (r"\b(goi y|nen di|di nuoc nao|nuoc nao|nuoc di manh|nuoc tot nhat"
     r"|nuoc di tiep theo|nuoc tiep theo|nuoc ke tiep|di gi|choi gi)\b", "goi_y"),
]


def by_keyword(text: str) -> Optional[Command]:
    low = _norm(text)
    lenh = next((c for pat, c in _CMD if re.search(pat, low)), None)
    return None if lenh is None else Command(lenh, "tu_khoa", text)


# ---------- lớp 3: LLM có ràng buộc ----------

def by_llm(text: str, timeout: float = 90.0) -> Command:
    """Đo trên board, trong lúc coach_server vẫn đang chạy nhận diện:

        402 token few-shot ĐÃ CACHE · 12–16 token mới · prefill 5–7 s ·
        sinh 4–5 token ở 2,2 tok/s (grammar) · TỔNG 7,6–8,7 s

    Prefill 14 token mà mất 6 s là vì tranh CPU: vòng nhận diện ăn ~4 nhân trong
    6 nhân, llama-server chỉ còn phần thừa. Timeout để rộng tay vì lần gọi ĐẦU
    TIÊN (chưa cache) tốn ~43 s — `warmup()` lúc khởi động dịch vụ là để không ai
    phải trả khoản đó khi đang nói.
    """
    t0 = time.time()
    messages = [{"role": "system", "content": SYSTEM}]
    for user, answer in EXAMPLES:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": text})

    out = llm.chat(messages, temperature=0.0, max_tokens=8,
                   timeout=timeout, grammar=GRAMMAR)
    ms = (time.time() - t0) * 1000
    if out is None:
        # LLM chết thì đường giọng nói không được chết theo: từ khoá vẫn chạy,
        # và câu này chỉ đơn giản là không hiểu được.
        return Command("khong_hieu", "llm_loi", text, ms)
    word = out.strip().split()[0] if out.strip() else ""
    return Command(word if word in COMMANDS else "khong_hieu", "llm", text, ms)


def resolve(text: str, use_llm: bool = True) -> Command:
    """Câu đã nghe → lệnh, lớp rẻ nhất trước."""
    if not text.strip():
        return Command("khong_hieu", "loai", text)
    hit = by_keyword(text)
    if hit is not None:
        return hit
    if not use_llm:
        return Command("khong_hieu", "tu_khoa_khong_khop", text)
    return by_llm(text)


# ---------- lớp 1: Whisper trên NPU ----------

def resolve_mic() -> str:
    """Thiết bị thu, dò theo TÊN card chứ không đóng đinh số thứ tự.

    `asr.py` mặc định `plughw:0,0`, mà số card do thứ tự kernel nhận thiết bị
    quyết định — cùng đúng lý do kit phải có `KIT_CAM` cho `/dev/video*`. Cắm
    thêm một thiết bị âm thanh, hoặc chỉ cần bật nguồn lại theo thứ tự khác, là
    C270 có thể thành card 1 và mọi bản thu ra `Device or resource busy` hoặc
    file rỗng — một lỗi trông y hệt lỗi phần cứng.

    Đè bằng `KIT_MIC=plughw:1,0` nếu cần.
    """
    env = os.environ.get("KIT_MIC")
    if env:
        return env
    try:
        # " 0 [WEBCAM         ]: USB-Audio - C270 HD WEBCAM"
        with open("/proc/asound/cards") as fh:
            for line in fh:
                m = re.match(r"\s*(\d+)\s+\[", line)
                if m and any(k in line.upper() for k in ("WEBCAM", "C270", "USB")):
                    return f"plughw:{m.group(1)},0"
    except OSError:
        pass
    return "plughw:0,0"


def _load_asr_module():
    """Nạp `asr.py` từ WHISPER_DIR mà không cần nó nằm trong PYTHONPATH.

    `asr` tự import `whisper_npu`, nên thư mục phải có mặt trong sys.path trước —
    nạp bằng spec thôi thì import lồng bên trong sẽ hỏng.
    """
    if WHISPER_DIR not in sys.path:
        sys.path.insert(0, WHISPER_DIR)
    path = os.path.join(WHISPER_DIR, "asr.py")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"khong thay {path} — chay tools/install_voice.sh de deploy bo Whisper")
    spec = importlib.util.spec_from_file_location("kit_asr", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kit_asr"] = mod
    spec.loader.exec_module(mod)
    return mod


class VoiceCommander:
    """Whisper thường trú + lớp hiểu lệnh. An toàn khi gọi từ nhiều luồng.

    Worker Whisper nạp 2 context binary (358 MB) mất ~1,4 s, trả MỘT LẦN rồi giữ.
    Vì vậy nó được khởi động ngầm lúc dịch vụ lên, không phải lúc người dùng bấm
    nút — nếu không, lần nói đầu tiên phải chờ cả phần nạp lẫn phần nhận dạng.
    """

    def __init__(self, seconds: float = 4.0, lang: str = "vi") -> None:
        self.seconds = seconds
        self.lang = lang
        self._asr = None
        self._lock = threading.Lock()      # worker chỉ phục vụ 1 yêu cầu/lần
        self.error = ""
        self.startup_s = 0.0
        self.mic = ""                      # đặt ở start(), xem resolve_mic()

    # -- vòng đời --
    def start(self) -> bool:
        """Khởi động worker. Trả False và ghi `self.error` nếu không lên được."""
        with self._lock:
            if self._asr is not None:
                return True
            try:
                mod = _load_asr_module()
                self.mic = resolve_mic()
                mod.MIC_DEVICE = self.mic      # asr.listen() đọc biến module này
                t0 = time.time()
                self._asr = mod.Recognizer(models=WHISPER_DIR, lang=self.lang)
                self.startup_s = time.time() - t0
                self.error = ""
                return True
            except Exception as exc:
                # Giữ nguyên văn lỗi: mọi kiểu chết của worker mà quy về một câu
                # "không khởi động được" là tự bịt mắt mình lúc gỡ rối.
                self.error = f"{type(exc).__name__}: {exc}"
                self._asr = None
                return False

    def close(self) -> None:
        with self._lock:
            if self._asr is not None:
                try:
                    self._asr.close()
                except Exception:
                    pass
                self._asr = None

    @property
    def ready(self) -> bool:
        return self._asr is not None

    # -- nhận dạng --
    def listen(self, seconds: Optional[float] = None) -> tuple[str, dict]:
        """Thu micro rồi phiên âm. Trả ("", stats) khi không nghe thấy tiếng nói."""
        if self._asr is None and not self.start():
            return "", {"error": self.error}
        with self._lock:
            t0 = time.time()
            try:
                text = self._asr.listen(seconds or self.seconds)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                return "", {"error": self.error}
            stats = dict(self._asr.last_stats)
            stats["peak"] = getattr(self._asr, "last_peak", 0)
            stats["wall_ms"] = (time.time() - t0) * 1000
        return text, stats

    def listen_command(self, seconds: Optional[float] = None,
                       use_llm: bool = True) -> tuple[Command, dict]:
        text, stats = self.listen(seconds)
        cmd = resolve(text, use_llm=use_llm)
        return cmd, stats


if __name__ == "__main__":       # python3 -m chess_ai.voice ["câu thử"]
    if len(sys.argv) > 1:
        for s in sys.argv[1:]:
            c = resolve(s)
            print(f"{'OK ' if c.understood else 'BO '} {c.lenh:<11} "
                  f"[{c.source:<8} {c.ms:5.0f}ms]  {s}")
    else:
        vc = VoiceCommander()
        if not vc.start():
            print("khong khoi dong duoc Whisper:", vc.error)
            sys.exit(1)
        print(f"worker san sang sau {vc.startup_s:.2f}s — NOI DI ({vc.seconds:.0f}s)...")
        cmd, st = vc.listen_command()
        print(f'nghe: "{cmd.text}"  (peak={st.get("peak")}, '
              f'{st.get("total_ms", 0)/1000:.2f}s)')
        print(f"lenh: {cmd.lenh}  [{cmd.source}]")
        vc.close()
