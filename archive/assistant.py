"""TRỢ LÝ ĐỐI THOẠI — trả lời câu hỏi của người chơi về thế cờ (tiếng Việt).

HAI TẦNG, tự động chuyển:
  1. Có LLM  -> đối thoại tự do, hiểu câu hỏi bất kỳ, giải thích như HLV thật.
  2. Không có LLM -> quay về bộ luật từ khóa cũ (chạy offline 100%, không bao giờ chết).

NGUYÊN TẮC CHỐNG BỊA: LLM KHÔNG được tự nghĩ nước đi. Mọi nước đi, điểm số,
cảnh báo đều do Stockfish + analysis.py cung cấp sẵn trong khối DỮ LIỆU. LLM chỉ
diễn đạt lại và trả lời câu hỏi mở dựa trên đúng khối dữ liệu đó.
"""
from __future__ import annotations
import chess
import unicodedata

from chess_ai import analysis
from chess_ai import llm
from chess_ai.engine import ChessEngine


def _norm(s: str) -> str:
    s = s.lower().strip()
    # "đ" (U+0111) không tách dấu bằng NFD -> phải thay tay
    s = s.replace("đ", "d").replace("Đ", "d")
    s = "".join(c for c in unicodedata.normalize("NFD", s)
                if unicodedata.category(c) != "Mn")
    return s


# ý định -> danh sách từ khóa (đã bỏ dấu). Dùng cho tầng dự phòng + gợi ý độ dài.
INTENTS = {
    "evaluate": ["danh gia", "the co the nao", "the tran", "ai dang thang", "ai loi",
                 "tinh hinh", "the nao roi", "danh gia the tran"],
    "suggest": ["nen di", "goi y", "nuoc nao", "di gi", "di nuoc nao", "khuyen",
                "nen danh", "choi gi", "tu van"],
    "why": ["tai sao", "vi sao", "ly do", "vi sao vay"],
    "threats": ["nguy hiem", "bi treo", "de doa", "canh bao", "coi chung", "co gi nguy",
                "quan nao bi", "an duoc gi"],
    "material": ["vat chat", "con bao nhieu quan", "hon quan", "kem quan", "so quan"],
    "phase": ["giai doan", "khai cuoc", "trung cuoc", "tan cuoc dang o dau"],
    "plan": ["ke hoach", "chien luoc", "lam gi tiep", "dinh huong", "y tuong"],
    "help": ["giup", "lam sao", "huong dan", "ban lam gi duoc", "hoi gi"],
    "repeat": ["lap lai", "noi lai", "nhac lai"],
}

# hỏi kiểu này thì cho phép trả lời dài hơn
DEEP_KEYWORDS = ["phan tich", "chi tiet", "giai thich", "day", "hoc", "vi sao", "tai sao",
                 "ke hoach", "chien luoc", "day cho", "giang"]

SYSTEM_PROMPT = """Bạn là một huấn luyện viên cờ vua người Việt, giàu kinh nghiệm, nói chuyện thân thiện và dễ hiểu.

QUY TẮC BẮT BUỘC:
1. CHỈ được dùng thông tin trong khối "DỮ LIỆU THẾ CỜ". Tuyệt đối KHÔNG tự nghĩ ra nước đi, không tự tính biến, không bịa tên quân hay ô cờ.
2. Khi nói về nước nên đi, CHỈ được nhắc các nước có trong mục "Phương án Stockfish". Nếu dữ liệu không có, hãy nói thẳng là chưa rõ.
3. Câu trả lời sẽ được ĐỌC THÀNH TIẾNG. Vì vậy: không markdown, không gạch đầu dòng, không ký hiệu lạ, không emoji. Viết thành câu văn xuôi liền mạch.
4. Không đọc số kỹ thuật thô như "centipawn", "+35 cp", "eval". Hãy quy đổi thành lời: "nhỉnh hơn chút ít", "hơn hẳn một quân Tốt", "ưu thế lớn".
5. Xưng "tôi", gọi người chơi là "bạn". Giọng khích lệ, không chê bai nặng nề.
6. Mặc định trả lời NGẮN GỌN 2 đến 4 câu. Chỉ khi người ta hỏi "phân tích", "giải thích", "tại sao", "kế hoạch" thì mới nói dài hơn, tối đa 8 câu.
7. Trả lời thẳng vào câu hỏi được hỏi. Không lặp lại toàn bộ dữ liệu."""


class Assistant:
    def __init__(self, settings, engine: ChessEngine, use_llm: bool = True):
        self.settings = settings
        self.engine = engine
        self.use_llm = use_llm
        self._last = ""
        self._history: list[dict] = []      # [{role, content}] các lượt hỏi–đáp gần đây
        self._max_turns = 6                 # nhớ 6 lượt gần nhất

    # ------------------------------------------------------------------ #
    #  ĐIỀU PHỐI
    # ------------------------------------------------------------------ #
    def answer(self, board: chess.Board, question: str) -> str:
        intent = self._match(question)

        # "nhắc lại" xử lý tại chỗ, không tốn một lượt LLM
        if intent == "repeat":
            return self._last or "Tôi chưa có gì để nhắc lại."

        ans = None
        if self.use_llm and llm.available():
            ans = self._answer_llm(board, question, intent)

        if not ans:
            ans = self._answer_template(board, intent)

        self._last = ans
        return ans

    def reset(self) -> None:
        """Ván mới — quên hội thoại cũ."""
        self._history.clear()
        self._last = ""

    @property
    def llm_ready(self) -> bool:
        return self.use_llm and llm.available()

    def status_line(self) -> str:
        if self.llm_ready:
            return f"Trợ lý: LLM {llm.model_name()}"
        return "Trợ lý: chế độ offline (mẫu câu)"

    # ------------------------------------------------------------------ #
    #  TẦNG 1 — LLM
    # ------------------------------------------------------------------ #
    def _answer_llm(self, board: chess.Board, question: str, intent: str) -> str | None:
        facts = self._build_facts(board)
        deep = any(k in _norm(question) for k in DEEP_KEYWORDS)

        user_msg = (
            f"DỮ LIỆU THẾ CỜ (do Stockfish tính, là sự thật duy nhất):\n"
            f"{facts}\n\n"
            f"CÂU HỎI CỦA NGƯỜI CHƠI: {question}\n\n"
            f"Trả lời bằng tiếng Việt, "
            + ("giải thích kỹ như đang dạy học." if deep else "ngắn gọn 2-3 câu.")
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self._history[-self._max_turns * 2:])
        messages.append({"role": "user", "content": user_msg})

        out = llm.chat(messages,
                       temperature=0.4,
                       max_tokens=600 if deep else 260)
        if not out:
            return None

        out = self._clean(out)
        if not out:
            return None

        # chỉ lưu câu hỏi gốc vào lịch sử (không lưu khối dữ liệu -> đỡ phình ngữ cảnh)
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": out})
        if len(self._history) > self._max_turns * 2:
            self._history = self._history[-self._max_turns * 2:]
        return out

    @staticmethod
    def _clean(text: str) -> str:
        """Bỏ rác markdown để đọc lên nghe không kỳ."""
        bad = ["**", "*", "#", "`", "- ", "•", "—", "###"]
        for b in bad:
            text = text.replace(b, " ")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        text = " ".join(lines)
        while "  " in text:
            text = text.replace("  ", " ")
        return text.strip()

    # ------------------------------------------------------------------ #
    #  XÂY KHỐI DỮ LIỆU CHO LLM  (đây là thứ quyết định chất lượng phân tích)
    # ------------------------------------------------------------------ #
    def _build_facts(self, board: chess.Board) -> str:
        r = analysis.build_report(board, self.engine)
        human = self._human_color()
        L: list[str] = []

        L.append(f"Người chơi cầm: {analysis.COLOR_VI[human]}")
        L.append(f"Đang tới lượt: {analysis.COLOR_VI[board.turn]}")
        L.append(f"Giai đoạn: {r.phase}")

        # điểm đánh giá — diễn giải sẵn để LLM khỏi tự quy đổi sai
        L.append(f"Đánh giá: {analysis._eval_sentence(r.eval)}")
        if r.eval.mate_in is not None:
            L.append(f"CHIẾU HẾT trong {abs(r.eval.mate_in)} nước "
                     f"({analysis.COLOR_VI[r.eval.mate_in > 0]} thắng)")

        # phương án — nguồn DUY NHẤT mà LLM được phép nhắc tới
        cands = getattr(r.eval, "candidates", None) or []
        if cands:
            parts = []
            for c in cands[:3]:
                pv_txt = ""
                if getattr(c, "pv", None):
                    pv_txt = ""     # PV dạng Move, để trống cho gọn; best_line_san bên dưới đủ dùng
                parts.append(f"{c.san}{pv_txt}")
            L.append("Phương án Stockfish (chỉ được nhắc các nước này): " + ", ".join(parts))
        elif r.best_line_san:
            L.append("Phương án Stockfish (chỉ được nhắc các nước này): " + r.best_line_san[0])
        else:
            L.append("Phương án Stockfish: chưa có")

        if r.best_line_san:
            L.append("Biến chính dự kiến: " + " ".join(r.best_line_san[:5]))

        # vật chất
        diff = r.white.material - r.black.material
        if diff == 0:
            L.append("Vật chất: cân bằng")
        else:
            lead = analysis.COLOR_VI[chess.WHITE] if diff > 0 else analysis.COLOR_VI[chess.BLACK]
            L.append(f"Vật chất: {lead} hơn {abs(diff)} điểm quân")

        # nguy hiểm
        warn = []
        for c in (chess.WHITE, chess.BLACK):
            h = r.hanging.get(c)
            if h:
                warn.append(f"{analysis.COLOR_VI[c]} đang treo {h}")
        if r.checks_available:
            warn.append("có nước chiếu đáng lưu ý")
        L.append("Nguy hiểm: " + ("; ".join(warn) if warn else "chưa thấy đòn tức thời"))

        # đặc điểm hai bên
        for col, sr in ((chess.WHITE, r.white), (chess.BLACK, r.black)):
            bits = [f"trung tâm {sr.center_control}", f"quân nhẹ đã phát triển {sr.developed}",
                    f"độ chủ động {sr.mobility}"]
            if sr.bishop_pair:
                bits.append("có cặp Tượng")
            if sr.doubled:
                bits.append(f"{sr.doubled} Tốt chồng")
            if sr.isolated:
                bits.append(f"{sr.isolated} Tốt cô lập")
            if sr.passed:
                bits.append(f"Tốt thông ở {', '.join(str(x) for x in sr.passed)}")
            if sr.open_rooks:
                bits.append(f"{sr.open_rooks} Xe trên cột mở")
            bits.append(f"Vua có {sr.king_shield} Tốt che, {sr.king_open_files} cột mở cạnh Vua")
            L.append(f"{analysis.COLOR_VI[col]}: " + ", ".join(bits))

        # lịch sử nước đi gần đây
        hist = self._recent_sans(board, 6)
        if hist:
            L.append("Các nước gần đây: " + " ".join(hist))

        if board.is_check():
            L.append("LƯU Ý: đang bị chiếu")

        return "\n".join(L)

    @staticmethod
    def _recent_sans(board: chess.Board, n: int) -> list[str]:
        """Lấy n nước cuối dạng SAN (phải phát lại ván vì SAN phụ thuộc thế cờ)."""
        try:
            tmp = chess.Board()
            sans = []
            for mv in board.move_stack:
                sans.append(tmp.san(mv))
                tmp.push(mv)
            return sans[-n:]
        except Exception:
            return []

    # ------------------------------------------------------------------ #
    #  TẦNG 2 — MẪU CÂU (offline, y như bản cũ)
    # ------------------------------------------------------------------ #
    def _match(self, text: str) -> str:
        t = _norm(text)
        best, score = "unknown", 0
        for intent, kws in INTENTS.items():
            for kw in kws:
                if kw in t:
                    if len(kw) > score:
                        best, score = intent, len(kw)
        return best

    def _answer_template(self, board: chess.Board, intent: str) -> str:
        rep = None

        def report():
            nonlocal rep
            if rep is None:
                rep = analysis.build_report(board, self.engine)
            return rep

        if intent == "help":
            return ("Bạn có thể hỏi tôi: 'đánh giá thế cờ', 'nên đi nước nào', "
                    "'có gì nguy hiểm không', 'vật chất thế nào', 'kế hoạch là gì', "
                    "hoặc 'tại sao'. Tôi vừa là đối thủ, vừa là huấn luyện viên của bạn.")

        if intent == "evaluate":
            r = report()
            lines = analysis.narrate(r, "normal", for_color=self._human_color())
            return " ".join(lines[:3])
        if intent == "suggest":
            r = report()
            if r.best_line_san:
                return (f"Theo tôi, bạn nên cân nhắc {r.best_line_san[0]}. "
                        + analysis._eval_sentence(r.eval))
            return "Chưa có nước đi rõ ràng."
        if intent == "why":
            r = report()
            if r.best_line_san:
                return (f"Vì phương án {' '.join(r.best_line_san[:3])} giữ hoặc tăng ưu thế. "
                        + analysis._eval_sentence(r.eval))
            return analysis._eval_sentence(r.eval)
        if intent == "threats":
            r = report()
            msgs = []
            for c in (chess.WHITE, chess.BLACK):
                h = r.hanging.get(c)
                if h:
                    msgs.append(f"{analysis.COLOR_VI[c]} đang treo {h}")
            if r.checks_available:
                msgs.append("có nước chiếu cần lưu ý")
            return ("Cảnh báo: " + "; ".join(msgs) + ".") if msgs else "Hiện chưa thấy đòn nguy hiểm tức thời."
        if intent == "material":
            r = report()
            diff = r.white.material - r.black.material
            if diff == 0:
                return "Vật chất cân bằng."
            lead = analysis.COLOR_VI[chess.WHITE] if diff > 0 else analysis.COLOR_VI[chess.BLACK]
            return f"{lead} đang hơn {abs(diff)} điểm quân."
        if intent == "phase":
            return f"Chúng ta đang ở giai đoạn {report().phase}."
        if intent == "plan":
            r = report()
            lines = analysis.narrate(r, "detailed", for_color=self._human_color())
            return " ".join(lines[-3:])

        return ("Xin lỗi, tôi chưa hiểu. Bạn thử hỏi 'đánh giá thế cờ', "
                "'nên đi nước nào', hay 'có gì nguy hiểm không'.")

    def _human_color(self):
        return chess.WHITE if self.settings.human_color == "white" else chess.BLACK
