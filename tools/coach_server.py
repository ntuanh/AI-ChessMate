#!/usr/bin/env python3
"""HLV CỜ VUA TRỰC TIẾP — bản MƯỢT + GIAO DIỆN ĐẸP.
Tách luồng quay/nhận diện; 720p + làm nét; dashboard sinh động (badge nước đi,
thanh đánh giá, trạng thái, hiệu ứng), phát tiếng trên trình duyệt. http://127.0.0.1:8090"""
import sys, os, time, json, threading, queue
from urllib.parse import parse_qs, urlparse
# Gốc dự án suy từ vị trí file này, không đóng đinh một đường dẫn cố định: code có thể nằm ở
# /home/<user>/kit. Đè được bằng KIT_ROOT nếu cần.
KIT_ROOT = os.environ.get("KIT_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAP_DIR = os.path.join(KIT_ROOT, "captures")
sys.path.insert(0, KIT_ROOT)
import numpy as np
import cv2
import chess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chess_ai import vision, config, render, gridfind, rectify, reader, piece_net, tracker3
from chess_ai.engine import ChessEngine
from chess_ai import analysis as ana
from chess_ai.speaker import Speaker

S = config.Settings(human_color="white", skill_level=12,
                    think_time=0.3, analysis_time=0.35, speak=False)
engine = ChessEngine(S)
speaker = Speaker(S)

STATE = {"jpeg": None, "raw": None, "board": None, "msg": "Đang khởi động...",
         "moves": [], "audio_id": 0, "wav": b"", "n": 0, "ready": False,
         "hint": "—", "eval": "", "cp": 0, "your_turn": True,
         "corners": None, "track": "KHỞI ĐỘNG", "caro": 0.0, "bver": 0,
         "cam_seen": 0.0}   # lần cuối trình duyệt xin ảnh camera
latest = {"frame": None, "seq": 0}
lock = threading.Lock()
board = chess.Board()
START_FEN = chess.STARTING_FEN
model = None            # gridfind.OccupancyModel (tự học lúc canh bàn)
tracker = None          # rectify.Tracker (bám 4 góc, camera rung vẫn đuổi)
# PieceNet đọc TỪNG QUÂN nên tốt hơn occupancy hẳn một bậc; thiếu model/onnxruntime
# thì trả None và cả pipeline tự lùi về đường occupancy cũ.
pnet = piece_net.load()
print(f"PieceNet: {type(pnet).__name__ if pnet else 'KHONG CO'}"
      + (f" ({pnet.path})" if pnet else ""), flush=True)
recal = threading.Event()
diag_reset = threading.Event()   # nút ↺ trên web: đo lại từ 0 cho một thử nghiệm mới
# C270 cắm vào AIBOX có thể ra /dev/video0|1|2 tuỳ thứ tự nhận thiết bị, nên để
# đặt được từ ngoài: KIT_CAM=0 python3 tools/coach_server.py
CAM = int(os.environ.get("KIT_CAM", "2"))
CALIB_PATH = os.path.join(CAP_DIR, "board_calib.json")

AUDIO_CACHE = {}          # {audio_id: bytes}
AUDIO_CACHE_LOCK = threading.Lock()
SPEAK_QUEUE = queue.Queue()
LAST_SPOKEN_TEXT = ""


def _speak_worker_loop():
    global LAST_SPOKEN_TEXT
    while True:
        try:
            text = SPEAK_QUEUE.get()
            if not text:
                continue
            if text == LAST_SPOKEN_TEXT:
                SPEAK_QUEUE.task_done()
                continue
            LAST_SPOKEN_TEXT = text

            with lock:
                next_id = STATE["audio_id"] + 1

            p = os.path.join(CAP_DIR, f"coach_speak_{next_id}.wav")
            try:
                if speaker.synth_to_wav(text, p):
                    with open(p, "rb") as f:
                        data = f.read()
                    with AUDIO_CACHE_LOCK:
                        AUDIO_CACHE[next_id] = data
                        if len(AUDIO_CACHE) > 30:
                            for k in sorted(AUDIO_CACHE.keys())[:-30]:
                                del AUDIO_CACHE[k]
                    with lock:
                        STATE["wav"] = data
                        STATE["audio_id"] = next_id
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            except Exception:
                pass
            finally:
                SPEAK_QUEUE.task_done()
        except Exception:
            pass


def set_msg(text, do_speak=True):
    with lock:
        STATE["msg"] = text
    if do_speak and text:
        SPEAK_QUEUE.put(text)


from chess_ai import commentary


HUMAN_COLOR = chess.WHITE

def push_advice(prefix, b):
    # Người chơi (ở phía dưới bàn cờ) cầm Trắng hoặc Đen (tự động dò)
    is_user_turn = (b.turn == HUMAN_COLOR)

    a = engine.analyse(b, multipv=3)
    cp = a.score_cp if a.score_cp is not None else (3000 if (a.mate_in or 0) > 0 else -3000)
    candidates = getattr(a, "candidates", []) or []
    cand_sans = [c.san for c in candidates if hasattr(c, "san")]

    if a.best_move is not None:
        try:
            best_san = b.san(a.best_move)
        except Exception:
            best_san = a.best_move.uci()
    else:
        best_san = "—"
    hint_str = cand_sans[0] if cand_sans else best_san

    if is_user_turn:
        # Quy tắc 1: Trạng thái bình thường — CHỈ hiển thị/đọc duy nhất nước đi tốt nhất
        speech_text = commentary.generate_short_move(b, a)
        do_speak = True
    else:
        speech_text = "Đang chờ đối thủ đi..."
        do_speak = False

    with lock:
        STATE["hint"] = hint_str if is_user_turn else "—"
        STATE["eval"] = "Thế cờ cân bằng" if abs(cp) < 80 else ("Trắng ưu thế" if cp > 0 else "Đen ưu thế")
        STATE["cp"] = int(max(-1000, min(1000, cp)))
        STATE["your_turn"] = is_user_turn
    set_msg(speech_text, do_speak=do_speak)


ADVICE_Q = queue.Queue(maxsize=1)


def _advice_async(prefix, b):
    """Đẩy việc phân tích sang luồng riêng.

    push_advice() gọi engine.analyse (analysis_time=0.35s) và nó đang nằm NGAY TRONG
    vòng nhận diện, nên mỗi lần chốt được một nước là vòng đó đứng ~0.35-0.7s: không
    frame nào được xử lý, và nước trả lời của đối thủ bị bắt muộn đúng bằng khoảng
    đó. Đây là lý do "lúc chậm lúc nhanh" — ván càng nhiều nước liên tiếp thì càng
    dồn. Chuyển sang hàng đợi 1 chỗ: giữ yêu cầu MỚI NHẤT, bỏ yêu cầu cũ đã lạc hậu,
    và vẫn chỉ một luồng gọi engine (SimpleEngine không dùng song song được).
    """
    try:
        ADVICE_Q.put_nowait((prefix, b.copy()))
        return
    except queue.Full:
        pass
    try:
        ADVICE_Q.get_nowait()
    except queue.Empty:
        pass
    try:
        ADVICE_Q.put_nowait((prefix, b.copy()))
    except queue.Full:
        pass


def _advice_worker():
    while True:
        prefix, b = ADVICE_Q.get()
        try:
            push_advice(prefix, b)
        except Exception as e:
            print("advice worker:", e, flush=True)
        finally:
            ADVICE_Q.task_done()


def board_grid(b, flipped=False):
    g = [["."] * 8 for _ in range(8)]
    for sq in chess.SQUARES:
        p = b.piece_at(sq)
        if p:
            r = chess.square_rank(sq)
            f = chess.square_file(sq)
            row = r if flipped else (7 - r)
            col = (7 - f) if flipped else f
            g[row][col] = p.symbol()
    return g


def _mismatch(b, occ):
    flipped = (HUMAN_COLOR == chess.BLACK)
    return int((gridfind.expected_mask(b, flipped=flipped) != occ).sum())


def detect_move_occ(b, occ):
    """Vỏ mỏng gọi reader.explain_occ — logic nằm ở chess_ai/reader.py để test được
    rời khỏi camera/engine (xem tests/test_reader.py)."""
    return reader.explain_occ(b, occ, flipped=(HUMAN_COLOR == chess.BLACK))


def try_resync(occ):
    """TỰ HỒI PHỤC khi game state lệch bàn thật kéo dài. Thử 4 giả thuyết:
      (a) bỏ lỡ 1 nước;  (b) bỏ lỡ 2 nước đi nhanh;
      (c) nước cuối là THỪA (chưa ai đi);  (d) nước cuối ghi NHẦM.

    Chấm bằng LỆCH NHỎ NHẤT, không đòi khớp 100%. Bản cũ đòi `_mismatch == 0`
    tuyệt đối, và đó là code chết: đo trên 1000 khung thật, occupancy báo THỪA
    291 ô so với ĐÚNG MỘT ô bỏ sót, phần lớn dồn vào một cột do lưới căn sát mép
    bàn. Ô thừa đó là lỗi HÌNH HỌC — không thế cờ nào trên đời làm nó biến mất,
    nên mọi lần gọi đều đốt ~122 ms rồi bảo đảm trả False.

    Nới chốt chặn thì phải siết chỗ khác, nếu không sẽ "tự sửa" thành thế SAI:
      - ứng viên phải lệch ÍT HƠN HẲN thế đang giữ (đỡ ít nhất 2 ô), và
      - phải THẮNG RÕ ứng viên nhì (cách ít nhất 2 ô) — hoà thì thà đứng yên, và
      - lệch dư còn lại phải <= 2 ô (bằng đúng mức nhiễu occupancy đo được)."""
    global board
    cur_d = _mismatch(board, occ)
    if cur_d == 0:
        return False
    cands = []                                            # [(lệch, board)]

    def thu(nb):
        cands.append((_mismatch(nb, occ), nb.copy()))

    b = board.copy()
    for mv1 in list(b.legal_moves):                       # (a) + (b)
        if mv1.promotion not in (None, chess.QUEEN):
            continue
        b.push(mv1)
        thu(b)
        # Chỉ nở tầng 2 khi tầng 1 còn lệch đáng kể. Nhánh đã khớp gần hết thì đi
        # thêm một nước nữa chỉ làm tệ đi, mà nở đủ 2 tầng tốn ~813 lần dựng mặt
        # nạ (đo 122 ms) — giữ nguyên phạm vi tìm kiếm, bỏ phần chắc chắn vô ích.
        if cands[-1][0] > 1:
            for mv2 in list(b.legal_moves):
                if mv2.promotion not in (None, chess.QUEEN):
                    continue
                b.push(mv2)
                thu(b)
                b.pop()
        b.pop()

    if board.move_stack:
        b = board.copy()
        wrong = b.pop()
        thu(b)                                            # (c) nước cuối thừa
        for mv in list(b.legal_moves):                    # (d) nước cuối ghi nhầm
            if mv == wrong or mv.promotion not in (None, chess.QUEEN):
                continue
            b.push(mv)
            thu(b)
            b.pop()

    uniq = {}
    for d, nb in cands:
        f = nb.fen()
        if f not in uniq or d < uniq[f][0]:
            uniq[f] = (d, nb)
    xep = sorted(uniq.values(), key=lambda t: t[0])
    if not xep:
        return False
    best_d, best_b = xep[0]
    if best_d > 2 or best_d > cur_d - 2:
        return False                        # không khá hơn đủ nhiều -> chưa sửa
    if len(xep) > 1 and xep[1][0] < best_d + 2:
        return False                        # có ứng viên bám sát -> mơ hồ, chưa sửa

    rb = chess.Board(START_FEN)
    sans = []
    for mv in best_b.move_stack:
        try:
            sans.append(rb.san(mv))
            rb.push(mv)
        except Exception:
            pass
    board = best_b
    with lock:
        STATE["moves"] = sans
    tail = " ".join(sans[-2:]) if sans else "(chưa có nước nào)"
    _advice_async(f"🔧 Đã tự động đồng bộ theo bàn thật (…{tail}). ", board)
    return True


def set_exposure():
    os.system(f"v4l2-ctl -d /dev/video{CAM} --set-ctrl=auto_exposure=3 2>/dev/null")
    # khử vằn do lệch tần số đèn/màn hình (điện lưới VN 50Hz)
    os.system(f"v4l2-ctl -d /dev/video{CAM} --set-ctrl=power_line_frequency=1 2>/dev/null")


def capture_thread():
    cap = vision.open_camera(CAM)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    for _ in range(8):
        cap.read(); time.sleep(0.03)
    set_exposure()
    aim_cor, aim_t, last_enc = None, 0.0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03); continue
        latest["frame"] = frame
        latest["seq"] += 1
        # cap.read() đã tự chặn theo nhịp camera, nên vòng này KHÔNG cần ngủ. Trước
        # đây ngủ 0.05s mỗi vòng trong khi driver vẫn xếp frame vào hàng đợi, nên
        # frame đọc ra ngày càng cũ so với thực tế — đó chính là phần "delay" nhìn
        # thấy trên ảnh, và nó cũng làm nước đi bị bắt muộn. Chỉ hãm việc NÉN ảnh
        # (20 ảnh/giây là đủ cho mắt); frame cấp cho nhận diện luôn là frame mới nhất.
        now = time.time()
        if now - last_enc < 0.05:
            continue
        last_enc = now
        disp = frame.copy()
        col = (80, 220, 120) if STATE["ready"] else (60, 160, 255)
        cor = STATE.get("corners")
        if cor is not None:
            cv2.polylines(disp, [np.array(cor, dtype=np.int32)], True, col, 2)
        elif not STATE["ready"]:
            # Khung cam giúp ngắm camera. Nhưng auto_detect_corners là một bộ dò
            # contour trên cả frame 720p, mà trước đây nó chạy MỖI frame (20 lần/s)
            # đúng lúc build_reference đang cần CPU để dò bàn thật — hai bên giành
            # nhau CPU nên chính pha canh bàn bị kéo dài. Nó chỉ để vẽ cho người xem
            # nên 2 lần/giây là đủ, và giữ lại kết quả cũ để khung không nháy.
            now = time.time()
            if now - aim_t >= 0.5:
                aim_cor, aim_t = vision.auto_detect_corners(frame), now
            if aim_cor is not None:
                cv2.polylines(disp, [aim_cor.astype(np.int32)], True, (255, 140, 0), 2)
                cv2.putText(disp, "Giu yen...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
        txt = f"{STATE['track']} | caro={STATE['caro']} | {STATE['n']} o co quan"
        cv2.putText(disp, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        # CHỈ encode ảnh mà trình duyệt đang xem. Bản /raw là endpoint gỡ lỗi, hầu
        # như không ai mở, nhưng trước đây nó vẫn bị nén JPEG 720p q88 hai mươi lần
        # mỗi giây — nay nén lúc có người gọi (xem handler /raw).
        # Không ai mở trình duyệt thì khỏi nén: nén 720p q80 hai mươi lần mỗi giây
        # tốn CPU thật, mà AIBOX chạy không màn hình gần như suốt ngày. Nén tiếp
        # thêm 3s sau lần xem cuối để lần mở lại không thấy ảnh đen.
        if now - STATE.get("cam_seen", 0.0) > 3.0:
            continue
        ok2, jpg = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok2:
            with lock:
                STATE["jpeg"] = jpg.tobytes()


def _set_track(mode, caro=None):
    with lock:
        STATE["track"] = mode
        if caro is not None:
            STATE["caro"] = round(float(caro), 1)


def build_reference():
    global HUMAN_COLOR, model, tracker, START_FEN, board
    sb = chess.Board()
    try:
        _fen = open(os.path.join(CAP_DIR, "start_fen.txt")).read().strip()
        if _fen:
            sb = chess.Board(_fen)
    except Exception:
        pass
    ref = latest["frame"]
    if ref is None:
        return False
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    if gridfind.sharpness(gray) < gridfind.SHARP_MIN:
        set_msg("Ảnh đang mờ — giữ camera yên giúp tôi.", do_speak=False)
        return False

    # Dò bàn bằng rectify: quad được TINH CHỈNH để tối đa hoá điểm caro rồi mới
    # được nhận, nên không còn cảnh warp vào viền laptop/khung browser như trước.
    # Thử QUAD ĐÃ LƯU trước: camera gắn cố định vào AIBOX nên giữa hai lần chạy
    # bàn hầu như không dịch. verify() vẫn đòi đủ ngưỡng caro + ràng buộc hình học
    # y như detect(), chỉ bỏ bước rải 600 mồi — không đạt thì rơi xuống detect()
    # đầy đủ ngay bên dưới, nên đây là đường NHANH chứ không phải đường dễ.
    corners, score = None, 0.0
    _saved = gridfind.load_calib(CALIB_PATH)
    if _saved is not None:
        corners, score = rectify.verify(ref, _saved[0])
        if corners is not None:
            print(f"canh bàn từ calib đã lưu: caro={score:.1f} (khỏi quét toàn khung)",
                  flush=True)
    if corners is None:
        corners, score = rectify.detect(ref)
    if corners is None or score < rectify.ACCEPT:
        set_msg("Chưa thấy bàn cờ trong khung hình — đưa bàn vào giữa, giảm loá.",
                do_speak=False)
        return False
    if rectify.frame_coverage(ref, corners) < rectify.MIN_INSIDE:
        set_msg("Bàn cờ tràn ra ngoài khung — lùi camera ra hoặc chỉnh lại góc.",
                do_speak=False)
        return False
    warped = rectify.rectify(ref, corners).board

    # Có PieceNet thì ĐỌC THẲNG thế cờ đang có: không cần đoán "chắc đang ở thế
    # khai cuộc" như đường occupancy, nên canh bàn được cả khi vào giữa ván.
    # PieceNet đọc không chắc thì KHÔNG được bỏ cuộc: phải rơi xuống đường
    # occupancy phía dưới. Trả False ở đây là treo vĩnh viễn, vì recog_thread chỉ
    # gọi lại build_reference chứ không có nhánh nào khác.
    b_read, wb = None, None
    if pnet is not None:
        grid, conf = pnet.predict(warped)
        wb = pnet.white_is_bottom(warped)
        n_unsure = int((np.asarray(conf) < reader.CONF_MIN).sum())
        if wb is None or n_unsure > reader.MAX_UNSURE:
            print(f"PieceNet chưa chắc ({n_unsure} ô mờ) -> dùng occupancy", flush=True)
        else:
            # Dò từ thế khai cuộc TRƯỚC. Ảnh tĩnh không nói được ai đang đi, mà
            # board_from_grid phải đoán và nó thử Trắng trước — nên khi bạn cầm Đen
            # và Trắng đã đi 1 nước, nó trả về "lượt Trắng", tức SAI LƯỢT. Lúc đó
            # bàn không chậm mà ĐỨNG IM: explain()/explain_occ() chỉ sinh nước của
            # bên đang đi, nên nước Đen của bạn không bao giờ khớp; Stockfish cũng
            # gợi ý cho sai bên. Đường này cho lượt đi, quyền nhập thành và ô bắt
            # tốt qua đường là dữ kiện chắc chắn thay vì suy đoán.
            b_read = reader.locate_from_start(grid, flipped=not wb, start=sb)
            if b_read is not None:
                nply = len(b_read.move_stack)
                print(f"khớp thế khai cuộc sau {nply} nước -> lượt "
                      f"{'Trắng' if b_read.turn else 'Đen'} (chắc chắn)", flush=True)
            else:
                b_read = reader.board_from_grid(grid, flipped=not wb)
                if b_read is None:
                    print("PieceNet đọc ra thế cờ không hợp lệ -> dùng occupancy",
                          flush=True)
                else:
                    print(f"vào giữa ván (không khớp khai cuộc) -> đoán lượt "
                          f"{'Trắng' if b_read.turn else 'Đen'}", flush=True)

    if b_read is not None:
        HUMAN_COLOR = chess.WHITE if wb else chess.BLACK
        # DÙNG CẢ HAI: thế cờ PieceNet đọc được chính là NHÃN đúng cho mặt nạ
        # ô-có-quân, nên hiệu chỉnh luôn mô hình occupancy bằng nhãn đó — không
        # phải đoán mò giữa 21 thế khai cuộc như đường cũ. Từ đó mỗi frame có hai
        # nguồn đọc độc lập: PieceNet cho danh tính quân, occupancy làm chỗ dựa
        # khi PieceNet đọc không chắc.
        occ_model, occ_margins = gridfind.OccupancyModel.fit(
            warped, gridfind.expected_mask(b_read, flipped=not wb))
        model = occ_model if occ_model.usable() else None
        print(f"occupancy hiệu chỉnh theo nhãn PieceNet: margins={occ_margins} "
              f"usable={occ_model.usable()}", flush=True)
        tracker = rectify.Tracker(corners, score)
        board = b_read
        # Khớp được từ thế khai cuộc thì gốc ván là `sb`, và các nước đã đi phải
        # hiện lên danh sách nước — nếu lấy thế hiện tại làm gốc thì nước mở màn của
        # đối thủ biến mất khỏi biên bản, và try_resync() (dựng lại SAN từ START_FEN
        # + move_stack) sẽ đọc lệch.
        init_sans = []
        if b_read.move_stack:
            START_FEN = sb.fen()
            _rb = sb.copy()
            for _mv in b_read.move_stack:
                init_sans.append(_rb.san(_mv))
                _rb.push(_mv)
        else:
            START_FEN = b_read.fen()
        try:
            gridfind.save_calib(CALIB_PATH, corners,
                                gridfind.OccupancyModel(), score)
        except Exception:
            pass
        with lock:
            STATE["ready"] = True
            STATE["corners"] = np.asarray(corners).tolist()
            STATE["moves"] = init_sans
        _set_track("BÁM", score)
        color_str = ("Bạn cầm Trắng (quân ở dưới)." if wb
                     else "Bạn cầm Đen (quân ở dưới).")
        move_str = f" Đối thủ đã đi {' '.join(init_sans)}." if init_sans else ""
        _advice_async(f"Đã đọc được bàn cờ bằng PieceNet! {color_str}{move_str} ", board)
        return True

    # 1. Dò màu quân ở phía DƯỚI bàn cờ camera (Trắng hay Đen)
    HUMAN_COLOR = gridfind.detect_human_color(warped)
    flipped = (HUMAN_COLOR == chess.BLACK)

    # 2. Thử danh sách thế cờ mở màn (đầu ván HOẶC đã đi 1 nước Trắng: 1. e4, 1. d4, 1. Nf3, ...)
    cand_boards = [(sb.copy(), None)]
    for mv in list(sb.legal_moves):
        b_tmp = sb.copy()
        b_tmp.push(mv)
        cand_boards.append((b_tmp, mv))

    best_m = None
    best_board = None
    best_mv = None
    best_margin = -1e9

    for b_cand, mv in cand_boards:
        exp_mask = gridfind.expected_mask(b_cand, flipped=flipped)
        m, margins = gridfind.OccupancyModel.fit(warped, exp_mask)
        if m.usable():
            tot_margin = sum(margins)
            if tot_margin > best_margin:
                best_margin = tot_margin
                best_m = m
                best_board = b_cand.copy()
                best_mv = mv

    if best_m is None or not best_m.usable():
        set_msg("Chưa tách được ô quân/ô trống — kiểm tra màn hình đúng thế cờ đã khai báo chưa.",
                do_speak=False)
        return False

    model = best_m
    tracker = rectify.Tracker(corners, score)
    board = best_board.copy()
    START_FEN = sb.fen()
    init_moves = [sb.san(best_mv)] if best_mv else []

    try:
        gridfind.save_calib(CALIB_PATH, corners, model, score)
    except Exception:
        pass
    with lock:
        STATE["ready"] = True
        STATE["corners"] = corners.tolist()
        STATE["moves"] = init_moves

    _set_track("BÁM", score)
    color_str = "Bạn cầm Trắng (quân ở dưới)." if HUMAN_COLOR == chess.WHITE else "Bạn cầm Đen (quân ở dưới)."
    move_str = f" Đối thủ đã mở màn {sb.san(best_mv)}." if best_mv else ""
    _advice_async(f"Đã nhận diện bàn cờ thành công! {color_str}{move_str} ", board)
    return True


def _occ_of(warped):
    """Mặt nạ 8×8 có/không quân. Ưu tiên mô hình occupancy vì nó nhẹ và đã được
    hiệu chỉnh đúng ánh sáng bàn này; thiếu nó thì lấy từ PieceNet."""
    if model is not None:
        return model.predict(warped)
    return pnet.occupancy(warped)


def _orient_after_redetect(fr, quad):
    """Gỡ nhập nhằng 180° sau một lần dò lại toàn khung.

    rectify.detect() chỉ bảo đảm ô (0,0) là ô sáng, tức còn đúng hai hướng lệch
    nhau 180°. Chọn hướng nào cho occupancy khớp thế cờ đang có hơn.
    """
    flipped = (HUMAN_COLOR == chess.BLACK)
    exp = gridfind.expected_mask(board, flipped=flipped)
    best, best_d = quad, None
    for q in (quad, rectify.rot_quad(quad, 2)):
        d = int((exp != _occ_of(rectify.rectify(fr, q).board)).sum())
        if best_d is None or d < best_d:
            best, best_d = q, d
    return best


def _board_view(occ):
    flipped = (HUMAN_COLOR == chess.BLACK)
    img = render.render_grid(board_grid(board, flipped=flipped))
    exp = gridfind.expected_mask(board, flipped=flipped)
    step = img.shape[0] // 8
    for r in range(8):
        for f in range(8):
            if occ[r][f] != exp[r][f]:
                cv2.circle(img, (f * step + step - 12, r * step + 12), 7, (0, 0, 255), -1)
    return img


def recog_thread():
    # `model` là biến toàn cục do build_reference đặt; hàm này vừa đọc vừa GHI nó
    # (tự hiệu chỉnh ngưỡng), nên thiếu dòng global là mọi lần đọc phía trên vỡ
    # thành UnboundLocalError và cả vòng nhận diện chết câm.
    global model
    while latest["frame"] is None:
        time.sleep(0.1)
    # Chờ phơi sáng ổn định. 3s là quá tay: capture_thread đã đọc bỏ 8 frame và
    # set_exposure() xong trước đó, nên đây là 3 giây chết trước lần dò đầu tiên.
    time.sleep(1.0)
    while not build_reference():
        _set_track("ĐANG DÒ BÀN...")
        time.sleep(0.4)      # dò lại sớm: detect đã rẻ đi ~2x

    # DATASET cho PieceNet (phiên train model nhờ): dump warp + FEN khi frame
    # bám tốt và khớp game state 100% -> nhãn tự động đúng tuyệt đối.
    dataset_dir = os.path.join(CAP_DIR, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    dump_count = len([f for f in os.listdir(dataset_dir) if f.endswith(".png")])
    last_dump = 0.0
    last_fit = 0.0      # lần cuối tự hiệu chỉnh ngưỡng occupancy
    stab = tracker3.Stabilizer()   # bỏ phiếu trên KẾT QUẢ ĐỌC TỪNG Ô

    pend_mv, pend_cnt = None, 0
    occ_prev, stuck = None, 0
    last_view, last_seq = None, -1
    # CHẨN ĐOÁN SỐNG — để soi ở /diag mà không phải dừng lại đọc log. Đây là các
    # con số mà cả buổi chỉnh occupancy vừa rồi đo được offline; đưa lên màn hình
    # thì mới kiểm chứng được trên camera thật, ánh sáng thật.
    dg = {"khung": 0, "khop": 0, "ge2": 0, "mat_ban": 0, "mo": 0,
          "cnn_khac": 0, "cnn_tot": 0, "pix_tot": 0, "t0": time.time()}
    while True:
        # Nhịp THÍCH ỨNG: đang có nước chờ đủ phiếu (hoặc đang nghi lệch) thì quay
        # nhanh để chốt sớm; bàn yên thì thả chậm cho đỡ tốn CPU. Số phiếu và mọi
        # ngưỡng giữ nguyên — chỉ bỏ khoảng chờ cứng 0.13s giữa hai phiếu.
        time.sleep(0.04 if (pend_cnt or stuck) else 0.12)
        try:
            if recal.is_set():
                recal.clear()
                while not build_reference():
                    _set_track("ĐANG DÒ BÀN...")
                    time.sleep(0.4)      # dò lại sớm: detect đã rẻ đi ~2x
                pend_mv, pend_cnt, occ_prev, stuck = None, 0, None, 0
                continue
            # Chờ canh bàn xong. `model` được phép là None (occupancy không tách
            # được ô mà PieceNet vẫn đọc tốt), nên chỉ chặn khi KHÔNG CÒN nguồn
            # đọc nào — chặn theo `model` như trước là treo cả vòng lặp.
            if tracker is None or (model is None and pnet is None):
                continue
        except Exception as e:
            print("recog loop error (recal):", e, flush=True)
            continue
        # Bọc TOÀN BỘ thân vòng lặp: bất kỳ lỗi nào cũng không được giết thread
        # (thread chết im lặng = server zombie, UI còn sống nhưng không nhận nước).
        try:
            fr = latest["frame"]
            if fr is None:
                continue
            # Phiếu chỉ được tính trên frame MỚI. Quay nhanh mà đếm nhiều phiếu trên
            # cùng một frame là tự lừa mình, nên chốt chặn "3 frame khác nhau" vẫn
            # nguyên giá trị như cũ, chỉ là không còn phải chờ 0.13s cho mỗi phiếu.
            if latest["seq"] == last_seq:
                continue
            last_seq = latest["seq"]
            t_loop = time.time()
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if gridfind.sharpness(gray) < gridfind.SHARP_MIN:
                _set_track("MỜ (đang di chuyển?)")
                dg["mo"] += 1
                continue

            corners, sc, mode = tracker.update(fr)
            if mode == "MẤT BÀN":
                _set_track("MẤT BÀN — đưa camera về bàn cờ")
                dg["mat_ban"] += 1
                continue
            if mode == "DÒ LẠI":
                # rectify chỉ bảo đảm parity (a8 sáng) nên sau khi dò lại vẫn còn
                # nhập nhằng 180°; không gỡ thì bàn có thể bị lật ngược âm thầm.
                corners = _orient_after_redetect(fr, corners)
                tracker.quad = corners
            _set_track(mode, sc)
            with lock:
                STATE["corners"] = np.asarray(corners).tolist()

            t_track = (time.time() - t_loop) * 1000
            t0 = time.time()
            warped = rectify.rectify(fr, corners).board
            t_rect = (time.time() - t0) * 1000
            # PieceNet nay chạy MỖI FRAME: trên NPU nó chỉ tốn 10ms (trước là
            # 495ms trên CPU, đó mới là lý do nó từng phải chạy lười và cả pipeline
            # phải xoay quanh occupancy).
            t0 = time.time()
            if model is not None:
                occ = model.predict(warped)
            elif pnet is not None:
                occ = pnet.occupancy(warped)     # chưa fit được ngưỡng thì nhờ model
            else:
                continue                          # không còn nguồn đọc nào
            t_occ = (time.time() - t0) * 1000
            # Bàn phụ chỉ đổi khi thế cờ hoặc mặt nạ ô-có-quân đổi. Trước đây nó được
            # vẽ lại + nén JPEG mỗi chu kỳ 0.13s dù y nguyên; bỏ được thì trình duyệt
            # cũng khỏi tải lại ảnh giống hệt (xem bver ở /status).
            view_key = (board.board_fen(), occ.tobytes())
            if view_key != last_view:
                last_view = view_key
                try:
                    bok, bj = cv2.imencode(".jpg", _board_view(occ),
                                           [cv2.IMWRITE_JPEG_QUALITY, 85])
                except Exception:
                    bok = False
                with lock:
                    if bok:
                        STATE["board"] = bj.tobytes()
                        STATE["bver"] = STATE.get("bver", 0) + 1
            with lock:
                STATE["n"] = int(occ.sum())

            # TỰ HỒI PHỤC: lệch >=2 ô, occupancy phải ỔN ĐỊNH (tay đang che/kéo
            # quân thì occ nhảy loạn — không đếm), kéo dài ~4 chu kỳ (~1.2s).
            base = _mismatch(board, occ)
            if base >= 2 and occ_prev is not None and np.array_equal(occ, occ_prev):
                stuck += 1
            elif base < 2:
                stuck = 0
            occ_prev = occ

            # dump dataset: khớp 100% game state, tối đa 1 mẫu/2s, trần 3000 mẫu
            # (~1.8GB) để không ăn hết đĩa
            # TỰ HIỆU CHỈNH NGƯỠNG: thế cờ đang giữ CHÍNH LÀ nhãn đúng cho mặt nạ
            # này — nhãn miễn phí. Nhưng bản cũ (`base == 0`, `model = m2`) đo
            # được là LÀM TỆ ĐI: 90,9% → 87,8% trên 1000 khung thật. Hai lý do,
            # cả hai đều đã sửa ở đây:
            #
            #  1. THAY HẲN model. Fit trên MỘT khung dùng percentile 95/5, mà
            #     giữa ván nhãn lệch hẳn (ít quân), nên ngưỡng bám vào nhiễu của
            #     đúng khung đó. Tệ hơn: nếu một kênh đặc trưng không tách được ở
            #     khung này, `m2.t[k]` = 1e9 và việc gán `model = m2` VỨT LUÔN
            #     ngưỡng tốt của kênh đó. Giờ trộn EMA từng kênh và chỉ trộn khi
            #     kênh mới thực sự tách được.
            #  2. Đòi `base == 0`. Lệch mãn tính (lưới sát mép, ánh sáng đổi) thì
            #     base không bao giờ về 0 → hiệu chỉnh KHÔNG BAO GIỜ chạy, đúng
            #     lúc cần nó nhất. Nới lên 1 ô, đúng mức nhiễu occupancy đo được.
            #
            # Quét trên 1000 khung thật: 90,9% → 92,3%, ô báo thừa 174 → 113,
            # và khung lệch ≥2 ô 47 → 24.
            if base <= 1 and model is not None and time.time() - last_fit >= 1.0:
                last_fit = time.time()
                m2, mg2 = gridfind.OccupancyModel.fit(
                    warped, gridfind.expected_mask(board, flipped=(HUMAN_COLOR == chess.BLACK)))
                for k in range(len(model.t)):
                    if m2.t[k] < 1e8 and model.t[k] < 1e8:
                        model.t[k] = 0.6 * model.t[k] + 0.4 * m2.t[k]
                    elif m2.t[k] < 1e8 and not model.usable():
                        model.t[k] = m2.t[k]      # chưa có gì để giữ -> nhận luôn

            if base == 0 and time.time() - last_dump >= 2.0 and dump_count < 3000:
                try:
                    ts = int(time.time() * 1000)
                    cv2.imwrite(f"{dataset_dir}/{ts}.png", warped)
                    flipped_flag = 1 if HUMAN_COLOR == chess.BLACK else 0
                    with open(f"{dataset_dir}/{ts}.fen", "w") as fh:
                        fh.write(board.fen() + f"\nflipped={flipped_flag}\n")
                    last_dump = time.time()
                    dump_count += 1
                except Exception:
                    pass

            if stuck >= 8:                        # ~1s ổn định ở nhịp 0.13s
                if try_resync(occ):
                    pend_mv, pend_cnt = None, 0
                stuck = 0

            # ĐỌC TRỰC TIẾP rồi so với thế cờ đang giữ (chess_ai/tracker3.py).
            # Thay hẳn đường cũ "liệt kê tổ hợp nước hợp lệ rồi bỏ phiếu".
            # Phân vai hai nguồn theo đúng thế mạnh, đo trên 1000 khung thật:
            #   occupancy pixel -> có quân hay không (0 ô bỏ sót)
            #   PieceNet 3 lớp  -> màu gì            (99,96%)
            # Màu ở ô ĐÍCH là thứ phá được nước ăn quân, chỗ occupancy vĩnh viễn
            # mù. Đo trên 274 cặp khung thật: chốt ngay 96,4% (cũ 91,6%), và
            # nhập nhằng 0% (cũ 2,2%). PieceNet giờ 10ms nên chạy được mỗi frame.
            flipped = (HUMAN_COLOR == chess.BLACK)
            det = None
            t_pnet = 0.0
            if pnet is not None:
                t0 = time.time()
                ps, pc = pnet.read3(warped)
                t_pnet = (time.time() - t0) * 1000
                # ĐỐI CHỨNG SỐNG pixel vs CNN. Đo offline trên 1000 khung dataset
                # cho thấy CNN đọc "có quân hay không" tốt hơn pixel rõ rệt (98,7%
                # so với 92,3% khung khớp cả bàn) — ngược hẳn giả định ban đầu của
                # dự án. Nhưng dataset đó do chính đường pixel dump ra, và ánh sáng
                # phòng có thể khác, nên trước khi đổi vai hai nguồn thì phải thấy
                # cùng kết luận trên camera thật. Chỉ ĐẾM, không đổi hành vi.
                o_cnn = (ps != tracker3.TRONG)
                exp_now = gridfind.expected_mask(board, flipped=flipped)
                dg["cnn_tot"] += int((o_cnn != exp_now).sum())
                dg["pix_tot"] += int((occ != exp_now).sum())
                dg["cnn_khac"] += int((o_cnn != occ).sum())
                S = tracker3.read_state(occ, ps, pc)
                stab.push(S)
                got, resid = tracker3.detect(board, S, stab.stable(), flipped)
                if len(got) == 1:
                    det = ([got[0]], resid)
                elif len(got) > 1:            # nhập nhằng -> chờ thêm frame, không đoán
                    print(f"tracker3 nhập nhằng {len(got)} nước -> chờ", flush=True)
            if det is None and model is not None and base < 4:
                # Lưới đỡ: thiếu PieceNet, hoặc đọc trực tiếp chưa ra gì thì vẫn
                # còn đường occupancy cũ. Không bỏ nguồn nào.
                #
                # CHẶN Ở `base < 4`: `explain_occ` nở cây sâu 2 khi base>=2, tức
                # dựng 27x27=729 bàn cờ — đo được 282 ms ở thế giữa ván (laptop;
                # AIBOX chậm hơn vài lần). Với base>=4 nó KHÔNG BAO GIỜ trả kết
                # quả: điều kiện nhận là residual<=1, mà 4 ô lệch thì một nước đi
                # (đổi tối đa 2-4 ô) không kéo nổi xuống. Nghĩa là toàn bộ 729
                # phép tính đó chắc chắn bị vứt. Đo trên 1000 khung thật: 4+ ô
                # lệch luôn là lỗi CĂN LƯỚI/ánh sáng (291 ô báo thừa so với 1 ô
                # bỏ sót), không phải lỡ nước — trả giá tổ hợp để tìm lời giải cờ
                # vua cho một lỗi camera. Để `stuck` lo, đừng nở cây.
                occ_res = detect_move_occ(board, occ)
                if occ_res is not None and len(occ_res[0]) == 1:
                    det = (occ_res[0][0], occ_res[1])
            if det is not None:
                mvs, bd = det
                # Stabilizer đã đòi mỗi ô đọc giống nhau 3 frame liên tiếp, tức
                # phần bỏ phiếu nằm ở TẦNG ĐỌC rồi. Ở đây chỉ cần 2 lần xác nhận
                # để loại nhiễu một-frame, thay vì 3-5 phiếu như trước.
                need = 2 if bd == 0 else 3
                if mvs == pend_mv:
                    pend_cnt += 1
                else:
                    pend_mv, pend_cnt = mvs, 1
                if pend_cnt >= need:
                    last_san = ""
                    for mv in mvs:
                        try:
                            last_san = board.san(mv)
                        except Exception:
                            last_san = mv.uci()
                        board.push(mv)
                        with lock:
                            STATE["moves"].append(last_san)
                    pend_mv, pend_cnt = None, 0
                    stab.reset()          # thế cờ đổi -> lịch sử đọc cũ hết nghĩa
                    if board.is_game_over():
                        set_msg(f"Nước {last_san}. Ván cờ kết thúc.")
                    else:
                        _advice_async(f"Nước {last_san}. ", board)
            else:
                pend_cnt = max(0, pend_cnt - 1)

            if diag_reset.is_set():
                diag_reset.clear()
                dg.update({"khung": 0, "khop": 0, "ge2": 0, "mat_ban": 0, "mo": 0,
                           "cnn_khac": 0, "cnn_tot": 0, "pix_tot": 0, "t0": time.time()})
            dg["khung"] += 1
            dg["khop"] += base == 0
            dg["ge2"] += base >= 2
            try:
                cho_san = board.san(pend_mv[0]) if pend_mv else ""
            except Exception:
                cho_san = ""          # nước chờ không còn hợp lệ -> đừng giết vòng
            with lock:
                STATE["diag"] = {
                    "lech": base, "phieu": pend_cnt, "stuck": stuck,
                    "cho": cho_san,
                    "khung": dg["khung"],
                    "khop_pc": round(100.0 * dg["khop"] / max(dg["khung"], 1), 1),
                    "ge2_pc": round(100.0 * dg["ge2"] / max(dg["khung"], 1), 1),
                    "mat_ban": dg["mat_ban"], "mo": dg["mo"],
                    "fps": round(dg["khung"] / max(time.time() - dg["t0"], 1e-6), 1),
                    "ms_track": round(t_track), "ms_rect": round(t_rect),
                    "ms_occ": round(t_occ), "ms_pnet": round(t_pnet),
                    "ms_vong": round((time.time() - t_loop) * 1000),
                    "pix_o": dg["pix_tot"], "cnn_o": dg["cnn_tot"],
                    "cnn_khac": dg["cnn_khac"],
                }
        except Exception as e:
            print("recog loop error:", e, flush=True)
            time.sleep(0.5)


PAGE = """<!doctype html><html lang=vi><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>HLV Cờ Vua AI</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:system-ui,'Segoe UI',sans-serif;color:#e8eef7;
 background:radial-gradient(1200px 600px at 20% -10%,#1c2b4a 0%,#0b0f1a 55%) fixed}
.top{display:flex;align-items:center;gap:14px;padding:14px 20px;
 background:linear-gradient(90deg,#12203c,#0d1526);border-bottom:1px solid #22314f}
.logo{font-size:26px}.title{font-weight:800;font-size:20px;letter-spacing:.3px}
.title small{display:block;font-weight:500;color:#8fa3c4;font-size:12px}
.pill{margin-left:auto;display:flex;gap:8px;align-items:center}
.chip{padding:6px 12px;border-radius:999px;font-size:13px;font-weight:600;
 background:#16203a;border:1px solid #26375c}
.chip.ok{background:#123524;border-color:#1f6b45;color:#79f0b0}
.chip.warn{background:#3a2416;border-color:#7a4a1f;color:#ffb877}
.btn{padding:8px 16px;border:0;border-radius:10px;font-size:14px;font-weight:700;cursor:pointer;
 background:linear-gradient(90deg,#2e9e5b,#33b866);color:#04120a}
.wrap{display:grid;grid-template-columns:1.1fr .9fr;gap:18px;padding:20px;max-width:1200px;margin:0 auto}
@media(max-width:860px){.wrap{grid-template-columns:1fr}}
.card{background:linear-gradient(180deg,#141d33,#0f1626);border:1px solid #223050;
 border-radius:16px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
.card h3{margin:0 0 10px;font-size:14px;color:#9db2d6;text-transform:uppercase;letter-spacing:1px}
.cam img{width:100%;border-radius:12px;display:block;border:1px solid #223050}
.camcol{display:flex;flex-direction:column;gap:18px}
.aiboard{width:100%;max-width:400px;display:block;margin:0 auto;border-radius:12px;border:1px solid #2b6ea0}
.hint2{font-size:12px;color:#7f93b5;margin-top:8px;text-align:center}
.hero{display:flex;align-items:center;gap:18px;margin-bottom:6px}
.movebox{min-width:120px;text-align:center;padding:14px 18px;border-radius:14px;
 background:linear-gradient(180deg,#0f3d27,#0b2b1c);border:1px solid #1f8a54}
.movebox .lbl{font-size:12px;color:#7fe0a8;letter-spacing:1px}
.movebox .mv{font-size:44px;font-weight:900;color:#effff5;line-height:1.05;
 text-shadow:0 2px 12px rgba(51,184,102,.5)}
.heroinfo{flex:1}
.turn{display:inline-block;padding:4px 10px;border-radius:8px;font-size:12px;font-weight:700;margin-bottom:8px}
.turn.you{background:#123a5a;color:#7cc4ff;border:1px solid #2b6ea0}
.turn.opp{background:#3a1622;color:#ff8fb0;border:1px solid #7a2b45}
.evalbar{height:14px;border-radius:8px;background:#0b1220;border:1px solid #223050;overflow:hidden}
.evalfill{height:100%;background:linear-gradient(90deg,#33b866,#7fe0a8);transition:width .5s}
.evaltxt{font-size:13px;color:#9db2d6;margin-top:6px}
.msg{margin-top:14px;font-size:19px;line-height:1.55;background:#0d1830;border:1px solid #22406b;
 border-radius:12px;padding:14px;min-height:70px;transition:background .3s}
.msg.flash{background:#123a24}
.moves{display:flex;flex-wrap:wrap;gap:8px}
.moves span{padding:6px 11px;border-radius:8px;background:#0f1c33;border:1px solid #24406b;
 font-family:ui-monospace,monospace;font-size:14px;color:#bfe0ff}
.empty{color:#5a6b8a;font-style:italic}
.dg{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));gap:8px}
.dg div{background:#0f1c33;border:1px solid #24406b;border-radius:10px;padding:8px 10px}
.dg .k{font-size:11px;color:#7f93b5;letter-spacing:.5px;text-transform:uppercase}
.dg .v{font-size:20px;font-weight:800;font-family:ui-monospace,monospace;color:#cfe6ff}
.dg .v.good{color:#79f0b0}.dg .v.bad{color:#ff9b9b}.dg .v.mid{color:#ffc978}
.bars{margin-top:10px;font-family:ui-monospace,monospace;font-size:12px;color:#8fa3c4}
.bars b{color:#cfe6ff;font-weight:700}
.spark{display:flex;align-items:flex-end;gap:2px;height:34px;margin-top:10px}
.spark i{flex:1;background:#2b6ea0;border-radius:2px 2px 0 0;min-height:2px}
.spark i.z{background:#1f6b45}.spark i.h{background:#8a3a3a}
.dghint{font-size:12px;color:#7f93b5;margin-top:10px;line-height:1.5}
</style></head><body>
<div class=top>
 <div class=logo>♟️</div>
 <div class=title>HLV Cờ Vua AI<small>Camera → nhận diện → khuyên nước &amp; phân tích</small></div>
 <div class=pill>
   <span id=stat class="chip warn">Đang canh…</span>
   <span id=cnt class=chip>0 quân</span>
   <button class="btn ok" onclick="newGame()">🆕 Ván mới</button>
   <button class="btn warn" onclick="recalibrate()">🔄 Đồng bộ</button>
   <button class=btn id=snd onclick=toggleSound()>🔇 Tiếng: TẮT</button>
 </div>
</div>
<div class=wrap>
 <div class=camcol>
   <div class="card cam"><h3>📷 Camera trực tiếp</h3><img id=cam src="/stream"></div>
   <div class=card><h3>🤖 Bàn cờ AI nhận diện (đối chiếu)</h3>
     <img id=aiboard class=aiboard alt="AI board">
     <div class=hint2>So với camera bên trên — lệch chỗ nào là nhận sai chỗ đó.</div>
   </div>
 </div>
 <div class=side>
   <div class=card>
     <h3>🧠 AI khuyên</h3>
     <div class=hero>
       <div class=movebox><div class=lbl>NƯỚC ĐI</div><div class=mv id=mv>—</div></div>
       <div class=heroinfo>
         <span class="turn you" id=turn>Lượt của bạn</span>
         <div class=evalbar><div class=evalfill id=evfill style="width:50%"></div></div>
         <div class=evaltxt id=evtxt>—</div>
       </div>
     </div>
      <div class=msg id=msg>…</div>
      <button class=btn style="background:linear-gradient(90deg,#2b6ea0,#1e4f75);margin-top:12px;width:100%;" onclick="triggerAnalysis()">🔍 Phân tích chi tiết</button>
    </div>
    <div class=card><h3>📜 Các nước đã đi</h3><div class=moves id=moves><span class=empty>chưa có</span></div></div>
    <div class=card><h3>🔬 Chẩn đoán trực tiếp</h3>
      <div class=dg id=dg></div>
      <div class=spark id=spark title="ô lệch mỗi khung — xanh = khớp 100%, đỏ = >=2 ô"></div>
      <div class=bars id=bars></div>
      <button class=btn id=dgr style="background:linear-gradient(90deg,#40507a,#2c3a5c);color:#dbe6ff;margin-top:10px;width:100%" onclick="fetch('/diag_reset');hist=[]">↺ Đo lại từ 0</button>
      <div class=dghint>Ô lệch = số chấm đỏ trên bàn AI. 0 là tốt; từ 2 ô trở lên là lúc hệ
        thống phải chờ thêm bằng chứng. Dòng cuối đối chiếu hai nguồn đọc trên chính
        camera của bạn — chỉ đếm, chưa đổi hành vi.</div>
    </div>
  </div>
</div>
<audio id=au></audio>
<script>
let lastSeenAudioId=0,sound=false,prevMsg='';
let audioQueue=[],isPlayingAudio=false;
const queuedAudioIds = {};

async function newGame(){
    document.getElementById('stat').textContent = 'Đang khởi tạo ván mới…';
    await fetch('/new_game');
}

async function recalibrate(){
    document.getElementById('stat').textContent = 'Đang đồng bộ lại…';
    await fetch('/recalibrate');
}

async function triggerAnalysis(){
    const mbox = document.getElementById('msg');
    mbox.textContent = '⏳ Đang phân tích chi tiết ván cờ…';
    try {
        const res = await fetch('/analyze');
        const data = await res.json();
        if(data.report){
            mbox.textContent = data.report;
        }
    } catch(e){
        mbox.textContent = 'Lỗi kết nối phân tích.';
    }
}

function toggleSound(){
    sound = !sound;
    const btn = document.getElementById('snd');
    if(btn){
        btn.textContent = sound ? '🔊 Tiếng: BẬT' : '🔇 Tiếng: TẮT';
    }
    if(sound) playNextAudio();
}

function playNextAudio(){
    if(!sound || isPlayingAudio || audioQueue.length===0) return;
    if(audioQueue.length>3){
        audioQueue=audioQueue.slice(-2);
    }
    const aid=audioQueue.shift();
    const a=document.getElementById('au');
    isPlayingAudio=true;
    a.src='/audio?i='+aid;
    a.play().then(()=>{}).catch(()=>{
        isPlayingAudio=false;
        setTimeout(playNextAudio, 300);
    });
}

document.addEventListener('DOMContentLoaded', ()=>{
    const a = document.getElementById('au');
    if(a){
        a.onended = ()=>{ isPlayingAudio = false; playNextAudio(); };
        a.onerror = ()=>{ isPlayingAudio = false; playNextAudio(); };
    }
});

let prevBver=-1;
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))}
async function poll(){try{
 const s=await (await fetch('/status')).json();
 document.getElementById('mv').textContent=s.hint||'—';
 document.getElementById('msg').textContent=s.msg;
 document.getElementById('cnt').textContent=s.n+' quân';
 const st=document.getElementById('stat');
 if(s.ready){st.textContent=(s.track||'Sẵn sàng')+' · caro '+s.caro;st.className=(s.track&&s.track.startsWith('MẤT'))?'chip warn':'chip ok';}else{st.textContent=s.track||'Đang canh…';st.className='chip warn';}
 const t=document.getElementById('turn');
 if(s.your_turn){t.textContent='Lượt của bạn';t.className='turn you';}else{t.textContent='Lượt đối thủ';t.className='turn opp';}
 document.getElementById('evtxt').textContent=s.eval||'—';
 let pct=50+(s.cp/1000)*50; pct=Math.max(2,Math.min(98,pct));
 document.getElementById('evfill').style.width=pct+'%';
 const mv=document.getElementById('moves');
 if(s.moves.length){mv.innerHTML=s.moves.map((m,i)=>'<span>'+(i%2==0?(i/2+1)+'. ':'')+esc(m)+'</span>').join('');}
 const mbox=document.getElementById('msg');
 if(s.msg!==prevMsg){prevMsg=s.msg;mbox.classList.add('flash');setTimeout(()=>mbox.classList.remove('flash'),450);}
 if(sound && s.audio_id > 0){
     if(lastSeenAudioId === 0){
         lastSeenAudioId = s.audio_id;
         if(!queuedAudioIds[s.audio_id]){
             queuedAudioIds[s.audio_id] = true;
             audioQueue.push(s.audio_id);
             playNextAudio();
         }
     } else if(s.audio_id > lastSeenAudioId){
         for(let i = lastSeenAudioId + 1; i <= s.audio_id; i++){
             if(!queuedAudioIds[i]){
                 queuedAudioIds[i] = true;
                 audioQueue.push(i);
             }
         }
         lastSeenAudioId = s.audio_id;
         playNextAudio();
     }
 }
 drawDiag(s.diag||{});
 if(s.bver!==prevBver){prevBver=s.bver;document.getElementById('aiboard').src='/board?v='+s.bver;}
}catch(e){}}
let hist=[];
function cell(k,v,cls){return '<div><div class=k>'+k+'</div><div class="v '+(cls||'')+'">'+v+'</div></div>';}
function drawDiag(d){
 if(d.khung===undefined){document.getElementById('dg').innerHTML='<div><div class=k>trạng thái</div><div class=v>chờ…</div></div>';return;}
 const lech=d.lech||0;
 const lc = lech===0?'good':(lech<2?'mid':'bad');
 const kc = d.khop_pc>=90?'good':(d.khop_pc>=75?'mid':'bad');
 const gc = d.ge2_pc<=2?'good':(d.ge2_pc<=8?'mid':'bad');
 document.getElementById('dg').innerHTML=
   cell('ô lệch',lech,lc)+cell('khớp 100%',d.khop_pc+'%',kc)+cell('≥2 ô',d.ge2_pc+'%',gc)+
   cell('fps xử lý',d.fps)+cell('vòng',d.ms_vong+'ms')+
   cell('phiếu',(d.phieu||0)+(d.cho?' · '+esc(d.cho):''))+
   cell('kẹt',d.stuck||0,(d.stuck>=4?'bad':''))+cell('mất bàn',d.mat_ban||0,(d.mat_ban?'mid':''));
 hist.push(lech); if(hist.length>60) hist.shift();
 const mx=Math.max(2,...hist);
 document.getElementById('spark').innerHTML=hist.map(v=>
   '<i class="'+(v===0?'z':(v>=2?'h':''))+'" style="height:'+Math.max(2,v/mx*34)+'px"></i>').join('');
 const tong=(d.pix_o||0)+(d.cnn_o||0);
 document.getElementById('bars').innerHTML=
   'thời gian: bám <b>'+d.ms_track+'ms</b> · nắn <b>'+d.ms_rect+'ms</b> · ô <b>'+d.ms_occ+
   'ms</b> · PieceNet <b>'+d.ms_pnet+'ms</b><br>'+
   'ô đọc sai tích luỹ — pixel <b>'+(d.pix_o||0)+'</b> vs CNN <b>'+(d.cnn_o||0)+
   '</b>' + (tong? ' (hai nguồn bất đồng '+(d.cnn_khac||0)+' ô)':'') +
   '<br>khung đã xử lý <b>'+d.khung+'</b> · mờ bỏ qua <b>'+(d.mo||0)+'</b>';
}
setInterval(poll,300);poll();
// MJPEG khong tu noi lai: restart service la ket noi dut va the <img> nam chet
// cho toi khi nguoi dung F5. Bat onerror + kiem tra dinh ky de tu noi lai.
(function(){
  const cam=document.getElementById('cam');
  let alive=0;
  function reconnect(){cam.src='/stream?t='+Date.now();}
  cam.onerror=()=>setTimeout(reconnect,800);
  cam.onload=()=>{alive=Date.now();};
  setInterval(()=>{ if(alive && Date.now()-alive>6000) reconnect(); },3000);
})();
</script></body></html>""".encode("utf-8")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def handle_one_request(self):
        # Trình duyệt đóng tab / curl ngắt giữa chừng là chuyện thường, nhưng nó
        # đổ nguyên traceback vào journal và lấp mất log nhận diện thật.
        try:
            BaseHTTPRequestHandler.handle_one_request(self)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            last = None
            try:
                while True:
                    with lock:
                        STATE["cam_seen"] = time.time()   # báo capture_thread: có người xem
                        j = STATE["jpeg"]
                    if j is not None and j is not last:   # khỏi đẩy trùng frame qua adb
                        last = j
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                return
        elif self.path.startswith("/status"):
            with lock:
                data = json.dumps({"msg": STATE["msg"], "moves": STATE["moves"],
                                   "audio_id": STATE["audio_id"], "n": STATE["n"],
                                   "ready": STATE["ready"], "hint": STATE["hint"],
                                   "eval": STATE["eval"], "cp": STATE["cp"],
                                   "your_turn": STATE["your_turn"],
                                   "track": STATE["track"],
                                   "caro": STATE["caro"],
                                   "diag": STATE.get("diag", {}),
                                   "bver": STATE["bver"]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(data)
        elif self.path.startswith("/raw"):
            fr = latest["frame"]                 # nén khi có người gọi, không nén sẵn
            r = None
            if fr is not None:
                okr, rj = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
                r = rj.tobytes() if okr else None
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            if r:
                self.wfile.write(r)
        elif self.path.startswith("/diag_reset"):
            diag_reset.set()
            self.send_response(200); self.send_header("Content-Type", "text/plain")
            self.end_headers(); self.wfile.write(b"ok")
        elif self.path.startswith("/recalibrate"):
            recal.set()
            self.send_response(200); self.send_header("Content-Type", "text/plain")
            self.end_headers(); self.wfile.write(b"recalibrating")
        elif self.path.startswith("/new_game"):
            global board, START_FEN, model, tracker
            try:
                for fpath in [os.path.join(CAP_DIR, "start_fen.txt"), CALIB_PATH]:
                    if os.path.exists(fpath):
                        os.remove(fpath)
            except Exception:
                pass
            board = chess.Board()
            START_FEN = board.fen()
            model = None
            tracker = None
            with lock:
                STATE["moves"] = []
                STATE["msg"] = "Đang khởi tạo ván cờ mới, vui lòng giữ camera ổn định..."
                STATE["ready"] = False
                STATE["hint"] = "—"
                STATE["board"] = None
                STATE["eval"] = ""
                STATE["cp"] = 0
            recal.set()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(b'{"status":"ok"}')
        elif self.path.startswith("/analyze"):
            a = engine.analyse(board, multipv=3)
            report = commentary.generate_detailed_analysis(board, a)
            set_msg(report, do_speak=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok", "report": report}).encode("utf-8"))
        elif self.path.startswith("/board"):
            with lock:
                bd = STATE.get("board")
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if bd:
                self.wfile.write(bd)
        elif self.path.startswith("/audio"):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            aid_str = params.get("i", [None])[0]
            wv = None
            if aid_str is not None:
                try:
                    aid = int(aid_str)
                    with AUDIO_CACHE_LOCK:
                        wv = AUDIO_CACHE.get(aid)
                except ValueError:
                    pass
                if wv is None:
                    self.send_response(204)
                    self.end_headers()
                    return
            else:
                with lock:
                    wv = STATE["wav"]
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if wv:
                self.wfile.write(wv)
        else:
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers(); self.wfile.write(PAGE)


if __name__ == "__main__":
    threading.Thread(target=_speak_worker_loop, daemon=True).start()
    threading.Thread(target=_advice_worker, daemon=True).start()
    threading.Thread(target=capture_thread, daemon=True).start()
    threading.Thread(target=recog_thread, daemon=True).start()
    # Cổng đặt được từ ngoài: laptop thường đã có `adb forward tcp:8090` trỏ sang
    # AIBOX, nên muốn chạy thêm một bản để đối chiếu thì phải đổi cổng.
    port = int(os.environ.get("KIT_PORT", "8090"))
    print(f"coach server (pretty UI) on :{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
