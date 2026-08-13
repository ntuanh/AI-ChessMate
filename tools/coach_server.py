#!/usr/bin/env python3
"""HLV CỜ VUA TRỰC TIẾP — bản MƯỢT + GIAO DIỆN ĐẸP.
Tách luồng quay/nhận diện; 720p + làm nét; dashboard sinh động (badge nước đi,
thanh đánh giá, trạng thái, hiệu ứng), phát tiếng trên trình duyệt. http://127.0.0.1:8090"""
import sys, os, time, json, threading, queue
from urllib.parse import parse_qs, urlparse
sys.path.insert(0, "/data/kit")
import numpy as np
import cv2
import chess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from chess_ai import vision, config, render, gridfind, rectify, reader, piece_net
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
         "corners": None, "track": "KHỞI ĐỘNG", "caro": 0.0}
latest = {"frame": None}
lock = threading.Lock()
board = chess.Board()
START_FEN = chess.STARTING_FEN
model = None            # gridfind.OccupancyModel (tự học lúc canh bàn)
tracker = None          # rectify.Tracker (bám 4 góc, camera rung vẫn đuổi)
# PieceNet đọc TỪNG QUÂN nên tốt hơn occupancy hẳn một bậc; thiếu model/onnxruntime
# thì trả None và cả pipeline tự lùi về đường occupancy cũ.
pnet = piece_net.load()
recal = threading.Event()
# C270 cắm vào AIBOX có thể ra /dev/video0|1|2 tuỳ thứ tự nhận thiết bị, nên để
# đặt được từ ngoài: KIT_CAM=0 python3 tools/coach_server.py
CAM = int(os.environ.get("KIT_CAM", "2"))

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

            p = f"/data/kit/captures/coach_speak_{next_id}.wav"
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
    rời khỏi camera/engine (xem tools/test_reader.py)."""
    return reader.explain_occ(b, occ, flipped=(HUMAN_COLOR == chess.BLACK))


def try_resync(occ):
    """TỰ HỒI PHỤC khi game state lệch bàn thật kéo dài. Thử 4 giả thuyết:
      (a) bỏ lỡ 1 nước;  (b) bỏ lỡ 2 nước đi nhanh;
      (c) nước cuối là THỪA (chưa ai đi);  (d) nước cuối ghi NHẦM.
    CHỈ nhận khi lời giải khớp ảnh 100% (residual=0) và DUY NHẤT — mơ hồ hay còn
    lệch dư thì thà đứng yên chờ thêm bằng chứng còn hơn "tự sửa" thành thế SAI."""
    global board
    if _mismatch(board, occ) == 0:
        return False
    cands = []

    b = board.copy()
    for mv1 in list(b.legal_moves):                       # (a) + (b)
        if mv1.promotion not in (None, chess.QUEEN):
            continue
        b.push(mv1)
        if _mismatch(b, occ) == 0:
            cands.append(b.copy())
        else:
            for mv2 in list(b.legal_moves):
                if mv2.promotion not in (None, chess.QUEEN):
                    continue
                b.push(mv2)
                if _mismatch(b, occ) == 0:
                    cands.append(b.copy())
                b.pop()
        b.pop()

    if board.move_stack:
        b = board.copy()
        wrong = b.pop()
        if _mismatch(b, occ) == 0:                        # (c) nước cuối thừa
            cands.append(b.copy())
        for mv in list(b.legal_moves):                    # (d) nước cuối ghi nhầm
            if mv == wrong or mv.promotion not in (None, chess.QUEEN):
                continue
            b.push(mv)
            if _mismatch(b, occ) == 0:
                cands.append(b.copy())
            b.pop()

    uniq = {}
    for nb in cands:
        uniq.setdefault(nb.fen(), nb)
    if len(uniq) != 1:
        return False                                      # không có / mơ hồ -> chưa sửa
    best_b = next(iter(uniq.values()))

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
    push_advice(f"🔧 Đã tự động đồng bộ theo bàn thật (…{tail}). ", board)
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
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.03); continue
        latest["frame"] = frame
        disp = frame.copy()
        col = (80, 220, 120) if STATE["ready"] else (60, 160, 255)
        cor = STATE.get("corners")
        if cor is not None:
            cv2.polylines(disp, [np.array(cor, dtype=np.int32)], True, col, 2)
        elif not STATE["ready"]:
            # Provide visual feedback to help user aim the camera
            temp_cor = vision.auto_detect_corners(frame)
            if temp_cor is not None:
                cv2.polylines(disp, [temp_cor.astype(np.int32)], True, (255, 140, 0), 2)
                cv2.putText(disp, "Giu yen...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 140, 0), 2)
        txt = f"{STATE['track']} | caro={STATE['caro']} | {STATE['n']} o co quan"
        cv2.putText(disp, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        ok2, jpg = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
        okr, rj = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
        with lock:
            if ok2:
                STATE["jpeg"] = jpg.tobytes()
            if okr:
                STATE["raw"] = rj.tobytes()
        time.sleep(0.05)


def _set_track(mode, caro=None):
    with lock:
        STATE["track"] = mode
        if caro is not None:
            STATE["caro"] = round(float(caro), 1)


def build_reference():
    global HUMAN_COLOR, model, tracker, START_FEN, board
    sb = chess.Board()
    try:
        _fen = open("/data/kit/captures/start_fen.txt").read().strip()
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
            b_read = reader.board_from_grid(grid, flipped=not wb)
            if b_read is None:
                print("PieceNet đọc ra thế cờ không hợp lệ -> dùng occupancy",
                      flush=True)

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
        START_FEN = b_read.fen()
        try:
            gridfind.save_calib("/data/kit/captures/board_calib.json", corners,
                                gridfind.OccupancyModel(), score)
        except Exception:
            pass
        with lock:
            STATE["ready"] = True
            STATE["corners"] = np.asarray(corners).tolist()
            STATE["moves"] = []
        _set_track("BÁM", score)
        color_str = ("Bạn cầm Trắng (quân ở dưới)." if wb
                     else "Bạn cầm Đen (quân ở dưới).")
        push_advice(f"Đã đọc được bàn cờ bằng PieceNet! {color_str} ", board)
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
        gridfind.save_calib("/data/kit/captures/board_calib.json", corners, model, score)
    except Exception:
        pass
    with lock:
        STATE["ready"] = True
        STATE["corners"] = corners.tolist()
        STATE["moves"] = init_moves

    _set_track("BÁM", score)
    color_str = "Bạn cầm Trắng (quân ở dưới)." if HUMAN_COLOR == chess.WHITE else "Bạn cầm Đen (quân ở dưới)."
    move_str = f" Đối thủ đã mở màn {sb.san(best_mv)}." if best_mv else ""
    push_advice(f"Đã nhận diện bàn cờ thành công! {color_str}{move_str} ", board)
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
    while latest["frame"] is None:
        time.sleep(0.1)
    time.sleep(3.0)
    while not build_reference():
        _set_track("ĐANG DÒ BÀN...")
        time.sleep(1.5)

    # DATASET cho PieceNet (phiên train model nhờ): dump warp + FEN khi frame
    # bám tốt và khớp game state 100% -> nhãn tự động đúng tuyệt đối.
    dataset_dir = "/data/kit/captures/dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    dump_count = len([f for f in os.listdir(dataset_dir) if f.endswith(".png")])
    last_dump = 0.0

    pend_mv, pend_cnt = None, 0
    occ_prev, stuck = None, 0
    while True:
        time.sleep(0.13)      # NHẠY hơn (~7-8 fps xử lý) -> bắt kịp nước đi nhanh, gần real-time
        try:
            if recal.is_set():
                recal.clear()
                while not build_reference():
                    _set_track("ĐANG DÒ BÀN...")
                    time.sleep(1.5)
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
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if gridfind.sharpness(gray) < gridfind.SHARP_MIN:
                _set_track("MỜ (đang di chuyển?)")
                continue

            corners, sc, mode = tracker.update(fr)
            if mode == "MẤT BÀN":
                _set_track("MẤT BÀN — đưa camera về bàn cờ")
                continue
            if mode == "DÒ LẠI":
                # rectify chỉ bảo đảm parity (a8 sáng) nên sau khi dò lại vẫn còn
                # nhập nhằng 180°; không gỡ thì bàn có thể bị lật ngược âm thầm.
                corners = _orient_after_redetect(fr, corners)
                tracker.quad = corners
            _set_track(mode, sc)
            with lock:
                STATE["corners"] = np.asarray(corners).tolist()

            warped = rectify.rectify(fr, corners).board
            # Hai nguồn đọc song song trên cùng một ảnh đã nắn.
            grid = conf = None
            n_unsure = 99
            if pnet is not None:
                grid, conf = pnet.predict(warped)
                n_unsure = int((np.asarray(conf) < reader.CONF_MIN).sum())
            if model is not None:
                occ = model.predict(warped)
            else:
                occ = np.array([[grid[r][f] != "." for f in range(8)]
                                for r in range(8)], bool)
            try:
                bok, bj = cv2.imencode(".jpg", _board_view(occ), [cv2.IMWRITE_JPEG_QUALITY, 85])
            except Exception:
                bok = False
            with lock:
                STATE["n"] = int(occ.sum())
                if bok:
                    STATE["board"] = bj.tobytes()

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

            # Ưu tiên PieceNet vì nó biết danh tính quân nên so được TRỌN THẾ CỜ:
            # phân biệt được nước ăn quân (exd5 vs exf5 cho cùng mặt nạ occupancy)
            # và cả phong cấp. PieceNet đọc không chắc thì mới hạ xuống occupancy,
            # nhờ vậy frame tối/mờ vẫn bắt được nước thay vì đứng im.
            # KẾT HỢP HAI NGUỒN, occupancy đi trước vì nó nhẹ và bền với ánh
            # sáng; PieceNet chỉ được gọi vào đúng hai chỗ occupancy bó tay:
            # (a) nhiều nước bằng điểm — mọi nước ăn quân đều thế;
            # (b) occupancy không giải thích được gì.
            flipped = (HUMAN_COLOR == chess.BLACK)
            id_ok = grid is not None and n_unsure <= reader.MAX_UNSURE
            det = None
            occ_res = detect_move_occ(board, occ) if model is not None else None
            if occ_res is not None:
                cands, bd = occ_res
                if len(cands) == 1:
                    det = (cands[0], bd)
                elif id_ok:
                    seq = reader.pick_by_identity(board, cands, grid, flipped)
                    if seq:
                        det = (seq, bd)
                        print(f"occupancy bằng điểm {len(cands)} phương án -> "
                              f"PieceNet chọn {[m.uci() for m in seq]}", flush=True)
            if det is None and id_ok:
                seq = reader.explain(board, grid, flipped)
                if seq:
                    det = (seq, 0)
            if det is not None:
                mvs, bd = det
                # BỎ PHIẾU bắt buộc, tính theo THỜI GIAN THỰC chứ không theo chu kỳ
                # (nhịp 0.13s): nước trọn vẹn 1 nước = 3 phiếu (~0.4s); chuỗi 2
                # nước / còn ô lệch dư = 5 phiếu (~0.65s). Nhiễu loá thoáng 0.3s
                # vẫn không đủ phiếu thành nước ma.
                need = 3 if (bd == 0 and len(mvs) == 1) else 5
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
                    if board.is_game_over():
                        set_msg(f"Nước {last_san}. Ván cờ kết thúc.")
                    else:
                        push_advice(f"Nước {last_san}. ", board)
            else:
                pend_cnt = max(0, pend_cnt - 1)
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
   <div class="card cam"><h3>📷 Camera trực tiếp</h3><img src="/stream"></div>
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
 document.getElementById('aiboard').src='/board?t='+Date.now();
}catch(e){}}
setInterval(poll,900);poll();
</script></body></html>""".encode("utf-8")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        j = STATE["jpeg"]
                    if j:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
                    time.sleep(0.05)
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
                                   "caro": STATE["caro"]}).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.end_headers(); self.wfile.write(data)
        elif self.path.startswith("/raw"):
            with lock:
                r = STATE.get("raw")
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            if r:
                self.wfile.write(r)
        elif self.path.startswith("/recalibrate"):
            recal.set()
            self.send_response(200); self.send_header("Content-Type", "text/plain")
            self.end_headers(); self.wfile.write(b"recalibrating")
        elif self.path.startswith("/new_game"):
            global board, START_FEN, model, tracker
            try:
                for fpath in ["/data/kit/captures/start_fen.txt", "/data/kit/captures/board_calib.json"]:
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
    threading.Thread(target=capture_thread, daemon=True).start()
    threading.Thread(target=recog_thread, daemon=True).start()
    print("coach server (pretty UI) on :8090", flush=True)
    ThreadingHTTPServer(("0.0.0.0", 8090), H).serve_forever()
