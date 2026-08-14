#!/usr/bin/env bash
# =============================================================================
#  build_npu.sh — dựng PieceNet cho NPU Hexagon (QNN HTP) TRÊN AIBOX. Chạy 1 lần.
#
#  Làm gì:
#    1. venv python3.12 + numpy/onnx/... (API QAIRT bắt buộc 3.12; hệ thống là 3.10)
#    2. ONNX  → DLC        (qairt-converter, cố định batch = 64 ô)
#    3. DLC   → DLC INT8   (qairt-quantizer, hiệu chuẩn bằng ẢNH THẬT đã dump)
#    4. đo tốc độ + đối chiếu độ chính xác với ONNX CPU trên tập held-out
#
#  Cần internet cho bước 1: chạy `python3 tools/usb_proxy.py` trên laptop trước
#  (script tự bắc `adb reverse`). Các bước sau chạy offline hoàn toàn.
# =============================================================================
set -euo pipefail

ADB="${ADB:-adb}"
DEST="${KIT_DEST:-/home/khacthu/kit}"
QSDK="${QSDK:-/opt/qcom/qairt-new/qairt/2.48.0.260626}"
ARCH=aarch64-oe-linux-gcc11.2
NCAL="${NCAL:-100}"     # số frame hiệu chuẩn INT8
NTEST="${NTEST:-100}"   # số frame kiểm định (KHÔNG trùng tập hiệu chuẩn)

"$ADB" wait-for-device
"$ADB" shell "mkdir -p $DEST/qnn_build"

echo ">> [0/5] bắc internet qua USB (cần tools/usb_proxy.py đang chạy ở laptop)"
"$ADB" reverse tcp:8899 tcp:8899 >/dev/null
PX="export http_proxy=http://127.0.0.1:8899 https_proxy=http://127.0.0.1:8899"

echo ">> [1/5] venv python3.12 + gói phụ thuộc"
# libpython3.12.so không nằm trong gói python3.12 của deadsnakes nhưng API QAIRT
# nạp nó bằng dlopen → phải cài thêm gói libpython3.12.
"$ADB" shell "$PX; set -e
  test -f /usr/lib/aarch64-linux-gnu/libpython3.12.so.1.0 || {
    curl -sL -o /tmp/libpy312.deb \
      https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu/pool/main/p/python3.12/libpython3.12_3.12.13-1+jammy1_arm64.deb
    dpkg -i /tmp/libpy312.deb >/dev/null; }
  test -x $DEST/qnnenv/bin/python || python3.12 -m venv $DEST/qnnenv
  $DEST/qnnenv/bin/pip install -q --upgrade pip
  # onnx 1.16: bản 1.22 bỏ onnx.version nên converter của QAIRT 2.48 gãy.
  # setuptools: python 3.12 bỏ distutils mà converter vẫn import.
  $DEST/qnnenv/bin/pip install -q numpy==1.26.4 'onnx==1.16.1' 'setuptools<81' \
      pydantic aenum tabulate paramiko
  echo '   ok'"

echo ">> [2/5] ghi env.sh"
"$ADB" shell "cat > $DEST/qnn_build/env.sh <<'EOF'
export Q=$QSDK
export A=$ARCH
export PATH=\$Q/bin/\$A:\$PATH
export LD_LIBRARY_PATH=\$Q/lib/\$A:\$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=\"\$Q/lib/hexagon-v73/unsigned;/usr/lib/rfsa/adsp;/dsp\"
export PYTHONPATH=\$Q/lib/python
EOF
echo '   ok'"

echo ">> [3/5] cắt ô từ ảnh thật (hiệu chuẩn $NCAL / kiểm định $NTEST frame)"
"$ADB" shell "cd $DEST && NCAL=$NCAL NTEST=$NTEST python3 - <<'PY'
import glob, os, numpy as np, cv2
from chess_ai.piece_net import PieceNet
ncal, ntest = int(os.environ['NCAL']), int(os.environ['NTEST'])
fs = sorted(glob.glob('captures/dataset/*.png'))
assert len(fs) >= ncal + ntest, f'chi co {len(fs)} frame, can {ncal+ntest}'
for tag, ss in (('calib', fs[:ncal]), ('test', fs[-ntest:])):
    os.makedirs(f'qnn_build/{tag}', exist_ok=True)
    paths = []
    for i, f in enumerate(ss):
        p = f'qnn_build/{tag}/{i:04d}.raw'
        PieceNet.cells(cv2.imread(f)).astype(np.float32).tofile(p)
        paths.append(p)
    open(f'qnn_build/{tag}_list.txt', 'w').write('cells:=' + '\ncells:='.join(paths) + '\n')
    print('  ', tag, len(paths))
PY"

echo ">> [4/5] ONNX → DLC → INT8"
"$ADB" shell "cd $DEST && . qnn_build/env.sh && set -e
  qnnenv/bin/python \$Q/bin/\$A/qairt-converter -i models/piece_net.onnx \
      -o qnn_build/piece_net.dlc --onnx_override_batch 64 --target_backend HTP 2>&1 | tail -2
  qnnenv/bin/python \$Q/bin/\$A/qairt-quantizer --input_dlc qnn_build/piece_net.dlc \
      --input_list qnn_build/calib_list.txt --output_dlc qnn_build/piece_net_q.dlc 2>&1 | tail -1"

echo ">> [5/5] đo tốc độ + đối chiếu độ chính xác trên tập held-out"
"$ADB" shell "cd $DEST && . qnn_build/env.sh && \
  qnn-net-run --backend \$Q/lib/\$A/libQnnHtp.so --dlc_path qnn_build/piece_net_q.dlc \
    --input_list qnn_build/test_list.txt --output_dir qnn_build/out_test --log_level error >/dev/null 2>&1
  qnnenv/bin/python - <<'PY' 2>/dev/null
import numpy as np, time
import qti.aisw.core.model_level_api as mlapi
from qti.aisw.tools.core.modules.api.backend.htp_backend import HtpBackend
from qti.aisw.tools.core.modules.api.definitions.common import ModelConfig
inf = mlapi.Inferencer(HtpBackend(target=mlapi.OELinuxTarget()),
                       ModelConfig(path='qnn_build/piece_net_q.dlc'),
                       executor=mlapi.NativeExecutor())
t = time.time(); inf.setup(); setup = (time.time() - t) * 1000
x = np.fromfile('qnn_build/test/0000.raw', np.float32).reshape(64, 3, 64, 64)
inf.run({'cells': x})
t = time.time()
for _ in range(20): inf.run({'cells': x})
open('/tmp/npu_bench.txt','w').write('setup %.0f ms | suy luan %.2f ms/ban\n'
                                     % (setup, (time.time() - t) / 20 * 1000))
PY
  cat /tmp/npu_bench.txt"

"$ADB" shell "cd $DEST && python3 - <<'PY' 2>/dev/null
import glob, numpy as np, cv2, chess
from chess_ai.piece_net import CLASSES, PieceNet, _MODEL_PATHS
import os
cpu = PieceNet(next(p for p in _MODEL_PATHS if os.path.exists(p)))
fs = sorted(glob.glob('captures/dataset/*.png'))
fs = fs[-len(glob.glob('qnn_build/test/*.raw')):]
same = accc = acch = n = 0
for i, f in enumerate(fs):
    a = cpu.logits(cv2.imread(f)).argmax(1)
    b = np.fromfile('qnn_build/out_test/Result_%d/logits.raw' % i, np.float32).reshape(64, 13).argmax(1)
    txt = open(f[:-4] + '.fen').read(); fl = 'flipped=1' in txt
    bd = chess.Board(txt.split('\n')[0]); y = []
    for r in range(8):
        for fi in range(8):
            rk = r if fl else 7 - r; fe = (7 - fi) if fl else fi
            pc = bd.piece_at(chess.square(fe, rk))
            y.append(CLASSES.index(pc.symbol() if pc else '.'))
    y = np.array(y); same += (a == b).sum(); accc += (a == y).sum(); acch += (b == y).sum(); n += 64
print('   HTP int8 trung khop CPU fp32 : %.3f%%' % (100 * same / n))
print('   do chinh xac CPU fp32        : %.3f%%' % (100 * accc / n))
print('   do chinh xac HTP int8        : %.3f%%' % (100 * acch / n))
PY"

echo
echo "XONG. Chạy 'bash tools/run_aibox.sh' — sidecar NPU sẽ tự lên."
