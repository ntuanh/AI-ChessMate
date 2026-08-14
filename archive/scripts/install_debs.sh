#!/bin/bash
# Cai stockfish + espeak-ng tren AIBOX bang cach tai .deb qua proxy USB (khong dung apt).
P="http://127.0.0.1:8899"
BASE="http://ports.ubuntu.com/ubuntu-ports"
WANT="stockfish espeak-ng libespeak-ng1 espeak-ng-data libpcaudio0 libsonic0"
mkdir -p /tmp/debs /tmp/idx
: > /tmp/idx/all.txt
for comp in main universe; do
  for suite in jammy jammy-updates; do
    echo ">> fetch $suite/$comp"
    curl -x "$P" -s "$BASE/dists/$suite/$comp/binary-arm64/Packages.gz" | gunzip -c >> /tmp/idx/all.txt 2>/dev/null || true
  done
done
echo ">> index lines: $(wc -l < /tmp/idx/all.txt)"
for pkg in $WANT; do
  fn=$(awk -v p="$pkg" '$1=="Package:"{cur=$2} $1=="Filename:"&&cur==p{print $2}' /tmp/idx/all.txt | tail -1)
  if [ -z "$fn" ]; then echo "!! khong thay goi: $pkg"; continue; fi
  echo ">> tai $pkg"
  curl -x "$P" -s -o "/tmp/debs/$(basename "$fn")" "$BASE/$fn"
done
echo ">> dpkg -i"
dpkg -i /tmp/debs/*.deb 2>&1 | tail -5
echo "===== KET QUA ====="
command -v stockfish >/dev/null && echo "stockfish: OK ($(printf 'uci\nquit\n' | stockfish 2>/dev/null | grep -m1 id))" || echo "stockfish: THIEU"
command -v espeak-ng >/dev/null && echo "espeak-ng: OK ($(espeak-ng --version 2>&1 | head -1))" || echo "espeak-ng: THIEU"
