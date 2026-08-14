#!/bin/bash
# Cong SSH cho CA NHOM vao AIBOX qua Tailscale cua laptop nay.
#
# Mo hinh:  Team --Tailscale--> laptop:2222 --adb forward--> AIBOX:22 (sshd)
# Yeu cau: laptop dang bat, Tailscale up, AIBOX cam USB vao laptop.
#
# Chay:   bash tools/aibox_ssh_gateway.sh
# Dung:   Ctrl+C
set -e

TSIP=$(tailscale ip -4 2>/dev/null | head -1)
if [ -z "$TSIP" ]; then echo "Loi: laptop chua bat Tailscale."; exit 1; fi

# 1) AIBOX co ket noi khong
if ! adb devices | grep -q "device$"; then
  echo "Loi: khong thay AIBOX (adb). Cam lai cap USB."; exit 1
fi

# 2) adb forward: laptop 127.0.0.1:2223 -> AIBOX:22
adb forward tcp:2223 tcp:22 >/dev/null
echo ">> adb forward 2223 -> AIBOX:22  OK"

# 3) socat: mo cong tren IP Tailscale cua laptop
echo ">> Cong SSH nhom mo tai:  $TSIP : 2222"
echo ">> Team SSH bang:  ssh -p 2222 thunopro@$TSIP"
echo ">> (hoac dung ten may Tailscale cua laptop thay cho IP)"
echo ">> Ctrl+C de dung."
exec socat TCP-LISTEN:2222,bind="$TSIP",fork,reuseaddr TCP:127.0.0.1:2223
