#!/bin/bash
# Tao 7 tai khoan thanh vien tren AIBOX (qua adb). Mat khau mac dinh: hust123
# Chay:  bash tools/add_members.sh
# Moi nguoi SSH:  ssh -p 2222 <user>@100.106.160.24   (mat khau hust123, nen doi bang: passwd)

USERS="khacthu tuananh khanhtung anhkiet dungtuan tramho truongduy"

if ! adb devices | grep -q "device$"; then
  echo "Loi: khong thay AIBOX (adb). Cam lai cap USB."; exit 1
fi

adb shell "for u in $USERS; do \
  if id \"\$u\" >/dev/null 2>&1; then echo \"\$u : da ton tai (giu nguyen)\"; \
  else useradd -m -s /bin/bash \"\$u\" && echo \"\$u:hust123\" | chpasswd && echo \"\$u : DA TAO\"; fi; \
done"

echo "----------------------------------------"
echo " User  |  Mat khau  |  Lenh SSH"
for u in $USERS; do
  printf " %-10s hust123    ssh -p 2222 %s@100.106.160.24\n" "$u" "$u"
done
echo "----------------------------------------"
echo "Moi nguoi vao xong nen doi mat khau: passwd"
