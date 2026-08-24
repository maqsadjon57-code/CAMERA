#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
#  AI HUD — ЗАПУСК СЕРВЕРА ПРЯМО НА ТЕЛЕФОНЕ (Android + Termux)
#
#  1) Установите Termux из F-Droid:  https://f-droid.org/packages/com.termux/
#     (версия из Google Play устарела и не поддерживается)
#  2) Скопируйте папку проекта (app.py и эти файлы) во внутреннюю память
#     телефона, например в /sdcard/ai-hud/
#  3) В Termux выполните:
#         cd /sdcard/ai-hud
#         bash run_phone.sh
#  4) Скрипт установит зависимости и запустит сервер.
#  5) Откройте в браузере ТЕЛЕФОНА:   http://127.0.0.1:5000
#     Нажмите «Разрешить доступ к камере» — localhost является защищённым
#     контекстом, поэтому камера телефона работает СРАЗУ, без HTTPS.
#  6) (Необязательно) Меню браузера → «Установить приложение» — AI HUD
#     появится на главном экране как полноценное приложение.
#
#  Примечание: YOLOv8 и MediaPipe на Termux не устанавливаются (нет сборок) —
#  приложение автоматически использует HOG-детектор людей OpenCV и эвристику поз.
# ============================================================================
set -e

echo "=== [1/4] Обновление пакетов Termux ==="
yes | pkg update -y || true
pkg install -y python python-pip || pkg install -y python

echo "=== [2/4] Установка NumPy и OpenCV (из репозитория Termux) ==="
pkg install -y python-numpy opencv-python || pip install "numpy" "opencv-python-headless>=4.8,<5"

echo "=== [3/4] Установка Flask и qrcode ==="
pip install --upgrade pip
pip install flask qrcode

echo "=== [4/4] Запуск AI HUD на порту 5000 ==="
echo "Откройте в браузере телефона:  http://127.0.0.1:5000"
echo
exec python app.py --host 0.0.0.0 --port 5000
