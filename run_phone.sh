#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
#  AI HUD — ЗАПУСК СЕРВЕРА ПРЯМО НА ТЕЛЕФОНЕ (Android + Termux)
#
#  Простой путь (одна команда, всё скачивает сам) — install_phone.sh:
#    curl -fsSL https://raw.githubusercontent.com/maqsadjon57-code/arbitrage-win-rate-90-/arena/01a031d0-arbitrage-win-rate-90/install_phone.sh | bash
#
#  Этот скрипт (run_phone.sh) — если вы уже скопировали папку проекта на телефон:
#  1) Установите Termux из F-Droid:  https://f-droid.org/packages/com.termux/
#     (версия из Google Play устарела и не поддерживается)
#  2) Скопируйте папку проекта (app.py и эти файлы) во внутреннюю память
#     телефона, например в /sdcard/ai-hud/
#  3) В Termux выполните:
#         cd /sdcard/ai-hud
#         bash run_phone.sh
#  4) Скрипт установит зависимости, запустит сервер и сам откроет браузер
#     на http://127.0.0.1:5000 — это localhost, поэтому камера телефона
#     работает СРАЗУ, без HTTPS.
#  5) (Необязательно) Меню браузера → «Установить приложение» — AI HUD
#     появится на главном экране как полноценное приложение.
#
#  Примечание: YOLOv8 и MediaPipe на Termux не устанавливаются (нет сборок) —
#  приложение автоматически использует HOG-детектор людей OpenCV и эвристику поз.
# ============================================================================
set -e

echo "=== [1/4] Обновление пакетов Termux ==="
apt update -y >/dev/null 2>&1 || pkg update -y || true
pkg install -y python python-pip || pkg install -y python

echo "=== [2/4] Установка NumPy и OpenCV (из репозитория Termux, ~1-2 минуты) ==="
# цепочка fallback-ов: пакет Termux → пользовательский репозиторий (tur) → pip
pkg install -y python-numpy opencv-python \
  || (pkg install -y tur-repo && pkg install -y opencv-python) \
  || pip install "numpy" "opencv-python-headless>=4.8,<5"

echo "=== [3/4] Установка Flask и qrcode ==="
pip install --upgrade pip >/dev/null 2>&1 || true
pip install flask qrcode

echo "=== [4/4] Запуск AI HUD на порту 5000 ==="
echo "Через несколько секунд браузер откроется сам (или откройте http://127.0.0.1:5000)"
echo
# открываем браузер через 8 секунд, когда сервер уже поднят
( sleep 8; termux-open-url "http://127.0.0.1:5000" 2>/dev/null || true ) &
exec python app.py --host 0.0.0.0 --port 5000
