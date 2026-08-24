#!/data/data/com.termux/files/usr/bin/bash
# ============================================================================
#  AI HUD — УСТАНОВКА ОДНОЙ КОМАНДОЙ НА ТЕЛЕФОН (Android + Termux)
#
#  Скопируйте в Termux и выполните:
#    curl -fsSL https://raw.githubusercontent.com/maqsadjon57-code/arbitrage-win-rate-90-/arena/01a031d0-arbitrage-win-rate-90/install_phone.sh | bash
#
#  Что делает скрипт:
#    1) ставит Python и зависимости (numpy, OpenCV из репозитория Termux);
#    2) скачивает проект (app.py и файлы) в ~/ai-hud;
#    3) запускает сервер и сам открывает браузер на http://127.0.0.1:5000.
# ============================================================================
set -e

REPO="maqsadjon57-code/arbitrage-win-rate-90-"
BRANCH="arena/01a031d0-arbitrage-win-rate-90"

echo "============================================================"
echo "  AI HUD — установка на телефон (Termux)"
echo "============================================================"

echo "[1/5] Обновление пакетов Termux..."
apt update -y >/dev/null 2>&1 || pkg update -y || true

echo "[2/5] Установка Python и curl..."
pkg install -y python python-pip curl || {
  echo "Не удалось установить Python. Обновите Termux из F-Droid и повторите."
  exit 1
}

echo "[3/5] Скачивание проекта в ~/ai-hud ..."
mkdir -p ~/ai-hud && cd ~/ai-hud
curl -fsSL "https://github.com/${REPO}/archive/${BRANCH}.tar.gz" -o /tmp/aihud.tar.gz
tar xzf /tmp/aihud.tar.gz --strip-components=1
rm -f /tmp/aihud.tar.gz
ls

echo "[4/5] Установка NumPy и OpenCV (пакеты Termux, ~1-2 минуты)..."
pkg install -y python-numpy opencv-python || {
  echo "Пакет opencv-python недоступен — пробую pip..."
  pip install "numpy" "opencv-python-headless>=4.8,<5"
}

echo "[4b/5] Установка Flask и qrcode..."
pip install flask qrcode

echo "[5/5] Запуск сервера..."
echo "============================================================"
echo "  Через несколько секунд браузер откроется сам."
echo "  Если нет — откройте вручную:  http://127.0.0.1:5000"
echo "  Там нажмите «Разрешить доступ к камере»."
echo "============================================================"
# открываем браузер через 8 секунд, когда сервер уже поднят
( sleep 8; termux-open-url "http://127.0.0.1:5000" 2>/dev/null || true ) &
exec python app.py --host 0.0.0.0 --port 5000
