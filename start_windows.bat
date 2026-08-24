@echo off
chcp 65001 >nul
title AI HUD - Real-Time Vision
cd /d "%~dp0"

echo ============================================================
echo   AI HUD - запуск на компьютере (Windows) одним файлом
echo ============================================================
echo.

rem ---------- 1. ищем Python ----------
set PY=
where py >nul 2>nul
if not errorlevel 1 set PY=py
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 set PY=python
)
if defined PY goto :found
echo.
echo   [!] Python не найден.
echo   Установите его с https://www.python.org/downloads/
echo   ВАЖНО: при установке отметьте галочку "Add Python to PATH",
echo   затем запустите этот файл ещё раз.
echo.
pause
exit /b 1

:found
echo [1/3] Python найден: %PY%

rem ---------- 2. если app.py нет рядом - скачиваем проект с GitHub ----------
if exist app.py goto :deps
echo [2/3] Скачиваю проект с GitHub...
powershell -NoProfile -Command "try { Invoke-WebRequest 'https://github.com/maqsadjon57-code/arbitrage-win-rate-90-/archive/refs/heads/arena/01a031d0-arbitrage-win-rate-90.zip' -OutFile '%TEMP%\aihud.zip'; Expand-Archive '%TEMP%\aihud.zip' -DestinationPath '%TEMP%\aihud' -Force; Copy-Item '%TEMP%\aihud\*\*' '.' -Recurse -Force } catch { exit 1 }"
if errorlevel 1 (
  echo   [!] Не удалось скачать проект. Проверьте интернет и повторите.
  pause
  exit /b 1
)
if not exist app.py (
  echo   [!] Файл app.py не найден после скачивания.
  pause
  exit /b 1
)

:deps
rem ---------- 3. зависимости (лёгкий набор: без тяжёлых ML) ----------
echo [3/3] Устанавливаю зависимости flask numpy qrcode opencv...
%PY% -m pip install --quiet --disable-pip-version-check flask numpy qrcode "opencv-python-headless>=4.8,<5"
echo.
echo ============================================================
echo   Готово! Открываю браузер:  http://127.0.0.1:5000
echo   Телефон в той же Wi-Fi: откройте http://IP-КОМПЬЮТЕРА:5000
echo   Для камеры телефона по Wi-Fi запустите: start_windows.bat --https
echo   Остановить сервер: закройте это окно или Ctrl+C
echo ============================================================
start "" http://127.0.0.1:5000
%PY% app.py --host 0.0.0.0 --port 5000 %*
pause
