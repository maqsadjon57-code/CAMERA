# -*- coding: utf-8 -*-
# ======================================================================================
#  AI HUD WEB APPLICATION — Real-Time Computer Vision Overlay (Flask + MJPEG)
#  Сценарии: A) «Автомобильный HUD» (OBD-II телеметрия + диагностика узлов)
#            Б) «Аналитика студентов» (позы MediaPipe + рейтинг успеваемости)
# ======================================================================================
#
#  ------------------------------- requirements.txt ---------------------------------
#  flask>=3.0
#  opencv-python>=4.8        # на сервере без дисплея: opencv-python-headless
#  ultralytics>=8.1.0        # YOLOv8 (torch ставится автоматически как зависимость)
#  mediapipe>=0.10.9
#  numpy>=1.24
#  ----------------------------------------------------------------------------------
#  Установка:  pip install -r requirements.txt
#
#  ============================== ИНСТРУКЦИЯ ПО ЗАПУСКУ (RU) =========================
#  1. Установите Python 3.10+ (https://python.org). При установке на Windows отметьте
#     галочку "Add Python to PATH".
#
#  2. Установите зависимости:
#         pip install -r requirements.txt
#     (или вручную: pip install flask opencv-python ultralytics mediapipe numpy)
#
#  3. Запуск (из папки с файлом app.py):
#         python app.py                          # по умолчанию: веб-камера (--input 0)
#         python app.py --input 0                # обычная веб-камера
#         python app.py --input http://192.168.1.100:8080/video
#                                                  # IP-камера смартфона (см. пункт 6)
#         python app.py --input synth            # встроенная демо-сцена (без камеры)
#         python app.py --input video.mp4        # видеофайл
#         python app.py --port 5000 --host 0.0.0.0 --width 960 --max-fps 24
#
#  4. Откройте в браузере на КОМПЬЮТЕРЕ:     http://127.0.0.1:5000
#     (или http://localhost:5000). Если порт 5000 занят, укажите --port 5001.
#
#  5. Как открыть на СМАРТФОНЕ / другом устройстве в той же Wi-Fi сети:
#       a) Узнайте локальный IP компьютера:
#            Windows:  выполните  ipconfig  и найдите "IPv4-адрес" (например 192.168.1.5)
#            Linux/mac:  ip addr   или  ifconfig  (например 192.168.1.5)
#       b) Запустите сервер с доступом извне:   python app.py --host 0.0.0.0
#       c) На телефоне откройте:                http://192.168.1.5:5000
#       d) Если не открывается — разрешите порт 5000 в брандмауэре (firewall):
#            Windows (cmd от администратора):
#            netsh advfirewall firewall add rule name="AI HUD" dir=in action=allow protocol=TCP localport=5000
#
#  6. Как настроить ТЕЛЕФОН как IP-КАМЕРУ (приложение "IP Webcam", Android):
#       a) Установите "IP Webcam" из Google Play (автор: Pavel Khlebovich).
#       b) Телефон и компьютер должны быть в ОДНОЙ Wi-Fi сети.
#       c) Откройте приложение → внизу выберите "Запустить сервер" (Start).
#       d) На экране появится адрес, например:  http://192.168.1.100:8080
#          Видео-поток MJPEG доступен по адресу: http://192.168.1.100:8080/video
#       e) Вставьте эту ссылку в поле "Источник видео" в веб-интерфейсе и нажмите
#          "Подключить" — источник сменится БЕЗ перезапуска программы.
#          Либо запустите сразу: python app.py --input http://192.168.1.100:8080/video
#       f) Аналоги для iOS: "iVCam", "DroidCam" (в DroidCam используйте
#          http://IP:4747/video), "EpocCam".
#
#  7. Если веб-камера недоступна, а ссылки нет — сервер автоматически включит
#     встроенную СИНТЕТИЧЕСКУЮ ДЕМО-СЦЕНУ, чтобы HUD продолжал работать
#     (вы всегда можете вписать источник в поле на странице и нажать "Подключить").
#
#  8. Первый запуск скачивает модель YOLOv8n (~6 МБ, нужен интернет). Без интернета
#     приложение продолжает работать на встроенном синтетическом детекторе.
#
#  ======================================================================================
#  ПРИМЕЧАНИЕ ПО ШРИФТАМ: cv2.putText (шрифты Hershey) не поддерживает кириллицу,
#  поэтому весь текст, отрисовываемый ПОВЕРХ ВИДЕО — на английском (стиль бортового HUD),
#  а весь веб-интерфейс (страница, боковая панель, кнопки) — на русском языке.
# ======================================================================================

import argparse
import io
import json
import os
import random
import re
import socket
import threading
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

# ОПИСАНИЕ ЛОГИКИ: подключаем OpenCV и NumPy (обязательные зависимости).
# OPENCV_FFMPEG_CAPTURE_OPTIONS задаёт таймаут чтения сетевых потоков (RTSP/IP-камер),
# чтобы "мёртвая" ссылка не подвешивала поток захвата на минуты.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000")
import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify, send_file

# ОПИСАНИЕ ЛОГИКИ: необязательные тяжёлые зависимости (ultralytics/mediapipe) импортируем
# "мягко" — если их нет или модель недоступна, приложение деградирует до fallback-режимов,
# но не падает. Флаги доступности пишутся в состояние и показываются в веб-интерфейсе.
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False

# ======================================================================================
#  ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ И СОСТОЯНИЕ ПРИЛОЖЕНИЯ
# ======================================================================================

# ОПИСАНИЕ ЛОГИКИ: глобальный словарь состояний. Веб-кнопки меняют app_state['mode'] через
# POST /api/mode, а поток инференса читает его на КАЖДОМ кадре — переключение мгновенное.
app_state = {
    "mode": "vehicle",            # 'vehicle' | 'student' — активный сценарий
    "source": "0",                # текущий источник (зеркалится из ThreadedVideoCapture)
    "source_synthetic": False,    # True — работает демо-сцена (камера недоступна)
    "clients": 0,                 # сколько браузеров сейчас смотрят MJPEG-поток
    "started": time.time(),
}
STATE_LOCK = threading.Lock()     # блокировка на изменение app_state из разных потоков

STOP_EVENT = threading.Event()    # общий флаг завершения всех фоновых потоков

# ОПИСАНИЕ ЛОГИКИ: словарь классов для автомобильного сценария (по заданию).
# ВАЖНО: предобученная COCO-модель YOLOv8n НЕ знает таких классов — для боевого
# применения нужно дообучить YOLO на своём датасете (data.yaml с этими 4 классами).
# В демо-режиме метки COCO-детекций отображаются на этот словарь (class_id % 4),
# а синтетический детектор использует словарь напрямую.
VEHICLE_CLASSES = {
    0: "Engine_Block",
    1: "Belt_Pulley",
    2: "Dashboard",
    3: "Check_Engine_Light",
}
VEHICLE_CLASS_COLORS = {
    "Engine_Block":       (200, 200, 60),   # BGR
    "Belt_Pulley":        (40, 220, 240),
    "Dashboard":          (230, 140, 60),
    "Check_Engine_Light": (60, 60, 255),
}

# Статусы студентов и их цвета (BGR)
STUDENT_STATUS_COLORS = {
    "Actively Participating": (80, 220, 80),
    "Focused":                (230, 220, 60),
    "Confused":               (230, 120, 220),
    "Distracted":             (80, 140, 255),
}

# ======================================================================================
#  ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ: FPS-измеритель и шина событий
# ======================================================================================

class FPSMeter:
    """ОПИСАНИЕ ЛОГИКИ: потокобезопасный измеритель FPS на скользящем окне."""

    def __init__(self, window: int = 30):
        self._timestamps = deque(maxlen=window)
        self._lock = threading.Lock()

    def tick(self) -> None:
        with self._lock:
            self._timestamps.append(time.time())

    def fps(self) -> float:
        with self._lock:
            if len(self._timestamps) < 2:
                return 0.0
            dt = self._timestamps[-1] - self._timestamps[0]
            if dt <= 0:
                return 0.0
            return (len(self._timestamps) - 1) / dt


class EventBus:
    """ОПИСАНИЕ ЛОГИКИ: потокобезопасный журнал событий (deque с блокировкой).
    Сюда пишут диагностика автомобиля и трекер студентов, а веб-интерфейс читает
    последние события через GET /api/events (боковая панель «Журнал»)."""

    def __init__(self, maxlen: int = 250):
        self._events = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._counter = 0

    def add(self, text: str, level: str = "info", code: str = "") -> None:
        with self._lock:
            self._counter += 1
            self._events.appendleft({
                "id": self._counter,
                "time": datetime.now().strftime("%H:%M:%S"),
                "level": level,        # info | warn | error | ok
                "code": code,          # например CHP-0402
                "text": text,
            })

    def list(self, limit: int = 60) -> list:
        with self._lock:
            return list(self._events)[:max(1, min(limit, len(self._events)))]


event_bus = EventBus()

# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: ТЕЛЕМЕТРИЯ (OBD-II симулятор)
#  Плавный «случайный блуждатель» (random walk) с ограничением диапазона — данные
#  меняются плавно, как настоящие датчики. Работает в отдельном потоке, не мешая
#  захвату видео и инференсу.
# ======================================================================================

class TelemetrySimulator(threading.Thread):
    """Базовый поток телеметрии: раз в interval секунд обновляет значения."""

    def __init__(self, interval: float = 0.1):
        super().__init__(daemon=True, name="TelemetrySimulator")
        self.interval = interval
        self._lock = threading.Lock()

    @staticmethod
    def _walk(value: float, lo: float, hi: float, step: float) -> float:
        """Один шаг случайного блуждания с отражением от границ диапазона."""
        value += random.uniform(-step, step)
        return max(lo, min(hi, value))

    def run(self):
        while not STOP_EVENT.is_set():
            self._update()
            time.sleep(self.interval)

    def _update(self):  # переопределяется в наследниках
        pass


class VehicleTelemetry(TelemetrySimulator):
    """ОПИСАНИЕ ЛОГИКИ: симулятор OBD-II данных автомобиля.
    RPM      — обороты двигателя, 800..5000 об/мин;
    TEMP     — температура охлаждающей жидкости, 80..115 °C (>105 — перегрев);
    VIBRATION— уровень вибрации, нормированный 0.0..1.0 (>0.9 — критично).
    Значения случайным образом «проплывают» по диапазону, поэтому аварийные
    состояния возникают периодически — это удобно для демонстрации HUD."""

    def __init__(self):
        super().__init__(interval=0.12)
        self.rpm = 1400.0
        self.temp = 92.0
        self.vibration = 0.35

    def _update(self):
        with self._lock:
            self.rpm = self._walk(self.rpm, 800, 5000, 90)
            self.temp = self._walk(self.temp, 80, 115, 0.55)
            self.vibration = self._walk(self.vibration, 0.0, 1.0, 0.045)

    def get(self) -> dict:
        with self._lock:
            return {
                "rpm": int(round(self.rpm)),
                "temp": round(self.temp, 1),
                "vibration": round(self.vibration, 2),
            }


# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: СИНТЕТИЧЕСКИЕ ДЕМО-СЦЕНЫ
#  Если камера недоступна (нет устройства / мёртвая ссылка), генерируем анимированную
#  сцену на чистом NumPy/OpenCV: «моторный отсек» для режима vehicle и «аудиторию»
#  для режима student. Сцена — источник истинных координат объектов для синтетического
#  детектора: рамки HUD точно совпадают с нарисованными объектами.
# ======================================================================================

class SyntheticScene:
    """Генератор демо-кадров. draw(mode) возвращает (frame, objects):
    objects — список dict(name, bbox, conf, hint) — «детекции» нарисованных объектов."""

    def __init__(self, width: int = 960, height: int = 540):
        self.w, self.h = width, height
        self.t0 = time.time()
        self._last_objects = {"vehicle": [], "student": []}
        self._lock = threading.Lock()

    # ------------------------- общий фон -------------------------------------------
    def _noise_bg(self, base: tuple, spread: int = 6) -> np.ndarray:
        bg = np.full((self.h, self.w, 3), base, dtype=np.uint8)
        noise = np.random.randint(-spread, spread + 1, (self.h, self.w, 1))
        return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # ------------------------- СЦЕНА «МОТОРНЫЙ ОТСЕК» --------------------------------
    def draw_vehicle(self, telemetry: dict) -> tuple:
        t = time.time() - self.t0
        vib = telemetry.get("vibration", 0.0)
        # «Тряска» камеры пропорциональна вибрации: чем выше вибрация — тем сильнее дёргается кадр
        shake = int(vib * 6)
        dx, dy = random.randint(-shake, shake), random.randint(-shake, shake)

        frame = self._noise_bg((34, 30, 26))
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(frame, M, (self.w, self.h))

        objects = []

        # --- Блок двигателя: большой «агрегат» слева по центру ---
        ex1, ey1, ex2, ey2 = 70, 170, 470, 460
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (70, 62, 50), -1)
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (110, 100, 80), 2)
        for bx in range(ex1 + 40, ex2 - 20, 90):            # «болты» по периметру блока
            for by in (ey1 + 30, ey2 - 30):
                cv2.circle(frame, (bx, by), 9, (150, 140, 120), -1)
                cv2.circle(frame, (bx, by), 3, (40, 38, 34), -1)
        cv2.putText(frame, "V8 4.6L", (ex1 + 120, ey1 + 70), 1, 1.6, (190, 180, 160), 2)
        cv2.putText(frame, "CHP POWERTRAIN", (ex1 + 60, ey2 - 50), 1, 0.9, (150, 140, 120), 1)
        objects.append({"name": "Engine_Block", "bbox": (ex1, ey1, ex2, ey2),
                        "conf": 0.88 + 0.06 * np.sin(t * 0.7), "hint": None})

        # --- Шкив ремня: вращающийся круг со спицами справа от блока ---
        pcx, pcy, r = 610, 315, 78
        cv2.circle(frame, (pcx, pcy), r, (60, 70, 90), -1)
        ang = t * (2.0 + vib * 14.0)                        # скорость вращения зависит от вибрации
        for k in range(6):
            a = ang + k * np.pi / 3
            p2 = (int(pcx + r * 0.85 * np.cos(a)), int(pcy + r * 0.85 * np.sin(a)))
            cv2.line(frame, (pcx, pcy), p2, (140, 160, 190), 4)
        cv2.circle(frame, (pcx, pcy), 16, (180, 200, 230), -1)
        # ремень к маленькому шкиву генератора
        scx, scy = 810, 200
        cv2.circle(frame, (scx, scy), 34, (60, 70, 90), -1)
        cv2.line(frame, (pcx + r - 4, pcy - 12), (scx + 20, scy + 6), (35, 30, 28), 12)
        cv2.line(frame, (pcx + r - 4, pcy + 30), (scx + 24, scy + 28), (35, 30, 28), 12)
        objects.append({"name": "Belt_Pulley", "bbox": (pcx - r, pcy - r, pcx + r, pcy + r),
                        # уверенность «пульсирует»: периодически превышает порог 0.80
                        "conf": 0.55 + 0.42 * abs(np.sin(t * 0.35)), "hint": None})

        # --- Панель приборов: полоса сверху с датчиками ---
        dx1, dy1, dx2, dy2 = 60, 40, 900, 120
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (45, 42, 38), -1)
        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (95, 90, 80), 2)
        cv2.putText(frame, "DASHBOARD", (dx1 + 15, dy1 + 32), 1, 0.9, (200, 195, 180), 1)
        rpm = telemetry.get("rpm", 0)
        cv2.putText(frame, "RPM %04d" % rpm, (dx1 + 260, dy1 + 45), 2, 0.8, (120, 220, 120), 1)
        cv2.putText(frame, "TEMP %.1fC" % telemetry.get("temp", 0), (dx1 + 430, dy1 + 45),
                    2, 0.8, (120, 220, 120), 1)
        objects.append({"name": "Dashboard", "bbox": (dx1, dy1, dx2, dy2),
                        "conf": 0.84 + 0.05 * np.cos(t * 0.5), "hint": None})

        # --- Лампа Check Engine: мигает при перегреве / высокой вибрации ---
        overheat = telemetry.get("temp", 0) > 105.0
        blink = overheat or (vib > 0.9)
        if blink and int(t * 3) % 2 == 0:
            lx1, ly1, lx2, ly2 = 760, 55, 820, 95
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (40, 40, 235), -1)
            cv2.putText(frame, "CHK", (lx1 + 6, ly1 + 28), 1, 0.8, (255, 255, 255), 1)
            objects.append({"name": "Check_Engine_Light", "bbox": (lx1, ly1, lx2, ly2),
                            "conf": 0.93, "hint": None})

        # декоративная сетка-подложка «технического сканера»
        for gx in range(0, self.w, 60):
            cv2.line(frame, (gx, 0), (gx, self.h), (42, 40, 38), 1)
        for gy in range(0, self.h, 60):
            cv2.line(frame, (0, gy), (self.w, gy), (42, 40, 38), 1)

        with self._lock:
            self._last_objects["vehicle"] = objects
        return frame, objects

    # ------------------------- СЦЕНА «АУДИТОРИЯ» ------------------------------------
    def draw_student(self) -> tuple:
        t = time.time() - self.t0
        frame = self._noise_bg((36, 34, 30))

        # задник класса: доска и стена
        cv2.rectangle(frame, (0, 0), (self.w, self.h), (52, 48, 40), -1)
        cv2.rectangle(frame, (180, 40), (780, 190), (38, 60, 70), -1)
        cv2.rectangle(frame, (180, 40), (780, 190), (90, 120, 130), 2)
        cv2.putText(frame, "AI CLASS ANALYTICS", (250, 90), 2, 0.8, (170, 200, 210), 1)
        for i, yy in enumerate((120, 145, 168)):
            cv2.line(frame, (230 + i * 30, yy), (700 - i * 40, yy), (120, 150, 160), 1)

        objects = []
        # три «студента» за партами; состояние каждого циклически меняется во времени,
        # чтобы все статусы (Participating / Distracted / Confused / Focused) показывались
        seats = [
            {"x": 170, "phase": 7.0, "base": "arm"},     # периодически поднимает руку
            {"x": 430, "phase": 9.0, "base": "down"},    # периодически «клюёт носом»
            {"x": 690, "phase": 13.0, "base": "side"},   # периодически смотрит в сторону
        ]
        for idx, s in enumerate(seats):
            cx = s["x"]
            bob = int(6 * np.sin(t * 1.4 + idx * 2.1))             # лёгкое покачивание
            head_cy = 300 + bob + (34 if self._head_down(s, t) else 0)
            arm_up = self._arm_up(s, t)
            head_side = self._head_side(s, t)

            # партa
            cv2.rectangle(frame, (cx - 95, 430), (cx + 95, 470), (90, 75, 60), -1)
            cv2.rectangle(frame, (cx - 95, 430), (cx + 95, 470), (130, 110, 90), 2)
            # туловище
            cv2.rectangle(frame, (cx - 55, 330), (cx + 55, 430), (60, 70, 90), -1)
            cv2.rectangle(frame, (cx - 55, 330), (cx + 55, 430), (95, 110, 135), 2)
            # руки: поднята / лежат на парте
            if arm_up:
                cv2.line(frame, (cx + 40, 340), (cx + 85, 215 + bob), (150, 160, 180), 12)
                cv2.circle(frame, (cx + 85, 208 + bob), 14, (170, 180, 200), -1)
            else:
                cv2.line(frame, (cx + 40, 350), (cx + 90, 428), (140, 150, 170), 10)
                cv2.line(frame, (cx - 40, 350), (cx - 90, 428), (140, 150, 170), 10)
            # голова
            hx = cx + (26 if head_side else 0)
            cv2.circle(frame, (hx, head_cy), 40, (140, 150, 175), -1)
            cv2.circle(frame, (hx, head_cy), 40, (190, 200, 220), 2)
            # глаза (при повороте головы видно один eye)
            if head_side:
                cv2.circle(frame, (hx + 16, head_cy - 6), 5, (40, 40, 40), -1)
            else:
                cv2.circle(frame, (hx - 13, head_cy - 6), 5, (40, 40, 40), -1)
                cv2.circle(frame, (hx + 13, head_cy - 6), 5, (40, 40, 40), -1)

            # состояние этого «студента» — подаётся детектору как hint для fallback-позы
            if arm_up:
                hint = {"status": "Actively Participating", "hand_raised": True, "head_roll": 0}
            elif self._head_down(s, t):
                hint = {"status": "Distracted", "hand_raised": False, "head_roll": 28}
            elif head_side:
                hint = {"status": "Confused", "hand_raised": False, "head_roll": -17}
            else:
                hint = {"status": "Focused", "hand_raised": False, "head_roll": 3}

            objects.append({"name": "person", "bbox": (cx - 100, head_cy - 60, cx + 100, 432),
                            "conf": 0.86 + 0.08 * np.sin(t + idx), "hint": hint})

        with self._lock:
            self._last_objects["student"] = objects
        return frame, objects

    # ---- детерминированные «поведенческие» функции демо-студентов ----
    @staticmethod
    def _arm_up(seat: dict, t: float) -> bool:
        # рука поднята в течение 3.5 секунд каждые seat['phase'] секунд
        return (t % seat["phase"]) < 3.5 and seat["base"] == "arm"

    @staticmethod
    def _head_down(seat: dict, t: float) -> bool:
        # голова опущена по 4 секунды каждые seat['phase'] секунд
        return (t % seat["phase"]) < 4.0 and seat["base"] == "down"

    @staticmethod
    def _head_side(seat: dict, t: float) -> bool:
        return (t % seat["phase"]) < 4.5 and seat["base"] == "side"


# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: ЗАХВАТ ВИДЕО В ОТДЕЛЬНОМ ПОТОКЕ (ThreadedVideoCapture)
#  Всегда держит ОДИН самый свежий кадр (Queue(maxsize=1), старые отбрасываются),
#  поэтому медленный инференс никогда не копит лаг. Поддерживает:
#    --input 0                          → индекс локальной веб-камеры
#    --input http://192.168.1.100:8080/video → MJPEG-поток приложения "IP Webcam"
#    --input synth                       → встроенная демо-сцена
#    --input file.mp4                    → видеофайл (зацикливается)
#  Смена источника на лету: метод switch() кладёт новый источник в очередь управления,
#  поток захвата переоткрывает cv2.VideoCapture(...) без перезапуска сервера.
# ======================================================================================

class ThreadedVideoCapture(threading.Thread):
    def __init__(self, source: str, width: int = 960, mode_provider=None):
        super().__init__(daemon=True, name="ThreadedVideoCapture")
        self.source_raw = str(source)
        self.width = width
        self.mode_provider = mode_provider or (lambda: "vehicle")
        self._cap = None
        self._frame_queue = None            # Queue(maxsize=1) — всегда самый свежий кадр
        self._pending_source = None         # источник, который нужно открыть на лету
        self._control_lock = threading.Lock()
        self.is_synthetic = True            # True, пока реальный источник не открыт
        self.active_source = "synth"        # что реально открыто сейчас
        self.resolution = (0, 0)
        self.fps_meter = FPSMeter(20)
        self.scene = SyntheticScene(width=width, height=int(width * 9 / 16))
        self.telemetry_ref = None           # телеметрия нужна демо-сцене «моторного отсека»
        self._last_frame_time = 0.0

    # ---------- разбор строки источника ----------
    @staticmethod
    def parse_source(src: str):
        s = str(src).strip()
        if s.lower() in ("synth", "synthetic", "demo", "test", ""):
            return "synth"
        if re.fullmatch(r"\d+", s):
            return int(s)                   # индекс камеры: 0, 1, 2...
        return s                            # URL (http/rtsp) или путь к файлу

    # ---------- попытка открыть реальный источник ----------
    def _open_real(self, src) -> bool:
        try:
            cap = cv2.VideoCapture(src)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                cap.release()
                return False
            ok, frame = cap.read()          # пробный кадр с таймаутом через FFMPEG-опции
            if not ok or frame is None:
                cap.release()
                return False
            self._cap = cap
            self.is_synthetic = False
            self.active_source = str(src)
            h, w = frame.shape[:2]
            self.resolution = (w, h)
            return True
        except Exception:
            return False

    # ---------- переключение источника из веб-интерфейса ----------
    def switch(self, new_source: str) -> None:
        with self._control_lock:
            self._pending_source = str(new_source)

    # ---------- главный цикл захвата ----------
    def run(self):
        import queue as _queue
        src = self.parse_source(self.source_raw)

        if src == "synth":
            self._enter_synthetic("synth (запрошен пользователем)")
        else:
            if not self._open_real(src):
                event_bus.add(
                    "Не удалось открыть источник «%s». Включена демо-сцена — "
                    "проверьте камеру/ссылку и нажмите «Подключить»." % self.source_raw,
                    level="warn", code="SOURCE")
                self._enter_synthetic("synth (fallback: источник недоступен)")
            else:
                event_bus.add("Источник видео подключён: %s" % self.active_source,
                              level="ok", code="SOURCE")

        while not STOP_EVENT.is_set():
            # (1) ОПИСАНИЕ ЛОГИКИ: проверяем запрос на смену источника без перезапуска
            with self._control_lock:
                pending = self._pending_source
                self._pending_source = None
            if pending is not None:
                parsed = self.parse_source(pending)
                if parsed == "synth":
                    self._release_real()
                    self._enter_synthetic("synth (переключено пользователем)")
                    event_bus.add("Включена демо-сцена (synth)", level="info", code="SOURCE")
                else:
                    if self._open_real(parsed):
                        event_bus.add("Источник видео переключён: %s" % self.active_source,
                                      level="ok", code="SOURCE")
                    else:
                        event_bus.add("Не удалось открыть «%s» — остаётся прежний источник." % pending,
                                      level="error", code="SOURCE")

            # (2) читаем кадр
            frame = None
            if self.is_synthetic:
                mode = self.mode_provider()
                if mode == "vehicle":
                    tele = self.telemetry_ref.get() if self.telemetry_ref else {}
                    frame, _ = self.scene.draw_vehicle(tele)
                else:
                    frame, _ = self.scene.draw_student()
                time.sleep(1.0 / 30.0)      # демо-сцена генерируется с ~30 FPS
            elif self._cap is not None:
                ok, f = self._cap.read()
                if ok:
                    frame = f
                else:
                    # ОПИСАНИЕ ЛОГИКИ: видеофайл закончился → зацикливаем; поток «умер» → демо
                    src_str = str(self.active_source)
                    if not src_str.startswith("http") and not src_str.startswith("rtsp"):
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok2, f2 = self._cap.read()
                        if ok2:
                            frame = f2
                    if frame is None:
                        event_bus.add("Поток «%s» прерван. Включена демо-сцена." % self.active_source,
                                      level="error", code="SOURCE")
                        self._release_real()
                        self._enter_synthetic("synth (fallback: поток прерван)")

            if frame is None:
                time.sleep(0.05)
                continue

            # (3) приводим к рабочей ширине (ускоряет инференс, HUD всегда аккуратный)
            target_w = self.width
            h, w = frame.shape[:2]
            if w != target_w:
                k = target_w / float(w)
                frame = cv2.resize(frame, (target_w, max(2, int(round(h * k)))),
                                   interpolation=cv2.INTER_AREA)
            self.resolution = (frame.shape[1], frame.shape[0])

            # (4) публикуем самый свежий кадр; старый выбрасываем (anti-lag)
            if self._frame_queue is None:
                self._frame_queue = _queue.Queue(maxsize=1)
            try:
                self._frame_queue.get_nowait()
            except _queue.Empty:
                pass
            self._frame_queue.put(frame)
            self.fps_meter.tick()
            self._last_frame_time = time.time()

    # ---------- вспомогательные ----------
    def _release_real(self):
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
        self._cap = None
        self.is_synthetic = True

    def _enter_synthetic(self, label: str):
        self._release_real()
        self.active_source = label

    def read(self, timeout: float = 1.5):
        """Забрать самый свежий кадр (блокирующе, с таймаутом)."""
        if self._frame_queue is None:
            time.sleep(0.05)
            return None
        try:
            return self._frame_queue.get(timeout=timeout)
        except Exception:
            return None

    def last_scene_objects(self):
        """Детекции демо-сцены для текущего режима (используются fallback-детектором)."""
        mode = self.mode_provider()
        with self.scene._lock:
            return list(self.scene._last_objects.get(mode, []))


# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: ИНФЕРЕНС YOLOv8 (класс YOLOInference)
#  Реальная модель, если доступна (первый запуск скачивает yolov8n.pt), иначе —
#  встроенный SyntheticDetector по объектам демо-сцены. Класс отдаёт список Detection
#  в координатах кадра — дальше работают рендерер HUD и логика сценариев.
# ======================================================================================

@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: float
    label: str
    hint: dict = field(default=None)  # доп. данные (например, статус демо-студента)


class YOLOInference:
    """Обёртка над Ultralytics YOLOv8 с потокобезопасным вызовом predict().
    model_available=False → работает в паре с SyntheticDetector (оффлайн-демо)."""

    def __init__(self, model_path: str = "yolov8n.pt", conf: float = 0.35, imgsz: int = 480):
        self.conf = conf
        self.imgsz = imgsz
        self.model = None
        self.model_available = False
        self.model_name = "SYNTHETIC-DEMO"
        self._lock = threading.Lock()   # модель нельзя дергать из двух потоков одновременно
        if ULTRALYTICS_AVAILABLE:
            try:
                if not os.path.exists(model_path) and not self._network_ok():
                    raise ConnectionError("нет доступа к интернету для скачивания весов %s" % model_path)
                model = YOLO(model_path)            # авто-скачивание весов при наличии интернета
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                model.predict(dummy, verbose=False) # прогрев
                self.model = model
                self.model_available = True
                self.model_name = os.path.basename(model_path)
            except Exception as e:
                event_bus.add("YOLOv8 недоступна (%s) — включён синтетический детектор." % type(e).__name__,
                              level="warn", code="MODEL")

    @staticmethod
    def _network_ok(timeout: float = 2.0) -> bool:
        """Быстрая проверка доступа в интернет (чтобы офлайн-старт не ждал ретраев)."""
        try:
            socket.create_connection(("github.com", 443), timeout=timeout).close()
            return True
        except Exception:
            return False

    # ---------- сырой инференс ----------
    def _raw_detect(self, frame: np.ndarray) -> list:
        out = []
        with self._lock:
            results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                                         verbose=False, device="cpu")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return out
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, clss):
            out.append(Detection(int(x1), int(y1), int(x2), int(y2), float(cf), int(cl)))
        return out

    def detect_vehicle(self, frame: np.ndarray) -> list:
        """ОПИСАНИЕ ЛОГИКИ: автомобильный режим. Предобученная COCO-модель не знает
        классов Engine_Block/Belt_Pulley/..., поэтому в ДЕМОНСТРАЦИОННЫХ целях метка
        берётся из словаря VEHICLE_CLASSES по (id % 4). Для боевого применения
        дообучите модель на своём датасете — тогда label подставится автоматически."""
        if not self.model_available:
            return []
        dets = []
        for d in self._raw_detect(frame):
            label = VEHICLE_CLASSES[d.label % len(VEHICLE_CLASSES)]
            dets.append(Detection(d.x1, d.y1, d.x2, d.y2, d.conf, label))
        return dets

    def detect_students(self, frame: np.ndarray) -> list:
        """Студенческий режим: интересует только класс 0 (person) COCO."""
        if not self.model_available:
            return []
        dets = []
        for d in self._raw_detect(frame):
            if d.label == 0:
                dets.append(Detection(d.x1, d.y1, d.x2, d.y2, d.conf, "person"))
        return dets


class SyntheticDetector:
    """Fallback-детектор: возвращает «детекции» объектов, нарисованных демо-сценой.
    Координаты и уверенность совпадают с тем, что реально видно в кадре."""

    def __init__(self, capture: ThreadedVideoCapture):
        self.capture = capture

    def detect(self) -> list:
        objs = self.capture.last_scene_objects()
        out = []
        for o in objs:
            x1, y1, x2, y2 = o["bbox"]
            out.append(Detection(x1, y1, x2, y2, float(min(0.99, max(0.30, o["conf"]))),
                                 o["name"], hint=o.get("hint")))
        return out


class DetectorFacade:
    """ОПИСАНИЕ ЛОГИКИ: фасад выбирает источник детекций для каждого кадра:
    реальная камера + доступная YOLO → настоящий инференс;
    демо-сцена (или нет модели) → синтетические детекции сцены."""

    def __init__(self, yolo: YOLOInference, synth: SyntheticDetector):
        self.yolo = yolo
        self.synth = synth

    def detect(self, frame: np.ndarray, mode: str, capture: ThreadedVideoCapture) -> list:
        if capture.is_synthetic or not self.yolo.model_available:
            return self.synth.detect()
        if mode == "vehicle":
            return self.yolo.detect_vehicle(frame)
        return self.yolo.detect_students(frame)

# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: АНАЛИЗ ПОЗ СТУДЕНТОВ (MediaPipe)
#  Три уровня работы (пробуем по очереди, первый успешный используется):
#    tier 1 'mediapipe-legacy' — классический API mp.solutions.pose (mediapipe <=0.10.x);
#    tier 2 'mediapipe-tasks'  — новый API mediapipe.tasks (PoseLandmarker), модель
#                                pose_landmarker_lite.task автоматически скачивается
#                                в папку assets/ при первом запуске;
#    tier 3 'heuristic'        — эвристика без MediaPipe: используется hint демо-сцены,
#                                а для реальной камеры — детерминированная функция времени
#                                (честный демо-режим, помечен в UI как «эвристика»).
#  Классификация по заданию (приоритет сверху вниз):
#    рука поднята            → "Actively Participating"
#    голова опущена (вниз)   → "Distracted"
#    голова назад/в сторону  → "Confused"
#    взгляд прямо            → "Focused"
# ======================================================================================

# индексы ключевых точек MediaPipe Pose (33 landmarks)
LM_NOSE, LM_LEYE, LM_REYE, LM_LEAR, LM_REAR = 0, 2, 5, 7, 8
LM_LSHOULDER, LM_RSHOULDER = 11, 12
LM_LELBOW, LM_RELBOW = 13, 14
LM_LWRIST, LM_RWRIST = 15, 16

POSE_TASK_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                       "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")


class StudentPoseAnalyzer:
    def __init__(self, assets_dir: str = "assets"):
        self.tier = "heuristic"
        self.legacy_pose = None
        self.task_landmarker = None
        self._lock = threading.Lock()

        # ---- tier 1: legacy mp.solutions.pose ----
        if MEDIAPIPE_AVAILABLE:
            try:
                solutions = getattr(mp, "solutions", None)
                if solutions is not None and hasattr(solutions, "pose"):
                    self.legacy_pose = solutions.pose.Pose(
                        static_image_mode=True, model_complexity=1,
                        min_detection_confidence=0.5)
                    self.tier = "mediapipe-legacy"
            except Exception:
                self.legacy_pose = None

        # ---- tier 2: mediapipe.tasks PoseLandmarker ----
        if self.tier == "heuristic" and MEDIAPIPE_AVAILABLE:
            try:
                model_path = os.path.join(assets_dir, "pose_landmarker_lite.task")
                if not os.path.exists(model_path):
                    os.makedirs(assets_dir, exist_ok=True)
                    urllib.request.urlretrieve(POSE_TASK_MODEL_URL, model_path)  # нужен интернет
                from mediapipe.tasks.python import vision as mp_vision
                from mediapipe.tasks.python import BaseOptions
                opts = mp_vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=model_path),
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_poses=1, min_pose_detection_confidence=0.5)
                self.task_landmarker = mp_vision.PoseLandmarker.create_from_options(opts)
                self.tier = "mediapipe-tasks"
            except Exception:
                self.task_landmarker = None

        event_bus.add("Анализатор поз: активен уровень «%s»" % self.tier, level="info", code="POSE")

    # ---------- извлечение landmark'ов из кропа человека ----------
    def _landmarks(self, frame: np.ndarray, det: Detection):
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        h, w = frame.shape[:2]
        # расширяем кроп на 15% — чтобы плечи гарантированно попали в кадр
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        mx, my = int(bw * 0.15), int(bh * 0.15)
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
        cx2, cy2 = min(w, x2 + mx), min(h, y2 + my)
        if cx2 - cx1 < 20 or cy2 - cy1 < 20:
            return None
        crop = frame[cy1:cy2, cx1:cx2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        lms = None
        try:
            if self.tier == "mediapipe-legacy" and self.legacy_pose is not None:
                res = self.legacy_pose.process(rgb)
                if res.pose_landmarks:
                    lms = [(p.x, p.y) for p in res.pose_landmarks.landmark]
            elif self.tier == "mediapipe-tasks" and self.task_landmarker is not None:
                import mediapipe as _mp
                mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=rgb)
                res = self.task_landmarker.detect(mp_image)
                if res.pose_landmarks:
                    lms = [(p.x, p.y) for p in res.pose_landmarks[0]]
        except Exception:
            lms = None
        if lms is None:
            return None
        # переводим нормализованные координаты кропа в координаты всего кадра
        cw, ch = crop.shape[1], crop.shape[0]
        return [(cx1 + px * cw, cy1 + py * ch) for (px, py) in lms]

    # ---------- ОПИСАНИЕ ЛОГИКИ: правила классификации позы ----------
    def _classify(self, lms) -> dict:
        (nx, ny) = lms[LM_NOSE]
        lsx, lsy = lms[LM_LSHOULDER]
        rsx, rsy = lms[LM_RSHOULDER]
        lw = lms[LM_LWRIST]
        rw = lms[LM_RWRIST]
        lex, ley = lms[LM_LEYE]
        rex, rey = lms[LM_REYE]

        shoulder_mid = ((lsx + rsx) / 2.0, (lsy + rsy) / 2.0)
        shoulder_w = max(1.0, abs(rsx - lsx))
        face_h = max(1.0, abs(shoulder_mid[1] - ny))          # расстояние нос-плечи

        # (1) Рука поднята: ЗАПЯСТЬЕ выше линии плеч (ось Y в изображении направлена вниз)
        hand_raised = (lw[1] < lsy - shoulder_w * 0.06) or (rw[1] < rsy - shoulder_w * 0.06)

        # (2) Наклон головы (roll): угол линии «глаза» относительно горизонта, градусы
        roll = float(np.degrees(np.arctan2(rey - ley, max(1.0, rex - lex))))

        # (3) Голова опущена: нос ОПУСТИЛСЯ близко к линии плеч (подбородок к груди)
        head_down = ny > shoulder_mid[1] - face_h * 0.18 and (shoulder_mid[1] - ny) < face_h * 0.55

        # (4) Голова в сторону/назад: сильный крен ИЛИ нос сильно смещён к одному плечу (yaw-прокси)
        nose_shift = abs(nx - shoulder_mid[0]) / shoulder_w
        head_side = (abs(roll) > 20.0) or (nose_shift > 0.42)

        # приоритет статусов согласно заданию
        if hand_raised:
            status = "Actively Participating"
        elif head_down:
            status = "Distracted"
        elif head_side:
            status = "Confused"
        else:
            status = "Focused"
        return {"status": status, "hand_raised": hand_raised, "head_roll": roll,
                "landmarks": lms}

    # ---------- публичный метод ----------
    def analyze(self, frame: np.ndarray, det: Detection, sid: str = None) -> dict:
        if self.tier != "heuristic":
            lms = self._landmarks(frame, det)
            if lms is not None:
                return self._classify(lms)
            # landmark'ы не найдены (человек частично в кадре / синтетический рисунок)
            # → переходим к эвристике ниже
        return self._heuristic(det, sid)

    def _heuristic(self, det: Detection, sid) -> dict:
        """Эвристика без MediaPipe. Для демо-сцены берёт истинный hint сцены;
        для реального потока — детерминированная функция времени по ID студента
        (просто показывает все категории статусов в отсутствии модели)."""
        if det.hint:
            return {"status": det.hint["status"], "hand_raised": det.hint.get("hand_raised", False),
                    "head_roll": float(det.hint.get("head_roll", 0.0)), "landmarks": None}
        sid = sid or "0000"
        seed = (int(re.sub(r"\D", "", sid) or 0) * 37 + int(time.time() / 9.0)) % 4
        table = [("Actively Participating", True, 0.0), ("Focused", False, 2.0),
                 ("Confused", False, -22.0), ("Distracted", False, 30.0)]
        status, hand, roll = table[seed]
        return {"status": status, "hand_raised": hand, "head_roll": roll, "landmarks": None}


# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: ТРЕКЕР СТУДЕНТОВ (SimpleTracker)
#  Centroid-трекер: сопоставляет детекции между кадрами по близости центров,
#  назначает стабильные ID вида STD-0001, STD-0002..., ведёт историю статусов
#  (с гистерезисом на 3 кадра, чтобы статус не «мигал»), генерирует сглаженный
#  PRED. SCORE (случайное блуждание 0..100) и индекс активности для рейтинга.
# ======================================================================================

@dataclass
class Student:
    track_id: int
    sid: str
    cx: float
    cy: float
    bbox: tuple
    score: float = 55.0                    # PRED. SCORE 0..100
    activity: float = 50.0                 # индекс активности 0..100 (для сортировки)
    status: str = "Focused"
    hand_raised: bool = False
    head_roll: float = 0.0
    landmarks: object = None
    status_history: deque = field(default_factory=lambda: deque(maxlen=5))
    lost: int = 0
    last_seen: float = field(default_factory=time.time)


STATUS_ACTIVITY_VALUE = {
    "Actively Participating": 100.0,
    "Focused": 72.0,
    "Confused": 40.0,
    "Distracted": 15.0,
}


class SimpleTracker:
    def __init__(self, max_lost: int = 25):
        self.students = {}                 # track_id -> Student
        self.next_id = 1
        self.max_lost = max_lost
        self._lock = threading.Lock()

    def _center(self, det: Detection):
        return ((det.x1 + det.x2) / 2.0, (det.y1 + det.y2) / 2.0)

    def update(self, detections: list, poses: dict) -> None:
        """poses: dict(det_index -> результат StudentPoseAnalyzer.analyze)."""
        now = time.time()
        with self._lock:
            # --- (1) сопоставление детекций с существующими треками (жадное, по дистанции) ---
            unmatched_dets = list(range(len(detections)))
            for st in self.students.values():
                st._matched = False
                best_j, best_d = None, 1e9
                for j in unmatched_dets:
                    c = self._center(detections[j])
                    dist = ((c[0] - st.cx) ** 2 + (c[1] - st.cy) ** 2) ** 0.5
                    bw = max(1.0, st.bbox[2] - st.bbox[0])
                    if dist < best_d and dist < bw * 1.1:
                        best_j, best_d = j, dist
                if best_j is not None:
                    unmatched_dets.remove(best_j)
                    self._apply(st, detections[best_j], poses.get(best_j), now)
                    st._matched = True
                else:
                    st.lost += 1

            # --- (2) новые треки получают свежие ID STD-XXXX ---
            for j in unmatched_dets:
                det = detections[j]
                tid = self.next_id
                self.next_id += 1
                st = Student(track_id=tid, sid="STD-%04d" % tid,
                             cx=(det.x1 + det.x2) / 2.0, cy=(det.y1 + det.y2) / 2.0,
                             bbox=(det.x1, det.y1, det.x2, det.y2),
                             score=random.uniform(35, 75))
                self._apply(st, det, poses.get(j), now, is_new=True)
                self.students[tid] = st
                event_bus.add("Новый студент в кадре: %s" % st.sid, level="info", code="TRACK")

            # --- (3) удаляем «потерявшихся» ---
            for tid in [t for t, s in self.students.items() if s.lost > self.max_lost]:
                event_bus.add("%s покинул(а) кадр" % self.students[tid].sid, level="info", code="TRACK")
                del self.students[tid]

    def _apply(self, st: Student, det: Detection, pose: dict, now: float, is_new: bool = False):
        c = self._center(det)
        # экспоненциальное сглаживание центра (EMA) — рамка не дёргается
        k = 0.35 if not is_new else 1.0
        st.cx += (c[0] - st.cx) * k
        st.cy += (c[1] - st.cy) * k
        st.bbox = (det.x1, det.y1, det.x2, det.y2)
        st.lost = 0
        st.last_seen = now
        if pose:
            st.hand_raised = pose.get("hand_raised", False)
            st.head_roll = pose.get("head_roll", 0.0)
            st.landmarks = pose.get("landmarks")
            # гистерезис: статус меняется, когда новый продержится 3 из последних 5 кадров
            st.status_history.append(pose.get("status", "Focused"))
            if not is_new and len(st.status_history) >= 3:
                counts = {}
                for s in st.status_history:
                    counts[s] = counts.get(s, 0) + 1
                cand = max(counts.items(), key=lambda kv: kv[1])
                if cand[1] >= 3 and cand[0] != st.status:
                    event_bus.add("%s: статус → %s" % (st.sid, cand[0]),
                                  level="info" if cand[0] in ("Focused", "Actively Participating") else "warn",
                                  code="STATUS")
                    st.status = cand[0]
            elif is_new:
                st.status = pose.get("status", "Focused")
        # PRED. SCORE — плавное случайное блуждание; активные статусы подталкивают вверх
        target = STATUS_ACTIVITY_VALUE.get(st.status, 50.0)
        st.score = float(np.clip(st.score + random.uniform(-4, 4) + (target - st.score) * 0.03, 0, 100))
        st.activity += (target - st.activity) * 0.12

    def leaderboard(self) -> list:
        """Рейтинг по активности (для боковой панели и мини-таблицы на HUD)."""
        with self._lock:
            rows = [{
                "id": s.sid, "status": s.status, "score": int(round(s.score)),
                "activity": int(round(s.activity)), "hand_raised": s.hand_raised,
                "head_roll": int(round(s.head_roll)),
            } for s in self.students.values() if s.lost == 0]
        rows.sort(key=lambda r: r["activity"], reverse=True)
        return rows

    def snapshot(self) -> list:
        with self._lock:
            return [dict(sid=s.sid, bbox=s.bbox, status=s.status, score=int(round(s.score)),
                         landmarks=s.landmarks, hand_raised=s.hand_raised,
                         head_roll=float(s.head_roll), lost=s.lost)
                    for s in self.students.values() if s.lost == 0]


# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: ДИАГНОСТИКА АВТОМОБИЛЯ (VehicleDiagnostic)
#  Конечный автомат по правилам задания:
#    Belt_Pulley (conf > 80%) + вибрация > 0.9 → CHP-0402 | Bolt Wear Detected  [ERROR]
#    температура > 105 °C                      → CHP-0031 | Overheating Risk     [WARN]
#    иначе                                     → Normal                          [OK]
#  Событие пишется в журнал при ИЗМЕНЕНИИ статуса + периодическое подтверждение
#  каждые 5 секунд, пока активная неисправность держится (журнал не заспамливается).
# ======================================================================================

class VehicleDiagnostic:
    def __init__(self, refresh: float = 5.0):
        self.current = ("Normal", "", "ok")
        self._last_emit = 0.0
        self.refresh = refresh

    def update(self, detections: list, telemetry: dict) -> dict:
        belt_conf = 0.0
        for d in detections:
            if d.label == "Belt_Pulley":
                belt_conf = max(belt_conf, d.conf)

        if belt_conf > 0.80 and telemetry["vibration"] > 0.9:
            new = ("CHP-0402", "Bolt Wear Detected", "error")
        elif telemetry["temp"] > 105.0:
            new = ("CHP-0031", "Overheating Risk", "warn")
        else:
            new = ("Normal", "", "ok")

        now = time.time()
        if new[:2] != self.current[:2]:
            if new[0] == "Normal":
                event_bus.add("Диагностика: все системы в норме", level="ok", code="DTC")
            else:
                event_bus.add("Диагностика: %s | %s (Belt conf %.0f%%, TEMP %.1f°C, VIB %.2f)"
                              % (new[0], new[1], belt_conf * 100, telemetry["temp"],
                                 telemetry["vibration"]),
                              level=new[2], code=new[0])
            self._last_emit = now
        elif new[0] != "Normal" and now - self._last_emit > self.refresh:
            event_bus.add("Подтверждение: %s | %s остаётся активной" % (new[0], new[1]),
                          level=new[2], code=new[0])
            self._last_emit = now
        self.current = new
        return {"code": new[0], "text": new[1], "level": new[2], "belt_conf": belt_conf}

# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: РЕНДЕРЕР HUD (OverlayRenderer)
#  Вся графика рисуется на стороне Python/OpenCV (только примитивы cv2: rectangle,
#  line, putText, circle, addWeighted) и уходит в браузер готовым JPEG-кадром в
#  MJPEG-потоке. Кириллица в Hershey-шрифтах не поддерживается, поэтому подписи
#  на HUD — латиницей (стиль бортовой электроники).
# ======================================================================================

# палитра (BGR)
C_BG = (12, 14, 18)
C_PANEL = (24, 28, 36)
C_EDGE = (70, 80, 95)
C_TEXT = (225, 230, 235)
C_DIM = (140, 150, 160)
C_CYAN = (235, 190, 80)
C_GREEN = (90, 210, 90)
C_YELLOW = (70, 200, 230)
C_ORANGE = (60, 150, 255)
C_RED = (60, 60, 255)
C_WHITE = (255, 255, 255)


class OverlayRenderer:
    def __init__(self):
        self.line = cv2.LINE_AA

    # ---------------- низкоуровневые помощники ----------------
    def _panel(self, img, x1, y1, x2, y2, color=C_PANEL, alpha=0.62, border=C_EDGE, bw=1):
        """Полупрозрачная панель: копируем ROI, заливаем цветом и смешиваем addWeighted."""
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = img[y1:y2, x1:x2]
        overlay = roi.copy()
        overlay[:] = color
        cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, roi)
        if border is not None:
            cv2.rectangle(img, (x1, y1), (x2, y2), border, bw, self.line)

    def _text(self, img, txt, org, scale=0.5, color=C_TEXT, thickness=1, shadow=True, font=None):
        font = font or cv2.FONT_HERSHEY_SIMPLEX
        if shadow:
            cv2.putText(img, txt, (org[0] + 1, org[1] + 1), font, scale, (0, 0, 0),
                        thickness + 1, self.line)
        cv2.putText(img, txt, org, font, scale, color, thickness, self.line)

    def _bar(self, img, x, y, w, h, ratio, color, bg=(40, 45, 55)):
        """Горизонтальная шкала-индикатор с фоном."""
        ratio = float(np.clip(ratio, 0.0, 1.0))
        cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1, self.line)
        cv2.rectangle(img, (x, y), (x + w, y + h), C_EDGE, 1, self.line)
        if ratio > 0:
            cv2.rectangle(img, (x + 1, y + 1), (x + 1 + int((w - 2) * ratio), y + h - 1),
                          color, -1, self.line)

    def _corner_brackets(self, img, x1, y1, x2, y2, color, ln=18, th=2):
        """Угловые «скобки» вместо сплошной рамки — стиль военного/авто HUD."""
        for (px, py, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (px, py), (px + sx * ln, py), color, th, self.line)
            cv2.line(img, (px, py), (px, py + sy * ln), color, th, self.line)

    # ======================================================================
    #  ОБЩИЕ ЭЛЕМЕНТЫ: верхняя строка, нижняя строка, декор
    # ======================================================================
    def _chrome(self, img, mode_label, status, fps):
        h, w = img.shape[:2]
        blink = int(time.time() * 2) % 2 == 0
        # --- нижняя строка: REC + время + разрешение ---
        bar_h = 26
        self._panel(img, 0, h - bar_h, w, h, C_BG, 0.7)
        if blink:
            cv2.circle(img, (16, h - bar_h // 2), 5, C_RED, -1, self.line)
        self._text(img, "REC  AI-HUD v2.1  MODE: %s  STATUS: %s" % (mode_label, status),
                   (30, h - 9), 0.42, C_DIM, 1)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._text(img, ts, (w - 175, h - 9), 0.42, C_DIM, 1)
        # --- уголки кадра ---
        self._corner_brackets(img, 6, 6, w - 6, h - 6, C_CYAN, 22, 1)

    def _status_banner(self, img, code, text, level):
        h, w = img.shape[:2]
        colors = {"ok": C_GREEN, "warn": C_YELLOW, "error": C_RED}
        c = colors.get(level, C_CYAN)
        label = code if code == "Normal" else "DTC %s | %s" % (code, text)
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        bw_ = tw + 40
        x1 = (w - bw_) // 2
        self._panel(img, x1, 8, x1 + bw_, 44, (20, 24, 30), 0.75, border=c, bw=2)
        cv2.circle(img, (x1 + 16, 26), 6, c, -1, self.line)
        self._text(img, label, (x1 + 30, 32), 0.62, c, 2)
        return label

    # ======================================================================
    #  СЦЕНАРИЙ А: АВТОМОБИЛЬНЫЙ HUD
    # ======================================================================
    def draw_vehicle(self, img, detections, telemetry, status, fps):
        h, w = img.shape[:2]

        # --- (1) панель телеметрии OBD-II слева ---
        px, py, pw, ph = 14, 56, 250, 178
        self._panel(img, px, py, px + pw, py + ph)
        self._text(img, "OBD-II TELEMETRY", (px + 12, py + 20), 0.5, C_CYAN, 1)

        rpm = telemetry["rpm"]
        self._text(img, "RPM", (px + 12, py + 44), 0.45, C_DIM, 1)
        self._text(img, "%04d" % rpm, (px + 60, py + 46), 0.55, C_TEXT, 2)
        # шкала оборотов с зонами: до 3500 зелёная, до 4500 жёлтая, дальше красная
        bx, by, bw2, bh = px + 130, py + 34, 105, 14
        self._bar(img, bx, by, bw2, bh, rpm / 5500.0,
                  C_GREEN if rpm < 3500 else (C_YELLOW if rpm < 4500 else C_RED))
        for k in range(1, 5):  # риски каждые 1100 RPM
            cv2.line(img, (bx + int(bw2 * k / 5.4), by), (bx + int(bw2 * k / 5.4), by + bh),
                     C_BG, 1, self.line)

        temp = telemetry["temp"]
        self._text(img, "TEMP", (px + 12, py + 76), 0.45, C_DIM, 1)
        self._text(img, "%.1f C" % temp, (px + 60, py + 78), 0.55,
                   C_RED if temp > 105 else C_TEXT, 2)
        self._bar(img, bx, by + 32, bw2, bh, (temp - 60) / 60.0,
                  C_GREEN if temp < 100 else (C_YELLOW if temp <= 105 else C_RED))

        vib = telemetry["vibration"]
        self._text(img, "VIB", (px + 12, py + 108), 0.45, C_DIM, 1)
        self._text(img, "%.2f" % vib, (px + 60, py + 110), 0.55,
                   C_RED if vib > 0.9 else C_TEXT, 2)
        # вибрация — сегментная шкала (10 сегментов)
        seg_on = int(round(vib * 10))
        for i in range(10):
            sx = bx + i * 11
            c = C_GREEN if i < 6 else (C_YELLOW if i < 9 else C_RED)
            if i < seg_on:
                cv2.rectangle(img, (sx, by + 60), (sx + 8, by + 74), c, -1, self.line)
            else:
                cv2.rectangle(img, (sx, by + 60), (sx + 8, by + 74), (45, 50, 60), 1, self.line)

        belt = status.get("belt_conf", 0.0) * 100
        self._text(img, "BELT CONF: %2.0f%%" % belt, (px + 12, py + 152), 0.45,
                   C_YELLOW if belt > 80 else C_DIM, 1)
        self._text(img, "FPS %4.1f" % fps, (px + 130, py + 152), 0.45, C_DIM, 1)
        self._text(img, "SCAN: LIVE  BUS: OK", (px + 12, py + 170), 0.45, C_DIM, 1)

        # --- (2) рамки детекций узлов ---
        for d in detections:
            c = VEHICLE_CLASS_COLORS.get(d.label, C_CYAN)
            self._corner_brackets(img, d.x1, d.y1, d.x2, d.y2, c, 14, 2)
            cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), c, 1, self.line)
            label = "%s %.0f%%" % (d.label, d.conf * 100)
            (tw, thh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            ly = d.y1 - 6 if d.y1 - 18 > 0 else d.y2 + 16
            self._panel(img, d.x1, ly - 15, d.x1 + tw + 10, ly + 3, C_BG, 0.7, border=c, bw=1)
            self._text(img, label, (d.x1 + 5, ly - 3), 0.42, c, 1)
            # перекрестие в центре рамки
            ccx, ccy = (d.x1 + d.x2) // 2, (d.y1 + d.y2) // 2
            cv2.line(img, (ccx - 6, ccy), (ccx + 6, ccy), c, 1, self.line)
            cv2.line(img, (ccx, ccy - 6), (ccx, ccy + 6), c, 1, self.line)

        # --- (3) верхний баннер статуса и общий декор ---
        self._status_banner(img, status["code"], status["text"], status["level"])
        self._chrome(img, "VEHICLE", status["code"], fps)
        return img

    # ======================================================================
    #  СЦЕНАРИЙ Б: АНАЛИТИКА СТУДЕНТОВ
    # ======================================================================
    def _skeleton(self, img, lms, color):
        """Мини-скелет: плечи-локти-запястья + нос — если MediaPipe дал landmarks."""
        if not lms:
            return
        pairs = [(LM_LSHOULDER, LM_RSHOULDER), (LM_LSHOULDER, LM_LELBOW),
                 (LM_LELBOW, LM_LWRIST), (LM_RSHOULDER, LM_RELBOW),
                 (LM_RELBOW, LM_RWRIST)]
        for a, b in pairs:
            try:
                p1, p2 = lms[a], lms[b]
                cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2, self.line)
            except Exception:
                pass
        try:
            nose = lms[LM_NOSE]
            cv2.circle(img, (int(nose[0]), int(nose[1])), 4, color, -1, self.line)
        except Exception:
            pass

    def draw_student(self, img, students, fps):
        h, w = img.shape[:2]
        n = len(students)
        participating = sum(1 for s in students if s["status"] == "Actively Participating")

        # --- (1) информационная панель слева ---
        px, py, pw, ph = 14, 56, 235, 108
        self._panel(img, px, py, px + pw, py + ph)
        self._text(img, "CLASS ANALYTICS", (px + 12, py + 20), 0.5, C_CYAN, 1)
        self._text(img, "TRACKED: %d" % n, (px + 12, py + 44), 0.48, C_TEXT, 1)
        part_pct = int(100 * participating / n) if n else 0
        self._text(img, "PARTICIPATION: %d%%" % part_pct, (px + 12, py + 68), 0.48, C_GREEN, 1)
        self._text(img, "FPS %4.1f" % fps, (px + 12, py + 92), 0.45, C_DIM, 1)

        # --- (2) рамки студентов: ID + статус + PRED.SCORE ---
        for s in students:
            x1, y1, x2, y2 = s["bbox"]
            c = STUDENT_STATUS_COLORS.get(s["status"], C_CYAN)
            cv2.rectangle(img, (x1, y1), (x2, y2), c, 2, self.line)
            self._corner_brackets(img, x1, y1, x2, y2, c, 12, 2)
            l1 = "%s" % s["sid"]
            l2 = "SCORE:%d" % s["score"]
            (w1, _), _ = cv2.getTextSize(l1, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
            (w2, _), _ = cv2.getTextSize(l2, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            bw_ = max(w1, w2) + 12
            ly = y1 - 34 if y1 - 34 > 40 else y2 + 4
            self._panel(img, x1, ly, x1 + bw_, ly + 32, C_BG, 0.72, border=c, bw=1)
            self._text(img, l1, (x1 + 6, ly + 13), 0.46, c, 1)
            self._text(img, l2, (x1 + 6, ly + 27), 0.42, C_TEXT, 1)
            # статус под рамкой
            st = s["status"].upper()
            (w3, _), _ = cv2.getTextSize(st, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            sy = y2 + 4 if ly == y1 - 34 else y2 + 40
            if sy + 16 < h - 30:
                self._panel(img, x1, sy, x1 + w3 + 12, sy + 17, C_BG, 0.7, border=c, bw=1)
                self._text(img, st, (x1 + 6, sy + 12), 0.40, c, 1)
            # иконка поднятой руки / наклона головы
            extra = ""
            if s["hand_raised"]:
                extra = "HAND UP"
            elif abs(s["head_roll"]) > 15:
                extra = "ROLL %+d" % int(s["head_roll"])
            if extra:
                self._text(img, extra, (x1, y2 + 16 if sy == y2 + 4 else y2 + 60),
                           0.38, c, 1)
            self._skeleton(img, s.get("landmarks"), c)

        # --- (3) мини-таблица лидеров справа ---
        rows = sorted(students, key=lambda s: s["score"], reverse=True)[:4]
        if rows:
            rw, rh = 215, 26 + 22 * len(rows)
            rx = w - rw - 14
            self._panel(img, rx, 56, rx + rw, 56 + rh)
            self._text(img, "TOP STUDENTS", (rx + 10, 74), 0.46, C_CYAN, 1)
            yy = 92
            for i, s in enumerate(rows):
                c = STUDENT_STATUS_COLORS.get(s["status"], C_CYAN)
                self._text(img, "%d." % (i + 1), (rx + 10, yy), 0.42, C_DIM, 1)
                self._text(img, s["sid"], (rx + 28, yy), 0.42, c, 1)
                self._text(img, "%3d" % s["score"], (rx + rw - 44, yy), 0.42, C_TEXT, 1)
                self._bar(img, rx + rw - 100, yy - 9, 52, 8, s["score"] / 100.0, c)
                yy += 22

        self._status_banner(img, "STUDENT MODE", "%d tracked" % n, "ok" if n else "warn")
        self._chrome(img, "STUDENT", "%d TRACKED" % n, fps)
        return img

    # ======================================================================
    #  КАДР «НЕТ СИГНАЛА»
    # ======================================================================
    def draw_no_signal(self, w=960, h=540):
        img = np.full((h, w, 3), C_BG, dtype=np.uint8)
        self._text(img, "NO SIGNAL", (w // 2 - 110, h // 2), 1.0, C_RED, 2)
        self._text(img, "Waiting for video source...", (w // 2 - 120, h // 2 + 30), 0.5, C_DIM, 1)
        self._corner_brackets(img, 6, 6, w - 6, h - 6, C_EDGE, 22, 1)
        return img


# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: КОНВЕЙЕР ОБРАБОТКИ (HUDPipeline)
#  Отдельный поток: берёт свежий кадр у ThreadedVideoCapture → детекция (по текущему
#  режиму из app_state) → позы/трекинг или диагностика → рендер HUD → JPEG в общий
#  буфер, откуда его забирают ВСЕ подключённые браузеры (MJPEG). Одна операция
#  кодирования на кадр независимо от числа клиентов.
# ======================================================================================

class HUDPipeline(threading.Thread):
    def __init__(self, capture, detector, pose_analyzer, tracker, diagnostic,
                 renderer, telemetry, max_fps=24):
        super().__init__(daemon=True, name="HUDPipeline")
        self.capture = capture
        self.detector = detector
        self.pose = pose_analyzer
        self.tracker = tracker
        self.diagnostic = diagnostic
        self.renderer = renderer
        self.telemetry = telemetry
        self.interval = 1.0 / max(1, max_fps)
        self.fps_meter = FPSMeter(30)
        self._jpeg_lock = threading.Lock()
        self._latest_jpeg = None
        self.frame_id = 0

    def get_jpeg(self):
        with self._jpeg_lock:
            return self._latest_jpeg

    def run(self):
        while not STOP_EVENT.is_set():
            try:
                self._loop_once()
            except Exception as e:
                # ОПИСАНИЕ ЛОГИКИ: защита конвейера — ошибка одного кадра (например,
                # некорректная детекция) не должна останавливать весь видеопоток.
                event_bus.add("Ошибка кадра конвейера: %s (%s) — поток продолжается"
                              % (type(e).__name__, e), level="error", code="PIPE")
                time.sleep(0.1)

    def _loop_once(self):
        t0 = time.time()
        frame = self.capture.read(timeout=2.0)
        if frame is None:
            placeholder = self.renderer.draw_no_signal()
            ok, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                with self._jpeg_lock:
                    self._latest_jpeg = buf.tobytes()
            time.sleep(0.2)
            return

        # ОПИСАНИЕ ЛОГИКИ: режим читается НА КАЖДОМ КАДРЕ — переключение мгновенное
        with STATE_LOCK:
            mode = app_state["mode"]

        detections = self.detector.detect(frame, mode, self.capture)

        if mode == "vehicle":
            tele = self.telemetry.get()
            status = self.diagnostic.update(detections, tele)
            self.renderer.draw_vehicle(frame, detections, tele, status,
                                       self.fps_meter.fps())
        else:
            poses = {}
            snap = self.tracker.snapshot()
            for i, det in enumerate(detections):
                # ищем sid уже известного студента рядом с детекцией (для эвристики)
                sid = None
                for s in snap:
                    if self._near(det, s["bbox"]):
                        sid = s["sid"]
                        break
                poses[i] = self.pose.analyze(frame, det, sid)
            self.tracker.update(detections, poses)
            self.renderer.draw_student(frame, self.tracker.snapshot(),
                                       self.fps_meter.fps())

        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self._jpeg_lock:
                self._latest_jpeg = buf.tobytes()
            self.frame_id += 1
        self.fps_meter.tick()

        # ограничение частоты цикла (чтобы не грузить CPU впустую)
        spent = time.time() - t0
        if spent < self.interval:
            time.sleep(self.interval - spent)

    @staticmethod
    def _near(det, bbox, k=1.6):
        dx = abs((det.x1 + det.x2) / 2 - (bbox[0] + bbox[2]) / 2)
        dy = abs((det.y1 + det.y2) / 2 - (bbox[1] + bbox[3]) / 2)
        w = max(1.0, bbox[2] - bbox[0])
        return dx < w * k and dy < w * k

# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: ВЕБ-ИНТЕРФЕЙС (HTML + CSS + JS в одной строке-шаблоне)
#  Страница отдаётся через render_template_string. Видео — <img src="/video_feed">,
#  который принимает MJPEG-поток. Кнопки режимов и поле источника шлют POST через
#  Fetch API; журнал событий, рейтинг студентов и телеметрия опрашиваются JSON-API.
#  Вёрстка адаптивная: на экранах уже 960px боковая панель уезжает под видео.
# ======================================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, viewport-fit=cover">
<title>AI HUD — Real-Time Computer Vision Overlay</title>
<style>
  :root{
    --bg:#0b0f14; --panel:#121820; --panel2:#0e141b; --line:#22303e;
    --text:#e6edf3; --dim:#8b98a5; --cyan:#22d3ee; --green:#4ade80;
    --yellow:#facc15; --orange:#fb923c; --red:#f87171; --purple:#c084fc;
    --mono:'Consolas','SF Mono','Menlo',monospace;
  }
  *{box-sizing:border-box; margin:0; padding:0}
  html,body{height:100%}
  body{
    background:
      radial-gradient(1200px 500px at 80% -10%, rgba(34,211,238,.07), transparent 60%),
      radial-gradient(900px 400px at 0% 110%, rgba(74,222,128,.05), transparent 60%),
      var(--bg);
    color:var(--text); font-family:-apple-system,'Segoe UI',Roboto,'Ubuntu',sans-serif;
    min-height:100vh; display:flex; flex-direction:column;
  }
  header{
    display:flex; align-items:center; gap:14px; flex-wrap:wrap;
    padding:10px 16px; border-bottom:1px solid var(--line);
    background:rgba(14,20,27,.85); backdrop-filter:blur(6px); position:sticky; top:0; z-index:20;
  }
  .brand{display:flex; align-items:center; gap:10px; font-weight:700; letter-spacing:.5px}
  .brand .dot{width:10px; height:10px; border-radius:50%; background:var(--cyan);
    box-shadow:0 0 12px var(--cyan); animation:pulse 2s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
  .brand small{color:var(--dim); font-weight:400; font-size:11px}
  .modes{display:flex; gap:8px; flex-wrap:wrap}
  .btn{
    border:1px solid var(--line); background:var(--panel); color:var(--text);
    padding:9px 16px; border-radius:9px; cursor:pointer; font-size:14px;
    transition:.15s; user-select:none;
  }
  .btn:hover{border-color:var(--cyan)}
  .btn.active{background:linear-gradient(135deg,rgba(34,211,238,.25),rgba(34,211,238,.08));
    border-color:var(--cyan); box-shadow:0 0 14px rgba(34,211,238,.25)}
  .btn.green.active{background:linear-gradient(135deg,rgba(74,222,128,.25),rgba(74,222,128,.08));
    border-color:var(--green); box-shadow:0 0 14px rgba(74,222,128,.25)}
  .chips{display:flex; gap:6px; flex-wrap:wrap; margin-left:auto}
  .chip{font-family:var(--mono); font-size:11px; color:var(--dim);
    border:1px solid var(--line); border-radius:6px; padding:4px 8px; background:var(--panel2)}
  .chip b{color:var(--text); font-weight:600}
  main{flex:1; display:grid; grid-template-columns:minmax(0,1fr) 350px; gap:14px;
    padding:14px; max-width:1700px; width:100%; margin:0 auto}
  .video-col{display:flex; flex-direction:column; gap:10px; min-width:0}
  .video-wrap{
    position:relative; border:1px solid var(--line); border-radius:12px; overflow:hidden;
    background:#05080c; box-shadow:0 10px 40px rgba(0,0,0,.45);
  }
  .video-wrap img{display:block; width:100%; height:auto; aspect-ratio:16/9; object-fit:contain; background:#05080c}
  .live-badge{
    position:absolute; top:10px; right:12px; font-family:var(--mono); font-size:11px;
    color:#fff; background:rgba(10,12,16,.7); border:1px solid var(--line);
    padding:4px 10px; border-radius:6px; display:flex; gap:7px; align-items:center;
  }
  .live-badge .r{width:8px; height:8px; border-radius:50%; background:var(--red); animation:pulse 1.4s infinite}
  .video-off{position:absolute; inset:0; display:none; align-items:center; justify-content:center;
    color:var(--dim); font-size:14px; background:rgba(5,8,12,.9); text-align:center; padding:20px}
  .video-tools{display:flex; gap:8px; flex-wrap:wrap}
  .video-tools .btn{font-size:13px; padding:7px 13px}
  aside{display:flex; flex-direction:column; gap:12px; min-width:0}
  .card{border:1px solid var(--line); background:var(--panel); border-radius:12px; overflow:hidden}
  .card h3{font-size:12px; letter-spacing:1.2px; text-transform:uppercase; color:var(--dim);
    padding:10px 14px; border-bottom:1px solid var(--line); display:flex; align-items:center; gap:8px}
  .card h3 .ind{width:7px; height:7px; border-radius:50%; background:var(--cyan)}
  .card .body{padding:12px 14px}
  .src-row{display:flex; gap:8px}
  .src-row input{
    flex:1; min-width:0; background:var(--panel2); border:1px solid var(--line); color:var(--text);
    padding:9px 11px; border-radius:8px; font-family:var(--mono); font-size:12px;
  }
  .src-row input:focus{outline:none; border-color:var(--cyan)}
  .src-quick{display:flex; gap:6px; margin-top:8px; flex-wrap:wrap}
  .src-quick .btn{font-size:12px; padding:5px 10px}
  .hint{font-size:11.5px; color:var(--dim); margin-top:8px; line-height:1.5}
  .hint code{color:var(--cyan); font-family:var(--mono); font-size:11px}
  .tabs{display:flex; border-bottom:1px solid var(--line)}
  .tab{flex:1; text-align:center; padding:10px; font-size:13px; cursor:pointer;
    color:var(--dim); border-bottom:2px solid transparent}
  .tab.active{color:var(--text); border-bottom-color:var(--cyan)}
  .telemetry-grid{display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px}
  .tele{background:var(--panel2); border:1px solid var(--line); border-radius:9px; padding:9px}
  .tele .lab{font-size:10px; letter-spacing:1px; color:var(--dim)}
  .tele .val{font-family:var(--mono); font-size:19px; font-weight:700; margin:3px 0 6px}
  .tele .val.warn{color:var(--yellow)} .tele .val.crit{color:var(--red)}
  .gauge{height:6px; background:#1c2632; border-radius:4px; overflow:hidden}
  .gauge i{display:block; height:100%; width:0%; background:var(--green); transition:width .4s}
  .dtc{margin-top:10px; font-family:var(--mono); font-size:12px; padding:8px 10px;
    border-radius:8px; border:1px solid var(--line); background:var(--panel2); display:none}
  .dtc.ok{display:block; color:var(--green); border-color:rgba(74,222,128,.4)}
  .dtc.warn{display:block; color:var(--yellow); border-color:rgba(250,204,21,.4)}
  .dtc.error{display:block; color:var(--red); border-color:rgba(248,113,113,.5);
    animation:pulse 1s infinite}
  .feed{max-height:420px; overflow-y:auto}
  .ev{display:flex; gap:9px; padding:8px 14px; border-bottom:1px solid rgba(34,48,62,.5);
    font-size:12.5px; align-items:flex-start}
  .ev:last-child{border-bottom:none}
  .ev .t{font-family:var(--mono); color:var(--dim); font-size:11px; min-width:56px; padding-top:1px}
  .ev .b{width:8px; height:8px; border-radius:50%; margin-top:4px; flex:none}
  .ev.info .b{background:var(--cyan)} .ev.ok .b{background:var(--green)}
  .ev.warn .b{background:var(--yellow)} .ev.error .b{background:var(--red)}
  .ev .c{font-family:var(--mono); font-size:10.5px; color:var(--dim); margin-top:2px}
  .stu{padding:10px 14px; border-bottom:1px solid rgba(34,48,62,.5)}
  .stu:last-child{border-bottom:none}
  .stu .row1{display:flex; justify-content:space-between; align-items:center; font-size:13px}
  .stu .id{font-family:var(--mono); font-weight:700}
  .stu .sc{font-family:var(--mono); color:var(--dim)}
  .stu .badge{display:inline-block; font-size:10.5px; padding:2px 8px; border-radius:20px;
    margin:6px 0 7px; border:1px solid}
  .badge.Actively-Participating{color:var(--green); border-color:var(--green); background:rgba(74,222,128,.08)}
  .badge.Focused{color:var(--yellow); border-color:var(--yellow); background:rgba(250,204,21,.08)}
  .badge.Confused{color:var(--purple); border-color:var(--purple); background:rgba(192,132,252,.08)}
  .badge.Distracted{color:var(--orange); border-color:var(--orange); background:rgba(251,146,60,.08)}
  .stu .meta{font-size:10.5px; color:var(--dim); font-family:var(--mono)}
  .gauge.mini{height:5px; margin-top:4px}
  .empty{color:var(--dim); font-size:12.5px; text-align:center; padding:22px 10px}
  details.help{border-top:1px solid var(--line)}
  details.help summary{cursor:pointer; padding:10px 14px; font-size:12px; color:var(--dim);
    list-style:none; outline:none}
  details.help[open] summary{border-bottom:1px solid var(--line)}
  details.help .body{font-size:12px; color:var(--dim); line-height:1.65}
  details.help b{color:var(--text)}
  details.help code{color:var(--cyan); font-family:var(--mono); font-size:11px}
  footer{padding:10px 16px; border-top:1px solid var(--line); color:var(--dim);
    font-size:11.5px; display:flex; gap:10px; flex-wrap:wrap; justify-content:space-between}
  footer span code{font-family:var(--mono); color:var(--cyan)}
  .toast{
    position:fixed; bottom:18px; left:50%; transform:translateX(-50%) translateY(80px);
    background:var(--panel); border:1px solid var(--cyan); color:var(--text);
    padding:10px 18px; border-radius:10px; font-size:13px; transition:.3s; z-index:50;
    box-shadow:0 8px 30px rgba(0,0,0,.5);
  }
  .toast.show{transform:translateX(-50%) translateY(0)}
  @media (max-width:960px){
    main{grid-template-columns:1fr; padding:10px}
    header{padding:8px 10px; gap:8px}
    .chips{margin-left:0; width:100%}
    .feed{max-height:300px}
    .modes{width:100%; order:3}
    .modes .btn{flex:1; text-align:center}
  }
</style>
</head>
<body>
<header>
  <div class="brand"><span class="dot"></span>AI&nbsp;HUD <small>real-time vision overlay</small></div>
  <div class="modes">
    <button class="btn active" id="btn-vehicle" onclick="setMode('vehicle')">&#128663; Автомобиль</button>
    <button class="btn green" id="btn-student" onclick="setMode('student')">&#128218; Студенты</button>
  </div>
  <div class="chips">
    <span class="chip">FPS <b id="chip-fps">--</b></span>
    <span class="chip">МОДЕЛЬ <b id="chip-model">--</b></span>
    <span class="chip">ПОЗЫ <b id="chip-pose">--</b></span>
    <span class="chip">ИСТОЧНИК <b id="chip-src">--</b></span>
    <span class="chip">КЛИЕНТЫ <b id="chip-clients">--</b></span>
  </div>
</header>

<main>
  <!-- ================= ЛЕВАЯ КОЛОНКА: ВИДЕОПОТОК MJPEG ================= -->
  <section class="video-col">
    <div class="video-wrap" id="videoWrap">
      <img id="stream" src="/video_feed" alt="AI HUD видеопоток">
      <div class="live-badge"><span class="r"></span>LIVE&nbsp;MJPEG</div>
      <div class="video-off" id="videoOff"><div>
        <b>Поток прерван</b><br>переподключение...
      </div></div>
    </div>
    <div class="video-tools">
      <button class="btn" onclick="reconnectStream()">&#128260; Переподключить поток</button>
      <button class="btn" onclick="window.open('/api/snapshot','_blank')">&#128247; Снимок кадра</button>
      <button class="btn" onclick="toggleFullscreen()">&#9974; Во весь экран</button>
    </div>
  </section>

  <!-- ================= ПРАВАЯ КОЛОНКА: ПАНЕЛИ ================= -->
  <aside>
    <!-- Источник видео: веб-камера / IP-камера телефона / демо-сцена -->
    <div class="card">
      <h3><span class="ind"></span>Источник видео</h3>
      <div class="body">
        <div class="src-row">
          <input id="srcInput" placeholder="0 (веб-камера) или http://192.168.1.100:8080/video" spellcheck="false">
          <button class="btn" onclick="setSource()">&#128279; Подключить</button>
        </div>
        <div class="src-quick">
          <button class="btn" onclick="quickSrc('0')">&#128421; Веб-камера</button>
          <button class="btn" onclick="quickSrc('synth')">&#127760; Демо-сцена</button>
        </div>
        <div class="hint">Телефон-камера: приложение <b>IP Webcam</b> (Android) → «Запустить сервер» →
          вставьте ссылку вида <code>http://192.168.1.100:8080/video</code>.<br>
          Источник меняется на лету, без перезапуска программы.</div>
      </div>
    </div>

    <!-- Телеметрия (режим «Автомобиль») -->
    <div class="card" id="teleCard">
      <h3><span class="ind"></span>Телеметрия OBD-II</h3>
      <div class="body">
        <div class="telemetry-grid">
          <div class="tele"><div class="lab">ОБОРОТЫ</div>
            <div class="val" id="t-rpm">--</div>
            <div class="gauge"><i id="g-rpm"></i></div></div>
          <div class="tele"><div class="lab">ТЕМП. °C</div>
            <div class="val" id="t-temp">--</div>
            <div class="gauge"><i id="g-temp"></i></div></div>
          <div class="tele"><div class="lab">ВИБРАЦИЯ</div>
            <div class="val" id="t-vib">--</div>
            <div class="gauge"><i id="g-vib"></i></div></div>
        </div>
        <div class="dtc ok" id="dtcBox">NORMAL — все системы в норме</div>
      </div>
    </div>

    <!-- Журнал событий / Рейтинг студентов (вкладки) -->
    <div class="card">
      <div class="tabs">
        <div class="tab active" id="tab-events" onclick="showTab('events')">&#128220; Журнал событий</div>
        <div class="tab" id="tab-students" onclick="showTab('students')">&#127942; Рейтинг студентов</div>
      </div>
      <div class="feed" id="pane-events"><div class="empty">Журнал пуст — ожидание событий...</div></div>
      <div class="feed" id="pane-students" style="display:none">
        <div class="empty" id="stuEmpty">Студенты не обнаружены.<br>Наведите камеру на аудиторию.</div>
      </div>
    </div>

    <!-- Справка -->
    <div class="card">
      <details class="help">
        <summary>&#9432; Как открыть с телефона / подключить IP-камеру</summary>
        <div class="body">
          <b>На компьютере:</b> откройте <code>http://127.0.0.1:5000</code><br>
          <b>На телефоне (та же Wi-Fi):</b> откройте адрес компьютера
          <span id="phoneUrls">— определяется автоматически —</span><br><br>
          <b>IP Webcam (Android):</b><br>
          1. Установите приложение «IP Webcam» из Google Play.<br>
          2. Телефон и компьютер — в одной сети Wi-Fi.<br>
          3. Нажмите «Запустить сервер» внизу экрана.<br>
          4. Вставьте <code>http://IP_ТЕЛЕФОНА:8080/video</code> в поле выше и нажмите «Подключить».
        </div>
      </details>
    </div>
  </aside>
</main>

<footer>
  <span>AI HUD Web Application — Flask + OpenCV + YOLOv8 + MediaPipe (MJPEG streaming)</span>
  <span id="uptime">UPTIME --</span>
</footer>

<div class="toast" id="toast"></div>

<script>
"use strict";
// ============================ ОПИСАНИЕ ЛОГИКИ (JS) ============================
// 1) setMode()     — POST /api/mode: переключает сценарий инференса (vehicle/student).
// 2) setSource()   — POST /api/source: смена источника видео на лету (веб-камера / IP-ссылка).
// 3) pollState()   — GET /api/state раз в 1с: телеметрия, DTC, FPS, источник, клиенты.
// 4) pollEvents()  — GET /api/events раз в 1.2с: журнал событий в боковую панель.
// 5) pollStudents()— GET /api/students раз в 1.5с: рейтинг студентов (режим «Студенты»).
// 6) Поток MJPEG   — <img src="/video_feed">; при обрыве авто-переподключение через 2с.

var lastEventId = 0;
var lastMode = null;

function $(id){ return document.getElementById(id); }

function toast(msg){
  var t = $('toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(t._h); t._h = setTimeout(function(){ t.classList.remove('show'); }, 2600);
}

// ---------- переключение режима (кнопки «Автомобиль» / «Студенты») ----------
function setMode(m){
  fetch('/api/mode', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode: m})
  })
  .then(function(r){ return r.json(); })
  .then(function(d){
    if(d.ok){ applyModeUI(d.mode); toast(m === 'vehicle' ? 'Режим: Автомобильный HUD' : 'Режим: Аналитика студентов'); }
  })
  .catch(function(){ toast('Ошибка сети при переключении режима'); });
}

function applyModeUI(m){
  $('btn-vehicle').classList.toggle('active', m === 'vehicle');
  $('btn-student').classList.toggle('active', m === 'student');
  $('teleCard').style.display = (m === 'vehicle') ? '' : 'none';
  showTab(m === 'student' ? 'students' : 'events');
}

// ---------- смена источника видео без перезапуска ----------
function setSource(){
  var v = $('srcInput').value.trim();
  if(!v){ toast('Введите источник: 0, synth или http://...:8080/video'); return; }
  fetch('/api/source', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source: v})
  })
  .then(function(r){ return r.json(); })
  .then(function(d){ toast(d.ok ? 'Подключение: ' + d.source : d.error); })
  .catch(function(){ toast('Ошибка сети при смене источника'); });
}
function quickSrc(s){ $('srcInput').value = s; setSource(); }

// ---------- вкладки боковой панели ----------
function showTab(t){
  $('tab-events').classList.toggle('active', t === 'events');
  $('tab-students').classList.toggle('active', t === 'students');
  $('pane-events').style.display  = (t === 'events')  ? '' : 'none';
  $('pane-students').style.display = (t === 'students') ? '' : 'none';
}

// ---------- MJPEG-поток: обрыв и переподключение ----------
function reconnectStream(){
  var img = $('stream');
  $('videoOff').style.display = 'none';
  img.src = '/video_feed?t=' + Date.now();
}
$('stream').onerror = function(){
  $('videoOff').style.display = 'flex';
  setTimeout(reconnectStream, 2000);
};
$('stream').onload = function(){ $('videoOff').style.display = 'none'; };

function toggleFullscreen(){
  var el = $('videoWrap');
  if(document.fullscreenElement){ document.exitFullscreen(); }
  else if(el.requestFullscreen){ el.requestFullscreen(); }
}

// ---------- опрос состояния (телеметрия, DTC, метрики) ----------
function pollState(){
  fetch('/api/state').then(function(r){ return r.json(); }).then(function(s){
    $('chip-fps').textContent = s.pipeline_fps.toFixed(1);
    $('chip-model').textContent = s.model;
    $('chip-pose').textContent = s.pose_tier;
    $('chip-src').textContent = s.source_synthetic ? 'DEMO-СЦЕНА' : s.source;
    $('chip-clients').textContent = s.clients;
    var up = Math.floor(s.uptime_sec);
    $('uptime').textContent = 'UPTIME ' + Math.floor(up/3600) + 'ч ' +
      Math.floor(up%3600/60) + 'м ' + (up%60) + 'с';
    if(s.server_urls && s.server_urls.length){
      var html = '';
      for(var i=0;i<s.server_urls.length;i++){ html += '<code>' + s.server_urls[i] + '</code> '; }
      $('phoneUrls').innerHTML = html;
    }
    // телеметрия OBD-II
    var rpm = s.telemetry.rpm, temp = s.telemetry.temp, vib = s.telemetry.vibration;
    $('t-rpm').textContent = rpm;
    $('t-rpm').className = rpm > 4200 ? 'val crit' : (rpm > 3400 ? 'val warn' : 'val');
    $('g-rpm').style.width = Math.min(100, rpm/5000*100) + '%';
    $('t-temp').textContent = temp.toFixed(1);
    $('t-temp').className = temp > 105 ? 'val crit' : (temp > 100 ? 'val warn' : 'val');
    $('g-temp').style.width = Math.min(100, (temp-60)/55*100) + '%';
    $('g-temp').style.background = temp > 105 ? '#f87171' : (temp > 100 ? '#facc15' : '#4ade80');
    $('t-vib').textContent = vib.toFixed(2);
    $('t-vib').className = vib > 0.9 ? 'val crit' : (vib > 0.7 ? 'val warn' : 'val');
    $('g-vib').style.width = Math.min(100, vib*100) + '%';
    $('g-vib').style.background = vib > 0.9 ? '#f87171' : (vib > 0.7 ? '#facc15' : '#4ade80');
    // статус диагностики
    var dtc = $('dtcBox');
    if(s.vehicle_status.code === 'Normal'){
      dtc.className = 'dtc ok'; dtc.textContent = 'NORMAL — все системы в норме';
    } else {
      dtc.className = 'dtc ' + s.vehicle_status.level;
      dtc.textContent = 'DTC ' + s.vehicle_status.code + ' | ' + s.vehicle_status.text;
    }
    if(lastMode !== null && lastMode !== s.mode){ applyModeUI(s.mode); }
    lastMode = s.mode;
  }).catch(function(){});
}

// ---------- опрос журнала событий ----------
function pollEvents(){
  fetch('/api/events?limit=40').then(function(r){ return r.json(); }).then(function(list){
    if(!list.length) return;
    var pane = $('pane-events');
    var fresh = [];
    for(var i=0;i<list.length;i++){ if(list[i].id > lastEventId){ fresh.push(list[i]); } }
    if(!fresh.length) return;
    lastEventId = list[0].id;
    var empty = pane.querySelector('.empty'); if(empty){ empty.remove(); }
    for(var j=fresh.length-1;j>=0;j--){
      var e = fresh[j];
      var div = document.createElement('div');
      div.className = 'ev ' + e.level;
      var code = e.code ? ' <span class="c">[' + e.code + ']</span>' : '';
      div.innerHTML = '<span class="t">' + e.time + '</span><span class="b"></span>' +
        '<div><div>' + e.text + code + '</div></div>';
      pane.insertBefore(div, pane.firstChild);
    }
    while(pane.children.length > 60){ pane.removeChild(pane.lastChild); }
  }).catch(function(){});
}

// ---------- опрос рейтинга студентов ----------
function pollStudents(){
  fetch('/api/students').then(function(r){ return r.json(); }).then(function(rows){
    var pane = $('pane-students');
    if(!rows.length){
      pane.innerHTML = '<div class="empty" id="stuEmpty">Студенты не обнаружены.<br>' +
        'Наведите камеру на аудиторию.</div>';
      return;
    }
    var html = '';
    for(var i=0;i<rows.length;i++){
      var s = rows[i];
      var badge = s.status.replace(/ /g, '-');
      html += '<div class="stu"><div class="row1"><span class="id">#' + (i+1) + '  ' + s.id +
        '</span><span class="sc">PRED. SCORE ' + s.score + '</span></div>' +
        '<span class="badge ' + badge + '">' + s.status + '</span>' +
        '<div class="meta">активность ' + s.activity + '% · наклон головы ' + s.head_roll +
        '° · ' + (s.hand_raised ? 'рука поднята' : 'руки опущены') + '</div>' +
        '<div class="gauge mini"><i style="width:' + s.activity + '%"></i></div></div>';
    }
    pane.innerHTML = html;
  }).catch(function(){});
}

applyModeUI('vehicle');
pollState(); pollEvents(); pollStudents();
setInterval(pollState, 1000);
setInterval(pollEvents, 1200);
setInterval(pollStudents, 1500);
</script>
</body>
</html>
"""

# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: FLASK-ПРИЛОЖЕНИЕ И МАРШРУТЫ (REST API + MJPEG)
# ======================================================================================

app = Flask(__name__)
# глобальные ссылки на объекты конвейера (создаются в main())
_capture = None
_detector = None
_pose_analyzer = None
_tracker = None
_diagnostic = None
_renderer = None
_telemetry = None
_pipeline = None
_SERVER_URLS = []


def _local_server_urls(port: int) -> list:
    """Определение локальных IP-адресов машины — чтобы показать пользователю,
    какой адрес вводить на телефоне (http://IP:PORT)."""
    urls = []
    try:
        hostnames = [socket.gethostname()]
        for hname in hostnames:
            for info in socket.getaddrinfo(hname, None, socket.AF_INET):
                ip = info[4][0]
                if ip.startswith(("192.168.", "10.", "172.")) and ip not in urls:
                    urls.append("http://%s:%d" % (ip, port))
    except Exception:
        pass
    if not urls:
        urls.append("http://<IP-КОМПЬЮТЕРА>:%d" % port)
    return urls


@app.route("/")
def index():
    """Главная страница: HUD + боковая панель (HTML/CSS/JS зашиты в шаблон выше)."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed():
    """ОПИСАНИЕ ЛОГИКИ: MJPEG-поток. multipart/x-mixed-replace: браузер получает
    бесконечную последовательность JPEG-кадров и рисует их как живое видео.
    Кадры берём из общего буфера HUDPipeline — кодирование одно на всех клиентов."""

    def generate():
        with STATE_LOCK:
            app_state["clients"] += 1
        try:
            while not STOP_EVENT.is_set():
                jpg = _pipeline.get_jpeg() if _pipeline else None
                if jpg is None:
                    time.sleep(0.08)
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1.0 / 30.0)
        finally:
            with STATE_LOCK:
                app_state["clients"] = max(0, app_state["clients"] - 1)

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/state")
def api_state():
    """Снимок состояния для веб-интерфейса: режим, источник, FPS, телеметрия, DTC."""
    tele = _telemetry.get() if _telemetry else {"rpm": 0, "temp": 0, "vibration": 0}
    code, text, level = _diagnostic.current if _diagnostic else ("Normal", "", "ok")
    return jsonify({
        "mode": app_state["mode"],
        "source": _capture.active_source if _capture else "-",
        "source_synthetic": bool(_capture.is_synthetic) if _capture else True,
        "resolution": list(_capture.resolution) if _capture else [0, 0],
        "capture_fps": round(_capture.fps_meter.fps(), 1) if _capture else 0.0,
        "pipeline_fps": round(_pipeline.fps_meter.fps(), 1) if _pipeline else 0.0,
        "frame_id": _pipeline.frame_id if _pipeline else 0,
        "model": _detector.yolo.model_name if _detector else "-",
        "pose_tier": _pose_analyzer.tier if _pose_analyzer else "-",
        "clients": app_state["clients"],
        "uptime_sec": int(time.time() - app_state["started"]),
        "telemetry": tele,
        "vehicle_status": {"code": code, "text": text, "level": level},
        "students_count": len(_tracker.leaderboard()) if _tracker else 0,
        "server_urls": _SERVER_URLS,
    })


@app.route("/api/events")
def api_events():
    """Журнал последних событий (боковая панель «Журнал событий»)."""
    limit = request.args.get("limit", default=40, type=int)
    return jsonify(event_bus.list(limit))


@app.route("/api/students")
def api_students():
    """Рейтинг студентов по активности (боковая панель «Рейтинг студентов»)."""
    return jsonify(_tracker.leaderboard() if _tracker else [])


@app.route("/api/mode", methods=["POST"])
def api_mode():
    """ОПИСАНИЕ ЛОГИКИ: переключение сценария по кнопке в интерфейсе. Меняем
    app_state['mode'] под блокировкой — поток инференса увидит это на след. кадре."""
    data = request.get_json(silent=True) or {}
    mode = str(data.get("mode", "")).lower()
    if mode not in ("vehicle", "student"):
        return jsonify({"ok": False, "error": "mode должен быть 'vehicle' или 'student'"}), 400
    with STATE_LOCK:
        app_state["mode"] = mode
    event_bus.add("Режим инференса переключён: %s" %
                  ("Автомобильный HUD" if mode == "vehicle" else "Аналитика студентов"),
                  level="info", code="MODE")
    return jsonify({"ok": True, "mode": mode})


@app.route("/api/source", methods=["POST"])
def api_source():
    """Смена источника видео без перезапуска: '0' | 'synth' | URL | путь к файлу."""
    data = request.get_json(silent=True) or {}
    source = str(data.get("source", "")).strip()
    if not source:
        return jsonify({"ok": False, "error": "Пустой источник"}), 400
    with STATE_LOCK:
        app_state["source"] = source
    _capture.switch(source)      # поток захвата переоткроет источник асинхронно
    return jsonify({"ok": True, "source": source})


@app.route("/api/snapshot")
def api_snapshot():
    """Снимок текущего кадра HUD (то, что видно в потоке, без перепаковки)."""
    jpg = _pipeline.get_jpeg() if _pipeline else None
    if jpg is None:
        return "Кадр ещё не готов", 503
    fname = "hud_snapshot_%s.jpg" % datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        os.makedirs("snapshots", exist_ok=True)
        with open(os.path.join("snapshots", fname), "wb") as f:
            f.write(jpg)
    except Exception:
        pass
    return send_file(io.BytesIO(jpg), mimetype="image/jpeg",
                     as_attachment=True, download_name=fname)


# ======================================================================================
#  ОПИСАНИЕ ЛОГИКИ: ТОЧКА ВХОДА
#  Разбираем аргументы (--input, --host, --port, ...), создаём все объекты,
#  запускаем фоновые потоки (телеметрия → захват → конвейер) и поднимаем Flask.
# ======================================================================================

def build_arg_parser():
    p = argparse.ArgumentParser(
        description="AI HUD Web Application — Flask + MJPEG + YOLOv8 + MediaPipe")
    p.add_argument("--input", default="0",
                   help="источник видео: 0 (веб-камера), http://192.168.1.100:8080/video "
                        "(IP Webcam), synth (демо-сцена), video.mp4 (файл)")
    p.add_argument("--host", default="0.0.0.0", help="адрес веб-сервера (0.0.0.0 — доступ с телефона)")
    p.add_argument("--port", type=int, default=5000, help="порт веб-сервера (по умолчанию 5000)")
    p.add_argument("--model", default="yolov8n.pt", help="модель YOLOv8 (yolov8n.pt / yolov8s.pt)")
    p.add_argument("--conf", type=float, default=0.35, help="порог уверенности детекции (0..1)")
    p.add_argument("--width", type=int, default=960, help="рабочая ширина кадра HUD")
    p.add_argument("--max-fps", type=int, default=24, help="ограничение FPS конвейера")
    return p


def print_banner(args):
    yolo_ok = bool(_detector and _detector.yolo.model_available)
    print("=" * 78)
    print("  AI HUD WEB APPLICATION — Flask + OpenCV + YOLOv8 + MediaPipe")
    print("=" * 78)
    print("  Источник видео : %s" % args.input)
    print("  YOLOv8         : %s" % ("модель загружена (%s)" % _detector.yolo.model_name if yolo_ok
                                    else "модель НЕ загружена — работает синтетический детектор"))
    print("  MediaPipe      : %s" % ("доступен (уровень: %s)" % _pose_analyzer.tier
                                    if MEDIAPIPE_AVAILABLE else "НЕ доступен (эвристика)"))
    print("-" * 78)
    print("  ОТКРЫТЬ В БРАУЗЕРЕ НА КОМПЬЮТЕРЕ:  http://127.0.0.1:%d" % args.port)
    for u in _SERVER_URLS:
        print("  ОТКРЫТЬ С ТЕЛЕФОНА (та же Wi-Fi):  %s" % u)
    print("  Переключение режимов и источника — кнопками в веб-интерфейсе.")
    print("=" * 78)
    print("  Остановить сервер: Ctrl+C")
    print()


def main():
    global _capture, _detector, _pose_analyzer, _tracker, _diagnostic, _renderer, _telemetry, _pipeline
    args = build_arg_parser().parse_args()
    _SERVER_URLS.extend(_local_server_urls(args.port))

    # ---- создаём все компоненты системы ----
    _telemetry = VehicleTelemetry()                                   # OBD-II симулятор
    _capture = ThreadedVideoCapture(args.input, width=args.width,
                                    mode_provider=lambda: app_state["mode"])
    _capture.telemetry_ref = _telemetry                               # сцене нужна телеметрия
    _yolo = YOLOInference(model_path=args.model, conf=args.conf)
    _detector = DetectorFacade(_yolo, SyntheticDetector(_capture))
    _pose_analyzer = StudentPoseAnalyzer()
    _tracker = SimpleTracker()
    _diagnostic = VehicleDiagnostic()
    _renderer = OverlayRenderer()
    _pipeline = HUDPipeline(_capture, _detector, _pose_analyzer, _tracker,
                            _diagnostic, _renderer, _telemetry, max_fps=args.max_fps)

    # ---- запускаем фоновые потоки: телеметрия, захват, конвейер HUD ----
    _telemetry.start()
    _capture.start()
    _pipeline.start()

    event_bus.add("Сервер AI HUD запущен (порт %d)" % args.port, level="ok", code="SYSTEM")
    event_bus.add("Источник при старте: %s" % args.input, level="info", code="SOURCE")

    print_banner(args)
    try:
        # threaded=True: MJPEG-поток + AJAX-опрос работают параллельно
        app.run(host=args.host, port=args.port, threaded=True, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        STOP_EVENT.set()
        print("\nСервер остановлен.")


if __name__ == "__main__":
    main()
