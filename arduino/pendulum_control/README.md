# Arduino Pendulum Control (ATmega328P)

Квантованная PPO политика для управления инвертированным маятником на Arduino Uno.

---

## Конфигурация

| Параметр | Значение |
|----------|----------|
| **Кейс** | Single pendulum, no position control |
| **Наблюдение** | 4 признака: `[sinθ1, cosθ1, dθ1, dx]` |
| **Действие** | Сила в `[-max_force, +max_force]` → PWM + DIR |
| **Частота управления** | 500 Гц (2 мс) |
| **Квантование** | Post-training INT8 asymmetric |
| **Архитектура НС** | MLP: `4 → 32 → 1` (Tanh) |

---

## Железо

| Компонент | Модель | Интерфейс | Пины Arduino |
|-----------|--------|-----------|--------------|
| Энкодер маятника (θ1) | AS5600 | I2C | A4 (SDA), A5 (SCL) |
| Энкодер тележки (x) | AS5600 | I2C | Тот же bus (адрес 0x36 / 0x37) |
| Драйвер мотора | TB6612FNG | PWM + DIR | PWM: D9 (OC1A), DIR: D2 |
| Питание Arduino | USB | — | — |
| Питание мотора | Отдельный блок (12-24V) | — | VM на TB6612 |

> **Важно:** AS5600 имеет фиксированный I2C адрес 0x36. Для двух энкодеров нужен мультиплексор (TCA9548A) или второй I2C bus (SoftI2C).

---

## Структура проекта

```
pendulum_control/
├── pendulum_control.ino       # Основной скетч
├── config.h                   # Пины, частоты, константы
├── model_weights.h            # INT8 веса в PROGMEM (генерируется)
├── inference.h / inference.cpp # INT8 forward pass (Q7.8 fixed-point)
├── as5600.h / as5600.cpp      # I2C драйвер AS5600
├── sensor_block.h / sensor_block.cpp # Квантование/шум/дифф/ФНЧ
└── tb6612.h / tb6612.cpp      # Драйвер TB6612FNG
```

---

## Сборка и загрузка

```bash
# Arduino IDE
# Файл → Открыть → pendulum_control.ino
# Инструменты → Плата: Arduino Uno
# Инструменты → Порт: /dev/ttyUSB0 (или COM3)
# Загрузить

# Или PlatformIO
pio run -t upload

# Или одной командой из корня проекта: веса + сборка
python -m packages firmware --model-dir /полный/путь/к/папке/с/моделью
```

---

## Генерация весов (на ПК)

После обучения модели на сервере:

```bash
# 1. Обучить модель
cd /home/gshjis/Projects/RL_control
poetry run python -m packages train --total-timesteps 3500000

# 2. Экспорт + квантование + сборка прошивки
#    (в папке должен лежать ровно один чекпоинт)
poetry run python -m packages firmware --model-dir /путь/к/папке/с/одной/моделью

# Только квантование, без сборки:
poetry run python tools/export_quantize.py \
  --model checkpoints/ppo/validation/model_3500000 \
  --output arduino/pendulum_control/model_weights.h
```

Скрипт `export_quantize.py`:
- Загружает PPO политику (weights + architecture)
- Применяет INT8 asymmetric quantization per-tensor
- Генерирует `model_weights.h` с массивами в PROGMEM

---

## Цикл управления (500 Гц)

```
Timer1 CTC interrupt (2 ms):
  1. Читаем AS5600 (I2C) → raw_angle (0-4095)
  2. Квантование + шум + дифференцирование + ФНЧ (SensorBlock)
  3. Строим features: [sinθ1, cosθ1, dθ1, dx] (Q7.8 fixed-point)
  4. INT8 inference: Linear(4→32) → Tanh → Linear(32→1)
  5. Деквантование выхода → force (float)
  6. TB6612: PWM duty + DIR
```

---

## Безопасность

- **Watchdog timer** (8 с) — перезагрузка при зависании
- **Лимиты силы** — клиппинг выхода НС
- **Liмиты позиции** — аварийная остановка при |x| > 0.5 м
- **Потеря I2C** — фолбэк в безопасное состояние (force = 0)

---

## Отладка (опционально)

Раскомментировать в `config.h`:
```cpp
#define DEBUG_SERIAL 1
```
Тогда в Serial (115200) будут выводиться:
- Сырые углы энкодеров
- Признаки после SensorBlock
- Выход НС (до и после деквантования)
- PWM duty cycle

---

## Зависимости

- `Wire.h` (I2C) — встроена
- `avr/pgmspace.h` (PROGMEM) — встроена
- `TimerOne` или ручная настройка Timer1 — для 500 Гц прерывания

---

## Лицензия

MIT — как основной проект RL_control.