# RL_control — Inverted Pendulum RL Simulator

Симулятор тележки с перевёрнутым маятником (1 или 2 звена) для обучения и развёртывания PPO-политик на микроконтроллерах.

**Что умеет симулятор:**
- Физика на C++ (RK4, dt=0.5мс): тележка + 1/2 звена, трение, инерция двигателя
- Реальные датчики (opt-in): квантование энкодеров 4096 CPR, гауссовский шум, оценка скоростей конечными разностями + ФНЧ 40/50Гц
- Идеальные датчики (по умолчанию): точные углы/скорости из физики — воспроизводимо, быстро
- Gymnasium-совместимая среда: 4/5/7/8-мерное наблюдение (авто по конфигу)
- PPO через Stable-Baselines3: обучение, чекпоинты, детерминированный инференс
- Экспорт в INT8 для Arduino: квантование весов, генерация `.h`.

---

## 4 варианта нейросети (память на Arduino)

| Вариант | Архитектура | Параметров | Flash (INT8) | RAM (активации) | Где используется |
|---------|-------------|------------|--------------|-----------------|------------------|
| **Tiny** | 4 → 16 → 1 | ~100 | 0.1 KB | 32 B | Тест квантования |
| **Small** | 4 → 32 → 32 → 1 | ~1.1K | 1.1 KB | 64 B | Текущий пресет |
| **Medium** | 7 → 64 → 64 → 1 | ~4.5K | 4.5 KB | 128 B | 2 звена |
| **Large** | 8 → 128 → 128 → 1 | ~17K | 17 KB | 256 B | 2 звена + pos control |

---

## Где инференсим

| Платформа | Код | Формат модели |
|-----------|-----|---------------|
| **Python (обучение/оценка)** | `packages/controllers/ppo/ppo.py` → `PPOController.act()` | SB3 `.zip` + `_vecnormalize.pkl` + `_config.json` |
| **Python (GUI)** | `packages/simulation/gui/gui.py` → `PendulumViewer` | то же |
| **Arduino (embedded)** | `tools/export_quantize.py` → `model_weights.h` | C header: `int8_t` веса + scales + `forward_int8()` |

---

## Команды (Make)

```bash
make build       # собрать C++ ядро (cmake + pybind11)
make train       # обучить PPO (1.5M шагов, ~5 мин)
make gui         # визуализация (Pygame), подхватит последний чекпоинт
make plot        # headless график cos(θ₁) за 20 прогона
make eval        # метрики + график для указанного чекпоинта
make check       # ruff + mypy + pytest (63 теста)
```

### Прошивка Arduino

```bash
python -m packages firmware --model-dir /полный/путь/к/папке/с/моделью
```

В папке должен лежать ровно один чекпоинт — `<name>_model.zip` (итоговое
обучение, рядом `_config.json`/`_vecnormalize.pkl`) или `<name>.zip` (снимок
валидации). Команда:

1. квантует веса в `arduino/pendulum_control/model_weights.h`;
2. собирает прошивку через PlatformIO (`pio run`);
3. копирует `firmware_<name>_<env>.hex` в `artifacts/firmware/<name>/`.

Флаги: `--sketch DIR` (другой скетч), `--output-dir DIR` (куда копировать
`.hex`), `--max-force N`, `--hidden-dim N`. Без PlatformIO сборка
завершается с понятной ошибкой (`pip install platformio`).

---

## Содержание (код)

| Что | Файл |
|-----|------|
| **Единый CLI** | [`packages/cli.py`](packages/cli.py) |
| **Конфиг эксперимента** | [`packages/experiment.py`](packages/experiment.py) |
| **Рабочие пресеты** | [`packages/presets.py`](packages/presets.py) |
| **PPO контроллер (инференс)** | [`packages/controllers/ppo/ppo.py`](packages/controllers/ppo/ppo.py) |
| **PPO тренер (обучение)** | [`packages/controllers/ppo/trainer.py`](packages/controllers/ppo/trainer.py) |
| **Gymnasium среда** | [`packages/simulation/env/env.py`](packages/simulation/env/env.py) |
| **Физика C++** | [`packages/simulation/co/cpp/co_physics.cpp`](packages/simulation/co/cpp/co_physics.cpp) |
| **Датчики C++** | [`packages/simulation/co/cpp/co_sensor.cpp`](packages/simulation/co/cpp/co_sensor.cpp) |
| **Признаки (4/5/7/8 dim)** | [`packages/simulation/co/features.py`](packages/simulation/co/features.py) |
| **Конфиги (Plant/Sensor/Controller)** | [`packages/simulation/co/datatypes.py`](packages/simulation/co/datatypes.py) |
| **Экспорт в Arduino** | [`tools/export_quantize.py`](tools/export_quantize.py) |
| **CI (GitHub Actions)** | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |

---

## Лицензия

MIT — см. [LICENSE](LICENSE).