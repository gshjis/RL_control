# CO — Control Object (физика и датчики перевёрнутого маятника)

Низкоуровневый слой симуляции: интегрирование физики тележки с одно- или
двухзвенным маятником (C++/pybind11) и блок измерений с квантованием
энкодеров, шумом и фильтрацией.

## Состав

| Файл | Что делает |
| --- | --- |
| `datatypes.py` | `PlantConfig`, `SensorConfig`, `ControllerConfig`, `NoiseForce`; расчёт длин `L1/L2` и инерций `J1/J2` в `__post_init__` |
| `pendulum.py` | `ObjectOfControl` — Python-обёртка над C++-блоками: `update_physics()`, `get_telemetry()`, `get_clean_state()`, `reset()` |
| `layout.py` | разметка векторов: телеметрия (8), цель (6), наблюдение (14), чистое состояние (6), константы индексов и `unpack`/`split` |
| `cpp/` | pybind11-модуль `co_cpp`: `co_physics` (RK4 + движок двигателя), `co_sensor` (квантование/шум/фильтр), `co_signal` |

## Сборка C++-ядра

Расширение собирается вручную и обязательно для симуляции и обучения:

```bash
make build     # из корня проекта (CMake + pybind11)
```

Готовый `co_cpp.*.so` ложится рядом с `co_physics.cpp`. Без него импорт
`packages.simulation.co` поднимает `ImportError` с подсказкой.

## Форматы данных

**Чистое состояние** (`ObjectOfControl.get_clean_state`, 6 компонент):
`[x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂]` — истинная физика, без шума датчика.

**Телеметрия** (`ObjectOfControl.get_telemetry`, 8 компонент):
`[x, cos θ₁, sin θ₁, cos θ₂, sin θ₂, ẋ, θ̇₁, θ̇₂]` — проходит квантование
энкодеров, добавление шума и фильтрацию в `SensorBlock`.

**Наблюдение среды** (14 компонент) = телеметрия ‖ цель (6 компонент).

Все индексы и подписи канализованы в `layout.py` — обращаться к
наблюдениям через `IDX_*`/`unpack_telemetry()`, а не числовыми литералами.

## Конфигурация

```python
from packages.simulation.co import PlantConfig, SensorConfig, ControllerConfig

plant = PlantConfig(m1=0.05, l1=0.2, dt=0.001, p_upright=0.05)
sensor = SensorConfig(encoder_resolution_1=4096, noise_std_q=(0.0005, 0.002, 0.002))
controller = ControllerConfig(dt=0.01, max_force=24.0)
```

`PlantConfig.init_q()` / `init_dq()` возвращает начальное состояние,
смешанное по вероятности `p_upright` между перевёрнутым и опущенным
маятником. `to_dict()` каждого конфига детерминирован и сериализуем.

Замечание: сейчас C++-блок измерений молча игнорирует шум и квантование —
в `co_sensor.cpp` оставлен режим точных показаний, а `noise_std_q`/
`noise_std_dq`/разрешение энкодеров на результат пока не влияют. Этап 8
плана рефакторинга (см. `docs/refactoring-plan.md`) возвращает честные
датчики и делает режим явным флагом.
