# C++ / PyBind11 backend

Папка содержит C++ ядро симуляции CO и pybind11-биндинг (`co_bindings.cpp`):

- `co_physics.cpp`/`.hpp` — физика (интегратор RK4, уравнения движения).
- `co_signal.cpp`/`.hpp` — обработка сигналов: `Differentiator` (численное
  дифференцирование + ФНЧ для оценки скорости) и `SignalFilter`
  (сглаживание первого порядка).
- `co_sensor.cpp`/`.hpp` — блок датчиков `SensorBlock` (квантование энкодеров
  + белый шум из предвычисленного пула).

Эти компоненты используются напрямую через pybind11 API (`co_cpp.SensorBlock`,
`co_cpp.Differentiator`, `co_cpp.SignalFilter`) — без Python-обёрток.

## Требования

- **cmake** ≥ 3.20
- **C++17** компилятор (gcc ≥ 9, clang ≥ 10)
- **pybind11** — уже объявлен в зависимостях корневого `pyproject.toml`,
  ставится обычным `poetry install --no-root`

Сборка использует Python из корневого poetry-окружения,
**не требует** создания отдельного `.venv` внутри `packages/simulation/co/`.

## Сборка

Из корня проекта достаточно одной команды:

```bash
make build
```

Разбор того, что эта команда делает, — ниже.

Используйте **внешнее виртуальное окружение из корня проекта** (`.venv`),
не создавайте новое внутри `packages/simulation/co/`.

```bash
# 1. Подготовить build-директорию
cd packages/simulation/co/cpp
rm -rf build
mkdir build && cd build

# 2. Запустить cmake с Python из корневого poetry-окружения
cmake .. \
  -DPython_EXECUTABLE="$(poetry env info -p)/bin/python" \
  -DCMAKE_BUILD_TYPE=Release

# 3. Собрать
cmake --build . -j "$(nproc)"
```

После успешной сборки бинарный модуль появится **сразу** здесь (CMake
настроен на вывод в каталог пакета, копировать вручную не нужно):

```
packages/simulation/co/co_cpp.so
```

## Проверка

```bash
poetry run python -c "
from packages.simulation.co import co_cpp as m
print('C++ backend OK:', m)
print('Functions:', [f for f in dir(m) if not f.startswith('_')])
"
```

Ожидаемый вывод (список отличается по версиям — ищите `SensorBlock`,
`rk4_step`, `update_physics_cpp`):
```
C++ backend OK: <module 'co_cpp' from '.../packages/simulation/co/co_cpp.so'>
Functions: ['Differentiator', 'NoiseForce', 'PlantParams', 'SensorBlock',
           'SignalFilter', 'State3', 'StateDot3',
           'rk4_step', 'set_noise_seed', 'update_physics_cpp']
```

`SensorBlock.set_seed(n)` пересеивает пул шума датчиков, а
`co_cpp.set_noise_seed(n)` — глобальный RNG шума силы; вместе они делают
эпизод воспроизводимым после `env.reset(seed=...)`.

## Многоступенчатое обновление физики

`update_physics_cpp` принимает дополнительный аргумент `n_updates` (по
умолчанию `1`) — целое число подшагов, выполняемых с одной и той же силой
`F_ideal`. На каждом подшаге заново применяются модель двигателя и шум,
поэтому эффект силы корректно накапливается. При `n_updates == 1` поведение
идентично исходному одноступенчатому вызову.

## Быстрая пересборка (если build уже настроен)

```bash
cd packages/simulation/co/cpp/build
cmake --build . -j "$(nproc)"
```

Без очистки `build`-директории — cmake кеш сохраняет `Python_EXECUTABLE`.
