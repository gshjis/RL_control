# C++ / PyBind11 backend

Папка содержит C++ ядро симуляции CO и pybind11-биндинг (`co_bindings.cpp`):

- `co_physics.cpp`/`.hpp` — физика (интегратор RK4, уравнения движения).
- `co_signal.cpp`/`.hpp` — обработка сигналов: `Differentiator` (численное
  дифференцирование + ФНЧ для оценки скорости) и `SignalFilter`
  (сглаживание первого порядка). Полные реализации, ранее жившие в
  `signal_processing.py`.
- `co_sensor.cpp`/`.hpp` — блок датчиков `SensorBlock` (квантование энкодеров
  + белый шум из предвычисленного пула). Полная реализация, ранее жившая
  в `sensor.py`.

Эти компоненты используются напрямую через pybind11 API (`co_cpp.SensorBlock`,
`co_cpp.Differentiator`, `co_cpp.SignalFilter`) — без Python-обёрток.

## Требования

- **cmake** ≥ 3.20
- **C++17** компилятор (gcc ≥ 9, clang ≥ 10)
- **pybind11** — устанавливается через poetry корневого проекта:
  ```bash
  poetry add pybind11
  ```

Сборка использует Python из корневого poetry-окружения,
**не требует** создания отдельного `.venv` внутри `packages/simulation/co/`.

## Сборка

Используйте **внешнее виртуальное окружение из корня проекта** (`.venv`),
не создавайте новое внутри `packages/simulation/co/`. Из **корня проекта**
(`/home/gshjis/Python_projects/RL`):

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

Ожидаемый вывод:
```
C++ backend OK: <module 'co_cpp' from '.../packages/simulation/co/co_cpp.so'>
Functions: ['Differentiator', 'NoiseForce', 'PlantParams', 'SensorBlock',
           'SignalFilter', 'State3', 'StateDot3',
           'rk4_step', 'update_physics_cpp']
```

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
