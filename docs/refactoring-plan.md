# План рефакторинга RL_control

Статус: выполняется в ветке `refactor/consolidation`.
Базовая точка: тег `refactor-baseline` (состояние `main` на момент начала).

Каждый этап отмечается `[x]` по мере завершения. Между этапами — зелёные
`make check` (`ruff check` + `ruff format --check` + `mypy` + `pytest`).

## Принятые решения

| # | Вопрос | Решение |
|---|--------|---------|
| 1 | Пространство имён | Оставить `packages.*`, без rename. Убрать path-deps и голые импорты. |
| 2 | Мёртвые контроллеры | Удалить всё, кроме PPO: `pid/`, `custom/`, `reinforce/`, `ddpg/`, `loggers/`, `co/controller.py`. |
| 3 | Датчики | Включить честный шум/квантование; `exact_mode: bool = False` в `SensorConfig`. |
| 4 | Скрипты | Единый CLI `python -m packages train\|gui\|eval\|plot\|profile` + тонкие обёртки `PPO_train.py`/`run_gui.py`. |
| 5 | CI / лицензия | CI — финальным этапом; лицензия MIT. |

## Карта проблем (снимок на `refactor-baseline`)

### Воспроизводимость

- `stable-baselines3` не объявлен ни в корневом `pyproject.toml`, ни в
  `packages/controllers/ppo/pyproject.toml` (`dependencies = []`), отсутствует в
  `poetry.lock` — при этом `ppo.py` его импортирует. `poetry install` не даёт
  рабочего окружения, `pytest` падает на сборе.
- Голые импорты, работающие только за счёт path-deps:
  `packages/controllers/ppo/ppo.py` (`from base import Controller`),
  `packages/simulation/env/env_orcestrator.py` (`from datatypes import ...`),
  `packages/controllers/pid/pid.py`, `packages/controllers/reinforce/reinforce.py`,
  `packages/controllers/custom/swing_up_block.py`. Остальной код использует
  `packages.*` → один файл грузится под двумя именами, классы становятся
  разными объектами.
- `packages/simulation/co/pyproject.toml` содержит `packages = [{ include = "*",
  from = "." }]` → в `site-packages` попадают голые модули `datatypes`,
  `pendulum`, `cpp` (коллизии имён).
- 11 вложенных poetry-пакетов + второй lock-файл
  `packages/simulation/co/poetry.lock`.
- Шум/квантование датчиков молча отключены: `co_sensor.cpp` —
  «Temporary exact-sensor mode», конструктор делает `(void)noise_std_q` и т.д.
  Все `noise_std_*`, `encoder_resolution_*`, `filter_cutoff_hz` из конфигов не
  влияют ни на что.
- Сидирование отсутствует: `PendulumEnv.reset(seed=...)` не пробрасывает seed,
  `PlantConfig.init_q/init_dq` используют глобальный `np.random`, C++
  `sample_noise_force` использует `std::random_device` (неуправляемо).
- Зависимости-призраки, не использованные ни в одном `.py`: `numba`, `scipy`,
  `scikit-optimize`, `statsmodels`, `tensorboard`, `mlflow`, `tqdm`. Плюс
  PyPI-пакет `ffmpeg` (это не бинарь; `gui.py` вызывает бинарь через
  `subprocess`).
- Захардкоженный артефакт `checkpoints/ppo/validation/model_1900000` в
  `run_gui.py` — файла в git нет.

### Модульность

- Точки входа импортируют точку входа: `run_gui.py`, `plot_creator.py`,
  `train_and_evaluate_4_agents.py` → `from PPO_train import reward_f, ...`
  (побочный эффект на уровне модуля-скрипта).
- Тройное дублирование конфигов: `configs.py`, `run_gui.py`,
  `plot_creator.py`, профиль в `profiling/profiling.py`. `TARGET` в `run_gui.py`
  расходится с `target()` из `configs.py`.
- Магические индексы наблюдения (`state[1]`, `state[6]`, `COS_INDEX = 1`) в
  пяти+ местах; layout описан только комментарием в `co_sensor.cpp`.
- Нарушение инкапсуляции: `env.py` → `self._plant._dt`; `gui.py` →
  `env._plant._l1`, `env._target`, `env._controller_dt`; `ppo.py` →
  `orchestrator._env_hub`, `orchestrator._make_env`.
- Интерфейс контроллера разъехался: `action()` в базе, `get_control()` в
  pid/custom, `get_action()` в reinforce, у всех разные `train()`.
- Три почти идентичных куска построения графика cos(θ1):
  `plot_creator.simulate_and_save_cos_plot`,
  `train_and_evaluate_4_agents.eval_agent_cos_and_reward`,
  `run_gui.save_reward_comparison`.

### Лишнее

- Пустые файлы: `packages/simulation/co/controller.py`, `custom/custom.py`,
  `ddpg/ddpg.py`, `ddpg/__init__.py`.
- Мёртвые поддеревья (вне своих тестов не используются нигде): `reinforce/`
  (сплошные `NotImplementedError` + `print(12)`), `ddpg/`, `pid/optimizers.py`
  (заглушки), `loggers/`, `custom/`, `pid/`.
- `pid.py` — печать нецензурного текста внутри публичного метода `train()`.
- `run_gui.py` содержит `save_reward_comparison()` (не относится к GUI) и
  `matplotlib.use("Agg")` в GUI-скрипте.
- 11 дублирующих `README.md` по пакетам.

### Гигиена

- `.gitignore` дублирует `.venv`; нет CI (`.github/` отсутствует), нет
  лицензии, `description = ""`.
- `Makefile` без целей `test`/`lint`/`typecheck`.
- Опечатки в именах: `env_orcestrator`, `reset_angel_integral`,
  `check_linerised_position`, `BEAST`.
- Нет `__init__.py` на уровнях `packages/`, `packages/simulation/`,
  `packages/controllers/` (namespace packages; mypy держится на
  `explicit_package_bases`).

## Этапы

- [ ] **1. Baseline — воспроизводимость.** Объявить `stable-baselines3`;
      выкинуть призрак-зависимости; удалить вложенный lock; цели
      `test/lint/typecheck/check` в `Makefile`; `conftest.py` с понятной
      ошибкой про `make build`; golden-тесты физики и reward.
- [ ] **2. Импорты.** Один poetry-пакет: удалить path-deps и вложенные
      `pyproject.toml`; убрать `packages = [{include="*"}]` из `co`; перевести
      голые импорты на `packages.*`; `__init__.py` на промежуточные уровни.
- [ ] **3. Конфигурация.** Frozen `ExperimentConfig` + именованные стратегии
      (reward/terminate/truncate); общий `presets.py`; убрать дубли
      конфигов и расходящийся `TARGET`.
- [ ] **4. Layout наблюдений.** `packages/simulation/co/layout.py`
      (константы + `unpack`/`split_obs`), заменить магические индексы.
- [ ] **5. Интерфейсы.** ABC `Controller.act()`; обучение — в `PPOTrainer`;
      публичные `env.plant`/`env.controller_dt`/`dt`; инверсия зависимости
      trainer ← env_factory.
- [ ] **6. Чистка.** Удалить `pid/`, `custom/`, `reinforce/`, `ddpg/`,
      `loggers/`, пустые файлы и их тесты; переименовать опечатки; свести
      README; почистить `.gitignore`.
- [ ] **7. CLI.** `packages/cli.py`; обёртки вместо точек входа; артефакты
      `artifacts/<run_id>/`; общий `rollout()` + `plot_rollout()`.
- [ ] **8. Сиды и датчики.** Seed → `np.random.Generator`; управляемый rng в
      C++ + `set_seed()`; `exact_mode: bool = False`; тест
      воспроизводимости; детерминированные сиды eval.
- [ ] **9. C++.** `make test` зависит от `make build`; golden-тесты
      энергосохранности; честный API сенсора (убрать `(void)noise_std_dq`).
- [ ] **10. Качество.** Аннотации типов; поднять ruff N-правила; тесты
      GUI/plot/PPO save-load; `.github/workflows/ci.yml`; `LICENSE` MIT;
      обновить README.

## Что не менять

- Численное ядро `compute_ddq` / `rk4_step` / матрицы M, C, G — только под
  golden-тестами.
- Формат чекпоинтов SB3 (`*.zip` + `*_vecnormalize.pkl`): старые модели
  должны грузиться.
- Смысл reward `-|E − U_max|` и условий terminate/truncate — рефакторинг их
  только переносит, не изменяет.

## Риски

- Решение №3 (включить шум датчиков) меняет цифры обучения: старые чекпоинты
  несопоставимы с новыми. Смена поведения идёт отдельным коммитом в этапе 8.
- Этапы 2 и 3 не объединять в один коммит: переезд имён и переезд конфигов
  должны быть раздельными, иначе бесполезный bisect.
