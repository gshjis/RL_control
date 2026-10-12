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

- [x] **1. Baseline — воспроизводимость.** Объявить `stable-baselines3`;
      выкинуть призрак-зависимости; удалить вложенный lock; цели
      `test/lint/typecheck/check` в `Makefile`; `conftest.py` с понятной
      ошибкой про `make build`; golden-тесты физики и reward.
      *Прогон отложен: `poetry lock` / `make build` / `make check` требуют
      сети и компиляции, запускать на этой машине нельзя. Сделано кодом.*
- [x] **2. Импорты.** Один poetry-пакет: удалить path-deps и вложенные
      `pyproject.toml`; убрать `packages = [{include="*"}]` из `co`; перевести
      голые импорты на `packages.*`; `__init__.py` на промежуточные уровни.
      *Порядок изменён: сначала удалены мёртвые поддеревья (этап 6.1/6.2),
      затем убраны path-deps — так чинить импорты пришлось в двух файлах.*
- [x] **3. Конфигурация.** Frozen `ExperimentConfig` + именованные стратегии
      (reward/terminate/truncate); общий `presets.py`; убрать дубли
      конфигов и расходящийся `TARGET`.
      *`configs.py` удалён полностью — шим не понадобился, все точки входа
      перешли на `packages.presets`/`packages.experiment`. `PPOConfig`
      переехал в `packages/controllers/ppo_config.py`, чтобы слой
      конфигурации не тянул torch/stable-baselines3 при импорте.*
- [x] **4. Layout наблюдений.** `packages/simulation/co/layout.py`
      (константы + `unpack`/`split_obs`), заменить магические индексы.
      *Публичные свойства `env.plant`, `env.controller_dt`, `env.target_fn`
      и `ObjectOfControl.dt` заведены уже здесь (GUI и среда заходили в
      приватные поля). Заодно починен баг HUD: он показывал `cos/sin`
      телеметрии вместо углов и `cos(theta2)` вместо `dx`.*
- [x] **5. Интерфейсы.** ABC `Controller.act()`; обучение — в `PPOTrainer`;
      публичные `env.plant`/`env.controller_dt`/`dt`; инверсия зависимости
      trainer ← env_factory.
      *`EnvOrchestrator` удалён: тренер получает фабрику сред, а не готовый
      пул. В `PPOController` больше нет ни обучения, ни callback'ов — только
      `act`/`save`/`load`. Заодно починена временная шкала графика
      `plot_creator` при `render_every > 1`.*
- [x] **6. Чистка.** Удалить `pid/`, `custom/`, `reinforce/`, `ddpg/`,
      `loggers/`, пустые файлы и их тесты; переименовать опечатки; свести
      README; почистить `.gitignore`.
      *Удалены поддеревья и пустые файлы, `save_reward_comparison`, пять
      пустых/устаревших README (в том числе gui/README, описывавший
      модули, которых уже нет). README `co` и `co/cpp` переписаны под
      фактическое устройство; `.gitignore` почистили, `experiments/`
      добавлен. Переименование опечаток не понадобилось: все файлы с
      опечатками (`env_orcestrator`, `reset_angel_integral`,
      `check_linerised_position`) удалены вместе с мёртвым кодом.*
- [x] **7. CLI.** `packages/cli.py`; обёртки вместо точек входа; артефакты
      `artifacts/<run_id>/`; общий `rollout()` + `plot_rollout()`.
      *`python -m packages train|gui|plot|eval|profile` (тяжёлые импорты
      отложены внутрь команд). `PPO_train.py`/`run_gui.py`/`plot_creator.py`
      — тонкие обёртки, так что `make train`/`make gui` не изменились.
      Общий прогон вынесен в `packages/rollout.py` — три дублирующих цикла
      (plot_creator, train_and_evaluate, удалённая функция GUI) свернулись
      в одну реализацию. Загрузка чекпоинтов —
      `packages/controllers/ppo/checkpoints.py`.*
- [x] **8. Сиды и датчики.** Seed → `np.random.Generator`; управляемый rng в
      C++ + `set_seed()`; `exact_mode: bool = False`; тест
      воспроизводимости; детерминированные сиды eval.
      *`ExperimentConfig.seed` (42) + `factory(rank)` → `seed+rank` на среду;
      `PendulumEnv(seed=...)` сеет первый reset; `init_q/init_dq` берут
      генератор у среды. В C++: `set_noise_seed()` (фиксированный дефолт,
      не `random_device`), `SensorBlock.set_seed()`. Датчик включён по
      умолчанию: квантование → шум позиций → дифференцирование → шум
      скоростей → ФНЧ; `noise_std_dq` работает (был «reserved»). Пул шума
      2M → 100k (~96 → ~5 МБ на среду). `tests/test_reproducibility.py`.*
- [x] **9. C++.** `make test` зависит от `make build`; golden-тесты
      энергосохранности; честный API сенсора (убрать `(void)noise_std_dq`).
      *`test: build` уже был в Makefile; аналитические проверки физики —
      `tests/test_physics_properties.py` (этап 1); API сенсора честный
      после этапа 8. Сборку прогнать не мог — см. «Что осталось».*
- [x] **10. Качество.** Аннотации типов; поднять ruff N-правила; тесты
      GUI/plot/PPO save-load; `.github/workflows/ci.yml`; `LICENSE` MIT;
      обновить README.
      *Убран `Any` из API GUI; `tests/test_rollout.py` и `tests/test_cli.py`
      (без физики); `LICENSE` (MIT); CI-workflow с `make build` + lint +
      mypy + pytest; README дополнен лицензией и разделом
      «Воспроизводимость». Тесты GUI (pygame headless) и подъём ruff
      N-правил отложены: требуют прогона, который сейчас недоступен.*

## Что не менять

- Численное ядро `compute_ddq` / `rk4_step` / матрицы M, C, G — только под
  golden-тестами.
- Формат чекпоинтов SB3 (`*.zip` + `*_vecnormalize.pkl`): старые модели
  должны грузиться.
- Смысл reward `-|E − U_max|` и условий terminate/truncate — рефакторинг их
  только переносит, не изменяет.

## Что осталось непроверенным

Рефакторинг выполнялся без установки/запуска (ограничение сессии), поэтому
нужен прогон после `poetry install --no-root && make build`:

1. `make build` — C++ ядро (сенсор впервые содержит рабочий конвейер и
   `set_seed`; код правился руками, компиляцию не гонял).
2. `make test` — все тесты, включая новый `tests/test_reproducibility.py`
   (проверит, что сиды реально дают бит-в-бит).
3. `make lint` / `make typecheck` — после переименований и удалений.
4. Тест GUI (pygame headless: логика цикла без окна) и подъём ruff-правил
   N801/N802/N803/N806/N812 — сделать после того, как прогон станет
   доступен.
5. Первый реальный прогон `make train` подтвердит, что честные датчики
   (квантование + шум + дифференцирование) дают обучаемую задачу; при
   необходимости подобрать `differentiator_cutoff_hz`/`filter_cutoff_hz`
   (сейчас 40/50 Гц) и `noise_std_*`.

## Риски

- Решение №3 (включить шум датчиков) меняет цифры обучения: старые чекпоинты
  несопоставимы с новыми. Смена поведения идёт отдельным коммитом в этапе 8.
- Этапы 2 и 3 не объединять в один коммит: переезд имён и переезд конфигов
  должны быть раздельными, иначе бесполезный bisect.
