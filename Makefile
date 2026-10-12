# Makefile для проекта RL
# Сборка C++-ядра (pybind11) и запуск обучения/GUI.

PYTHON  := $(shell poetry env info -p)/bin/python
CPP_DIR := packages/simulation/co/cpp
BUILD_DIR := $(CPP_DIR)/build
SO      := packages/simulation/co/co_cpp.so
JOBS    := $(shell nproc)

.PHONY: all build rebuild train gui plot eval profile clean install test lint format typecheck check

all: build

## Установить зависимости окружения
install:
	poetry install --no-root

## Собрать C++-ядро (cmake + make)
build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DPython_EXECUTABLE="$(PYTHON)" \
		-DCMAKE_BUILD_TYPE=Release
	@cd $(BUILD_DIR) && cmake --build . -j "$(JOBS)"
	@echo "C++ core built: $(SO)"

## Полная пересборка C++-ядра (очистка build-директории)
rebuild:
	@rm -rf $(BUILD_DIR)
	@$(MAKE) build

## Запустить обучение PPO (сначала собирает C++-ядро)
train: build
	poetry run python PPO_train.py

## Запустить GUI (сначала собирает C++-ядро)
gui: build
	poetry run python run_gui.py

## Сохранить график cos(theta1) по прогонам политики (headless)
plot: build
	poetry run python -m packages plot

## Оценить политику: метрики прогона + график (headless)
eval: build
	poetry run python -m packages eval

## Профилирование физического ядра
profile: build
	poetry run python -m packages profile

## Прогнать тесты (сначала собирает C++-ядро: часть тестов требует расширения)
test: build
	poetry run pytest

## Проверка стиля
lint:
	poetry run ruff check .
	poetry run ruff format --check .

## Автоисправление форматирования
format:
	poetry run ruff format .
	poetry run ruff check --fix .

## Проверка типов
typecheck:
	poetry run mypy .

## Полная проверка (та же, что в CI)
check: lint typecheck test

## Очистить артефакты сборки
clean:
	@rm -rf $(BUILD_DIR)
	@rm -f $(SO)
	@echo "Cleaned build artifacts"
