# Makefile для проекта RL
# Сборка C++-ядра (pybind11) и запуск обучения/GUI.

PYTHON  := $(shell poetry env info -p)/bin/python
CPP_DIR := packages/simulation/co/cpp
BUILD_DIR := $(CPP_DIR)/build
SO      := packages/simulation/co/co_cpp.so
JOBS    := $(shell nproc)

.PHONY: all build rebuild train gui clean

all: build

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

## Очистить артефакты сборки
clean:
	@rm -rf $(BUILD_DIR)
	@rm -f $(SO)
	@echo "Cleaned build artifacts"
