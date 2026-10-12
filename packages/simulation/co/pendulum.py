import numpy as np

try:
    import packages.simulation.co.co_cpp as _co_cpp
except ImportError as exc:  # pragma: no cover - depends on the build state
    raise ImportError(
        "C++ extension 'packages.simulation.co.co_cpp' is not available. "
        "Build it with 'make build' (CMake + pybind11) before using the "
        "simulation."
    ) from exc

from packages.simulation.co.datatypes import PlantConfig, SensorConfig


class ObjectOfControl:
    """Объединяет физическую модель и блок измерения состояния."""

    def __init__(self, plant_config: PlantConfig, sensor_config: SensorConfig) -> None:
        """Принимает конфигурации растения и датчика, создаёт объект модели."""
        self.__plant_config: PlantConfig = plant_config
        self._M: float = plant_config.M
        self._m1: float = plant_config.m1
        self._m2: float = plant_config.m2
        self._l1: float = plant_config.l1
        self._l2: float = plant_config.l2
        self._L1: float = plant_config.L1
        self._L2: float = plant_config.L2
        self._J1: float = plant_config.J1
        self._J2: float = plant_config.J2
        self._g: float = plant_config.g
        self._b_c: float = plant_config.b_c
        self._b_1: float = plant_config.b_1
        self._b_2: float = plant_config.b_2
        self._single_mode: bool = plant_config.single_pendulum_mode
        self._q_init: np.ndarray = plant_config.init_q()
        self._dq_init: np.ndarray = plant_config.init_dq()
        self._q: np.ndarray = self._q_init.copy()
        self._dq: np.ndarray = self._dq_init.copy()
        self._dt: float = plant_config.dt
        self._motor_tau: float = float(plant_config.motor_time_constant)
        self._motor_force: float = 0.0
        self._mean_f: float = plant_config.mean_f
        self._std_f: float = plant_config.std_f

        self._cpp_noise = _co_cpp.NoiseForce(self._mean_f, self._std_f)
        self._cpp_params = _co_cpp.PlantParams()
        self._cpp_q = _co_cpp.State3()
        self._cpp_dq = _co_cpp.StateDot3()
        self._cpp_params.M = self._M
        self._cpp_params.m1 = self._m1
        self._cpp_params.m2 = self._m2
        self._cpp_params.l1 = self._l1
        self._cpp_params.l2 = self._l2
        self._cpp_params.L1 = self._L1
        self._cpp_params.L2 = self._L2
        self._cpp_params.J1 = self._J1
        self._cpp_params.J2 = self._J2
        self._cpp_params.g = self._g
        self._cpp_params.b_c = self._b_c
        self._cpp_params.b_1 = self._b_1
        self._cpp_params.b_2 = self._b_2
        self._cpp_params.motor_tau = self._motor_tau

        self._cpp_sensor = _co_cpp.SensorBlock(
            float(sensor_config.cart_sensor_resolution),
            int(sensor_config.encoder_resolution_1),
            int(sensor_config.encoder_resolution_2),
            list(sensor_config.noise_std_q),
            list(sensor_config.noise_std_dq),
            int(sensor_config.seed if sensor_config.seed is not None else 0),
            int(sensor_config.noise_pool_size),
            self._dt,
            float(sensor_config.differentiator_cutoff_hz or 0.0),
            float(sensor_config.filter_cutoff_hz),
            bool(sensor_config.exact_mode),
        )

    @property
    def q(self) -> np.ndarray:
        """Вектор обобщённых координат ``[x, θ₁, θ₂]``."""
        return self._q.copy()

    @property
    def dq(self) -> np.ndarray:
        """Вектор обобщённых скоростей ``[ẋ, θ̇₁, θ̇₂]``."""
        return self._dq.copy()

    @property
    def dt(self) -> float:
        """Возвращает шаг интегрирования физики, с."""
        return self._dt

    @property
    def m1(self) -> float:
        """Возвращает массу первого стержня, кг."""
        return self._m1

    @property
    def m2(self) -> float:
        """Возвращает массу второго стержня, кг."""
        return self._m2

    @property
    def l1(self) -> float:
        """Возвращает полную длину первого стержня, м."""
        return self._l1

    @property
    def l2(self) -> float:
        """Возвращает полную длину второго стержня, м."""
        return self._l2

    @property
    def L1(self) -> float:
        """Возвращает расстояние ось → центр масс первого стержня, м."""
        return self._L1

    @property
    def L2(self) -> float:
        """Возвращает расстояние ось → центр масс второго стержня, м."""
        return self._L2

    @property
    def motor_force(self) -> float:
        """Возвращает текущую силу на выходе двигателя (с учётом инерции), Н."""
        return self._motor_force

    def update_physics(self, F_ideal: float, n_updates: int) -> None:
        """Принимает силу и число подшагов, обновляет состояние модели."""
        self._q, self._dq, self._motor_force = _co_cpp.update_physics_cpp(
            self._q,
            self._dq,
            float(F_ideal),
            self._mean_f,
            self._std_f,
            self._dt,
            self._cpp_params,
            self._single_mode,
            self._motor_force,
            int(n_updates),
        )

    def get_telemetry(self) -> np.ndarray:
        """Возвращает состояние после квантования, шума и фильтрации датчика."""
        return np.asarray(
            self._cpp_sensor.get_telemetry(self._q, self._dq), dtype=np.float64
        )

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Сбрасывает координаты, скорости, силу двигателя и датчики.

        Принимает опциональный генератор: среда передаёт сюда
        ``self.np_random``, чтобы начальное состояние эпизода зависело
        от ``reset(seed=...)``.
        """
        self._q = self.__plant_config.init_q(rng)
        self._dq = self.__plant_config.init_dq(rng)
        self._motor_force = 0.0
        self._cpp_sensor.reset()

    def set_seed(self, seed: int) -> None:
        """Пересеивает шум силы и шумовой пул датчиков."""
        _co_cpp.set_noise_seed(int(seed))
        self._cpp_sensor.set_seed(int(seed))

    def get_clean_state(self) -> np.ndarray:
        """Возвращает объединённый вектор истинных координат и скоростей."""
        return np.concatenate((self._q, self._dq))
