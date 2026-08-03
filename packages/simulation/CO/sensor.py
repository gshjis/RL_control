import numpy as np
from numpy.typing import NDArray
from packages.simulation.CO.datatypes import SensorConfig


class SensorBlock:
    """
    Блок получения телеметрии с датчиков.

    Моделирует:
    - **Квантование** энкодеров (углы) и оптической линейки (позиция тележки)
    - **Белый шум** заданной интенсивности, предвычисленный в пул

    Parameters
    ----------
    config : SensorConfig
        Конфигурация датчиков (разрядность, шаг квантования, СКО шума, seed).

    Notes
    -----
    Шум генерируется один раз при создании блока (пул 2 млн значений)
    и циклически выбирается на каждом такте. Это ускоряет симуляцию
    за счёт исключения вызовов RNG из горячего цикла.

    Optimization potential:
        - ``pool_size = 2_000_000`` захардкожен; при ``max_steps > pool_size``
          шум начнёт повторяться (циклический индекс). Хорошо бы вынести
          в ``SensorConfig``.
        - ``get_telemetry`` модифицирует внутренний буфер ``_meas``,
          затем прибавляет шум — это **одна** аллокация на такт (оптимально).
        - ``np.rint(q[0] / cs) * cs`` для квантования быстрее явного
          округления ``round()`` благодаря векторизации.
    """

    def __init__(self, config: SensorConfig) -> None:
        self._cart_step: float = config.cart_sensor_resolution
        self._angle_step_1: float = 2.0 * np.pi / config.encoder_resolution_1
        self._angle_step_2: float = 2.0 * np.pi / config.encoder_resolution_2

        self._rng = np.random.default_rng(config.seed)

        noise_std_q: NDArray[np.float64] = np.asarray(config.noise_std_q, dtype=np.float64)
        noise_std_dq: NDArray[np.float64] = np.asarray(config.noise_std_dq, dtype=np.float64)
        self._std: NDArray[np.float64] = np.concat([noise_std_q, noise_std_dq])

        self._pool_size: int = int(config.noise_pool_size)
        self._noise_pool: NDArray[np.float64] = self._rng.normal(
            0.0, self._std,
            size=(self._pool_size, 6)
        )
        self._noise_index: int = 0

        # Буфер для результата (переиспользуется)
        self._meas: NDArray[np.float64] = np.empty(6, dtype=np.float64)

    def get_telemetry(self, raw_q: np.ndarray, raw_dq: np.ndarray) -> np.ndarray:
        """
        Получить зашумлённые и квантованные измерения.

        Выполняет:
        1. Квантование координат ``(x, θ₁, θ₂)`` согласно шагу энкодеров
        2. Добавление предвычисленного белого шума

        Parameters
        ----------
        raw_q : np.ndarray
            Истинные обобщённые координаты ``(x, θ₁, θ₂)``.
        raw_dq : np.ndarray
            Истинные обобщённые скорости ``(ẋ, θ̇₁, θ̇₂)``.

        Returns
        -------
        np.ndarray
            Измеренный вектор ``(x, θ₁, θ₂, ẋ, θ̇₁, θ̇₂)``
            с квантованием и шумом (формат (6,)).

        Examples
        --------
        >>> cfg = SensorConfig(seed=42)
        >>> sensor = SensorBlock(cfg)
        >>> q = np.array([0.0, np.pi, 0.0])
        >>> dq = np.array([0.0, 0.0, 0.0])
        >>> meas = sensor.get_telemetry(q, dq)
        >>> meas.shape
        (6,)
        """
        # ── Квантование ────────────────────────────────────────────────
        cs = self._cart_step
        a1 = self._angle_step_1
        a2 = self._angle_step_2

        meas = self._meas
        meas[0] = np.rint(raw_q[0] / cs) * cs
        meas[1] = np.rint(raw_q[1] / a1) * a1
        meas[2] = np.rint(raw_q[2] / a2) * a2
        meas[3] = raw_dq[0]
        meas[4] = raw_dq[1]
        meas[5] = raw_dq[2]

        # ── Шум из пула ────────────────────────────────────────────────
        noise = self._noise_pool[self._noise_index]
        self._noise_index += 1
        if self._noise_index >= self._pool_size:
            self._noise_index = 0

        meas += noise
        return meas
