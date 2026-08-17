import numpy as np

from packages.simulation.co.datatypes import (
    ControllerConfig,
    NoiseForce,
    PlantConfig,
    SensorConfig,
)


def test_noise_force_defaults() -> None:
    noise = NoiseForce()

    assert noise.mean == 0.0
    assert noise.std == 0.0


def test_plant_config_derives_geometry_and_inertia() -> None:
    config = PlantConfig(m1=0.6, l1=0.4, m2=0.2, l2=0.8)

    assert config.L1 == 0.2
    assert config.L2 == 0.4
    assert np.isclose(config.J1, 0.6 * 0.4**2 / 12.0)
    assert np.isclose(config.J2, 0.2 * 0.8**2 / 12.0)


def test_plant_initial_position_uses_configured_mixture() -> None:
    config = PlantConfig(init_q_stats=np.array([[0.0, np.pi, 0.0], [0.0, 0.0, 0.0]]))

    samples = [config.init_q()[1] for _ in range(20)]

    assert set(samples).issubset({0.0, np.pi})


def test_plant_copy_preserves_values_and_is_independent() -> None:
    config = PlantConfig()
    copied = config.copy()

    copied.init_q_stats[0, 0] = 42.0

    assert copied.M == config.M
    assert copied.init_q_stats[0, 0] != config.init_q_stats[0, 0]


def test_config_to_dict_contains_serializable_fields() -> None:
    plant = PlantConfig()
    sensor = SensorConfig()
    controller = ControllerConfig()

    assert plant.to_dict()["M"] == plant.M
    assert sensor.to_dict()["encoder_resolution_1"] == sensor.encoder_resolution_1
    assert controller.to_dict()["max_force"] == controller.max_force
