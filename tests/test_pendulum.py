import numpy as np

from packages.simulation.co import ObjectOfControl, PlantConfig, SensorConfig


def make_plant() -> ObjectOfControl:
    plant = PlantConfig(
        init_q_stats=np.zeros((2, 3)),
        init_dq_stats=np.zeros((2, 3)),
        dt=0.001,
        mean_f=0.0,
        std_f=0.0,
    )
    sensor = SensorConfig(
        noise_std_q=(0.0, 0.0, 0.0),
        noise_std_dq=(0.0, 0.0, 0.0),
        noise_pool_size=100,
    )
    return ObjectOfControl(plant, sensor)


def test_object_of_control_state_shapes() -> None:
    plant = make_plant()

    assert plant.q.shape == (3,)
    assert plant.dq.shape == (3,)
    assert plant.get_clean_state().shape == (6,)
    assert plant.get_telemetry().shape == (8,)


def test_object_of_control_updates_and_resets() -> None:
    plant = make_plant()

    plant.update_physics(F_ideal=1.0, n_updates=1)
    changed = plant.get_clean_state()
    assert np.all(np.isfinite(changed))

    plant.reset()
    reset_state = plant.get_clean_state()
    assert np.all(np.isfinite(reset_state))
    assert reset_state[1] in (0.0, np.pi)
    assert np.allclose(reset_state[[0, 2, 3, 4, 5]], 0.0)
