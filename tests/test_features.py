import numpy as np

from packages.simulation.co.datatypes import ControllerConfig, PlantConfig
from packages.simulation.co.features import (
    FeatureConfig,
    build_features,
    build_features_auto,
    get_obs_dim,
)


def test_feature_config():
    config = FeatureConfig(n_links=1, position_controlled=True)
    assert config.dim == 6
    config = FeatureConfig(n_links=1, position_controlled=False)
    assert config.dim == 5
    config = FeatureConfig(n_links=2, position_controlled=True)
    assert config.dim == 9
    config = FeatureConfig(n_links=2, position_controlled=False)
    assert config.dim == 8


def test_build_features_auto_single_no_pos():
    """Single pendulum, no position control: sinθ1, cosθ1, dθ1, dx (4)."""
    plant = PlantConfig(single_pendulum_mode=True)
    controller = ControllerConfig(position_controlled=False)
    telemetry = np.array([1.0, 0.5, 0.5, 0.0, 0.0, 0.2, 0.1, 0.0])
    target = np.array([1.0])
    features = build_features_auto(plant, controller, telemetry, target)
    # [sinθ1, cosθ1, dθ1, dx]
    expected = np.array([0.5, 0.5, 0.1, 0.2])
    assert features.shape == (4,)
    np.testing.assert_allclose(features, expected)


def test_build_features_auto_single_with_pos():
    """Single pendulum, position control: sinθ1, cosθ1, dθ1, dx, x - x_target (5)."""
    plant = PlantConfig(single_pendulum_mode=True)
    controller = ControllerConfig(position_controlled=True)
    telemetry = np.array([1.0, 0.5, 0.5, 0.0, 0.0, 0.2, 0.1, 0.0])
    target = np.array([1.0])
    features = build_features_auto(plant, controller, telemetry, target)
    # [sinθ1, cosθ1, dθ1, dx, x - x_target]
    expected = np.array([0.5, 0.5, 0.1, 0.2, 0.0])
    assert features.shape == (5,)
    np.testing.assert_allclose(features, expected)


def test_build_features_auto_double_no_pos():
    """Double pendulum, no position control: 7 features."""
    plant = PlantConfig(single_pendulum_mode=False)
    controller = ControllerConfig(position_controlled=False)
    telemetry = np.array([1.0, 0.5, 0.5, 0.3, 0.7, 0.2, 0.1, 0.05])
    target = np.array([1.0])
    features = build_features_auto(plant, controller, telemetry, target)
    # [sinθ1, cosθ1, dθ1, dx, sinθ2, cosθ2, dθ2]
    expected = np.array([0.5, 0.5, 0.1, 0.2, 0.7, 0.3, 0.05])
    assert features.shape == (7,)
    np.testing.assert_allclose(features, expected)


def test_build_features_auto_double_with_pos():
    """Double pendulum, position control: 8 features."""
    plant = PlantConfig(single_pendulum_mode=False)
    controller = ControllerConfig(position_controlled=True)
    telemetry = np.array([1.0, 0.5, 0.5, 0.3, 0.7, 0.2, 0.1, 0.05])
    target = np.array([2.0])  # x_target != x
    features = build_features_auto(plant, controller, telemetry, target)
    # [sinθ1, cosθ1, dθ1, dx, sinθ2, cosθ2, dθ2, x - x_target]
    expected = np.array([0.5, 0.5, 0.1, 0.2, 0.7, 0.3, 0.05, -1.0])
    assert features.shape == (8,)
    np.testing.assert_allclose(features, expected)


def test_get_obs_dim():
    assert (
        get_obs_dim(
            PlantConfig(single_pendulum_mode=True),
            ControllerConfig(position_controlled=False),
        )
        == 4
    )
    assert (
        get_obs_dim(
            PlantConfig(single_pendulum_mode=True),
            ControllerConfig(position_controlled=True),
        )
        == 5
    )
    assert (
        get_obs_dim(
            PlantConfig(single_pendulum_mode=False),
            ControllerConfig(position_controlled=False),
        )
        == 7
    )
    assert (
        get_obs_dim(
            PlantConfig(single_pendulum_mode=False),
            ControllerConfig(position_controlled=True),
        )
        == 8
    )


def test_legacy_build_features_compatibility():
    """Legacy FeatureConfig API работает и выдаёт новый формат (без x)."""
    # single, no pos -> 4 features (sin, cos, dth1, dx)
    telemetry = np.array([1.0, 0.5, 0.5, 0.0, 0.0, 0.2, 0.1, 0.0])
    target = np.array([1.0])
    out = build_features(
        telemetry, target, FeatureConfig(n_links=1, position_controlled=False)
    )
    assert out.shape == (4,)
    np.testing.assert_allclose(out, [0.5, 0.5, 0.1, 0.2])

    # single, pos -> 5 features
    out = build_features(
        telemetry, target, FeatureConfig(n_links=1, position_controlled=True)
    )
    assert out.shape == (5,)
    np.testing.assert_allclose(out, [0.5, 0.5, 0.1, 0.2, 0.0])

    # double, no pos -> 7 features
    telemetry2 = np.array([1.0, 0.5, 0.5, 0.3, 0.7, 0.2, 0.1, 0.05])
    out = build_features(
        telemetry2, target, FeatureConfig(n_links=2, position_controlled=False)
    )
    assert out.shape == (7,)
    np.testing.assert_allclose(out, [0.5, 0.5, 0.1, 0.2, 0.7, 0.3, 0.05])

    # double, pos -> 8 features
    out = build_features(
        telemetry2, target, FeatureConfig(n_links=2, position_controlled=True)
    )
    assert out.shape == (8,)
    np.testing.assert_allclose(out, [0.5, 0.5, 0.1, 0.2, 0.7, 0.3, 0.05, 0.0])
