import numpy as np
from packages.simulation.co.features import FeatureConfig, build_features

def test_feature_config():
    config = FeatureConfig(n_links=1, position_controlled=True)
    assert config.dim == 6
    config = FeatureConfig(n_links=1, position_controlled=False)
    assert config.dim == 5
    config = FeatureConfig(n_links=2, position_controlled=True)
    assert config.dim == 9
    config = FeatureConfig(n_links=2, position_controlled=False)
    assert config.dim == 8


def test_build_features_single_link():
    telemetry = np.array([1, 0.5, 0.5, 0, 0, 0.2, 0.1, 0])
    target = np.array([1]))
    expected_output = np.array([0.5, 0.5, 0.1, 1, 0.2, 1 - 1])
    assert np.array_equal(build_features(telemetry, target, FeatureConfig(n_links=1, position_controlled=True)), expected_output)


def test_build_features_two_links():
    telemetry = np.array([1, 0.5, 0.5, 0.3, 0.7, 0.2, 0.1, 0.05])
    target = np.array([1])
    expected_output = np.array([0.5, 0.5, 0.1, 1, 0.2, 0.3, 0.7, 0.05])
    assert np.array_equal(build_features(telemetry, target, FeatureConfig(n_links=2, position_controlled=True)), expected_output)
