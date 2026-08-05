#pragma once

/**
 * @file co_sensor.hpp
 * @brief C++ sensor telemetry block: encoder quantization + white noise.
 *
 * Provides the full built-in implementation of the `SensorBlock` previously
 * living in Python (`sensor.py`). It models:
 *   - Quantization of the encoders (angles) and the optical ruler (cart
 *     position).
 *   - White noise of a given intensity, pre-computed into a pool at
 *     construction time and cycled through on each call (avoids RNG calls in
 *     the hot loop).
 *
 * Memory management: the noise pool is owned by the instance (RAII) and
 * allocated once in the constructor. The measurement buffer is reused across
 * calls to avoid per-call allocations.
 */

#include <vector>

namespace co {

/**
 * @brief Simulates sensor telemetry: quantization + additive white noise.
 *
 * Mirrors the Python `SensorBlock` in `sensor.py`. The output vector has the
 * layout `(x, theta1, theta2, x_dot, theta1_dot, theta2_dot)`.
 */
class SensorBlock {
public:
    /**
     * @param cart_resolution     Cart position sensor resolution (m).
     * @param encoder_resolution_1 First pendulum encoder resolution (counts/rev).
     * @param encoder_resolution_2 Second pendulum encoder resolution (counts/rev).
     * @param noise_std_q          Std of noise for coordinates (x, th1, th2).
     * @param noise_std_dq         Std of noise for velocities (dx, dth1, dth2).
     * @param seed                 RNG seed for the noise pool.
     * @param pool_size            Number of pre-generated noise samples.
     */
    SensorBlock(double cart_resolution,
                int encoder_resolution_1,
                int encoder_resolution_2,
                const std::vector<double>& noise_std_q,
                const std::vector<double>& noise_std_dq,
                int seed,
                int pool_size);

    /**
     * @brief Produce a noisy, quantized measurement from the true state.
     * @param raw_q True generalized coordinates (x, theta1, theta2).
     * @param raw_dq True generalized velocities (x_dot, theta1_dot, theta2_dot).
     * @return Measured vector (x, theta1, theta2, x_dot, theta1_dot, theta2_dot).
     */
    std::vector<double> get_telemetry(const std::vector<double>& raw_q,
                                      const std::vector<double>& raw_dq);

    /** @brief Reset the cyclic noise-pool index. */
    void reset();

private:
    double cart_step_;
    double angle_step_1_;
    double angle_step_2_;
    std::vector<double> std_;
    int pool_size_;
    std::vector<std::vector<double>> noise_pool_;
    int noise_index_;
    std::vector<double> meas_;
};

} // namespace co
