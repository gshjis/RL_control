#pragma once

/**
 * @file co_sensor.hpp
 * @brief C++ sensor telemetry block.
 *
 * Implements the measurement pipeline replacing the old Python `sensor.py`:
 *
 *   1. Quantization of the encoders (angles) and the optical ruler (cart
 *      position).
 *   2. Additive white noise for the coordinates, pre-computed into a pool at
 *      construction time and cycled through on each call to avoid RNG calls
 *      in the hot loop.
 *   3. Velocity estimation by a `Differentiator` — finite differences of the
 *      noisy positions, low-pass filtered inside the differentiator.
 *   4. Additive white noise for the velocities (second pool) and smoothing of
 *      all six channels by a `SignalFilter`.
 *
 * An explicit `exact_mode` switch keeps the legacy noise-free telemetry: the
 * true coordinates and the exact velocities from the physics backend are
 * returned untouched. It defaults to `false` (honest sensors); set it to
 * `true` only when you deliberately want a perfect-observer experiment.
 *
 * Memory management: the noise pools, the differentiator and the filter are
 * owned by the instance (RAII). The measurement buffer is reused across calls
 * to avoid per-call allocations.
 */

#include <vector>

#include "co_signal.hpp"

namespace co {

/**
 * @brief Simulates sensor telemetry: quantization + noise + differentiation
 *        + filtering.
 *
 * The output vector has the layout
 * `(x, cos(theta1), sin(theta1), cos(theta2), sin(theta2), d(x)/dt,
 * d(theta1)/dt, d(theta2)/dt)`.
 */
class SensorBlock {
public:
    /**
     * @param cart_resolution      Cart position sensor resolution (m).
     * @param encoder_resolution_1 First pendulum encoder resolution (counts/rev).
     * @param encoder_resolution_2 Second pendulum encoder resolution (counts/rev).
     * @param noise_std_q          Std of noise for coordinates (x, th1, th2).
     * @param noise_std_dq         Std of noise for velocities (dx, dth1, dth2).
     * @param seed                 RNG seed for the noise pools.
     * @param pool_size            Number of pre-generated noise samples.
     * @param dt                   Sampling period (s).
     * @param differentiator_cutoff_hz Low-pass cutoff for the differentiator (Hz).
     * @param filter_cutoff_hz     Low-pass cutoff for the output filter (Hz).
     * @param exact_mode           True — noise-free exact telemetry.
     */
    SensorBlock(double cart_resolution,
                int encoder_resolution_1,
                int encoder_resolution_2,
                const std::vector<double>& noise_std_q,
                const std::vector<double>& noise_std_dq,
                int seed,
                int pool_size,
                double dt,
                double differentiator_cutoff_hz,
                double filter_cutoff_hz,
                bool exact_mode = false);

    /**
     * @brief Produce telemetry from the true state.
     * @param raw_q True generalized coordinates (x, theta1, theta2).
     * @param raw_dq True generalized velocities (x_dot, theta1_dot, theta2_dot).
     * @return Measured vector (x, cos(theta1), sin(theta1), cos(theta2),
     *         sin(theta2), dx, dth1, dth2).
     */
    std::vector<double> get_telemetry(const std::vector<double>& raw_q,
                                      const std::vector<double>& raw_dq);

    /** @brief Reset the noise-pool index, the differentiator and the filter. */
    void reset();

    /** @brief Regenerate the noise pools from a new seed. */
    void set_seed(int seed);

private:
    /** @brief Fill both noise pools from `seed` (zeros in exact_mode). */
    void build_noise_pool(int seed);

    double cart_step_;
    double angle_step_1_;
    double angle_step_2_;
    std::vector<double> std_q_;
    std::vector<double> std_dq_;
    int pool_size_;
    std::vector<std::vector<double>> noise_pool_;
    std::vector<std::vector<double>> noise_pool_dq_;
    int noise_index_;
    std::vector<double> meas_;
    double dt_;
    double differentiator_cutoff_hz_;
    double filter_cutoff_hz_;
    bool exact_mode_;
    Differentiator differentiator_;
    SignalFilter filter_;
};

} // namespace co
