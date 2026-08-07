#pragma once

/**
 * @file co_sensor.hpp
 * @brief C++ sensor telemetry block: quantization + noise + differentiation
 *        + filtering.
 *
 * Provides the full built-in implementation of the `SensorBlock` previously
 * living in Python (`sensor.py`). The telemetry pipeline is:
 *
 *   1. Quantization of the encoders (angles) and the optical ruler (cart
 *      position).
 *   2. Additive white noise (pre-computed into a pool at construction time and
 *      cycled through on each call to avoid RNG calls in the hot loop).
 *   3. Velocity estimation by a `Differentiator` — it compares the current
 *      noisy position with the previous one and computes the derivative by
 *      finite differences.
 *   4. Smoothing of the noisy states and their derivatives by a `SignalFilter`.
 *
 * Memory management: the noise pool, the differentiator and the filter are
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
 * d(theta1)/dt, d(theta2)/dt)` — the position, the sine/cosine of the
 * pendulum angles, and the time derivatives of the position and the angles
 * (computed by finite differences of the noisy state and smoothed by the
 * filter).
 */
class SensorBlock {
public:
    /**
     * @param cart_resolution      Cart position sensor resolution (m).
     * @param encoder_resolution_1 First pendulum encoder resolution (counts/rev).
     * @param encoder_resolution_2 Second pendulum encoder resolution (counts/rev).
     * @param noise_std_q          Std of noise for coordinates (x, th1, th2).
     * @param noise_std_dq         Std of noise for velocities (dx, dth1, dth2).
     *                             (Reserved; velocity noise arises from the
     *                             differentiation of noisy positions.)
     * @param seed                 RNG seed for the noise pool.
     * @param pool_size            Number of pre-generated noise samples.
     * @param dt                   Sampling period (s).
     * @param differentiator_cutoff_hz Low-pass cutoff for the differentiator (Hz).
     * @param filter_cutoff_hz     Low-pass cutoff for the output filter (Hz).
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
                double filter_cutoff_hz);

    /**
     * @brief Produce a noisy, quantized, differentiated and filtered
     *        measurement from the true state.
     * @param raw_q True generalized coordinates (x, theta1, theta2).
     * @param raw_dq True generalized velocities (x_dot, theta1_dot, theta2_dot).
     * @return Measured vector (x, cos(theta1), sin(theta1), cos(theta2),
     *         sin(theta2), d(x)/dt, d(theta1)/dt, d(theta2)/dt).
     */
    std::vector<double> get_telemetry(const std::vector<double>& raw_q,
                                      const std::vector<double>& raw_dq);

    /** @brief Reset the noise-pool index, the differentiator and the filter. */
    void reset();

private:
    double cart_step_;
    double angle_step_1_;
    double angle_step_2_;
    std::vector<double> std_q_;
    int pool_size_;
    std::vector<std::vector<double>> noise_pool_;
    int noise_index_;
    std::vector<double> meas_;
    Differentiator differentiator_;
    SignalFilter filter_;
};

} // namespace co
