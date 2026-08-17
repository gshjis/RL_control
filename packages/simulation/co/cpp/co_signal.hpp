#pragma once

/**
 * @file co_signal.hpp
 * @brief C++ signal-processing primitives: numerical differentiation and
 *        first-order low-pass filtering (smoothing).
 *
 * These classes provide the full built-in implementation of the components
 * previously living in Python (`signal_processing.py`). They are exposed to
 * Python through pybind11 in `co_bindings.cpp` and are used to estimate
 * velocities from position measurements and to smooth noisy telemetry.
 *
 * Memory management: all internal state is owned by the class instances
 * (RAII). Buffers are `std::vector<double>` and are resized lazily to match
 * the dimensionality of the processed signal, so the same instance can be
 * reused across calls without per-call allocations.
 */

#include <vector>

namespace co {

/**
 * @brief Numerically differentiates a signal and low-pass filters the
 *        resulting velocity estimate (first-order IIR / exponential smoothing).
 *
 * Mirrors the Python `Differentiator` in `signal_processing.py`:
 *   raw_vel = (positions - prev_positions) / dt
 *   filtered = (1 - alpha) * filtered + alpha * raw_vel
 *
 * On the first call it returns a zero vector and only stores the reference
 * positions. `alpha` is derived from the cutoff frequency:
 *   tau = 1 / (2*pi*cutoff_hz),  alpha = dt / (tau + dt)
 * If `cutoff_hz <= 0`, `alpha = 1` (no filtering, pure differentiation).
 */
class Differentiator {
public:
    /**
     * @param dt        Sampling period (s).
     * @param cutoff_hz Low-pass cutoff frequency (Hz). <= 0 disables filtering.
     */
    explicit Differentiator(double dt, double cutoff_hz = 0.0);

    /**
     * @brief Compute the filtered velocity for the given positions.
     * @param positions Current position samples (any length, kept consistent
     *                  across calls).
     * @return Filtered velocity vector of the same length as `positions`.
     */
    std::vector<double> calculate_velocity(const std::vector<double>& positions);

    /** @brief Reset internal state (previous positions and filter memory). */
    void reset();

private:
    double dt_;
    double alpha_;
    bool has_prev_;
    std::vector<double> prev_positions_;
    bool has_filtered_;
    std::vector<double> filtered_velocity_;
};

/**
 * @brief First-order low-pass filter (exponential smoothing) for a signal.
 *
 * Mirrors the Python `SignalFilter` in `signal_processing.py`:
 *   filtered = (1 - alpha) * filtered + alpha * measurement
 *
 * On the first call the measurement is copied verbatim into the filter state.
 */
class SignalFilter {
public:
    /**
     * @param cutoff_hz Low-pass cutoff frequency (Hz).
     * @param dt        Sampling period (s).
     */
    SignalFilter(double cutoff_hz, double dt);

    /**
     * @brief Filter a measurement sample.
     * @param measurement Current measurement vector.
     * @return Smoothed vector of the same length as `measurement`.
     */
    std::vector<double> filter_signal(const std::vector<double>& measurement);

    /** @brief Reset the filter memory. */
    void reset();

private:
    double alpha_;
    bool has_filtered_;
    std::vector<double> filtered_;
};

} // namespace co
