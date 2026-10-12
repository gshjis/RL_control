#include "co_sensor.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace co {
namespace {

/** @brief Round `value` to the nearest multiple of `step` (encoder/ruler). */
double quantize(double value, double step) {
    if (step <= 0.0) {
        return value;
    }
    return std::round(value / step) * step;
}

/** @brief Std for channel `index`, or 0 when the vector is too short. */
double channel_std(const std::vector<double>& stds, size_t index) {
    return index < stds.size() ? stds[index] : 0.0;
}

} // namespace

SensorBlock::SensorBlock(double cart_resolution,
                         int encoder_resolution_1,
                         int encoder_resolution_2,
                         const std::vector<double>& noise_std_q,
                         const std::vector<double>& noise_std_dq,
                         int seed,
                         int pool_size,
                         double dt,
                         double differentiator_cutoff_hz,
                         double filter_cutoff_hz,
                         bool exact_mode)
    : cart_step_(cart_resolution),
      angle_step_1_(2.0 * M_PI / static_cast<double>(encoder_resolution_1)),
      angle_step_2_(2.0 * M_PI / static_cast<double>(encoder_resolution_2)),
      std_q_(noise_std_q),
      std_dq_(noise_std_dq),
      pool_size_(pool_size > 0 ? pool_size : 1),
      noise_pool_(),
      noise_pool_dq_(),
      noise_index_(0),
      meas_(6, 0.0),
      dt_(dt),
      differentiator_cutoff_hz_(differentiator_cutoff_hz),
      filter_cutoff_hz_(filter_cutoff_hz),
      exact_mode_(exact_mode),
      differentiator_(dt, differentiator_cutoff_hz),
      filter_(filter_cutoff_hz, dt) {
    build_noise_pool(seed);
}

void SensorBlock::build_noise_pool(int seed) {
    noise_pool_.assign(static_cast<size_t>(pool_size_),
                       std::vector<double>(3, 0.0));
    noise_pool_dq_.assign(static_cast<size_t>(pool_size_),
                          std::vector<double>(3, 0.0));
    noise_index_ = 0;

    if (exact_mode_) {
        // Exact sensors: the pools stay zero, no RNG involved.
        return;
    }

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::normal_distribution<double> gauss(0.0, 1.0);

    for (int i = 0; i < pool_size_; ++i) {
        auto& position_noise = noise_pool_[static_cast<size_t>(i)];
        auto& velocity_noise = noise_pool_dq_[static_cast<size_t>(i)];
        for (size_t channel = 0; channel < 3; ++channel) {
            position_noise[channel] =
                gauss(rng) * channel_std(std_q_, channel);
            velocity_noise[channel] =
                gauss(rng) * channel_std(std_dq_, channel);
        }
    }
}

std::vector<double> SensorBlock::get_telemetry(
    const std::vector<double>& raw_q,
    const std::vector<double>& raw_dq) {
    if (raw_q.size() < 3 || raw_dq.size() < 3) {
        throw std::runtime_error(
            "SensorBlock::get_telemetry expects q and dq of size 3");
    }

    // 1. Quantization: encoders for the angles, optical ruler for the cart.
    const double x_q = quantize(raw_q[0], cart_step_);
    const double th1_q = quantize(raw_q[1], angle_step_1_);
    const double th2_q = quantize(raw_q[2], angle_step_2_);

    // 2. Take the next noise sample from the pre-computed pools.
    const std::vector<double>& position_noise =
        noise_pool_[static_cast<size_t>(noise_index_)];
    const std::vector<double>& velocity_noise =
        noise_pool_dq_[static_cast<size_t>(noise_index_)];
    noise_index_ = (noise_index_ + 1) % pool_size_;

    meas_[0] = x_q + position_noise[0];
    meas_[1] = th1_q + position_noise[1];
    meas_[2] = th2_q + position_noise[2];

    if (exact_mode_) {
        // Legacy perfect observer: true coordinates and true velocities.
        return {
            raw_q[0],
            std::cos(raw_q[1]),
            std::sin(raw_q[1]),
            std::cos(raw_q[2]),
            std::sin(raw_q[2]),
            raw_dq[0],
            raw_dq[1],
            raw_dq[2],
        };
    }

    // 3. Velocity estimation by finite differences of the noisy positions.
    const std::vector<double> positions = {meas_[0], meas_[1], meas_[2]};
    const std::vector<double> velocities =
        differentiator_.calculate_velocity(positions);

    // 4. Velocity noise, then smoothing of all six measured channels.
    meas_[3] = velocities[0] + velocity_noise[0];
    meas_[4] = velocities[1] + velocity_noise[1];
    meas_[5] = velocities[2] + velocity_noise[2];
    const std::vector<double> smoothed = filter_.filter_signal(meas_);

    return {
        smoothed[0],
        std::cos(smoothed[1]),
        std::sin(smoothed[1]),
        std::cos(smoothed[2]),
        std::sin(smoothed[2]),
        smoothed[3],
        smoothed[4],
        smoothed[5],
    };
}

void SensorBlock::reset() {
    noise_index_ = 0;
    differentiator_.reset();
    filter_.reset();
}

void SensorBlock::set_seed(int seed) {
    build_noise_pool(seed);
}

} // namespace co
