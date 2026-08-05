#include "co_sensor.hpp"

#include <cmath>
#include <random>

namespace co {

SensorBlock::SensorBlock(double cart_resolution,
                         int encoder_resolution_1,
                         int encoder_resolution_2,
                         const std::vector<double>& noise_std_q,
                         const std::vector<double>& noise_std_dq,
                         int seed,
                         int pool_size)
    : cart_step_(cart_resolution),
      angle_step_1_(2.0 * M_PI / static_cast<double>(encoder_resolution_1)),
      angle_step_2_(2.0 * M_PI / static_cast<double>(encoder_resolution_2)),
      pool_size_(pool_size),
      noise_index_(0),
      meas_(6, 0.0) {
    // Combined std vector: [q_std (3), dq_std (3)].
    std_.reserve(6);
    std_.insert(std_.end(), noise_std_q.begin(), noise_std_q.end());
    std_.insert(std_.end(), noise_std_dq.begin(), noise_std_dq.end());

    // Pre-generate the noise pool once (avoids RNG calls in the hot loop).
    std::mt19937 rng(static_cast<unsigned int>(seed));
    noise_pool_.resize(pool_size_);
    for (int i = 0; i < pool_size_; ++i) {
        noise_pool_[i].resize(6);
        for (int j = 0; j < 6; ++j) {
            std::normal_distribution<double> dist(0.0, std_[j]);
            noise_pool_[i][j] = dist(rng);
        }
    }
}

std::vector<double> SensorBlock::get_telemetry(
    const std::vector<double>& raw_q,
    const std::vector<double>& raw_dq) {
    const double cs = cart_step_;
    const double a1 = angle_step_1_;
    const double a2 = angle_step_2_;

    // Quantization of coordinates.
    meas_[0] = std::rint(raw_q[0] / cs) * cs;
    meas_[1] = std::rint(raw_q[1] / a1) * a1;
    meas_[2] = std::rint(raw_q[2] / a2) * a2;
    // Velocities pass through unquantized.
    meas_[3] = raw_dq[0];
    meas_[4] = raw_dq[1];
    meas_[5] = raw_dq[2];

    // Add noise from the cyclic pool.
    const std::vector<double>& noise = noise_pool_[noise_index_];
    noise_index_++;
    if (noise_index_ >= pool_size_) {
        noise_index_ = 0;
    }

    for (int i = 0; i < 6; ++i) {
        meas_[i] += noise[i];
    }
    return meas_;
}

void SensorBlock::reset() {
    noise_index_ = 0;
}

} // namespace co
