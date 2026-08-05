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
                         int pool_size,
                         double dt,
                         double differentiator_cutoff_hz,
                         double filter_cutoff_hz)
    : cart_step_(cart_resolution),
      angle_step_1_(2.0 * M_PI / static_cast<double>(encoder_resolution_1)),
      angle_step_2_(2.0 * M_PI / static_cast<double>(encoder_resolution_2)),
      std_q_(noise_std_q),
      pool_size_(pool_size),
      noise_index_(0),
      meas_(6, 0.0),
      differentiator_(dt, differentiator_cutoff_hz),
      filter_(filter_cutoff_hz, dt) {
    (void)noise_std_dq;  // reserved: velocity noise arises from differentiation

    // Pre-generate the position-noise pool once (3 values per sample).
    std::mt19937 rng(static_cast<unsigned int>(seed));
    noise_pool_.resize(pool_size_);
    for (int i = 0; i < pool_size_; ++i) {
        noise_pool_[i].resize(3);
        for (int j = 0; j < 3; ++j) {
            std::normal_distribution<double> dist(0.0, std_q_[j]);
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

    // 1. Quantize coordinates.
    meas_[0] = std::rint(raw_q[0] / cs) * cs;
    meas_[1] = std::rint(raw_q[1] / a1) * a1;
    meas_[2] = std::rint(raw_q[2] / a2) * a2;

    // 2. Add position noise from the cyclic pool.
    const std::vector<double>& noise = noise_pool_[noise_index_];
    noise_index_++;
    if (noise_index_ >= pool_size_) {
        noise_index_ = 0;
    }
    for (int i = 0; i < 3; ++i) {
        meas_[i] += noise[i];
    }

    // 3. Transform the noisy coordinates into the feature vector:
    //    [x, sin(theta1), sin(theta2)].
    std::vector<double> feat(3);
    feat[0] = meas_[0];
    feat[1] = std::sin(meas_[1]);
    feat[2] = std::sin(meas_[2]);

    // 4. Differentiator: derivatives of the features (finite differences
    //    between the current and previous noisy sample).
    std::vector<double> vel = differentiator_.calculate_velocity(feat);

    // 5. Assemble the raw 6-vector [feat, vel].
    std::vector<double> raw6(6);
    for (int i = 0; i < 3; ++i) {
        raw6[i] = feat[i];
        raw6[3 + i] = vel[i];
    }

    // 6. Filter the noisy states and their derivatives.
    return filter_.filter_signal(raw6);
}

void SensorBlock::reset() {
    noise_index_ = 0;
    differentiator_.reset();
    filter_.reset();
}

} // namespace co
