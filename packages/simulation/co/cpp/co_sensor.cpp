#include "co_sensor.hpp"

#include <cmath>

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

    // Temporary exact-sensor mode: retain the constructor API, but bypass
    // quantization, noise, differentiation and filtering in get_telemetry().
    (void)noise_std_q;
    (void)noise_std_dq;
    (void)seed;
    (void)pool_size;
}

std::vector<double> SensorBlock::get_telemetry(
    const std::vector<double>& raw_q,
    const std::vector<double>& raw_dq) {
    // Exact telemetry with the existing observation layout:
    // [x, cos(theta1), sin(theta1), cos(theta2), sin(theta2), dq].
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

void SensorBlock::reset() {
    noise_index_ = 0;
    differentiator_.reset();
    filter_.reset();
}

} // namespace co
