#include "co_signal.hpp"

#include <cmath>

namespace co {

// ── Differentiator ─────────────────────────────────────────────────────────

Differentiator::Differentiator(double dt, double cutoff_hz)
    : dt_(dt),
      alpha_(1.0),
      has_prev_(false),
      has_filtered_(false) {
    if (cutoff_hz > 0.0) {
        const double tau = 1.0 / (2.0 * M_PI * cutoff_hz);
        alpha_ = dt_ / (tau + dt_);
    }
}

std::vector<double> Differentiator::calculate_velocity(
    const std::vector<double>& positions) {
    if (!has_prev_) {
        prev_positions_ = positions;
        has_prev_ = true;
        return std::vector<double>(positions.size(), 0.0);
    }

    // raw_vel = (positions - prev_positions) / dt
    std::vector<double> raw_vel(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        raw_vel[i] = (positions[i] - prev_positions_[i]) / dt_;
    }

    if (!has_filtered_) {
        filtered_velocity_ = raw_vel;
        has_filtered_ = true;
    } else {
        for (size_t i = 0; i < positions.size(); ++i) {
            filtered_velocity_[i] =
                (1.0 - alpha_) * filtered_velocity_[i] + alpha_ * raw_vel[i];
        }
    }

    prev_positions_ = positions;
    return filtered_velocity_;
}

void Differentiator::reset() {
    has_prev_ = false;
    has_filtered_ = false;
    prev_positions_.clear();
    filtered_velocity_.clear();
}

// ── SignalFilter ───────────────────────────────────────────────────────────

SignalFilter::SignalFilter(double cutoff_hz, double dt)
    : alpha_(0.0), has_filtered_(false) {
    const double tau = 1.0 / (2.0 * M_PI * cutoff_hz);
    alpha_ = dt / (tau + dt);
}

std::vector<double> SignalFilter::filter_signal(
    const std::vector<double>& measurement) {
    if (!has_filtered_) {
        filtered_ = measurement;
        has_filtered_ = true;
    } else {
        for (size_t i = 0; i < measurement.size(); ++i) {
            filtered_[i] =
                (1.0 - alpha_) * filtered_[i] + alpha_ * measurement[i];
        }
    }
    return filtered_;
}

void SignalFilter::reset() {
    has_filtered_ = false;
    filtered_.clear();
}

} // namespace co
