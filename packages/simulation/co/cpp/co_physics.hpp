#pragma once

/**
 * @file co_physics.hpp
 * @brief C++ core for inverted pendulum physics simulation (RK4 integrator).
 *
 * Provides data structures (State3, StateDot3, PlantParams, NoiseForceCPP)
 * and functions (compute_ddq, rk4_step) used by the pybind11 bindings.
 *
 * @note Optimization potential:
 *       - Cramer's rule in compute_ddq (3x3) could be replaced with
 *         Gaussian elimination for numerical stability near singularity.
 *       - In single_pendulum_mode, only a 2x2 system is solved;
 *         the code is already branch-efficient.
 */

/**
 * @brief State vector: generalized coordinates of the cart-pendulum system.
 */
struct State3 {
    double x;       /**< Cart position (m). */
    double theta1;  /**< First pendulum angle (rad). */
    double theta2;  /**< Second pendulum angle (rad). */
};

/**
 * @brief State velocity vector: time derivatives of State3.
 */
struct StateDot3 {
    double x_dot;        /**< Cart velocity (m/s). */
    double theta1_dot;   /**< First pendulum angular velocity (rad/s). */
    double theta2_dot;   /**< Second pendulum angular velocity (rad/s). */
};

/**
 * @brief Physical parameters of the plant (cart + pendulum).
 *
 * Maps 1:1 to PlantConfig in datatypes.py.
 */
struct PlantParams {
    double M;   /**< Cart mass (kg). */
    double m1;  /**< First link mass (kg). */
    double m2;  /**< Second link mass (kg). */
    double l1;  /**< First link full length (m). */
    double l2;  /**< Second link full length (m). */
    double L1;  /**< Distance from pivot to first link CoM (m). */
    double L2;  /**< Distance from joint to second link CoM (m). */
    double J1;  /**< First link moment of inertia about CoM (kg·m²). */
    double J2;  /**< Second link moment of inertia about CoM (kg·m²). */
    double g;   /**< Gravitational acceleration (m/s²). */
    double b_c; /**< Cart viscous friction coefficient. */
    double b_1; /**< First joint viscous friction coefficient. */
    double b_2; /**< Second joint viscous friction coefficient. */
    double motor_tau; /**< Motor time constant (s). 0 = instant response. */
};

/**
 * @brief Noise force parameters (API parity with Python NoiseForce).
 */
struct NoiseForceCPP {
    double mean; /**< Mean value of noise (N). */
    double std;  /**< Standard deviation of noise (N). */
};

/**
 * @brief Sample a random force from a normal distribution.
 * @param n Noise parameters (mean, std).
 * @return Sampled force (N). Returns mean if std == 0.
 *
 * Uses thread-local RNG (std::mt19937) for safety in multi-threaded
 * simulations.
 */
double sample_noise_force(const NoiseForceCPP& n);

/**
 * @brief Compute the time derivative of the state vector (ddq = M^{-1} * rhs).
 *
 * Solves the equations of motion derived from the Lagrangian formalism:
 *   M(q) * ddq + C(q, dq) * dq + G(q) = tau
 *
 * @param q         Current generalized coordinates.
 * @param dq        Current generalized velocities.
 * @param F_total   Total force applied to the cart (N).
 * @param p         Physical parameters of the plant.
 * @param single_mode If true, locks the second pendulum (θ₂ = 0, dθ₂ = 0).
 *
 * @return Acceleration vector (ddx, d²θ₁/dt², d²θ₂/dt²).
 *
 * @note Uses Cramer's rule to solve the 3x3 (or 2x2) linear system.
 *       Returns zeros on near-singular determinant.
 */
StateDot3 compute_ddq(const State3& q, const StateDot3& dq, double F_total,
                      const PlantParams& p, bool single_mode);

/**
 * @brief Perform one RK4 micro-step of the physics simulation.
 *
 * Updates q and dq in-place using the classical 4th-order Runge–Kutta
 * method with fixed step size dt.
 *
 * @param q         [in/out] Generalized coordinates (updated).
 * @param dq        [in/out] Generalized velocities (updated).
 * @param F_total   Total force applied to the cart (N).
 * @param dt        Integration time step (s).
 * @param p         Physical parameters of the plant.
 * @param single_mode If true, locks the second pendulum.
 */
void rk4_step(State3& q, StateDot3& dq, double F_total, double dt,
              const PlantParams& p, bool single_mode);
