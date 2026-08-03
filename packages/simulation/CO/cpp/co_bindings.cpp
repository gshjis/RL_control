#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "co_physics.hpp"

namespace py = pybind11;

struct PyState3 {
    State3 v;
};

PYBIND11_MODULE(co_cpp, m) {
    m.doc() = "C++ physics core for CO simulation (pybind11)";

    // NoiseForce (API parity with Python: mean/std)
    py::class_<NoiseForceCPP>(m, "NoiseForce")
        .def(py::init<double, double>(), py::arg("mean") = 0.0, py::arg("std") = 0.0)
        .def_readwrite("mean", &NoiseForceCPP::mean)
        .def_readwrite("std", &NoiseForceCPP::std)
        .def("get_force", &sample_noise_force);

    // Main step function: updates q/dq using RK4.
    // This mirrors ObjectOfControl.update_physics(F_ideal, noise).
    m.def(
        "rk4_step",
        [](State3 q, StateDot3 dq,
           double F_ideal,
           NoiseForceCPP noise,
           double dt,
           PlantParams params,
           bool backslash_mode,
           bool single_mode,
           double backlash_alpha,
           double backlash_m_mot,
           double& backlash_gap_pos,
           double& motor_force) {
            // Motor inertia (first-order lag)
            double F_actual = F_ideal;
            if (params.motor_tau > 0.0) {
                F_actual = motor_force + (F_ideal - motor_force) * (dt / params.motor_tau);
            }
            motor_force = F_actual;

            // Backlash model in C++.
            double F_real = F_actual;
            if (backslash_mode) {
                const double half_gap = backlash_alpha / 2.0;
                const double a_rel = F_actual / backlash_m_mot;
                backlash_gap_pos += a_rel * dt;
                if (backlash_gap_pos > half_gap) {
                    backlash_gap_pos = half_gap;
                    F_real = F_actual;
                } else if (backlash_gap_pos < -half_gap) {
                    backlash_gap_pos = -half_gap;
                    F_real = F_actual;
                } else {
                    F_real = 0.0;
                }
            }

            // Sample noise from normal distribution N(noise.mean, noise.std²)
            const double F_noise = sample_noise_force(noise);
            const double F_total = F_real + F_noise;

            rk4_step(q, dq, F_total, dt, params, single_mode);
            return py::make_tuple(q, dq);
        },
        py::arg("q"), py::arg("dq"), py::arg("F_ideal"), py::arg("noise"), py::arg("dt"),
        py::arg("params"), py::arg("backslash_mode"), py::arg("single_mode"),
        py::arg("backlash_alpha"), py::arg("backlash_m_mot"), py::arg("backlash_gap_pos"),
        py::arg("motor_force"));

    // update_physics_cpp: performance-oriented wrapper that updates q/dq in-place
    // and returns updated backlash_gap_pos and motor_force.
    // Noise is sampled from normal distribution N(noise_mean, noise_std²).
    m.def(
        "update_physics_cpp",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> q_arr,
           py::array_t<double, py::array::c_style | py::array::forcecast> dq_arr,
           double F_ideal,
           double noise_mean,
           double noise_std,
           double dt,
           PlantParams params,
           bool backslash_mode,
           bool single_mode,
           double backlash_alpha,
           double backlash_m_mot,
           double backlash_gap_pos,
           double motor_force) {
            if (q_arr.size() != 3 || dq_arr.size() != 3) {
                throw std::runtime_error("update_physics_cpp expects q/dq arrays of size 3");
            }

            // Ensure arrays are writable
            if (!q_arr.mutable_data() || !dq_arr.mutable_data()) {
                throw std::runtime_error("update_physics_cpp expects writable q/dq numpy arrays");
            }

            auto q_ptr = q_arr.mutable_data();
            auto dq_ptr = dq_arr.mutable_data();

            State3 q;
            StateDot3 dq;
            q.x = q_ptr[0];
            q.theta1 = q_ptr[1];
            q.theta2 = q_ptr[2];
            dq.x_dot = dq_ptr[0];
            dq.theta1_dot = dq_ptr[1];
            dq.theta2_dot = dq_ptr[2];

            // Motor inertia (first-order lag): F_actual = F_old + (F_ideal - F_old) * (dt / tau)
            double F_actual = F_ideal;
            if (params.motor_tau > 0.0) {
                F_actual = motor_force + (F_ideal - motor_force) * (dt / params.motor_tau);
            }
            motor_force = F_actual;

            // Backlash (exactly as in rk4_step wrapper)
            double F_real = F_actual;
            double gap_pos = backlash_gap_pos;
            if (backslash_mode) {
                const double half_gap = backlash_alpha / 2.0;
                const double a_rel = F_actual / backlash_m_mot;
                gap_pos += a_rel * dt;
                if (gap_pos > half_gap) {
                    gap_pos = half_gap;
                    F_real = F_actual;
                } else if (gap_pos < -half_gap) {
                    gap_pos = -half_gap;
                    F_real = F_actual;
                } else {
                    F_real = 0.0;
                }
            }

            // Sample noise from normal distribution N(noise_mean, noise_std²)
            const double F_noise = (noise_std > 0.0)
                ? sample_noise_force({noise_mean, noise_std})
                : noise_mean;
            const double F_total = F_real + F_noise;
            rk4_step(q, dq, F_total, dt, params, single_mode);

            // write back
            q_ptr[0] = q.x;
            q_ptr[1] = q.theta1;
            q_ptr[2] = q.theta2;
            dq_ptr[0] = dq.x_dot;
            dq_ptr[1] = dq.theta1_dot;
            dq_ptr[2] = dq.theta2_dot;

            return py::make_tuple(q_arr, dq_arr, gap_pos, motor_force);
        },
        py::arg("q"), py::arg("dq"), py::arg("F_ideal"),
        py::arg("noise_mean"), py::arg("noise_std"), py::arg("dt"),
        py::arg("params"), py::arg("backslash_mode"), py::arg("single_mode"),
        py::arg("backlash_alpha"), py::arg("backlash_m_mot"),
        py::arg("backlash_gap_pos"), py::arg("motor_force"));

    py::class_<State3>(m, "State3")
        .def(py::init<>())
        .def_readwrite("x", &State3::x)
        .def_readwrite("theta1", &State3::theta1)
        .def_readwrite("theta2", &State3::theta2);

    py::class_<StateDot3>(m, "StateDot3")
        .def(py::init<>())
        .def_readwrite("x_dot", &StateDot3::x_dot)
        .def_readwrite("theta1_dot", &StateDot3::theta1_dot)
        .def_readwrite("theta2_dot", &StateDot3::theta2_dot);

    py::class_<PlantParams>(m, "PlantParams")
        .def(py::init<>())
        .def_readwrite("M", &PlantParams::M)
        .def_readwrite("m1", &PlantParams::m1)
        .def_readwrite("m2", &PlantParams::m2)
        .def_readwrite("l1", &PlantParams::l1)
        .def_readwrite("l2", &PlantParams::l2)
        .def_readwrite("L1", &PlantParams::L1)
        .def_readwrite("L2", &PlantParams::L2)
        .def_readwrite("J1", &PlantParams::J1)
        .def_readwrite("J2", &PlantParams::J2)
        .def_readwrite("g", &PlantParams::g)
        .def_readwrite("b_c", &PlantParams::b_c)
        .def_readwrite("b_1", &PlantParams::b_1)
        .def_readwrite("b_2", &PlantParams::b_2)
        .def_readwrite("motor_tau", &PlantParams::motor_tau);
}
