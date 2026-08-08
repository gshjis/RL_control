#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <cmath>

#include "co_physics.hpp"
#include "co_sensor.hpp"
#include "co_signal.hpp"

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

    // ── Signal processing (full C++ implementations) ──────────────────────
    py::class_<co::Differentiator>(m, "Differentiator")
        .def(py::init<double, double>(), py::arg("dt"), py::arg("cutoff_hz") = 0.0)
        .def("calculate_velocity", &co::Differentiator::calculate_velocity)
        .def("reset", &co::Differentiator::reset);

    py::class_<co::SignalFilter>(m, "SignalFilter")
        .def(py::init<double, double>(), py::arg("cutoff_hz"), py::arg("dt"))
        .def("filter_signal", &co::SignalFilter::filter_signal)
        .def("reset", &co::SignalFilter::reset);

    // ── Sensor block (full C++ implementation) ─────────────────────────────
    py::class_<co::SensorBlock>(m, "SensorBlock")
        .def(py::init<double, int, int, std::vector<double>, std::vector<double>,
                      int, int, double, double, double>(),
             py::arg("cart_resolution"),
             py::arg("encoder_resolution_1"),
             py::arg("encoder_resolution_2"),
             py::arg("noise_std_q"),
             py::arg("noise_std_dq"),
             py::arg("seed"),
             py::arg("pool_size"),
             py::arg("dt"),
             py::arg("differentiator_cutoff_hz"),
             py::arg("filter_cutoff_hz"))
        .def("get_telemetry", &co::SensorBlock::get_telemetry)
        .def("reset", &co::SensorBlock::reset);

    // Main step function: updates q/dq using RK4.
    // This mirrors ObjectOfControl.update_physics(F_ideal, noise).
    m.def(
        "rk4_step",
        [](State3 q, StateDot3 dq,
           double F_ideal,
           NoiseForceCPP noise,
           double dt,
           PlantParams params,
           bool single_mode,
           double& motor_force) {
            // Exact zero-order-hold update of the first-order motor lag.
            double F_actual = F_ideal;
            if (params.motor_tau > 0.0) {
                const double alpha = -std::expm1(-dt / params.motor_tau);
                F_actual = motor_force + (F_ideal - motor_force) * alpha;
            }
            motor_force = F_actual;

            // Sample noise from normal distribution N(noise.mean, noise.std²)
            const double F_noise = sample_noise_force(noise);
            const double F_total = F_actual + F_noise;

            rk4_step(q, dq, F_total, dt, params, single_mode);
            return py::make_tuple(q, dq);
        },
        py::arg("q"), py::arg("dq"), py::arg("F_ideal"), py::arg("noise"), py::arg("dt"),
        py::arg("params"), py::arg("single_mode"), py::arg("motor_force"));

    // update_physics_cpp: performance-oriented multi-step wrapper that updates
    // q/dq in-place and returns the updated motor_force.
    //
    // The applied force F_ideal is held constant and the physics is advanced
    // `n_updates` times (substeps), each of duration `dt`. Motor inertia and
    // noise are re-evaluated on every substep so the effect of the force is
    // correctly accumulated. When `n_updates == 1` the behaviour is identical
    // to the original single-step call.
    //
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
           bool single_mode,
           double motor_force,
           int n_updates) {
            if (q_arr.size() != 3 || dq_arr.size() != 3) {
                throw std::runtime_error("update_physics_cpp expects q/dq arrays of size 3");
            }
            if (n_updates < 1) {
                throw std::runtime_error("update_physics_cpp expects n_updates >= 1");
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

            for (int i = 0; i < n_updates; ++i) {
                // Exact zero-order-hold update of the first-order motor lag.
                double F_actual = F_ideal;
                if (params.motor_tau > 0.0) {
                    const double alpha = -std::expm1(-dt / params.motor_tau);
                    F_actual = motor_force + (F_ideal - motor_force) * alpha;
                }
                motor_force = F_actual;

                // Sample noise from normal distribution N(noise_mean, noise_std²)
                const double F_noise = (noise_std > 0.0)
                    ? sample_noise_force({noise_mean, noise_std})
                    : noise_mean;
                const double F_total = F_actual + F_noise;
                rk4_step(q, dq, F_total, dt, params, single_mode);
            }

            // write back
            q_ptr[0] = q.x;
            q_ptr[1] = q.theta1;
            q_ptr[2] = q.theta2;
            dq_ptr[0] = dq.x_dot;
            dq_ptr[1] = dq.theta1_dot;
            dq_ptr[2] = dq.theta2_dot;

            return py::make_tuple(q_arr, dq_arr, motor_force);
        },
        py::arg("q"), py::arg("dq"), py::arg("F_ideal"),
        py::arg("noise_mean"), py::arg("noise_std"), py::arg("dt"),
        py::arg("params"), py::arg("single_mode"), py::arg("motor_force"),
        py::arg("n_updates") = 1);

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
