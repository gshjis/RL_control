cmake_minimum_required(VERSION 3.20)
project(co_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build a Python extension with pybind11.
# Assumes pybind11 is available via system or python packaging.

find_package(Python REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_include())"
  OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# If pybind11 is not importable from the Python interpreter, CMake can't build.
if (PYBIND11_INCLUDE_DIR STREQUAL "")
  message(FATAL_ERROR "pybind11 include directory not found. Ensure 'pybind11' is installed in the same interpreter used by CMake: ${Python_EXECUTABLE}")
endif()

include_directories(${PYBIND11_INCLUDE_DIR})
include_directories(${Python_INCLUDE_DIRS})

set(SRC
  co_bindings.cpp
  co_physics.cpp
  co_signal.cpp
  co_sensor.cpp
)

add_library(co_cpp_ext MODULE ${SRC})

set_target_properties(co_cpp_ext PROPERTIES
  PREFIX ""
  OUTPUT_NAME "co_cpp"
)

if (WIN32)
  set_target_properties(co_cpp_ext PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(co_cpp_ext PROPERTIES SUFFIX ".so")
endif()

target_link_libraries(co_cpp_ext PRIVATE ${Python_LIBRARIES})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python)

# Install into the Python package folder for local import.
set_target_properties(co_cpp_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/../"
)
