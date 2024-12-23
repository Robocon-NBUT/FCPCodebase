cmake_minimum_required(VERSION 3.10)

project(cppextension)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_SHARED_LIBRARY_PREFIX "")

find_package(pybind11 REQUIRED)

add_subdirectory(a_star)
add_subdirectory(ball_predictor)
add_subdirectory(localization)
