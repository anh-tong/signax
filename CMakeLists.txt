cmake_minimum_required(VERSION 3.10...3.18)
project(signax LANGUAGES CXX)

message(STATUS "Using CMake version " ${CMAKE_VERSION})

find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/signax/backend)

pybind11_add_module(cpu_ops ${CMAKE_CURRENT_LIST_DIR}/signax/backend/cpu_ops.cpp)
install(TARGETS cpu_ops DESTINATION signax)

if (BUILD_WITH_CUDA)
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    pybind11_add_module(
            gpu_ops
            ${CMAKE_CURRENT_LIST_DIR}/signax/backend/cuda_kernels.cc.cu
            ${CMAKE_CURRENT_LIST_DIR}/signax/backend/gpu_ops.cc)
    install(TARGETS gpu_ops DESTINATION signax)
else()
    message(STATUS "Building without CUDA")
endif()