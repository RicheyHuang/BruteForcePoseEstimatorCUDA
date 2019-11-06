cmake_minimum_required(VERSION 3.5)
project(Cudartographer)

set(CMAKE_CXX_STANDARD 14)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;")

find_package(CUDA QUIET REQUIRED)
add_definitions("-Wno-deprecated-gpu-targets")

include_directories(/usr/include/eigen3)

cuda_add_executable(Cudartographer main.cpp utils.cu)
target_link_libraries(Cudartographer ${CUDA_LIBRARIES})