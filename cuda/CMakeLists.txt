cmake_minimum_required(VERSION 3.24)
project(mobility_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# Override for another machine: -DCMAKE_CUDA_ARCHITECTURES=86 (for example).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
add_executable(cuda_smoke smoke.cu)
target_include_directories(cuda_smoke PRIVATE ../src/mobility/backends/kernels)
target_compile_options(cuda_smoke PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
enable_testing()
add_test(NAME cuda_smoke COMMAND cuda_smoke)
