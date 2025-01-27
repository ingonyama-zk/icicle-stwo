cmake_minimum_required(VERSION 3.28)
project(stwo_playground CUDA)

set(CMAKE_CUDA_STANDARD 20)

include_directories(icicle/icicle/include/)

add_executable(stwo_playground main.cu)

set_target_properties(stwo_playground PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)
