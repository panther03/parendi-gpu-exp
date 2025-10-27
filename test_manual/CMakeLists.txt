cmake_minimum_required(VERSION 3.20)

project(test_manual LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

include_directories(../include)

# Source file
# Add target for vectorAdd
add_executable(test_manual main.cu Vtest__codelet__0.cu)

target_compile_options(test_manual PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(test_manual PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(test_manual PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
