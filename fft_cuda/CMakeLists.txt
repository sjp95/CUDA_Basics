cmake_minimum_required(VERSION 3.18)
project(CUDA_Basics_FFT LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CTest)
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(CUDA_ENABLED)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        add_compile_definitions(ENABLE_CUFFT)
    endif()
    set(FFT_CORE_SOURCES
        src/fft_cpu.cpp
        src/fft_cuda.cu
        src/cufft_wrapper.cu
        src/image_processing.cpp
    )
else()
    message(STATUS "CUDA compiler not found. Building with C++ CPU fallback implementation.")
    set(FFT_CORE_SOURCES
        src/fft_cpu.cpp
        src/fft_cuda.cpp
        src/cufft_wrapper.cpp
        src/image_processing.cpp
    )
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_library(fft_cuda_core ${FFT_CORE_SOURCES})

if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    target_link_libraries(fft_cuda_core PUBLIC CUDA::cufft CUDA::cudart)
endif()

# Executables
add_executable(fft_benchmark src/benchmark.cpp)
target_link_libraries(fft_benchmark PRIVATE fft_cuda_core)

add_executable(fft_test tests/test_fft.cpp)
target_link_libraries(fft_test PRIVATE fft_cuda_core)

add_test(NAME unit_tests COMMAND fft_test)
