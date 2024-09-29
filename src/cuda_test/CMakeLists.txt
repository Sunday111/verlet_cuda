cmake_minimum_required(VERSION 3.20)
enable_language(CUDA)
include(set_compiler_options)
set(module_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/cuda/cuda_algorithms.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/cuda/cuda_max_element.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/cuda/cuda_vector_lengths.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/cuda/util.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/time.hpp)
add_executable(cuda_test ${module_source_files})
set_generic_compiler_options(cuda_test PRIVATE)
target_link_libraries(cuda_test PUBLIC fmt
                                       edt)
target_include_directories(cuda_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/code/public)
target_include_directories(cuda_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/code/private)
include(${CMAKE_CURRENT_SOURCE_DIR}/add_cuda.cmake)
