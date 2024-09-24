cmake_minimum_required(VERSION 3.20)
enable_language(CUDA)
include(set_compiler_options)
set(module_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/saxpy.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/saxpy.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/util.hpp)
add_executable(cuda_test ${module_source_files})
set_generic_compiler_options(cuda_test PRIVATE)
target_include_directories(cuda_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/code/public)
target_include_directories(cuda_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/code/private)
include(${CMAKE_CURRENT_SOURCE_DIR}/add_cuda.cmake)
