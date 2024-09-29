cmake_minimum_required(VERSION 3.20)
enable_language(CUDA)
include(set_compiler_options)
set(module_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/camera.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/constants.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/cuda/cuda_helpers.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/cuda/gl_interop.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/cuda/gl_interop.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/cuda_verlet.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/kernels.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/time.hpp)
add_executable(cuda_gl ${module_source_files})
set_generic_compiler_options(cuda_gl PRIVATE)
target_link_libraries(cuda_gl PUBLIC fmt
                                     edt
                                     klgl)
target_include_directories(cuda_gl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/code/public)
target_include_directories(cuda_gl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/code/private)
include(${CMAKE_CURRENT_SOURCE_DIR}/add_cuda.cmake)
add_custom_target(cuda_gl_copy_files ALL
    ${CMAKE_COMMAND} -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/content" ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/content
)
add_dependencies(cuda_gl_copy_files cuda_gl)
