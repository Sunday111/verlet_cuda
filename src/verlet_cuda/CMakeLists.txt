cmake_minimum_required(VERSION 3.20)
enable_language(CUDA)
include(set_compiler_options)
set(module_source_files
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/coloring/object_color_function.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/coloring/spawn_color/spawn_color_strategy.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/coloring/spawn_color/spawn_color_strategy_array.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/coloring/spawn_color/spawn_color_strategy_array.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/coloring/spawn_color/spawn_color_strategy_rainbow.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/coloring/spawn_color/spawn_color_strategy_rainbow.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/constants.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/cuda_util.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/cuda_util.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/emitters/emitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/emitters/emitter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/emitters/emitter_type.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/emitters/radial_emitter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/emitters/radial_emitter.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/kernels.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/kernels.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/verlet_cuda_app.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/verlet_cuda_app.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/code/private/verlet_object.hpp)
add_executable(verlet_cuda ${module_source_files})
set_generic_compiler_options(verlet_cuda PRIVATE)
target_link_libraries(verlet_cuda PUBLIC fmt
                                         edt
                                         klvk)
target_include_directories(verlet_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/code/public)
target_include_directories(verlet_cuda PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/code/private)
include(${CMAKE_CURRENT_SOURCE_DIR}/add_cuda.cmake)
add_custom_target(verlet_cuda_copy_files ALL
    COMMAND ${Python3_EXECUTABLE} "${YAE_SUPPORT_ROOT}/scripts/stage_directories.py"
            --destination "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/content"
            --source "${CMAKE_CURRENT_SOURCE_DIR}/content"
            --manifest "${CMAKE_CURRENT_BINARY_DIR}/verlet_cuda_copy_files_content.manifest"
    COMMENT "stage content of verlet_cuda"
    VERBATIM)
add_dependencies(verlet_cuda verlet_cuda_copy_files)
