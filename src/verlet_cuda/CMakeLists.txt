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
set(verlet_cuda_copy_files_outputs)
file(GLOB_RECURSE verlet_cuda_copy_files_sources_0 CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/content/*")
foreach(source_file ${verlet_cuda_copy_files_sources_0})
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_SOURCE_DIR}/content" "${source_file}")
    set(copied_file "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/content/${rel_path}")
    add_custom_command(
        OUTPUT "${copied_file}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_newer "${source_file}" "${copied_file}"
        DEPENDS "${source_file}"
        COMMENT "stage ${rel_path}")
    list(APPEND verlet_cuda_copy_files_outputs "${copied_file}")
endforeach()
add_custom_target(verlet_cuda_copy_files ALL DEPENDS ${verlet_cuda_copy_files_outputs})
add_dependencies(verlet_cuda verlet_cuda_copy_files)
