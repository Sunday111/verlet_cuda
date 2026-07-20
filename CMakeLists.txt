cmake_minimum_required(VERSION 3.20)

project(verlet_cuda)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Set output directories for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
# Set output directories for archives
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
# Set output directories for libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# Set path to cloned repositories sources
set(YAE_CLONED_REPOSITORIES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/cloned_repositories" CACHE PATH "Path to YAE cloned repository checkouts")

set(YAE_SUPPORT_ROOT "${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main")
set(YAE_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${YAE_SUPPORT_ROOT}/cmake")


include(yae_lto)
enable_lto_globally()

find_package(Python3 REQUIRED COMPONENTS Interpreter)


# https://github.com/fmtlib/fmt 11.1.1
set(YAE_fmtlib_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/fmtlib/fmt/11.1.1)
add_subdirectory(${YAE_fmtlib_SOURCES} yae_modules/fmtlib/fmt/11.1.1 SYSTEM)

# https://github.com/google/googletest v1.15.2
set(YAE_gtest_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/google/googletest/v1.15.2)
option(INSTALL_GTEST "" OFF)
option(BUILD_GMOCK "" OFF)
add_subdirectory(${YAE_gtest_SOURCES} yae_modules/google/googletest/v1.15.2 SYSTEM)

# https://github.com/Sunday111/EverydayTools main
set(YAE_edt_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/EverydayTools/main)
option(EDT_WITH_TESTS "" ON)
option(EDT_FETCH_GOOGLE_TESTS "" OFF)
option(EDT_FIND_GTEST_PACKAGE "" OFF)
add_subdirectory(${YAE_edt_SOURCES} yae_modules/Sunday111/EverydayTools/main SYSTEM)

# https://github.com/Sunday111/ass main
set(YAE_ass_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/ass/main)
option(ASS_ENABLE_TESTING "" ON)
option(ASS_FETCH_GOOGLE_TESTS "" OFF)
add_subdirectory(${YAE_ass_SOURCES} yae_modules/Sunday111/ass/main SYSTEM)

# https://github.com/Sunday111/CppReflection for_yae
set(YAE_CppReflection_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/CppReflection/for_yae)
add_subdirectory(${YAE_CppReflection_SOURCES} yae_modules/Sunday111/CppReflection/for_yae SYSTEM)

# https://github.com/jeremy-rifkin/cpptrace main
set(YAE_cpptrace_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/jeremy-rifkin/cpptrace/main)
add_subdirectory(${YAE_cpptrace_SOURCES} yae_modules/jeremy-rifkin/cpptrace/main SYSTEM)

# https://github.com/TartanLlama/expected v1.1.0
set(YAE_expected_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/TartanLlama/expected/v1.1.0)
option(EXPECTED_BUILD_TESTS "" OFF)
add_subdirectory(${YAE_expected_SOURCES} yae_modules/TartanLlama/expected/v1.1.0 SYSTEM)

# https://github.com/Neargye/magic_enum v0.9.7
set(YAE_magic_enum_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Neargye/magic_enum/v0.9.7)
add_subdirectory(${YAE_magic_enum_SOURCES} yae_modules/Neargye/magic_enum/v0.9.7 SYSTEM)

# https://github.com/martinus/unordered_dense v4.1.2
set(YAE_unordered_dense_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/martinus/unordered_dense/v4.1.2)
add_subdirectory(${YAE_unordered_dense_SOURCES} yae_modules/martinus/unordered_dense/v4.1.2 SYSTEM)

# https://github.com/nlohmann/json develop
set(YAE_nlohmann_json_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/nlohmann/json/develop)
add_subdirectory(${YAE_nlohmann_json_SOURCES} yae_modules/nlohmann/json/develop SYSTEM)

# https://github.com/glfw/glfw master
set(YAE_glfw_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/glfw/glfw/master)
option(GLFW_BUILD_EXAMPLES "" OFF)
option(GLFW_BUILD_TESTS "" OFF)
option(GLFW_BUILD_DOCS "" OFF)
add_subdirectory(${YAE_glfw_SOURCES} yae_modules/glfw/glfw/master SYSTEM)

# https://github.com/KhronosGroup/Vulkan-Headers vulkan-sdk-1.4.350.1
set(YAE_vulkan_headers_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/KhronosGroup/Vulkan-Headers/vulkan-sdk-1.4.350.1)
add_subdirectory(${YAE_vulkan_headers_SOURCES} yae_modules/KhronosGroup/Vulkan-Headers/vulkan-sdk-1.4.350.1 SYSTEM)

# https://github.com/zeux/volk vulkan-sdk-1.4.350.1
set(YAE_volk_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/zeux/volk/vulkan-sdk-1.4.350.1)
add_subdirectory(${YAE_volk_SOURCES} yae_modules/zeux/volk/vulkan-sdk-1.4.350.1 SYSTEM)

# https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator v3.4.0
set(YAE_vma_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/v3.4.0)
add_subdirectory(${YAE_vma_SOURCES} yae_modules/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/v3.4.0 SYSTEM)

# https://github.com/ocornut/imgui v1.90.9
set(YAE_imgui_sources_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/ocornut/imgui/v1.90.9)
set(YAE_imgui_vulkan_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/imgui_vulkan)
add_subdirectory(${YAE_imgui_vulkan_SOURCES} yae_modules/Sunday111/klvk/main/imgui_vulkan SYSTEM)

set(YAE_klvk_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/klvk)
add_subdirectory(${YAE_klvk_SOURCES} yae_modules/Sunday111/klvk/main/klvk SYSTEM)

include(${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/klvk/shader_cache.cmake)
include(${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/klvk/vulkan_api.cmake)
set(YAE_verlet_cuda_SOURCES src/verlet_cuda)
add_subdirectory(${YAE_verlet_cuda_SOURCES} yae_modules/src/verlet_cuda SYSTEM)

include(${CMAKE_CURRENT_SOURCE_DIR}/src/verlet_cuda/add_cuda.cmake)
set(YAE_ecs_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/ecs)
add_subdirectory(${YAE_ecs_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/ecs SYSTEM)

# https://github.com/google/benchmark v1.9.4
set(YAE_gbench_main_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/google/benchmark/v1.9.4)
option(BENCHMARK_ENABLE_TESTING "" OFF)
add_subdirectory(${YAE_gbench_main_SOURCES} yae_modules/google/benchmark/v1.9.4 SYSTEM)

include(${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/third_party/gbench_main.module.cmake)
set(YAE_ecs_benchmark_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/ecs_benchmark)
add_subdirectory(${YAE_ecs_benchmark_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/ecs_benchmark SYSTEM)

set(YAE_ecs_tests_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/ecs_tests)
add_subdirectory(${YAE_ecs_tests_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/ecs_tests SYSTEM)

set(YAE_yae_example_benchmark_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_benchmark)
add_subdirectory(${YAE_yae_example_benchmark_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_benchmark SYSTEM)

set(YAE_yae_example_benchmark_main_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_benchmark_main)
add_subdirectory(${YAE_yae_example_benchmark_main_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_benchmark_main SYSTEM)

set(YAE_yae_example_library_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_library)
add_subdirectory(${YAE_yae_example_library_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_library SYSTEM)

set(YAE_yae_example_header_only_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_header_only)
add_subdirectory(${YAE_yae_example_header_only_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_header_only SYSTEM)

set(YAE_yae_example_executable_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_executable)
add_subdirectory(${YAE_yae_example_executable_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_executable SYSTEM)

set(YAE_yae_example_tests_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_tests)
add_subdirectory(${YAE_yae_example_tests_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_tests SYSTEM)

set(YAE_yae_example_tests_main_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/yae-support/main/modules/examples/yae_example_tests_main)
add_subdirectory(${YAE_yae_example_tests_main_SOURCES} yae_modules/Sunday111/yae-support/main/modules/examples/yae_example_tests_main SYSTEM)

set(YAE_klvk_compute_shader_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/compute_shader)
add_subdirectory(${YAE_klvk_compute_shader_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/compute_shader SYSTEM)

set(YAE_klvk_curve_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/curve)
add_subdirectory(${YAE_klvk_curve_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/curve SYSTEM)

set(YAE_klvk_curve_fractal_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/curve_fractal)
add_subdirectory(${YAE_klvk_curve_fractal_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/curve_fractal SYSTEM)

set(YAE_klvk_diagnostic_config_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/diagnostic_config)
add_subdirectory(${YAE_klvk_diagnostic_config_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/diagnostic_config SYSTEM)

set(YAE_klvk_falling_sand_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/falling_sand)
add_subdirectory(${YAE_klvk_falling_sand_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/falling_sand SYSTEM)

set(YAE_klvk_fractal_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/fractal)
add_subdirectory(${YAE_klvk_fractal_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/fractal SYSTEM)

set(YAE_klvk_geometry_shader_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/geometry_shader)
add_subdirectory(${YAE_klvk_geometry_shader_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/geometry_shader SYSTEM)

set(YAE_klvk_minimal_cube_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/minimal_cube)
add_subdirectory(${YAE_klvk_minimal_cube_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/minimal_cube SYSTEM)

set(YAE_klvk_minimal_quad_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/minimal_quad)
add_subdirectory(${YAE_klvk_minimal_quad_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/minimal_quad SYSTEM)

set(YAE_klvk_painter2d_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/painter2d)
add_subdirectory(${YAE_klvk_painter2d_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/painter2d SYSTEM)

set(YAE_klvk_pendulum_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/pendulum)
add_subdirectory(${YAE_klvk_pendulum_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/pendulum SYSTEM)

set(YAE_klvk_post_processing_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/post_processing)
add_subdirectory(${YAE_klvk_post_processing_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/post_processing SYSTEM)

set(YAE_klvk_render_to_texture_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/render_to_texture)
add_subdirectory(${YAE_klvk_render_to_texture_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/render_to_texture SYSTEM)

set(YAE_klvk_shader_cache_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/shader_cache)
add_subdirectory(${YAE_klvk_shader_cache_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/shader_cache SYSTEM)

set(YAE_klvk_simple_lit_cube_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/simple_lit_cube)
add_subdirectory(${YAE_klvk_simple_lit_cube_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/simple_lit_cube SYSTEM)

set(YAE_klvk_tetris_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/tetris)
add_subdirectory(${YAE_klvk_tetris_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/tetris SYSTEM)

set(YAE_klvk_textured_quad_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/textured_quad)
add_subdirectory(${YAE_klvk_textured_quad_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/textured_quad SYSTEM)

set(YAE_klvk_timer_manager_test_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/tests/timer_manager)
add_subdirectory(${YAE_klvk_timer_manager_test_SOURCES} yae_modules/Sunday111/klvk/main/tests/timer_manager SYSTEM)

set(YAE_klvk_two_textures_example_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/klvk/main/examples/two_textures)
add_subdirectory(${YAE_klvk_two_textures_example_SOURCES} yae_modules/Sunday111/klvk/main/examples/two_textures SYSTEM)

set(YAE_imgui_glfw_cmake_SOURCES ${YAE_CLONED_REPOSITORIES_DIR}/Sunday111/imgui_glfw_cmake/yae)
add_subdirectory(${YAE_imgui_glfw_cmake_SOURCES} yae_modules/Sunday111/imgui_glfw_cmake/yae SYSTEM)


enable_testing()
