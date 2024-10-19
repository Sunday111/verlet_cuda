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

# Set path to external modules sources
set(YAE_EXTERNAL_MODULES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cloned_repositories)

set(YAE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/yae")
set(YAE_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${YAE_ROOT}/cmake")


include(yae_lto)
enable_lto_globally()


# https://github.com/Sunday111/EverydayTools master
set(YAE_edt_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/Sunday111/EverydayTools)
option(EDT_WITH_TESTS "" ON)
add_subdirectory(${YAE_edt_SOURCES} yae_modules/Sunday111/EverydayTools SYSTEM)

# https://github.com/Sunday111/CppReflection for_yae
set(YAE_CppReflection_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/Sunday111/CppReflection)
add_subdirectory(${YAE_CppReflection_SOURCES} yae_modules/Sunday111/CppReflection SYSTEM)

# https://github.com/martinus/unordered_dense v4.1.2
set(YAE_unordered_dense_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/martinus/unordered_dense)
add_subdirectory(${YAE_unordered_dense_SOURCES} yae_modules/martinus/unordered_dense SYSTEM)

# https://github.com/fmtlib/fmt 10.1.1
set(YAE_fmtlib_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/fmt/10_1_1)
add_subdirectory(${YAE_fmtlib_SOURCES} yae_modules/fmt/10_1_1 SYSTEM)

set(YAE_ecs_SOURCES yae/modules/examples/ecs)
add_subdirectory(${YAE_ecs_SOURCES} yae_modules/yae/modules/examples/ecs SYSTEM)

# https://github.com/google/benchmark v1.8.3
set(YAE_gbench_main_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/google/benchmark/1_8_3)
option(BENCHMARK_ENABLE_TESTING "" OFF)
add_subdirectory(${YAE_gbench_main_SOURCES} yae_modules/google/benchmark/1_8_3 SYSTEM)

set(YAE_ecs_benchmark_SOURCES yae/modules/examples/ecs_benchmark)
add_subdirectory(${YAE_ecs_benchmark_SOURCES} yae_modules/yae/modules/examples/ecs_benchmark SYSTEM)

# https://github.com/google/googletest v1.15.2
set(YAE_gtest_main_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/google/googletest/1_15_2)
option(INSTALL_GTEST "" OFF)
option(BUILD_GMOCK "" OFF)
add_subdirectory(${YAE_gtest_main_SOURCES} yae_modules/google/googletest/1_15_2 SYSTEM)

set(YAE_ecs_tests_SOURCES yae/modules/examples/ecs_tests)
add_subdirectory(${YAE_ecs_tests_SOURCES} yae_modules/yae/modules/examples/ecs_tests SYSTEM)

set(YAE_yae_example_benchmark_SOURCES yae/modules/examples/yae_example_benchmark)
add_subdirectory(${YAE_yae_example_benchmark_SOURCES} yae_modules/yae/modules/examples/yae_example_benchmark SYSTEM)

set(YAE_yae_example_benchmark_main_SOURCES yae/modules/examples/yae_example_benchmark_main)
add_subdirectory(${YAE_yae_example_benchmark_main_SOURCES} yae_modules/yae/modules/examples/yae_example_benchmark_main SYSTEM)

set(YAE_yae_example_library_SOURCES yae/modules/examples/yae_example_library)
add_subdirectory(${YAE_yae_example_library_SOURCES} yae_modules/yae/modules/examples/yae_example_library SYSTEM)

set(YAE_yae_example_header_only_SOURCES yae/modules/examples/yae_example_header_only)
add_subdirectory(${YAE_yae_example_header_only_SOURCES} yae_modules/yae/modules/examples/yae_example_header_only SYSTEM)

set(YAE_yae_example_executable_SOURCES yae/modules/examples/yae_example_executable)
add_subdirectory(${YAE_yae_example_executable_SOURCES} yae_modules/yae/modules/examples/yae_example_executable SYSTEM)

set(YAE_yae_example_tests_SOURCES yae/modules/examples/yae_example_tests)
add_subdirectory(${YAE_yae_example_tests_SOURCES} yae_modules/yae/modules/examples/yae_example_tests SYSTEM)

set(YAE_yae_example_tests_main_SOURCES yae/modules/examples/yae_example_tests_main)
add_subdirectory(${YAE_yae_example_tests_main_SOURCES} yae_modules/yae/modules/examples/yae_example_tests_main SYSTEM)

# https://github.com/Sunday111/ass main
set(YAE_ass_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/Sunday111/ass)
option(ASS_ENABLE_TESTING "" ON)
option(ASS_FETCH_GOOGLE_TESTS "" OFF)
add_subdirectory(${YAE_ass_SOURCES} yae_modules/Sunday111/ass SYSTEM)

# https://github.com/nlohmann/json develop
set(YAE_nlohmann_json_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/nlohmann/json)
add_subdirectory(${YAE_nlohmann_json_SOURCES} yae_modules/nlohmann/json SYSTEM)

# https://github.com/Sunday111/glad main
set(YAE_glad_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/Sunday111/glad)
add_subdirectory(${YAE_glad_SOURCES} yae_modules/Sunday111/glad SYSTEM)

# https://github.com/glfw/glfw master
set(YAE_glfw_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/glfw/glfw)
option(GLFW_BUILD_EXAMPLES "" OFF)
option(GLFW_BUILD_TESTS "" OFF)
option(GLFW_BUILD_DOCS "" OFF)
add_subdirectory(${YAE_glfw_SOURCES} yae_modules/glfw/glfw SYSTEM)

# https://github.com/ocornut/imgui v1.90.9
set(YAE_imgui_sources_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/ocornut/imgui)
set(YAE_imgui_glfw_cmake_SOURCES cloned_repositories/Sunday111/imgui_glfw_cmake)
add_subdirectory(${YAE_imgui_glfw_cmake_SOURCES} yae_modules/cloned_repositories/Sunday111/imgui_glfw_cmake SYSTEM)

# https://github.com/jeremy-rifkin/cpptrace main
set(YAE_cpptrace_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/jeremy-rifkin/cpptrace)
add_subdirectory(${YAE_cpptrace_SOURCES} yae_modules/jeremy-rifkin/cpptrace SYSTEM)

# https://github.com/Neargye/magic_enum v0.9.6
set(YAE_magic_enum_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/Neargye/magic_enum)
add_subdirectory(${YAE_magic_enum_SOURCES} yae_modules/Neargye/magic_enum SYSTEM)

# https://github.com/TartanLlama/expected v1.1.0
set(YAE_expected_SOURCES ${YAE_EXTERNAL_MODULES_DIR}/TartanLlama/expected)
option(EXPECTED_BUILD_TESTS "" OFF)
add_subdirectory(${YAE_expected_SOURCES} yae_modules/TartanLlama/expected SYSTEM)

set(YAE_klgl_SOURCES cloned_repositories/Sunday111/klgl/klgl)
add_subdirectory(${YAE_klgl_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/klgl SYSTEM)

set(YAE_verlet_cuda_SOURCES src/verlet_cuda)
add_subdirectory(${YAE_verlet_cuda_SOURCES} yae_modules/src/verlet_cuda SYSTEM)

set(YAE_klgl_geometry_shader_SOURCES cloned_repositories/Sunday111/klgl/examples/geometry_shader)
add_subdirectory(${YAE_klgl_geometry_shader_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/examples/geometry_shader SYSTEM)

set(YAE_klgl_matmul_sse_benchmark_SOURCES cloned_repositories/Sunday111/klgl/benchmarks/mat_mul_sse)
add_subdirectory(${YAE_klgl_matmul_sse_benchmark_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/benchmarks/mat_mul_sse SYSTEM)

set(YAE_klgl_minimal_cube_example_SOURCES cloned_repositories/Sunday111/klgl/examples/minimal_cube)
add_subdirectory(${YAE_klgl_minimal_cube_example_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/examples/minimal_cube SYSTEM)

set(YAE_klgl_minimal_quad_example_SOURCES cloned_repositories/Sunday111/klgl/examples/minimal_quad)
add_subdirectory(${YAE_klgl_minimal_quad_example_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/examples/minimal_quad SYSTEM)

set(YAE_klgl_painter2d_example_SOURCES cloned_repositories/Sunday111/klgl/examples/painter2d)
add_subdirectory(${YAE_klgl_painter2d_example_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/examples/painter2d SYSTEM)

set(YAE_klgl_rotator_benchmark_SOURCES cloned_repositories/Sunday111/klgl/benchmarks/rotators)
add_subdirectory(${YAE_klgl_rotator_benchmark_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/benchmarks/rotators SYSTEM)

set(YAE_klgl_simple_lit_cube_example_SOURCES cloned_repositories/Sunday111/klgl/examples/simple_lit_cube)
add_subdirectory(${YAE_klgl_simple_lit_cube_example_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/examples/simple_lit_cube SYSTEM)

set(YAE_klgl_tests_SOURCES cloned_repositories/Sunday111/klgl/tests)
add_subdirectory(${YAE_klgl_tests_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/tests SYSTEM)

set(YAE_klgl_textured_quad_example_SOURCES cloned_repositories/Sunday111/klgl/examples/textured_quad)
add_subdirectory(${YAE_klgl_textured_quad_example_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/examples/textured_quad SYSTEM)

set(YAE_klgl_two_textures_example_SOURCES cloned_repositories/Sunday111/klgl/examples/two_textures)
add_subdirectory(${YAE_klgl_two_textures_example_SOURCES} yae_modules/cloned_repositories/Sunday111/klgl/examples/two_textures SYSTEM)


enable_testing()
