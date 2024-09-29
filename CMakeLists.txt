cmake_minimum_required(VERSION 3.20)

project(Cuda)

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

set(YAE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/yae")
set(YAE_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${YAE_ROOT}/cmake")


include(yae_lto)
enable_lto_globally()


# ---------------------------- External Dependencies -----------------------------

# https://github.com/fmtlib/fmt 10.1.1
set(YAE_CLONED_fmtlib ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/fmt/10_1_1)
add_subdirectory(cloned_dependencies/fmt/10_1_1 SYSTEM)

# https://github.com/Sunday111/EverydayTools master
set(YAE_CLONED_edt ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/Sunday111/EverydayTools)
option(EDT_WITH_TESTS "" ON)
add_subdirectory(cloned_dependencies/Sunday111/EverydayTools SYSTEM)

# https://github.com/Sunday111/ass main
set(YAE_CLONED_ass ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/Sunday111/ass)
option(ASS_ENABLE_TESTING "" ON)
option(ASS_FETCH_GOOGLE_TESTS "" OFF)
add_subdirectory(cloned_dependencies/Sunday111/ass SYSTEM)

# https://github.com/Sunday111/CppReflection for_yae
set(YAE_CLONED_CppReflection ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/Sunday111/CppReflection)
add_subdirectory(cloned_dependencies/Sunday111/CppReflection SYSTEM)

# https://github.com/nlohmann/json develop
set(YAE_CLONED_nlohmann_json ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/nlohmann/json)
add_subdirectory(cloned_dependencies/nlohmann/json SYSTEM)

# https://github.com/Sunday111/glad main
set(YAE_CLONED_glad ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/Sunday111/glad)
add_subdirectory(cloned_dependencies/Sunday111/glad SYSTEM)

# https://github.com/glfw/glfw master
set(YAE_CLONED_glfw ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/glfw/glfw)
option(GLFW_BUILD_EXAMPLES "" OFF)
option(GLFW_BUILD_TESTS "" OFF)
option(GLFW_BUILD_DOCS "" OFF)
add_subdirectory(cloned_dependencies/glfw/glfw SYSTEM)

# https://github.com/ocornut/imgui v1.90.9
set(YAE_CLONED_imgui_sources ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/ocornut/imgui)
# https://github.com/jeremy-rifkin/cpptrace main
set(YAE_CLONED_cpptrace ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/jeremy-rifkin/cpptrace)
add_subdirectory(cloned_dependencies/jeremy-rifkin/cpptrace SYSTEM)

# https://github.com/martinus/unordered_dense v4.1.2
set(YAE_CLONED_unordered_dense ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/martinus/unordered_dense)
add_subdirectory(cloned_dependencies/martinus/unordered_dense SYSTEM)

# https://github.com/Neargye/magic_enum v0.9.6
set(YAE_CLONED_magic_enum ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/Neargye/magic_enum)
add_subdirectory(cloned_dependencies/Neargye/magic_enum SYSTEM)

# https://github.com/TartanLlama/expected v1.1.0
set(YAE_CLONED_expected ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/TartanLlama/expected)
option(EXPECTED_BUILD_TESTS "" OFF)
add_subdirectory(cloned_dependencies/TartanLlama/expected SYSTEM)

# https://github.com/google/benchmark v1.8.3
set(YAE_CLONED_gbench_main ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/google/benchmark/1_8_3)
option(BENCHMARK_ENABLE_TESTING "" OFF)
add_subdirectory(cloned_dependencies/google/benchmark/1_8_3 SYSTEM)

# https://github.com/google/googletest v1.15.2
set(YAE_CLONED_gtest_main ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/google/googletest/1_15_2)
option(INSTALL_GTEST "" OFF)
option(BUILD_GMOCK "" OFF)
add_subdirectory(cloned_dependencies/google/googletest/1_15_2 SYSTEM)

# --------------------------------- Own Modules ----------------------------------
add_subdirectory(src/atomic_cas)
add_subdirectory(cloned_dependencies/Sunday111/imgui_glfw_cmake)
add_subdirectory(cloned_dependencies/Sunday111/klgl/klgl)
add_subdirectory(src/cuda_gl)
add_subdirectory(src/cuda_test)
add_subdirectory(yae/modules/examples/ecs)
add_subdirectory(yae/modules/examples/ecs_benchmark)
add_subdirectory(yae/modules/examples/ecs_tests)
add_subdirectory(yae/modules/examples/yae_example_benchmark)
add_subdirectory(yae/modules/examples/yae_example_benchmark_main)
add_subdirectory(yae/modules/examples/yae_example_library)
add_subdirectory(yae/modules/examples/yae_example_header_only)
add_subdirectory(yae/modules/examples/yae_example_executable)
add_subdirectory(yae/modules/examples/yae_example_tests)
add_subdirectory(yae/modules/examples/yae_example_tests_main)
add_subdirectory(cloned_dependencies/Sunday111/klgl/benchmarks/mat_mul_sse)
add_subdirectory(cloned_dependencies/Sunday111/klgl/benchmarks/rotators)
add_subdirectory(cloned_dependencies/Sunday111/klgl/examples/geometry_shader)
add_subdirectory(cloned_dependencies/Sunday111/klgl/examples/minimal_cube)
add_subdirectory(cloned_dependencies/Sunday111/klgl/examples/minimal_quad)
add_subdirectory(cloned_dependencies/Sunday111/klgl/examples/painter2d)
add_subdirectory(cloned_dependencies/Sunday111/klgl/examples/simple_lit_cube)
add_subdirectory(cloned_dependencies/Sunday111/klgl/examples/textured_quad)
add_subdirectory(cloned_dependencies/Sunday111/klgl/examples/two_textures)
add_subdirectory(cloned_dependencies/Sunday111/klgl/tests)

enable_testing()
