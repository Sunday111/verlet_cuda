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

# https://github.com/Sunday111/EverydayTools master
set(YAE_CLONED_edt ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/Sunday111/EverydayTools)
option(EDT_WITH_TESTS "" ON)
add_subdirectory(cloned_dependencies/Sunday111/EverydayTools SYSTEM)

# https://github.com/Sunday111/CppReflection for_yae
set(YAE_CLONED_CppReflection ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/Sunday111/CppReflection)
add_subdirectory(cloned_dependencies/Sunday111/CppReflection SYSTEM)

# https://github.com/martinus/unordered_dense v4.1.2
set(YAE_CLONED_unordered_dense ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/martinus/unordered_dense)
add_subdirectory(cloned_dependencies/martinus/unordered_dense SYSTEM)

# https://github.com/fmtlib/fmt 10.1.1
set(YAE_CLONED_fmtlib ${CMAKE_CURRENT_SOURCE_DIR}/cloned_dependencies/fmt/10_1_1)
add_subdirectory(cloned_dependencies/fmt/10_1_1 SYSTEM)

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

enable_testing()
